"""Training loop skeleton for self-play-only updates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import time
import argparse

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset

from bridge_ai.data.buffer import ReplayBuffer, Transition
from bridge_ai.data.manifest import append_manifest_entry, compute_config_signature, write_config_snapshot
from bridge_ai.data.replay_store import load_replay_window
from bridge_ai.common.runtime_paths import resolve_runtime_path
from bridge_ai.models.monolithic_transformer import BridgeInputEncoder, BridgeMonolithTransformer, ModelConfig
from bridge_ai.training.checkpoint_store import (
    load_checkpoint_payload,
    resolve_latest_checkpoint,
    resolve_latest_snapshot,
    save_checkpoint_bundle,
)


class TransitionDataset(Dataset):
    def __init__(self, buffer: ReplayBuffer):
        self.buffer = buffer

    def __len__(self) -> int:
        return len(self.buffer._items)

    def __getitem__(self, idx: int):
        item = self.buffer._items[idx]
        return item.state, item.action, item.policy_target, item.value_target


def _tokenize_batch(state_batch):
    encoder = BridgeInputEncoder()
    tensors = [encoder.encode_dict(s, perspective=int(s.get("current_player", 0))) for s in state_batch]
    return torch.cat(tensors, dim=0).to(torch.long)


@dataclass
class TrainConfig:
    epochs: int = 1
    iterations: int = 1
    batch_size: int = 16
    lr: float = 1e-4
    replay_path: str = "replays/latest.json"
    init_checkpoint: str | None = None
    checkpoint_name: str = "latest.pt"
    ckpt_dir: str = "checkpoints"
    checkpoint_every: int = 0
    snapshot_every: int = 0
    save_final_snapshot: bool = True
    device: str = "cpu"
    online_refresh: bool = False
    online_refresh_every: int = 1
    resume_latest: bool = True
    replay_window_shards: int = 4
    replay_window_items: int = 0


def train_one_epoch(
    model: BridgeMonolithTransformer,
    dataset: Dataset,
    opt: torch.optim.Optimizer,
    batch_size: int = 16,
):
    model.train()
    total = 0.0
    total_batches = 0
    for start in range(0, len(dataset), batch_size):
        batch = [dataset[i] for i in range(start, min(len(dataset), start + batch_size))]
        states, _, policy_targets, value_targets = zip(*batch)
        device = next(model.parameters()).device
        states = _tokenize_batch(states).to(device)
        policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=device)
        value_targets = torch.tensor(value_targets, dtype=torch.float32, device=device)
        legal_mask = None
        phase = torch.zeros((states.shape[0],), dtype=torch.long, device=device)
        logits, values = model(states, phase, legal_action_mask=legal_mask)
        policy_denom = policy_targets.sum(dim=-1, keepdim=True).clamp(min=1.0)
        policy_targets_norm = policy_targets / policy_denom
        log_probs = F.log_softmax(logits, dim=-1)
        policy_loss = F.kl_div(log_probs, policy_targets_norm, reduction="batchmean")
        value_loss = F.mse_loss(values, value_targets)
        loss = policy_loss + value_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += float(loss.item())
        total_batches += 1
    return total / max(1, total_batches)


def _maybe_refresh_replay(
    *,
    do_refresh: bool,
    iteration: int,
    refresh_every: int,
    config_path: str,
    model: BridgeMonolithTransformer,
    model_cfg: ModelConfig,
    ckpt_dir: str,
    checkpoint_name: str,
    run_id: str,
    source_checkpoint: str | None,
    parent_checkpoint: str | None,
) -> None:
    if not do_refresh or refresh_every <= 0:
        return
    if iteration % refresh_every != 0:
        return
    from bridge_ai.selfplay.runner import run as run_selfplay

    save_checkpoint_bundle(
        model=model,
        model_cfg=model_cfg,
        ckpt_dir=ckpt_dir,
        checkpoint_name=checkpoint_name,
        iteration=iteration - 1,
        run_id=run_id,
        source_checkpoint=source_checkpoint,
        parent_checkpoint=parent_checkpoint,
        save_snapshot=False,
    )
    run_selfplay(config_path=config_path)


def _load_checkpoint_if_available(model: BridgeMonolithTransformer, path: str | None) -> tuple[int, str | None]:
    """Load model state and return `(last_completed_iteration, immutable_identity)`."""
    if not path:
        return -1, None
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return -1, None
    payload = load_checkpoint_payload(checkpoint_path)
    state_dict = payload.get("state_dict")
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict)
    return int(payload.get("iteration", -1)), payload.get("snapshot_path") or str(checkpoint_path)


def train(config_path: str = "configs/default.yaml") -> None:
    config_source = resolve_runtime_path(config_path)
    cfg = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    train_cfg = TrainConfig(**cfg.get("training", {}))
    manifest_path = str(resolve_runtime_path(cfg.get("storage", {}).get("manifest_path", "artifacts/manifest.json")))
    storage = cfg.get("storage", {})
    default_ckpt_dir = storage.get("checkpoint_dir", train_cfg.ckpt_dir)
    default_replay = storage.get("replay_dir", train_cfg.replay_path)
    default_replay_path = default_replay.rstrip("/") + "/latest.json"
    if train_cfg.replay_path == "replays/latest.json":
        train_cfg.replay_path = default_replay_path
    if train_cfg.ckpt_dir == "checkpoints":
        train_cfg.ckpt_dir = default_ckpt_dir
    replay_root = str(resolve_runtime_path(train_cfg.replay_path).parent)
    train_cfg.replay_path = str(resolve_runtime_path(train_cfg.replay_path))
    train_cfg.ckpt_dir = str(resolve_runtime_path(train_cfg.ckpt_dir))
    model_cfg = ModelConfig(**cfg.get("model", {}))
    model = BridgeMonolithTransformer(model_cfg).to(train_cfg.device)
    run_id = f"train_{int(time.time() * 1_000_000)}"
    resolved_init_checkpoint = str(resolve_runtime_path(train_cfg.init_checkpoint)) if train_cfg.init_checkpoint else None
    if not resolved_init_checkpoint and train_cfg.resume_latest:
        resolved_init_checkpoint = resolve_latest_checkpoint(train_cfg.ckpt_dir, checkpoint_name=train_cfg.checkpoint_name)
    last_completed_iteration, parent_checkpoint = _load_checkpoint_if_available(model, resolved_init_checkpoint)
    start_iteration = last_completed_iteration + 1
    run_signature = compute_config_signature(
        str(config_source),
        extra={
            "epochs": train_cfg.epochs,
            "iterations": train_cfg.iterations,
            "batch_size": train_cfg.batch_size,
            "init_checkpoint": resolved_init_checkpoint,
        },
    )
    config_snapshot = write_config_snapshot(str(config_source), manifest_path, run_id)
    buffer, replay_records = load_replay_window(
        replay_dir=replay_root,
        latest_path=train_cfg.replay_path,
        max_shards=train_cfg.replay_window_shards,
        max_items=train_cfg.replay_window_items,
    )
    dataset = TransitionDataset(buffer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    losses = []
    last_loss = 0.0
    iterations_to_run = max(1, train_cfg.iterations)
    last_iteration = last_completed_iteration
    latest_save_paths: dict[str, str | None] = {"latest_path": None, "snapshot_path": None}
    last_snapshot_iteration: int | None = None
    for iteration in range(start_iteration, start_iteration + iterations_to_run):
        _maybe_refresh_replay(
            do_refresh=train_cfg.online_refresh,
            iteration=iteration,
            refresh_every=train_cfg.online_refresh_every,
            config_path=str(config_source),
            model=model,
            model_cfg=model_cfg,
            ckpt_dir=train_cfg.ckpt_dir,
            checkpoint_name=train_cfg.checkpoint_name,
            run_id=run_id,
            source_checkpoint=resolved_init_checkpoint,
            parent_checkpoint=parent_checkpoint,
        )

        buffer, replay_records = load_replay_window(
            replay_dir=replay_root,
            latest_path=train_cfg.replay_path,
            max_shards=train_cfg.replay_window_shards,
            max_items=train_cfg.replay_window_items,
        )
        dataset = TransitionDataset(buffer)

        for epoch in range(train_cfg.epochs):
            loss = train_one_epoch(model, dataset, optimizer, batch_size=train_cfg.batch_size)
            last_loss = loss
            losses.append(float(loss))
            print({"iteration": iteration, "epoch": epoch, "loss": loss})
        last_iteration = iteration
        if train_cfg.checkpoint_every > 0 and (iteration + 1) % train_cfg.checkpoint_every == 0:
            save_snapshot = False
            if train_cfg.snapshot_every > 0:
                save_snapshot = (iteration + 1) % train_cfg.snapshot_every == 0
            elif train_cfg.checkpoint_every > 0:
                save_snapshot = True
            latest_save_paths = save_checkpoint_bundle(
                model=model,
                model_cfg=model_cfg,
                ckpt_dir=train_cfg.ckpt_dir,
                checkpoint_name=train_cfg.checkpoint_name,
                iteration=iteration,
                run_id=run_id,
                source_checkpoint=resolved_init_checkpoint,
                parent_checkpoint=parent_checkpoint,
                save_snapshot=save_snapshot,
            )
            if save_snapshot:
                last_snapshot_iteration = iteration
        if len(dataset) == 0:
            break

    if last_iteration < 0:
        last_iteration = 0
    save_final_snapshot = train_cfg.save_final_snapshot and last_snapshot_iteration != last_iteration
    latest_save_paths = save_checkpoint_bundle(
        model=model,
        model_cfg=model_cfg,
        ckpt_dir=train_cfg.ckpt_dir,
        checkpoint_name=train_cfg.checkpoint_name,
        iteration=last_iteration,
        run_id=run_id,
        source_checkpoint=resolved_init_checkpoint,
        parent_checkpoint=parent_checkpoint,
        save_snapshot=save_final_snapshot,
        snapshot_tag=None,
    )
    current_snapshot = latest_save_paths.get("snapshot_path") or resolve_latest_snapshot(train_cfg.ckpt_dir)
    append_manifest_entry(
        manifest_path,
        run_type="train",
        run_id=run_id,
        config_path=str(config_source),
        run_signature=run_signature,
        config_snapshot=config_snapshot,
        outputs={
            "checkpoint": latest_save_paths.get("latest_path"),
            "snapshot_checkpoint": current_snapshot,
            "epochs": train_cfg.epochs,
            "items": len(buffer),
            "replay_shards": [record.path for record in replay_records],
            "init_checkpoint": resolved_init_checkpoint,
        },
        metrics={
            "loss": last_loss,
            "loss_history": losses,
            "iterations": train_cfg.iterations,
            "start_iteration": start_iteration,
            "end_iteration": last_iteration,
        },
        status="ok",
    )


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Train monolithic bridge model from replay data.")
    parser.add_argument("--config-path", "--config_path", default="configs/default.yaml")
    args = parser.parse_args()
    train(config_path=args.config_path)


if __name__ == "__main__":
    main()

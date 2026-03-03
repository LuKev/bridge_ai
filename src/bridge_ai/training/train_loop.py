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
from bridge_ai.models.monolithic_transformer import BridgeInputEncoder, BridgeMonolithTransformer, ModelConfig


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
    device: str = "cpu"
    online_refresh: bool = False
    online_refresh_every: int = 1


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
) -> None:
    if not do_refresh or refresh_every <= 0:
        return
    if iteration % refresh_every != 0:
        return
    from bridge_ai.selfplay.runner import run as run_selfplay

    run_selfplay(config_path=config_path)


def _load_checkpoint_if_available(model: BridgeMonolithTransformer, path: str | None) -> int:
    """Load model state and return the iteration this checkpoint represents."""
    if not path:
        return 0
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return 0
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload.get("state_dict")
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict)
    return int(payload.get("iteration", 0))


def _save_checkpoint(
    model: BridgeMonolithTransformer,
    model_cfg: ModelConfig,
    ckpt_dir: str,
    checkpoint_name: str,
    iteration: int,
) -> None:
    ckpt = Path(ckpt_dir)
    ckpt.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt / checkpoint_name
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model_cfg.__dict__,
            "iteration": iteration,
        },
        ckpt_file,
    )


def train(config_path: str = "configs/default.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    train_cfg = TrainConfig(**cfg.get("training", {}))
    manifest_path = cfg.get("storage", {}).get("manifest_path", "artifacts/manifest.json")
    storage = cfg.get("storage", {})
    default_ckpt_dir = storage.get("checkpoint_dir", train_cfg.ckpt_dir)
    default_replay = storage.get("replay_dir", train_cfg.replay_path)
    default_replay_path = default_replay.rstrip("/") + "/latest.json"
    if train_cfg.replay_path == "replays/latest.json":
        train_cfg.replay_path = default_replay_path
    if train_cfg.ckpt_dir == "checkpoints":
        train_cfg.ckpt_dir = default_ckpt_dir
    model_cfg = ModelConfig(**cfg.get("model", {}))
    model = BridgeMonolithTransformer(model_cfg).to(train_cfg.device)
    start_iteration = _load_checkpoint_if_available(model, train_cfg.init_checkpoint)
    run_id = f"train_{int(time.time() * 1_000_000)}"
    run_signature = compute_config_signature(
        config_path,
        extra={"epochs": train_cfg.epochs, "iterations": train_cfg.iterations, "batch_size": train_cfg.batch_size},
    )
    config_snapshot = write_config_snapshot(config_path, manifest_path, run_id)
    buffer = ReplayBuffer.load_json(Path(train_cfg.replay_path))
    dataset = TransitionDataset(buffer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)

    losses = []
    last_loss = 0.0
    target_iterations = max(1, train_cfg.iterations)
    last_iteration = start_iteration - 1
    for iteration in range(start_iteration, target_iterations):
        _maybe_refresh_replay(
            do_refresh=train_cfg.online_refresh,
            iteration=iteration,
            refresh_every=train_cfg.online_refresh_every,
            config_path=config_path,
        )

        if not Path(train_cfg.replay_path).exists():
            raise RuntimeError(f"replay file does not exist: {train_cfg.replay_path}")

        buffer = ReplayBuffer.load_json(Path(train_cfg.replay_path))
        dataset = TransitionDataset(buffer)

        for epoch in range(train_cfg.epochs):
            loss = train_one_epoch(model, dataset, optimizer, batch_size=train_cfg.batch_size)
            last_loss = loss
            losses.append(float(loss))
            print({"iteration": iteration, "epoch": epoch, "loss": loss})
        last_iteration = iteration
        if train_cfg.checkpoint_every > 0 and (iteration + 1) % train_cfg.checkpoint_every == 0:
            _save_checkpoint(
                model,
                model_cfg,
                train_cfg.ckpt_dir,
                train_cfg.checkpoint_name,
                iteration=iteration,
            )
        if len(dataset) == 0:
            break

    if last_iteration < 0:
        last_iteration = 0
    checkpoint_path = str(Path(train_cfg.ckpt_dir).resolve() / train_cfg.checkpoint_name)
    _save_checkpoint(
        model,
        model_cfg,
        train_cfg.ckpt_dir,
        train_cfg.checkpoint_name,
        iteration=last_iteration,
    )
    append_manifest_entry(
        manifest_path,
        run_type="train",
        run_id=run_id,
        config_path=config_path,
        run_signature=run_signature,
        config_snapshot=config_snapshot,
        outputs={"checkpoint": checkpoint_path, "epochs": train_cfg.epochs, "items": len(buffer)},
        metrics={
            "loss": last_loss,
            "loss_history": losses,
            "iterations": train_cfg.iterations,
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

"""Training loop for the bidding-plus-belief transformer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import json
import time
from typing import List

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset

from bridge_ai.data.belief_dataset import BeliefExample, load_examples
from bridge_ai.data.manifest import append_manifest_entry, compute_config_signature, write_config_snapshot
from bridge_ai.eval.evaluator import EvalConfig, evaluate_model
from bridge_ai.infra.plots import write_accuracy_svg
from bridge_ai.models.bidding_belief_transformer import (
    BiddingBeliefConfig,
    BiddingBeliefEncoder,
    BiddingBeliefTransformer,
)


class BeliefExampleDataset(Dataset):
    def __init__(self, examples: List[BeliefExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> BeliefExample:
        return self.examples[idx]


@dataclass
class TrainConfig:
    epochs: int = 2
    iterations: int = 1
    batch_size: int = 32
    lr: float = 3e-4
    replay_path: str = "replays/latest.json"
    init_checkpoint: str | None = None
    checkpoint_name: str = "latest.pt"
    ckpt_dir: str = "checkpoints"
    device: str = "cpu"
    eval_every_epochs: int = 1
    holdout_max_examples: int = 256
    holdout_sample_count: int = 1


def _load_checkpoint_if_available(model: BiddingBeliefTransformer, path: str | None) -> int:
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
    model: BiddingBeliefTransformer,
    config: BiddingBeliefConfig,
    ckpt_dir: str,
    checkpoint_name: str,
    iteration: int,
) -> str:
    destination = Path(ckpt_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / checkpoint_name
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config.__dict__,
            "iteration": iteration,
        },
        checkpoint_path,
    )
    return str(checkpoint_path)


def _batch_tensors(
    examples: List[BeliefExample],
    encoder: BiddingBeliefEncoder,
    device: torch.device,
    action_vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    token_ids = torch.stack([encoder.encode_example(example) for example in examples]).to(device)
    legal_masks = torch.stack(
        [encoder.action_mask(example.legal_actions, action_vocab_size=action_vocab_size) for example in examples]
    ).to(device)
    bid_targets = torch.tensor([example.target_bid for example in examples], dtype=torch.long, device=device)
    bid_target_mask = torch.tensor([bool(example.has_bid_target) for example in examples], dtype=torch.bool, device=device)
    owner_targets = torch.tensor([example.card_owners for example in examples], dtype=torch.long, device=device)
    owner_mask = torch.tensor([example.belief_target_mask for example in examples], dtype=torch.bool, device=device)
    return token_ids, legal_masks, bid_targets, bid_target_mask, owner_targets, owner_mask


def train_one_epoch(
    model: BiddingBeliefTransformer,
    dataset: BeliefExampleDataset,
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
) -> dict:
    model.train()
    encoder = model.encoder
    device = next(model.parameters()).device
    total_loss = 0.0
    total_bid_loss = 0.0
    total_owner_loss = 0.0
    total_batches = 0
    for start in range(0, len(dataset), batch_size):
        examples = [dataset[idx] for idx in range(start, min(len(dataset), start + batch_size))]
        token_ids, legal_masks, bid_targets, bid_target_mask, owner_targets, owner_mask = _batch_tensors(
            examples,
            encoder,
            device,
            model.config.action_vocab_size,
        )
        bid_logits, owner_logits = model(token_ids, legal_action_mask=legal_masks)
        if bid_target_mask.any():
            bid_loss = F.cross_entropy(bid_logits[bid_target_mask], bid_targets[bid_target_mask])
        else:
            bid_loss = torch.tensor(0.0, device=device)

        masked_owner_logits = owner_logits[owner_mask]
        masked_owner_targets = owner_targets[owner_mask]
        if masked_owner_logits.numel() == 0:
            owner_loss = torch.tensor(0.0, device=device)
        else:
            owner_loss = F.cross_entropy(masked_owner_logits, masked_owner_targets)
        loss = bid_loss + owner_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_bid_loss += float(bid_loss.item())
        total_owner_loss += float(owner_loss.item())
        total_batches += 1
    return {
        "loss": total_loss / max(1, total_batches),
        "bid_loss": total_bid_loss / max(1, total_batches),
        "belief_loss": total_owner_loss / max(1, total_batches),
    }


def train(config_path: str = "configs/default.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    train_cfg = TrainConfig(**cfg.get("training", {}))
    manifest_path = cfg.get("storage", {}).get("manifest_path", "artifacts/manifest.json")
    storage = cfg.get("storage", {})
    if train_cfg.replay_path == "replays/latest.json":
        train_cfg.replay_path = f"{storage.get('replay_dir', 'replays').rstrip('/')}/latest.json"
    if train_cfg.ckpt_dir == "checkpoints":
        train_cfg.ckpt_dir = storage.get("checkpoint_dir", "checkpoints")

    model_cfg = BiddingBeliefConfig(**cfg.get("model", {}))
    model = BiddingBeliefTransformer(model_cfg).to(train_cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr)
    start_iteration = _load_checkpoint_if_available(model, train_cfg.init_checkpoint)

    examples = load_examples(train_cfg.replay_path, split="train")
    if not examples:
        raise RuntimeError(f"training dataset is empty: {train_cfg.replay_path}")
    dataset = BeliefExampleDataset(examples)
    holdout_examples = load_examples(train_cfg.replay_path, split="holdout")

    run_id = f"train_{int(time.time() * 1_000_000)}"
    run_signature = compute_config_signature(
        config_path,
        extra={"epochs": train_cfg.epochs, "iterations": train_cfg.iterations, "batch_size": train_cfg.batch_size},
    )
    config_snapshot = write_config_snapshot(config_path, manifest_path, run_id)

    history = []
    last_metrics = {"loss": 0.0, "bid_loss": 0.0, "belief_loss": 0.0}
    last_iteration = start_iteration
    eval_cfg = EvalConfig(
        replay_path=train_cfg.replay_path,
        max_examples=train_cfg.holdout_max_examples,
        preview_examples=0,
        sample_count=train_cfg.holdout_sample_count,
    )
    if holdout_examples:
        initial_eval = evaluate_model(model, holdout_examples, eval_cfg)
        history.append(
            {
                "epoch": 0,
                "iteration": start_iteration,
                "train_loss": None,
                "train_bid_loss": None,
                "train_belief_loss": None,
                "holdout_bid_accuracy": initial_eval["bid_accuracy"],
                "holdout_belief_accuracy": initial_eval["belief_accuracy"],
                "holdout_auction_belief_accuracy": initial_eval.get("auction_belief_accuracy"),
                "holdout_play_belief_accuracy": initial_eval.get("play_belief_accuracy"),
                "holdout_bid_loss": initial_eval["bid_loss"],
                "holdout_belief_loss": initial_eval["belief_loss"],
            }
        )
    global_epoch = 0
    for iteration in range(start_iteration, max(1, train_cfg.iterations)):
        for _ in range(train_cfg.epochs):
            last_metrics = train_one_epoch(
                model,
                dataset,
                optimizer,
                batch_size=train_cfg.batch_size,
            )
            global_epoch += 1
            row = {
                "epoch": global_epoch,
                "iteration": iteration,
                "train_loss": last_metrics["loss"],
                "train_bid_loss": last_metrics["bid_loss"],
                "train_belief_loss": last_metrics["belief_loss"],
            }
            if holdout_examples and train_cfg.eval_every_epochs > 0 and global_epoch % train_cfg.eval_every_epochs == 0:
                holdout_eval = evaluate_model(model, holdout_examples, eval_cfg)
                row.update(
                    {
                        "holdout_bid_accuracy": holdout_eval["bid_accuracy"],
                        "holdout_belief_accuracy": holdout_eval["belief_accuracy"],
                        "holdout_auction_belief_accuracy": holdout_eval.get("auction_belief_accuracy"),
                        "holdout_play_belief_accuracy": holdout_eval.get("play_belief_accuracy"),
                        "holdout_bid_loss": holdout_eval["bid_loss"],
                        "holdout_belief_loss": holdout_eval["belief_loss"],
                    }
                )
            history.append(row)
        last_iteration = iteration

    checkpoint_path = _save_checkpoint(
        model,
        model_cfg,
        train_cfg.ckpt_dir,
        train_cfg.checkpoint_name,
        iteration=last_iteration,
    )
    artifacts_dir = Path(storage.get("artifacts_dir", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    history_path = artifacts_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_path = artifacts_dir / "accuracy_curves.svg"
    write_accuracy_svg(history, plot_path)
    append_manifest_entry(
        manifest_path,
        run_type="train",
        run_id=run_id,
        config_path=config_path,
        run_signature=run_signature,
        config_snapshot=config_snapshot,
        outputs={
            "checkpoint": checkpoint_path,
            "items": len(dataset),
            "history_path": str(history_path),
            "plot_path": str(plot_path),
        },
        metrics={
            "loss": last_metrics["loss"],
            "bid_loss": last_metrics["bid_loss"],
            "belief_loss": last_metrics["belief_loss"],
            "loss_history": history,
        },
        status="ok",
    )


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Train bidding/belief model.")
    parser.add_argument("--config-path", "--config_path", default="configs/default.yaml")
    args = parser.parse_args()
    train(config_path=args.config_path)


if __name__ == "__main__":
    main()

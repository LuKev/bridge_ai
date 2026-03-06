"""Evaluation harness for bidding and hidden-hand belief checkpoints."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import pstdev
import time
from typing import Dict, List

import torch
import torch.nn.functional as F
import yaml

from bridge_ai.common.actions import bid_to_string
from bridge_ai.data.belief_dataset import BeliefExample, load_examples
from bridge_ai.data.manifest import append_manifest_entry, compute_config_signature, write_config_snapshot
from bridge_ai.inference.posterior_sampler import sample_hidden_deal
from bridge_ai.models.bidding_belief_transformer import (
    BiddingBeliefConfig,
    BiddingBeliefTransformer,
)


@dataclass
class EvalConfig:
    checkpoint: str = "checkpoints/latest.pt"
    replay_path: str = "replays/latest.json"
    max_examples: int = 64
    preview_examples: int = 3
    sample_count: int = 4


def _load_checkpoint(model_cfg: BiddingBeliefConfig, checkpoint: str) -> BiddingBeliefTransformer:
    model = BiddingBeliefTransformer(model_cfg)
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = payload.get("state_dict")
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
    return model


def _owner_entropy(owner_logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(owner_logits, dim=-1)
    return -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1)


def _preview_for_example(example: BeliefExample, bid_probs: torch.Tensor, owner_probs: torch.Tensor) -> Dict[str, object]:
    top_indices = (
        torch.topk(bid_probs, k=min(5, bid_probs.shape[-1])).indices.tolist()
        if example.has_bid_target
        else []
    )
    preview_cards = []
    for card_index in range(52):
        if not example.belief_target_mask[card_index]:
            continue
        probs = owner_probs[card_index]
        preview_cards.append(
            {
                "card_index": card_index,
                "true_owner": int(example.card_owners[card_index]),
                "top_owner": int(torch.argmax(probs).item()),
                "top_owner_prob": float(torch.max(probs).item()),
            }
        )
        if len(preview_cards) >= 6:
            break
    return {
        "record_id": example.record_id,
        "bid_index": example.bid_index,
        "phase": example.phase,
        "played_count": example.played_count,
        "target_bid": bid_to_string(int(example.target_bid)) if example.has_bid_target else None,
        "top_bids": [bid_to_string(int(idx)) for idx in top_indices] if example.has_bid_target else [],
        "top_bid_probs": [float(bid_probs[idx].item()) for idx in top_indices] if example.has_bid_target else [],
        "belief_preview_cards": preview_cards,
    }


def evaluate_model(model: BiddingBeliefTransformer, examples: List[BeliefExample], cfg: EvalConfig) -> Dict[str, object]:
    model.eval()
    device = next(model.parameters()).device
    examples = examples[: cfg.max_examples]
    if not examples:
        return {
            "examples": 0.0,
            "bid_accuracy": 0.0,
            "bid_loss": 0.0,
            "belief_accuracy": 0.0,
            "belief_loss": 0.0,
            "avg_true_owner_prob": 0.0,
            "avg_owner_entropy": 0.0,
            "sampler_validity_rate": 0.0,
            "preview": [],
        }

    bid_hits = 0
    bid_total = 0
    bid_losses: List[float] = []
    owner_hits = 0
    owner_total = 0
    owner_losses: List[float] = []
    auction_owner_hits = 0
    auction_owner_total = 0
    play_owner_hits = 0
    play_owner_total = 0
    true_owner_probs: List[float] = []
    entropies: List[float] = []
    play_progress_hits: Dict[int, int] = {}
    play_progress_total: Dict[int, int] = {}
    preview: List[Dict[str, object]] = []
    valid_samples = 0
    sample_attempts = 0

    with torch.no_grad():
        for example_idx, example in enumerate(examples):
            token_ids = model.encoder.encode_example(example).unsqueeze(0).to(device)
            legal_mask = model.encoder.action_mask(
                example.legal_actions,
                action_vocab_size=model.config.action_vocab_size,
            ).unsqueeze(0).to(device)
            bid_logits, owner_logits = model(token_ids, legal_action_mask=legal_mask)
            bid_probs = torch.softmax(bid_logits.squeeze(0), dim=-1)
            owner_probs = torch.softmax(owner_logits.squeeze(0), dim=-1)

            if example.has_bid_target:
                bid_target = torch.tensor([example.target_bid], dtype=torch.long, device=device)
                bid_losses.append(float(F.cross_entropy(bid_logits, bid_target).item()))
                bid_prediction = int(torch.argmax(bid_probs).item())
                bid_hits += int(bid_prediction == int(example.target_bid))
                bid_total += 1

            owner_target = torch.tensor(example.card_owners, dtype=torch.long, device=device)
            owner_mask = torch.tensor(example.belief_target_mask, dtype=torch.bool, device=device)
            masked_probs = owner_probs[owner_mask]
            masked_targets = owner_target[owner_mask]
            if masked_probs.numel() > 0:
                masked_logits = owner_logits.squeeze(0)[owner_mask]
                owner_losses.append(float(F.cross_entropy(masked_logits, masked_targets).item()))
                predictions = torch.argmax(masked_probs, dim=-1)
                hits = int((predictions == masked_targets).sum().item())
                total = int(masked_targets.numel())
                owner_hits += hits
                owner_total += total
                if example.phase == "auction":
                    auction_owner_hits += hits
                    auction_owner_total += total
                elif example.phase == "play":
                    play_owner_hits += hits
                    play_owner_total += total
                    progress = int(example.played_count)
                    play_progress_hits[progress] = play_progress_hits.get(progress, 0) + hits
                    play_progress_total[progress] = play_progress_total.get(progress, 0) + total
                true_owner_probs.extend(masked_probs[torch.arange(masked_targets.numel()), masked_targets].tolist())
                entropies.extend(_owner_entropy(masked_logits).tolist())

            if len(preview) < cfg.preview_examples:
                preview.append(_preview_for_example(example, bid_probs, owner_probs))

            for sample_idx in range(cfg.sample_count):
                sampled = sample_hidden_deal(example, owner_logits.squeeze(0), seed=example_idx * 100 + sample_idx)
                sample_attempts += 1
                if sampled.valid:
                    valid_samples += 1

    return {
        "examples": float(len(examples)),
        "bid_examples": float(bid_total),
        "bid_accuracy": float(bid_hits / max(1, bid_total)),
        "bid_loss": float(sum(bid_losses) / max(1, len(bid_losses))),
        "belief_accuracy": float(owner_hits / max(1, owner_total)),
        "auction_belief_accuracy": float(auction_owner_hits / max(1, auction_owner_total)),
        "play_belief_accuracy": float(play_owner_hits / max(1, play_owner_total)),
        "belief_loss": float(sum(owner_losses) / max(1, len(owner_losses))),
        "avg_true_owner_prob": float(sum(true_owner_probs) / max(1, len(true_owner_probs))),
        "avg_owner_entropy": float(sum(entropies) / max(1, len(entropies))),
        "owner_entropy_std": float(pstdev(entropies) if len(entropies) > 1 else 0.0),
        "sampler_validity_rate": float(valid_samples / max(1, sample_attempts)),
        "play_belief_accuracy_by_played_count": {
            str(progress): play_progress_hits[progress] / max(1, play_progress_total[progress])
            for progress in sorted(play_progress_hits)
        },
        "preview": preview,
    }


def run(config_path: str = "configs/default.yaml") -> Dict[str, object]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    eval_cfg = EvalConfig(**cfg.get("evaluation", {}))
    manifest_path = cfg.get("storage", {}).get("manifest_path", "artifacts/manifest.json")
    storage = cfg.get("storage", {})
    if eval_cfg.checkpoint == "checkpoints/latest.pt":
        eval_cfg.checkpoint = f"{storage.get('checkpoint_dir', 'checkpoints').rstrip('/')}/latest.pt"
    if eval_cfg.replay_path == "replays/latest.json":
        eval_cfg.replay_path = f"{storage.get('replay_dir', 'replays').rstrip('/')}/latest.json"

    model_cfg = BiddingBeliefConfig(**cfg.get("model", {}))
    model = _load_checkpoint(model_cfg, eval_cfg.checkpoint)
    examples = load_examples(eval_cfg.replay_path, split="holdout")
    if not examples:
        examples = load_examples(eval_cfg.replay_path, split="train")
    results = evaluate_model(model, examples, eval_cfg)

    run_id = f"eval_{int(time.time() * 1_000_000)}"
    run_signature = compute_config_signature(
        config_path,
        extra={"checkpoint": eval_cfg.checkpoint, "max_examples": eval_cfg.max_examples},
    )
    config_snapshot = write_config_snapshot(config_path, manifest_path, run_id)

    artifacts_dir = Path(storage.get("artifacts_dir", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    preview_path = artifacts_dir / "belief_eval_preview.json"
    preview_path.write_text(json.dumps(results["preview"], indent=2), encoding="utf-8")

    append_manifest_entry(
        manifest_path,
        run_type="eval",
        run_id=run_id,
        config_path=config_path,
        run_signature=run_signature,
        config_snapshot=config_snapshot,
        outputs={"checkpoint": eval_cfg.checkpoint, "preview_path": str(preview_path)},
        metrics={k: v for k, v in results.items() if k != "preview"},
        status="ok",
    )
    return results


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Evaluate bidding/belief checkpoint.")
    parser.add_argument("--config-path", "--config_path", default="configs/default.yaml")
    args = parser.parse_args()
    print(run(config_path=args.config_path))


if __name__ == "__main__":
    main()

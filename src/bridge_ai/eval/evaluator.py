"""Evaluation harness for checkpoint comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import pstdev
from typing import Dict, List
import time
import argparse

import yaml
import torch

from bridge_ai.env.bridge_env import BridgeEnv
from bridge_ai.models.monolithic_transformer import BridgeInputEncoder, BridgeMonolithTransformer, ModelConfig
from bridge_ai.search.ismcts import ISMCTS, ISMCTSConfig
from bridge_ai.data.manifest import append_manifest_entry, compute_config_signature, write_config_snapshot


@dataclass
class EvalConfig:
    rounds: int = 20
    seed: int = 0
    seed_sequence: list[int] | None = None
    checkpoint: str = "checkpoints/latest.pt"
    use_search: bool = True
    search_simulations: int = 12
    rollout_depth: int = 16
    num_determinizations: int = 4
    baseline_checkpoint: str | None = None
    seed_steps: int = 200


def evaluate_model(
    model_cfg: ModelConfig,
    checkpoint: str,
    rounds: int,
    seed: int,
    seed_sequence: list[int] | None = None,
    use_search: bool = True,
    search_simulations: int = 12,
    rollout_depth: int = 16,
    num_determinizations: int = 4,
) -> Dict[str, float]:
    checkpoint_path = Path(checkpoint)
    model = BridgeMonolithTransformer(model_cfg)
    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(payload["state_dict"])
    return evaluate(
        model,
        rounds=rounds,
        seed=seed,
        seed_sequence=seed_sequence,
        use_search=use_search,
        search_simulations=search_simulations,
        rollout_depth=rollout_depth,
        num_determinizations=num_determinizations,
    )


def evaluate(
    model: BridgeMonolithTransformer,
    rounds: int = 20,
    seed: int = 0,
    seed_sequence: list[int] | None = None,
    use_search: bool = True,
    search_simulations: int = 12,
    rollout_depth: int = 16,
    num_determinizations: int = 4,
) -> Dict[str, float]:
    env = BridgeEnv(seed=seed)
    search = ISMCTS(
        config=ISMCTSConfig(
            num_simulations=search_simulations,
            rollout_depth=rollout_depth,
            num_determinizations=num_determinizations,
        )
    ) if use_search else None
    encoder = BridgeInputEncoder()
    scores: List[float] = []

    if seed_sequence:
        if len(seed_sequence) < rounds:
            seeds = seed_sequence + [seed + i for i in range(rounds - len(seed_sequence))]
        else:
            seeds = seed_sequence[:rounds]
    else:
        seeds = [seed + i for i in range(rounds)]

    for s in seeds[:rounds]:
        state = env.reset(seed=s)
        total = 0.0
        for _ in range(260):
            legal = env.legal_actions()
            if not legal:
                break
            legal_actions = [int(a) for a in legal]
            token_ids = encoder.encode(state, perspective=state.current_player)
            legal_mask = torch.zeros(model.config.action_vocab_size, dtype=torch.bool)
            for a in legal_actions:
                if 0 <= a < model.config.action_vocab_size:
                    legal_mask[a] = True
            legal_mask = legal_mask.unsqueeze(0)
            logits, value = model(token_ids, torch.tensor([0], dtype=torch.long), legal_action_mask=legal_mask)
            if search is None:
                action_mask = legal_mask.squeeze(0)
                legal_logits = logits.masked_fill(~action_mask, float("-inf"))
                action = int(torch.argmax(legal_logits, dim=-1).item())
            else:
                result = search.select_action(state, logits, legal_actions=legal_actions, model=model)
                action = int(result.action)
                if action not in legal_actions:
                    action = int(legal_actions[0])
            if action not in legal_actions:
                action = int(legal_actions[0])
            _, reward, done, _ = env.step(action)
            total += reward
            if done:
                break
        scores.append(total)
    if not scores:
        scores = [0.0]
    return {
        "rounds": float(rounds),
        "mean_score": float(sum(scores) / max(1, len(scores))),
        "win_rate_vs_zero": float(sum(s > 0.0 for s in scores) / max(1, len(scores))),
        "score_std": float(pstdev(scores) if len(scores) > 1 else 0.0),
        "score_min": float(min(scores)),
        "score_max": float(max(scores)),
    }


def run(config_path: str = "configs/default.yaml") -> Dict[str, float]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    eval_cfg = EvalConfig(**cfg.get("evaluation", {}))
    manifest_path = cfg.get("storage", {}).get("manifest_path", "artifacts/manifest.json")
    model_cfg = ModelConfig(**cfg.get("model", {}))
    checkpoint_dir = cfg.get("storage", {}).get("checkpoint_dir", "checkpoints")
    if eval_cfg.checkpoint == "checkpoints/latest.pt":
        eval_cfg.checkpoint = f"{checkpoint_dir.rstrip('/')}/latest.pt"
    if eval_cfg.baseline_checkpoint == "checkpoints/latest.pt":
        eval_cfg.baseline_checkpoint = f"{checkpoint_dir.rstrip('/')}/latest.pt"
    run_id = f"eval_{int(time.time() * 1_000_000)}"
    run_signature = compute_config_signature(
        config_path,
        extra={"rounds": eval_cfg.rounds, "seed": eval_cfg.seed, "seed_sequence": eval_cfg.seed_sequence},
    )
    config_snapshot = write_config_snapshot(config_path, manifest_path, run_id)
    primary = evaluate_model(
        model_cfg,
        eval_cfg.checkpoint,
        eval_cfg.rounds,
        eval_cfg.seed,
        seed_sequence=eval_cfg.seed_sequence,
        use_search=eval_cfg.use_search,
        search_simulations=eval_cfg.search_simulations,
        rollout_depth=eval_cfg.rollout_depth,
        num_determinizations=eval_cfg.num_determinizations,
    )
    if eval_cfg.seed_sequence:
        primary["seed_sequence"] = eval_cfg.seed_sequence
    result = dict(primary)
    if eval_cfg.baseline_checkpoint:
        baseline = evaluate_model(
            model_cfg,
            eval_cfg.baseline_checkpoint,
            eval_cfg.rounds,
            eval_cfg.seed,
            seed_sequence=eval_cfg.seed_sequence,
            use_search=eval_cfg.use_search,
            search_simulations=eval_cfg.search_simulations,
            rollout_depth=eval_cfg.rollout_depth,
            num_determinizations=eval_cfg.num_determinizations,
        )
        result["baseline_rounds"] = baseline["rounds"]
        result["baseline_mean_score"] = baseline["mean_score"]
        result["baseline_win_rate_vs_zero"] = baseline["win_rate_vs_zero"]
        result["delta_vs_baseline"] = result["mean_score"] - baseline["mean_score"]
        result["checkpoint_baseline"] = eval_cfg.baseline_checkpoint
        result["baseline_seed_sequence"] = baseline.get("seed_sequence", None)
    checkpoint = Path(eval_cfg.checkpoint)
    append_manifest_entry(
        manifest_path,
        run_type="eval",
        run_id=run_id,
        config_path=config_path,
        run_signature=run_signature,
        config_snapshot=config_snapshot,
        outputs={"checkpoint": str(checkpoint), "rounds": eval_cfg.rounds},
        metrics=result,
        status="ok",
    )
    return result


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Evaluate bridge checkpoint(s).")
    parser.add_argument("--config-path", "--config_path", default="configs/default.yaml")
    args = parser.parse_args()
    print(run(config_path=args.config_path))


if __name__ == "__main__":
    main()

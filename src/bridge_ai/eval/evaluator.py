"""Evaluation harness for checkpoint comparison and rating updates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import pstdev
from typing import Dict, List, Sequence
import argparse
import json
import time

import torch
import yaml

from bridge_ai.common.types import Phase, Seat
from bridge_ai.data.manifest import append_manifest_entry, compute_config_signature, write_config_snapshot
from bridge_ai.env.bridge_env import BridgeEnv
from bridge_ai.eval.benchmark import resolve_benchmark_suite
from bridge_ai.eval.ratings import (
    DEFAULT_ELO,
    append_rating_history,
    load_rating_table,
    score_from_diff,
    update_elo,
    write_rating_table,
)
from bridge_ai.models.monolithic_transformer import BridgeInputEncoder, BridgeMonolithTransformer, ModelConfig
from bridge_ai.search.ismcts import ISMCTS, ISMCTSConfig
from bridge_ai.common.runtime_paths import resolve_runtime_path
from bridge_ai.training.checkpoint_store import (
    load_checkpoint_index,
    resolve_anchor_snapshot,
    resolve_checkpoint_identity,
    resolve_latest_checkpoint,
    resolve_previous_snapshot,
)


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
    mode: str = "auto"
    suite_name: str | None = None
    opponent_pool_size: int = 3
    include_anchor: bool = True
    include_previous_snapshot: bool = True
    duplicate_draw_margin: float = 0.0
    rating_k_factor: float = 24.0


def _phase_id(phase: Phase) -> int:
    return {
        Phase.AUCTION: 0,
        Phase.LEAD: 1,
        Phase.PLAY: 2,
        Phase.DEFENSE: 3,
    }.get(phase, 0)


def _load_model(model_cfg: ModelConfig, checkpoint: str, cache: dict[str, BridgeMonolithTransformer]) -> BridgeMonolithTransformer:
    key = str(checkpoint)
    if key in cache:
        return cache[key]

    model = BridgeMonolithTransformer(model_cfg)
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = payload.get("state_dict")
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
    model.eval()
    cache[key] = model
    return model


def _policy_action(
    *,
    model: BridgeMonolithTransformer,
    env: BridgeEnv,
    use_search: bool,
    search: ISMCTS | None,
    encoder: BridgeInputEncoder,
) -> int:
    state = env.state
    if state is None:
        raise RuntimeError("environment has no state")
    legal = tuple(int(action) for action in env.legal_actions())
    if not legal:
        raise RuntimeError("no legal actions available")
    legal_mask = encoder.action_mask(legal, action_vocab_size=model.config.action_vocab_size).unsqueeze(0)
    token_ids = encoder.encode(state, perspective=state.current_player)
    phase = torch.tensor([_phase_id(state.phase)], dtype=torch.long)
    logits, _ = model(token_ids, phase, legal_action_mask=legal_mask)

    if use_search and search is not None:
        result = search.select_action(state, logits, legal_actions=legal, model=model)
        if int(result.action) in legal:
            return int(result.action)

    masked = logits.masked_fill(~legal_mask, float("-inf"))
    action = int(torch.argmax(masked, dim=-1).item())
    if action not in legal:
        return int(legal[0])
    return action


def _evaluate_single_model(
    *,
    model: BridgeMonolithTransformer,
    rounds: int,
    seeds: Sequence[int],
    use_search: bool,
    search_simulations: int,
    rollout_depth: int,
    num_determinizations: int,
) -> Dict[str, float]:
    env = BridgeEnv(seed=seeds[0] if seeds else 0)
    search = None
    if use_search:
        search = ISMCTS(
            config=ISMCTSConfig(
                num_simulations=search_simulations,
                rollout_depth=rollout_depth,
                num_determinizations=num_determinizations,
            )
        )
    encoder = BridgeInputEncoder()
    scores: List[float] = []

    for seed in list(seeds)[:rounds]:
        env.reset(seed=seed)
        for _ in range(260):
            legal = env.legal_actions()
            if not legal:
                break
            action = _policy_action(
                model=model,
                env=env,
                use_search=use_search,
                search=search,
                encoder=encoder,
            )
            _, reward, done, _ = env.step(action)
            if done:
                scores.append(float(env.state.result if env.state and env.state.result is not None else reward))
                break
        else:
            scores.append(0.0)

    if not scores:
        scores = [0.0]
    return {
        "rounds": float(rounds),
        "mean_score": float(sum(scores) / max(1, len(scores))),
        "win_rate_vs_zero": float(sum(score > 0.0 for score in scores) / max(1, len(scores))),
        "score_std": float(pstdev(scores) if len(scores) > 1 else 0.0),
        "score_min": float(min(scores)),
        "score_max": float(max(scores)),
    }


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
    _, seeds = resolve_benchmark_suite(
        suite_name=None,
        rounds=rounds,
        seed=seed,
        seed_sequence=seed_sequence,
    )
    return _evaluate_single_model(
        model=model,
        rounds=len(seeds),
        seeds=seeds,
        use_search=use_search,
        search_simulations=search_simulations,
        rollout_depth=rollout_depth,
        num_determinizations=num_determinizations,
    )


def _play_game(
    *,
    seed: int,
    ns_model: BridgeMonolithTransformer,
    ew_model: BridgeMonolithTransformer,
    cfg: EvalConfig,
) -> float:
    env = BridgeEnv(seed=seed)
    env.reset(seed=seed)
    encoder = BridgeInputEncoder()
    search = None
    if cfg.use_search:
        search = ISMCTS(
            config=ISMCTSConfig(
                num_simulations=cfg.search_simulations,
                rollout_depth=cfg.rollout_depth,
                num_determinizations=cfg.num_determinizations,
            )
        )

    for _ in range(260):
        legal = env.legal_actions()
        if not legal:
            break
        if env.state is None:
            break
        actor = ns_model if env.state.current_player in (Seat.NORTH, Seat.SOUTH) else ew_model
        action = _policy_action(
            model=actor,
            env=env,
            use_search=cfg.use_search,
            search=search,
            encoder=encoder,
        )
        _, reward, done, _ = env.step(action)
        if done:
            if env.state is not None and env.state.result is not None:
                return float(env.state.result)
            return float(reward)
    return float(env.state.result if env.state and env.state.result is not None else 0.0)


def _run_duplicate_match(
    *,
    model_cfg: ModelConfig,
    checkpoint_a: str,
    checkpoint_b: str,
    seeds: Sequence[int],
    cfg: EvalConfig,
    model_cache: dict[str, BridgeMonolithTransformer],
) -> dict:
    model_a = _load_model(model_cfg, checkpoint_a, model_cache)
    model_b = _load_model(model_cfg, checkpoint_b, model_cache)

    board_results: list[dict] = []
    pair_diffs: list[float] = []
    for seed in seeds:
        score_a_ns = _play_game(seed=seed, ns_model=model_a, ew_model=model_b, cfg=cfg)
        score_b_ns = _play_game(seed=seed, ns_model=model_b, ew_model=model_a, cfg=cfg)
        pair_diff = float(score_a_ns - score_b_ns)
        pair_diffs.append(pair_diff)
        board_results.append(
            {
                "seed": int(seed),
                "score_a_as_ns": float(score_a_ns),
                "score_b_as_ns": float(score_b_ns),
                "pair_diff_a": pair_diff,
                "winner": "a" if pair_diff > 0.0 else "b" if pair_diff < 0.0 else "draw",
            }
        )

    total_diff = float(sum(pair_diffs))
    actual_a = score_from_diff(total_diff, draw_margin=cfg.duplicate_draw_margin)
    return {
        "checkpoint_a": checkpoint_a,
        "checkpoint_b": checkpoint_b,
        "boards": board_results,
        "rounds": len(board_results),
        "pair_diff_total": total_diff,
        "pair_diff_mean": float(total_diff / max(1, len(board_results))),
        "pair_diff_std": float(pstdev(pair_diffs) if len(pair_diffs) > 1 else 0.0),
        "board_win_rate_a": float(sum(diff > 0.0 for diff in pair_diffs) / max(1, len(pair_diffs))),
        "board_draw_rate": float(sum(diff == 0.0 for diff in pair_diffs) / max(1, len(pair_diffs))),
        "match_score_a": float(actual_a),
    }


def _resolve_eval_checkpoint(eval_cfg: EvalConfig, checkpoint_dir: str) -> str:
    if eval_cfg.checkpoint != "checkpoints/latest.pt":
        return str(resolve_runtime_path(eval_cfg.checkpoint))
    return resolve_latest_checkpoint(checkpoint_dir) or f"{checkpoint_dir.rstrip('/')}/latest.pt"


def _select_opponents(eval_cfg: EvalConfig, checkpoint_dir: str, current_identity: str) -> list[str]:
    if eval_cfg.baseline_checkpoint:
        return [str(resolve_runtime_path(eval_cfg.baseline_checkpoint))]

    index = load_checkpoint_index(checkpoint_dir)
    if not index:
        return []

    current_set = {current_identity}
    opponents: list[str] = []
    if eval_cfg.include_previous_snapshot:
        previous = resolve_previous_snapshot(checkpoint_dir)
        if previous and previous not in current_set:
            opponents.append(previous)
    if eval_cfg.include_anchor:
        anchor = resolve_anchor_snapshot(checkpoint_dir)
        if anchor and anchor not in current_set and anchor not in opponents:
            opponents.append(anchor)

    for record in reversed(index):
        if record.path in current_set or record.path in opponents:
            continue
        opponents.append(record.path)
        if len(opponents) >= max(1, eval_cfg.opponent_pool_size):
            break

    return opponents[: max(1, eval_cfg.opponent_pool_size)]


def _write_match_artifacts(artifacts_dir: str, run_id: str, payload: dict) -> tuple[str, str]:
    root = Path(artifacts_dir) / "evaluation"
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / f"{run_id}.json"
    boards_path = root / f"{run_id}.boards.jsonl"
    summary_payload = dict(payload)
    board_rows = summary_payload.pop("board_results", [])
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    with boards_path.open("w", encoding="utf-8") as handle:
        for row in board_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return str(summary_path), str(boards_path)


def run(config_path: str = "configs/default.yaml") -> Dict[str, float]:
    config_source = resolve_runtime_path(config_path)
    cfg = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    eval_cfg = EvalConfig(**cfg.get("evaluation", {}))
    storage = cfg.get("storage", {})
    manifest_path = str(resolve_runtime_path(storage.get("manifest_path", "artifacts/manifest.json")))
    artifacts_dir = str(resolve_runtime_path(storage.get("artifacts_dir", str(Path(manifest_path).parent))))
    checkpoint_dir = str(resolve_runtime_path(storage.get("checkpoint_dir", "checkpoints")))
    model_cfg = ModelConfig(**cfg.get("model", {}))
    run_id = f"eval_{int(time.time() * 1_000_000)}"

    resolved_checkpoint = _resolve_eval_checkpoint(eval_cfg, checkpoint_dir)
    checkpoint_identity = resolve_checkpoint_identity(resolved_checkpoint, ckpt_dir=checkpoint_dir)
    suite_name, seeds = resolve_benchmark_suite(
        suite_name=eval_cfg.suite_name,
        rounds=eval_cfg.rounds,
        seed=eval_cfg.seed,
        seed_sequence=eval_cfg.seed_sequence,
    )

    run_signature = compute_config_signature(
        str(config_source),
        extra={
            "checkpoint": resolved_checkpoint,
            "checkpoint_identity": checkpoint_identity,
            "seeds": list(seeds),
            "suite_name": suite_name,
        },
    )
    config_snapshot = write_config_snapshot(str(config_source), manifest_path, run_id)

    model_cache: dict[str, BridgeMonolithTransformer] = {}
    opponents = _select_opponents(eval_cfg, checkpoint_dir, checkpoint_identity)
    should_use_duplicate = eval_cfg.mode == "duplicate" or (eval_cfg.mode == "auto" and bool(opponents))

    result: dict = {
        "mode": "duplicate" if should_use_duplicate else "single_model",
        "checkpoint": resolved_checkpoint,
        "checkpoint_identity": checkpoint_identity,
        "suite_name": suite_name,
        "rounds": len(seeds),
        "seed_sequence": list(seeds),
    }

    if should_use_duplicate and opponents:
        matches: list[dict] = []
        board_rows: list[dict] = []
        ratings_path = Path(artifacts_dir) / "ratings" / "current.json"
        history_path = Path(artifacts_dir) / "ratings" / "history.jsonl"
        ratings = load_rating_table(ratings_path)
        ratings.setdefault(checkpoint_identity, DEFAULT_ELO)
        rating_events: list[dict] = []

        for idx, opponent in enumerate(opponents):
            match = _run_duplicate_match(
                model_cfg=model_cfg,
                checkpoint_a=resolved_checkpoint,
                checkpoint_b=opponent,
                seeds=seeds,
                cfg=eval_cfg,
                model_cache=model_cache,
            )
            opponent_identity = resolve_checkpoint_identity(opponent, ckpt_dir=checkpoint_dir)
            ratings.setdefault(opponent_identity, DEFAULT_ELO)
            rating_before_a = ratings[checkpoint_identity]
            rating_before_b = ratings[opponent_identity]
            rating_after_a, rating_after_b, expected_a = update_elo(
                rating_before_a,
                rating_before_b,
                actual_a=float(match["match_score_a"]),
                k_factor=eval_cfg.rating_k_factor,
            )
            ratings[checkpoint_identity] = rating_after_a
            ratings[opponent_identity] = rating_after_b

            match_id = f"{run_id}:{idx}"
            summary = {
                "match_id": match_id,
                "checkpoint_identity": checkpoint_identity,
                "opponent_identity": opponent_identity,
                "expected_score_a": expected_a,
                "rating_before_a": rating_before_a,
                "rating_after_a": rating_after_a,
                "rating_before_b": rating_before_b,
                "rating_after_b": rating_after_b,
                **{key: value for key, value in match.items() if key != "boards"},
            }
            matches.append(summary)
            for board in match["boards"]:
                board_rows.append({"match_id": match_id, **board})
            rating_events.extend(
                [
                    {
                        "match_id": match_id,
                        "checkpoint": checkpoint_identity,
                        "opponent": opponent_identity,
                        "before": rating_before_a,
                        "after": rating_after_a,
                        "actual": float(match["match_score_a"]),
                        "expected": expected_a,
                    },
                    {
                        "match_id": match_id,
                        "checkpoint": opponent_identity,
                        "opponent": checkpoint_identity,
                        "before": rating_before_b,
                        "after": rating_after_b,
                        "actual": 1.0 - float(match["match_score_a"]),
                        "expected": 1.0 - expected_a,
                    },
                ]
            )

        write_rating_table(ratings_path, ratings)
        append_rating_history(history_path, rating_events)

        match_payload = {
            "run_id": run_id,
            "checkpoint": resolved_checkpoint,
            "checkpoint_identity": checkpoint_identity,
            "suite_name": suite_name,
            "seed_sequence": list(seeds),
            "matches": matches,
            "board_results": board_rows,
            "rating_path": str(ratings_path),
            "rating_history_path": str(history_path),
        }
        summary_path, boards_path = _write_match_artifacts(artifacts_dir, run_id, match_payload)
        result.update(
            {
                "opponents": [resolve_checkpoint_identity(opponent, ckpt_dir=checkpoint_dir) for opponent in opponents],
                "num_matches": len(matches),
                "match_pair_diff_total": float(sum(match["pair_diff_total"] for match in matches)),
                "match_pair_diff_mean": float(
                    sum(match["pair_diff_mean"] for match in matches) / max(1, len(matches))
                ),
                "current_elo": float(ratings[checkpoint_identity]),
                "rating_path": str(ratings_path),
                "rating_history_path": str(history_path),
                "match_summary_path": summary_path,
                "board_results_path": boards_path,
            }
        )
        if matches:
            result["baseline_checkpoint"] = matches[0]["checkpoint_b"]
            result["baseline_identity"] = matches[0]["opponent_identity"]
            result["delta_vs_baseline"] = float(matches[0]["pair_diff_total"])
    else:
        model = _load_model(model_cfg, resolved_checkpoint, model_cache)
        result.update(
            _evaluate_single_model(
                model=model,
                rounds=len(seeds),
                seeds=seeds,
                use_search=eval_cfg.use_search,
                search_simulations=eval_cfg.search_simulations,
                rollout_depth=eval_cfg.rollout_depth,
                num_determinizations=eval_cfg.num_determinizations,
            )
        )

    append_manifest_entry(
        manifest_path,
        run_type="eval",
        run_id=run_id,
        config_path=str(config_source),
        run_signature=run_signature,
        config_snapshot=config_snapshot,
        outputs={
            "checkpoint": resolved_checkpoint,
            "checkpoint_identity": checkpoint_identity,
            "rounds": len(seeds),
            "suite_name": suite_name,
            "artifacts_dir": artifacts_dir,
        },
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

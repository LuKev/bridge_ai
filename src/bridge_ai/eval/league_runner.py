"""Pairwise duplicate-match league runner for explicit checkpoint sets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import json
import shutil
import time
from typing import Any, Sequence

import yaml

from bridge_ai.common.runtime_paths import resolve_runtime_path
from bridge_ai.eval.evaluator import run as run_eval


@dataclass(frozen=True)
class LeagueParticipant:
    name: str
    checkpoint: str


def _load_config(config_path: str) -> dict[str, Any]:
    source = resolve_runtime_path(config_path)
    return yaml.safe_load(source.read_text(encoding="utf-8"))


def _load_league_participants(cfg: dict[str, Any]) -> list[LeagueParticipant]:
    league_cfg = cfg.get("league", {})
    participants = []
    for row in league_cfg.get("participants", []):
        name = str(row.get("name", "")).strip()
        checkpoint = str(row.get("checkpoint", "")).strip()
        if not name or not checkpoint:
            continue
        participants.append(LeagueParticipant(name=name, checkpoint=str(resolve_runtime_path(checkpoint))))
    return participants


def _matrix_rows(participants: Sequence[LeagueParticipant], matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["a"], row["b"]): row for row in matches}
    rows: list[dict[str, Any]] = []
    for participant in participants:
        row: dict[str, Any] = {"participant": participant.name}
        for opponent in participants:
            if participant.name == opponent.name:
                row[opponent.name] = 0.0
                continue
            forward = lookup.get((participant.name, opponent.name))
            backward = lookup.get((opponent.name, participant.name))
            if forward is not None:
                row[opponent.name] = float(forward["pair_diff_total"])
            elif backward is not None:
                row[opponent.name] = float(-backward["pair_diff_total"])
            else:
                row[opponent.name] = None
        rows.append(row)
    return rows


def run_checkpoint_league(config_path: str, *, output_dir: str | None = None) -> dict[str, Any]:
    cfg = _load_config(config_path)
    league_cfg = cfg.get("league", {})
    participants = _load_league_participants(cfg)
    if len(participants) < 2:
        raise ValueError("league.participants must contain at least two checkpoints")

    base_storage = cfg.get("storage", {})
    base_artifacts = base_storage.get("artifacts_dir", "artifacts")
    league_root = resolve_runtime_path(output_dir or league_cfg.get("output_dir", f"{base_artifacts.rstrip('/')}/league"))
    run_root = league_root / f"league_{int(time.time() * 1_000_000)}"
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    manifest_path = run_root / "manifest.json"
    ratings_dir = run_root / "ratings"
    run_config_dir = run_root / "run_configs"
    run_config_dir.mkdir(parents=True, exist_ok=True)

    suite_name = league_cfg.get("suite_name", cfg.get("evaluation", {}).get("suite_name", "gating"))
    rounds = int(league_cfg.get("rounds", cfg.get("evaluation", {}).get("rounds", 32)))
    seed = int(league_cfg.get("seed", cfg.get("evaluation", {}).get("seed", 20_000)))
    pair_results: list[dict[str, Any]] = []

    for idx, (left, right) in enumerate(combinations(participants, 2)):
        pair_cfg = json.loads(json.dumps(cfg))
        pair_cfg.setdefault("storage", {})
        pair_cfg["storage"]["artifacts_dir"] = str(run_root)
        pair_cfg["storage"]["manifest_path"] = str(manifest_path)
        pair_cfg.setdefault("evaluation", {})
        pair_cfg["evaluation"].update(
            {
                "checkpoint": left.checkpoint,
                "baseline_checkpoint": right.checkpoint,
                "mode": "duplicate",
                "suite_name": suite_name,
                "rounds": rounds,
                "seed": seed,
                "opponent_pool_size": 1,
                "include_anchor": False,
                "include_previous_snapshot": False,
            }
        )
        pair_config_path = run_config_dir / f"pair_{idx:02d}_{left.name}_vs_{right.name}.yaml"
        pair_config_path.write_text(yaml.safe_dump(pair_cfg, sort_keys=False), encoding="utf-8")
        result = run_eval(str(pair_config_path))
        pair_results.append(
            {
                "a": left.name,
                "b": right.name,
                "checkpoint_a": left.checkpoint,
                "checkpoint_b": right.checkpoint,
                "pair_diff_total": float(result.get("delta_vs_baseline", 0.0)),
                "match_pair_diff_mean": float(result.get("match_pair_diff_mean", 0.0)),
                "match_summary_path": result.get("match_summary_path"),
                "board_results_path": result.get("board_results_path"),
                "checkpoint_identity": result.get("checkpoint_identity"),
                "baseline_identity": result.get("baseline_identity"),
            }
        )

    ratings_path = ratings_dir / "current.json"
    ratings = {}
    if ratings_path.exists():
        ratings = json.loads(ratings_path.read_text(encoding="utf-8"))

    ranking = []
    for participant in participants:
        matched_identity = None
        for row in pair_results:
            if row["a"] == participant.name:
                matched_identity = row.get("checkpoint_identity") or matched_identity
            if row["b"] == participant.name:
                matched_identity = row.get("baseline_identity") or matched_identity
        ranking.append(
            {
                "name": participant.name,
                "checkpoint": participant.checkpoint,
                "checkpoint_identity": matched_identity or participant.checkpoint,
                "elo": float(ratings.get(matched_identity or participant.checkpoint, 1500.0)),
            }
        )
    ranking.sort(key=lambda row: (-row["elo"], row["name"]))

    summary = {
        "run_root": str(run_root),
        "suite_name": suite_name,
        "rounds": rounds,
        "participants": [participant.__dict__ for participant in participants],
        "matches": pair_results,
        "matrix": _matrix_rows(participants, pair_results),
        "ranking": ranking,
        "ratings_path": str(ratings_path),
        "manifest_path": str(manifest_path),
    }
    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an explicit checkpoint duplicate-match league.")
    parser.add_argument("--config-path", "--config_path", default="configs/default.yaml")
    parser.add_argument("--output-dir", "--output_dir", default="")
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = _parse_args()
    print(
        run_checkpoint_league(
            config_path=args.config_path,
            output_dir=args.output_dir or None,
        )
    )


if __name__ == "__main__":
    main()

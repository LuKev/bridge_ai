"""Bootstrap dataset generation for bidding and hidden-hand belief training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import yaml

from bridge_ai.data.belief_dataset import build_dataset, build_tournament_dataset, save_dataset
from bridge_ai.data.tournament_bootstrap import DEFAULT_EVENT_FILES, DEFAULT_RELEASE_URL
from bridge_ai.data.manifest import append_manifest_entry, compute_config_signature, write_config_snapshot


@dataclass
class BootstrapConfig:
    output_path: str = "replays/latest.json"
    seed: int = 42
    data_sources: tuple[str, ...] = ()
    max_games: int = 8
    holdout_fraction: float = 0.25
    checkpoint: str | None = None
    archive_url: str | None = None
    event_files: tuple[str, ...] = ()
    cache_dir: str | None = None
    extract_dir: str | None = None
    records_per_event: int | None = None


def run(config_path: str = "configs/default.yaml") -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    selfplay_cfg = BootstrapConfig(**cfg.get("selfplay", {}))
    manifest_path = cfg.get("storage", {}).get("manifest_path", "artifacts/manifest.json")
    replay_dir = cfg.get("storage", {}).get("replay_dir", "replays")
    if selfplay_cfg.output_path == "replays/latest.json":
        selfplay_cfg.output_path = f"{replay_dir.rstrip('/')}/latest.json"

    run_id = f"selfplay_{int(time.time() * 1_000_000)}"
    run_signature = compute_config_signature(
        config_path,
        extra={
            "seed": selfplay_cfg.seed,
            "data_sources": list(selfplay_cfg.data_sources),
            "max_games": selfplay_cfg.max_games,
            "holdout_fraction": selfplay_cfg.holdout_fraction,
            "archive_url": selfplay_cfg.archive_url,
            "event_files": list(selfplay_cfg.event_files),
            "records_per_event": selfplay_cfg.records_per_event,
        },
    )
    config_snapshot = write_config_snapshot(config_path, manifest_path, run_id)

    storage = cfg.get("storage", {})
    artifacts_dir = storage.get("artifacts_dir", "artifacts")
    cache_dir = selfplay_cfg.cache_dir or f"{artifacts_dir.rstrip('/')}/bootstrap_cache"
    extract_dir = selfplay_cfg.extract_dir or f"{artifacts_dir.rstrip('/')}/bootstrap_extract"

    if selfplay_cfg.archive_url or selfplay_cfg.event_files:
        payload = build_tournament_dataset(
            archive_url=selfplay_cfg.archive_url or DEFAULT_RELEASE_URL,
            event_files=selfplay_cfg.event_files or DEFAULT_EVENT_FILES,
            cache_dir=cache_dir,
            extract_dir=extract_dir,
            max_games=selfplay_cfg.max_games,
            holdout_fraction=selfplay_cfg.holdout_fraction,
            seed=selfplay_cfg.seed,
            records_per_event=selfplay_cfg.records_per_event,
        )
    else:
        payload = build_dataset(
            data_sources=selfplay_cfg.data_sources,
            max_games=selfplay_cfg.max_games,
            holdout_fraction=selfplay_cfg.holdout_fraction,
            seed=selfplay_cfg.seed,
        )
    output_path = Path(selfplay_cfg.output_path)
    save_dataset(payload, output_path)
    summary = payload["summary"]
    append_manifest_entry(
        manifest_path,
        run_type="selfplay",
        run_id=run_id,
        config_path=config_path,
        run_signature=run_signature,
        config_snapshot=config_snapshot,
        outputs={
            "dataset_path": str(output_path),
            "num_records": summary["num_records"],
            "train_examples": summary["train_examples"],
            "holdout_examples": summary["holdout_examples"],
            "used_embedded_records": summary.get("used_embedded_records", False),
        },
        metrics={
            "num_records": float(summary["num_records"]),
            "train_examples": float(summary["train_examples"]),
            "holdout_examples": float(summary["holdout_examples"]),
        },
        status="ok",
    )
    return payload


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Generate bootstrap bidding/belief dataset.")
    parser.add_argument("--config-path", "--config_path", default="configs/default.yaml")
    args = parser.parse_args()
    print(run(config_path=args.config_path)["summary"])


if __name__ == "__main__":
    main()

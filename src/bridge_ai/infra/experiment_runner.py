"""Utilities for orchestrating one-line experimental batches."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from bridge_ai.training.train_loop import train
from bridge_ai.eval.evaluator import run as run_eval
from bridge_ai.selfplay.runner import run as run_selfplay
from bridge_ai.data.manifest import validate_manifest
from bridge_ai.common.runtime_paths import resolve_runtime_path


@dataclass
class StepResult:
    ok: bool
    error: str | None = None


@dataclass
class SmokeResult:
    config_path: str
    selfplay: StepResult = field(default_factory=lambda: StepResult(ok=False))
    train: StepResult = field(default_factory=lambda: StepResult(ok=False))
    evaluate: StepResult = field(default_factory=lambda: StepResult(ok=False))
    manifest_issues: list[Dict[str, Any]] = field(default_factory=list)


def run_smoke(config_path: str = "configs/default.yaml", manifest_path: str = "artifacts/manifest.json") -> Dict[str, Any]:
    """Run self-play, training, and evaluation in sequence."""
    config_source = resolve_runtime_path(config_path)
    if not manifest_path:
        import yaml

        cfg_dict = yaml.safe_load(config_source.read_text(encoding="utf-8"))
        manifest_path = cfg_dict.get("storage", {}).get("manifest_path", manifest_path)
    manifest_path = str(resolve_runtime_path(manifest_path))
    result = SmokeResult(config_path=config_path)

    try:
        run_selfplay(config_path=str(config_source))
        result.selfplay = StepResult(ok=True)
    except Exception as exc:  # pragma: no cover
        result.selfplay = StepResult(ok=False, error=str(exc))

    try:
        train(config_path=str(config_source))
        result.train = StepResult(ok=True)
    except Exception as exc:  # pragma: no cover
        result.train = StepResult(ok=False, error=str(exc))

    try:
        run_eval(config_path=str(config_source))
        result.evaluate = StepResult(ok=True)
    except Exception as exc:  # pragma: no cover
        result.evaluate = StepResult(ok=False, error=str(exc))

    manifest_obj = resolve_runtime_path(manifest_path)
    if manifest_obj.exists():
        result.manifest_issues = validate_manifest(manifest_path)

    return {
        "config_path": result.config_path,
        "selfplay_ok": result.selfplay.ok,
        "selfplay_error": result.selfplay.error,
        "train_ok": result.train.ok,
        "train_error": result.train.error,
        "eval_ok": result.evaluate.ok,
        "eval_error": result.evaluate.error,
        "manifest_issues": result.manifest_issues,
    }


def _parse_args() -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run one-line bridge experiment sequence.")
    parser.add_argument("--config-path", "--config_path", default="configs/default.yaml")
    parser.add_argument("--manifest-path", "--manifest_path", default="")
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = _parse_args()
    print(run_smoke(config_path=args.config_path, manifest_path=args.manifest_path))


if __name__ == "__main__":
    main()

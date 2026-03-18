"""Higher-level experiment orchestration across actor, learner, and evaluator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

from bridge_ai.eval.evaluator import run as run_eval
from bridge_ai.common.runtime_paths import resolve_runtime_path
from bridge_ai.selfplay.runner import run as run_selfplay
from bridge_ai.training.train_loop import train


@dataclass
class PipelineIterationResult:
    iteration: int
    selfplay_ok: bool
    train_ok: bool
    eval_ok: bool
    eval_metrics: dict


def run_pipeline(config_path: str = "configs/default.yaml", iterations: int | None = None) -> List[dict]:
    config_source = resolve_runtime_path(config_path)
    cfg = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    iteration_count = iterations if iterations is not None else cfg.get("pipeline", {}).get("iterations", 1)

    results: List[dict] = []
    for iteration in range(max(1, iteration_count)):
        out = {"iteration": iteration}
        try:
            run_selfplay(config_path=str(config_source))
            out["selfplay_ok"] = True
        except Exception as exc:  # pragma: no cover
            out["selfplay_ok"] = False
            out["selfplay_error"] = str(exc)

        try:
            train(config_path=str(config_source))
            out["train_ok"] = True
        except Exception as exc:  # pragma: no cover
            out["train_ok"] = False
            out["train_error"] = str(exc)

        try:
            out["eval"] = run_eval(config_path=str(config_source))
            out["eval_ok"] = True
        except Exception as exc:  # pragma: no cover
            out["eval_ok"] = False
            out["eval_error"] = str(exc)
            out["eval"] = {}
        results.append(out)
    return results


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run actor-learner-evaluator cycles.")
    parser.add_argument("--config-path", "--config_path", default="configs/default.yaml")
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args()
    print(run_pipeline(config_path=args.config_path, iterations=args.iterations))


if __name__ == "__main__":
    main()

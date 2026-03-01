"""Modal deployment entrypoints and lightweight orchestration helpers."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import yaml

from bridge_ai.eval.evaluator import run as run_eval
from bridge_ai.selfplay.runner import run as run_selfplay
from bridge_ai.training.train_loop import train as run_train

try:
    import modal
    _HAS_MODAL = True
except Exception:  # pragma: no cover
    _HAS_MODAL = False


def _build_images() -> list[str]:
    return ["torch", "pyyaml", "numpy", "pydantic", "streamlit"]


def app(config_path: str = "configs/default.yaml"):
    if not _HAS_MODAL:
        raise RuntimeError("modal package is not installed. Install with extra modal to use this interface.")

    image = modal.Image.debian_slim().pip_install(*_build_images())
    mapp = modal.App("bridge-ai")
    settings = yaml.safe_load(open(config_path, encoding="utf-8"))
    compute = settings.get("compute", {})

    actor_concurrency = compute.get("selfplay_concurrency", 1)
    actor_type = compute.get("actor_type", "CPU")
    trainer_type = compute.get("trainer_type", "CPU")
    evaluator_type = compute.get("evaluator_type", "CPU")

    @mapp.function(image=image, timeout=60 * 60, retries=1, cpu=actor_concurrency)
    def actor_worker(cfg_path: str):
        return run_selfplay(cfg_path)

    @mapp.function(image=image, timeout=60 * 60, retries=1)
    def trainer_worker(cfg_path: str):
        return run_train(cfg_path)

    @mapp.function(image=image, timeout=60 * 60, retries=1)
    def evaluator_worker(cfg_path: str):
        return run_eval(cfg_path)

    @mapp.function(image=image, timeout=60 * 60, retries=0)
    def pipeline(config: str):
        actor_worker.remote(config)
        trainer_worker.remote(config)
        return evaluator_worker.remote(config)

    app_obj = {
        "actor_worker": actor_worker,
        "trainer_worker": trainer_worker,
        "evaluator_worker": evaluator_worker,
        "pipeline": pipeline,
        "actor_type": actor_type,
        "trainer_type": trainer_type,
        "evaluator_type": evaluator_type,
    }
    return mapp, app_obj


def local_smoke(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Synchronous fallback path when Modal is unavailable."""
    run_selfplay(config_path)
    run_train(config_path)
    result = run_eval(config_path)
    return {"status": "ok", "eval": result, "config_path": config_path}


def run_local_pipeline(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Run actor + trainer + evaluator sequentially for local smoke checks."""
    run_selfplay(config_path)
    run_train(config_path)
    return {"result": run_eval(config_path), "config_path": config_path}


def manifest_workers(config_path: str) -> Dict[str, str]:
    settings = yaml.safe_load(open(config_path, encoding="utf-8"))
    return settings.get("storage", {})


if __name__ == "__main__":
    if _HAS_MODAL:
        print("Construct app with modal_app.app().")
    else:
        print("Modal not installed; local_smoke() path is available.")

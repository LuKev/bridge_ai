"""Modal deployment entrypoints and lightweight orchestration helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict
from uuid import uuid4

import yaml

from bridge_ai.eval.evaluator import run as run_eval
from bridge_ai.infra.pipeline import run_pipeline
from bridge_ai.selfplay.runner import run as run_selfplay
from bridge_ai.training.train_loop import train as run_train

DEFAULT_CONFIG_PATH = "configs/modal.yaml"
DEFAULT_APP_NAME = os.environ.get("BRIDGE_AI_MODAL_APP_NAME", "bridge-ai")
DEFAULT_VOLUME_NAME = os.environ.get("BRIDGE_AI_MODAL_VOLUME", "bridge-ai-artifacts")
DEFAULT_VOLUME_MOUNT = os.environ.get("BRIDGE_AI_MODAL_VOLUME_MOUNT", "/vol/bridge-ai")
DEFAULT_TIMEOUT_SECONDS = int(os.environ.get("BRIDGE_AI_MODAL_TIMEOUT_SECONDS", "7200"))
DEFAULT_SELFPLAY_CPU = int(os.environ.get("BRIDGE_AI_MODAL_SELFPLAY_CPU", "8"))
DEFAULT_TRAINER_CPU = int(os.environ.get("BRIDGE_AI_MODAL_TRAINER_CPU", "4"))
DEFAULT_EVALUATOR_CPU = int(os.environ.get("BRIDGE_AI_MODAL_EVALUATOR_CPU", "2"))
DEFAULT_TRAINER_GPU = os.environ.get("BRIDGE_AI_MODAL_TRAINER_GPU", "A100")
DEFAULT_SELFPLAY_GPU = os.environ.get("BRIDGE_AI_MODAL_SELFPLAY_GPU", "CPU")
DEFAULT_EVALUATOR_GPU = os.environ.get("BRIDGE_AI_MODAL_EVALUATOR_GPU", "CPU")
MODAL_RUNTIME_PACKAGES = (
    "modal>=1.3.5",
    "numpy>=1.26,<3",
    "pydantic>=2.0,<3",
    "pyyaml>=6.0,<7",
    "torch>=2.2,<2.11",
)

try:
    import modal

    _HAS_MODAL = True
except Exception:  # pragma: no cover
    modal = None
    _HAS_MODAL = False


def _normalize_gpu(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned or cleaned.upper() == "CPU":
        return None
    return cleaned


def _load_config(config_path: str) -> Dict[str, Any]:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}


def _default_storage_paths(mount_path: str) -> Dict[str, str]:
    base = mount_path.rstrip("/")
    return {
        "artifacts_dir": f"{base}/artifacts",
        "replay_dir": f"{base}/replays",
        "checkpoint_dir": f"{base}/checkpoints",
        "manifest_path": f"{base}/artifacts/manifest.json",
    }


def _prepare_modal_config(config_path: str) -> str:
    cfg = _load_config(config_path)
    modal_cfg = cfg.setdefault("modal", {})
    mount_path = str(modal_cfg.get("mount_path", DEFAULT_VOLUME_MOUNT)).rstrip("/")
    modal_cfg.setdefault("app_name", DEFAULT_APP_NAME)
    modal_cfg.setdefault("volume_name", DEFAULT_VOLUME_NAME)
    modal_cfg["mount_path"] = mount_path

    storage = cfg.setdefault("storage", {})
    default_storage = _default_storage_paths(mount_path)
    for key, value in default_storage.items():
        storage.setdefault(key, value)

    selfplay_cfg = cfg.setdefault("selfplay", {})
    if selfplay_cfg.get("output_path", "replays/latest.json") == "replays/latest.json":
        selfplay_cfg["output_path"] = f"{storage['replay_dir'].rstrip('/')}/latest.json"
    if "checkpoint" not in selfplay_cfg or selfplay_cfg.get("checkpoint") == "auto":
        selfplay_cfg["checkpoint"] = f"{storage['checkpoint_dir'].rstrip('/')}/latest.pt"

    training_cfg = cfg.setdefault("training", {})
    if training_cfg.get("replay_path", "replays/latest.json") == "replays/latest.json":
        training_cfg["replay_path"] = f"{storage['replay_dir'].rstrip('/')}/latest.json"
    if training_cfg.get("ckpt_dir", "checkpoints") == "checkpoints":
        training_cfg["ckpt_dir"] = storage["checkpoint_dir"]
    if "init_checkpoint" not in training_cfg or training_cfg.get("init_checkpoint") == "auto":
        training_cfg["init_checkpoint"] = f"{storage['checkpoint_dir'].rstrip('/')}/latest.pt"

    trainer_gpu = _normalize_gpu(
        cfg.get("compute", {}).get("trainer_gpu", cfg.get("compute", {}).get("trainer_type", DEFAULT_TRAINER_GPU))
    )
    if trainer_gpu and training_cfg.get("device", "cpu") == "cpu":
        training_cfg["device"] = "cuda"

    evaluation_cfg = cfg.setdefault("evaluation", {})
    if evaluation_cfg.get("checkpoint", "checkpoints/latest.pt") == "checkpoints/latest.pt":
        evaluation_cfg["checkpoint"] = f"{storage['checkpoint_dir'].rstrip('/')}/latest.pt"
    if evaluation_cfg.get("baseline_checkpoint") == "checkpoints/latest.pt":
        evaluation_cfg["baseline_checkpoint"] = f"{storage['checkpoint_dir'].rstrip('/')}/latest.pt"

    return yaml.safe_dump(cfg, sort_keys=False)


def _write_runtime_config(config_text: str) -> str:
    if _HAS_MODAL and _VOLUME is not None:
        runtime_dir = Path(DEFAULT_VOLUME_MOUNT) / "runtime_configs"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        config_path = runtime_dir / f"{uuid4().hex}.yaml"
        config_path.write_text(config_text, encoding="utf-8")
        _volume_commit()
        return str(config_path)

    runtime_dir = Path(tempfile.mkdtemp(prefix="bridge_ai_modal_"))
    config_path = runtime_dir / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    return str(config_path)


def _volume_reload() -> None:
    if _HAS_MODAL and _VOLUME is not None:
        _VOLUME.reload()


def _volume_commit() -> None:
    if _HAS_MODAL and _VOLUME is not None:
        _VOLUME.commit()


def _run_step(
    *,
    config_text: str,
    runner: Callable[..., Any],
    summarize: Callable[[Any, Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    config_path = _write_runtime_config(config_text)
    cfg = yaml.safe_load(config_text) or {}
    _volume_reload()
    try:
        result = runner(config_path=config_path)
        _volume_commit()
        return summarize(result, cfg)
    finally:
        _volume_commit()


def _summarize_selfplay(result: Any, cfg: Dict[str, Any]) -> Dict[str, Any]:
    output_path = cfg.get("selfplay", {}).get("output_path")
    summary = result.get("summary", {}) if isinstance(result, dict) else {}
    return {
        "status": "ok",
        "step": "selfplay",
        "replay_path": output_path,
        "records": summary.get("num_records", 0),
        "train_examples": summary.get("train_examples", 0),
        "holdout_examples": summary.get("holdout_examples", 0),
    }


def _summarize_train(_: Any, cfg: Dict[str, Any]) -> Dict[str, Any]:
    checkpoint_dir = cfg.get("training", {}).get("ckpt_dir") or cfg.get("storage", {}).get("checkpoint_dir")
    checkpoint_name = cfg.get("training", {}).get("checkpoint_name", "latest.pt")
    checkpoint_path = None
    if checkpoint_dir:
        checkpoint_path = str(Path(checkpoint_dir) / checkpoint_name)
    return {
        "status": "ok",
        "step": "train",
        "checkpoint_path": checkpoint_path,
        "device": cfg.get("training", {}).get("device", "cpu"),
    }


def _summarize_eval(result: Any, cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "ok",
        "step": "eval",
        "checkpoint_path": cfg.get("evaluation", {}).get("checkpoint"),
        "metrics": result,
    }


def _summarize_pipeline(result: Any, cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "ok",
        "step": "pipeline",
        "iterations": len(result) if isinstance(result, list) else 0,
        "results": result,
        "manifest_path": cfg.get("storage", {}).get("manifest_path"),
    }


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


def manifest_workers(config_path: str) -> Dict[str, Any]:
    cfg = _load_config(config_path)
    return {
        "compute": cfg.get("compute", {}),
        "storage": cfg.get("storage", {}),
        "modal": cfg.get("modal", {}),
    }


if _HAS_MODAL:
    _IMAGE = (
        modal.Image.debian_slim()
        .pip_install(*MODAL_RUNTIME_PACKAGES)
        .add_local_python_source("bridge_ai")
    )
    _VOLUME = modal.Volume.from_name(DEFAULT_VOLUME_NAME, create_if_missing=True)
    app = modal.App(DEFAULT_APP_NAME)

    @app.function(
        image=_IMAGE,
        volumes={DEFAULT_VOLUME_MOUNT: _VOLUME},
        timeout=DEFAULT_TIMEOUT_SECONDS,
        retries=1,
        cpu=DEFAULT_SELFPLAY_CPU,
        gpu=_normalize_gpu(DEFAULT_SELFPLAY_GPU),
    )
    def selfplay_worker(config_text: str) -> Dict[str, Any]:
        return _run_step(config_text=config_text, runner=run_selfplay, summarize=_summarize_selfplay)


    @app.function(
        image=_IMAGE,
        volumes={DEFAULT_VOLUME_MOUNT: _VOLUME},
        timeout=DEFAULT_TIMEOUT_SECONDS,
        retries=1,
        cpu=DEFAULT_TRAINER_CPU,
        gpu=_normalize_gpu(DEFAULT_TRAINER_GPU),
    )
    def train_worker(config_text: str) -> Dict[str, Any]:
        return _run_step(config_text=config_text, runner=run_train, summarize=_summarize_train)


    @app.function(
        image=_IMAGE,
        volumes={DEFAULT_VOLUME_MOUNT: _VOLUME},
        timeout=DEFAULT_TIMEOUT_SECONDS,
        retries=1,
        cpu=DEFAULT_EVALUATOR_CPU,
        gpu=_normalize_gpu(DEFAULT_EVALUATOR_GPU),
    )
    def eval_worker(config_text: str) -> Dict[str, Any]:
        return _run_step(config_text=config_text, runner=run_eval, summarize=_summarize_eval)


    @app.function(
        image=_IMAGE,
        volumes={DEFAULT_VOLUME_MOUNT: _VOLUME},
        timeout=DEFAULT_TIMEOUT_SECONDS,
        retries=0,
        cpu=DEFAULT_TRAINER_CPU,
        gpu=_normalize_gpu(DEFAULT_TRAINER_GPU),
    )
    def pipeline_worker(config_text: str, iterations: int = 1) -> Dict[str, Any]:
        return _run_step(
            config_text=config_text,
            runner=lambda *, config_path: run_pipeline(config_path=config_path, iterations=iterations),
            summarize=_summarize_pipeline,
        )


    @app.local_entrypoint()
    def main(config_path: str = DEFAULT_CONFIG_PATH, job: str = "pipeline", iterations: int = 1) -> None:
        config_text = _prepare_modal_config(config_path)
        if job == "selfplay":
            result: Any = selfplay_worker.remote(config_text)
        elif job == "train":
            result = train_worker.remote(config_text)
        elif job == "eval":
            result = eval_worker.remote(config_text)
        elif job == "pipeline":
            result = pipeline_worker.remote(config_text, iterations=iterations)
        elif job == "pipeline_worker":
            result = pipeline_worker.remote(config_text, iterations=iterations)
        else:
            raise ValueError(f"unknown modal job: {job}")
        print(yaml.safe_dump(result, sort_keys=False))
else:  # pragma: no cover
    _IMAGE = None
    _VOLUME = None
    app = None


if __name__ == "__main__":  # pragma: no cover
    if _HAS_MODAL:
        print("Run with: modal run -m bridge_ai.infra.modal_app --config-path configs/modal.yaml --job pipeline")
    else:
        print("Modal is not installed. Install the optional dependency with `pip install -e .[modal]`.")

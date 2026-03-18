"""Bazel-invoked Modal continuation runner with local artifact sync."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import json
import shutil
import tempfile
from typing import Any

import torch
import yaml

from bridge_ai.common.runtime_paths import resolve_runtime_path
from bridge_ai.data.manifest import validate_manifest
from bridge_ai.eval.evaluator import run as run_eval
from bridge_ai.eval.league_runner import run_checkpoint_league
from bridge_ai.models.monolithic_transformer import ModelConfig
from bridge_ai.selfplay.runner import run as run_selfplay
from bridge_ai.training.checkpoint_bootstrap import create_initial_checkpoint, import_legacy_checkpoint
from bridge_ai.training.checkpoint_store import resolve_latest_checkpoint, resolve_latest_snapshot
from bridge_ai.training.train_loop import train

try:
    import modal
    from modal.volume import FileEntryType
except Exception:  # pragma: no cover
    modal = None
    FileEntryType = None


@dataclass
class ModalRuntimeConfig:
    app_name: str = "bridge-ai"
    volume_name: str = "bridge-ai"
    volume_mount_path: str = "/mnt/bridge_ai"
    remote_root: str = "runs/default"
    timeout_seconds: int = 12 * 60 * 60
    cpu: float = 8.0
    gpu: str | None = None
    bootstrap_checkpoint: str = ""
    bootstrap_iteration: int | None = None
    bootstrap_snapshot_tag: str | None = None
    target_step: int | None = None
    environment_name: str | None = None
    force_bootstrap: bool = False
    sync_back: bool = True
    nonpreemptible: bool = False


def _require_modal() -> None:
    if modal is None:
        raise RuntimeError("modal package is not installed in this runtime")


def _build_modal_image(runtime: ModalRuntimeConfig):
    _require_modal()
    image = modal.Image.debian_slim(python_version="3.11")
    if runtime.gpu:
        image = image.pip_install(
            "numpy==1.26.4",
            "pydantic>=2.0",
            "pyyaml>=6.0",
            "torch==2.2.2",
            extra_index_url="https://download.pytorch.org/whl/cu121",
        )
    else:
        image = image.pip_install(
            "numpy==1.26.4",
            "pydantic>=2.0",
            "pyyaml>=6.0",
            "torch>=2.2",
        )
    return (
        image.add_local_python_source("bridge_ai")
    )


def _replace_remote_prefix(value: str, mappings: list[tuple[str, str]]) -> str:
    normalized = value.replace("\\", "/")
    for remote_prefix, local_prefix in mappings:
        remote_prefix = remote_prefix.rstrip("/")
        if normalized == remote_prefix:
            return local_prefix
        if normalized.startswith(remote_prefix + "/"):
            suffix = normalized[len(remote_prefix) + 1 :]
            return str(Path(local_prefix) / Path(suffix))
    return value


def _rewrite_paths(obj: Any, mappings: list[tuple[str, str]]) -> Any:
    if isinstance(obj, str):
        return _replace_remote_prefix(obj, mappings)
    if isinstance(obj, list):
        return [_rewrite_paths(item, mappings) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_rewrite_paths(item, mappings) for item in obj)
    if isinstance(obj, dict):
        return {key: _rewrite_paths(value, mappings) for key, value in obj.items()}
    return obj


def _rewrite_json_file(path: Path, mappings: list[tuple[str, str]]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(_rewrite_paths(payload, mappings), indent=2, sort_keys=True), encoding="utf-8")


def _rewrite_jsonl_file(path: Path, mappings: list[tuple[str, str]]) -> None:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(_rewrite_paths(json.loads(line), mappings))
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _rewrite_yaml_file(path: Path, mappings: list[tuple[str, str]]) -> None:
    text = path.read_text(encoding="utf-8")
    for remote_prefix, local_prefix in mappings:
        text = text.replace(remote_prefix, local_prefix)
    path.write_text(text, encoding="utf-8")


def _rewrite_checkpoint_payloads(checkpoint_dir: Path, mappings: list[tuple[str, str]]) -> None:
    for payload_path in checkpoint_dir.rglob("*.pt"):
        payload = torch.load(payload_path, map_location="cpu")
        rewritten = _rewrite_paths(payload, mappings)
        torch.save(rewritten, payload_path)


def _rewrite_synced_files(*, local_dirs: list[Path], mappings: list[tuple[str, str]]) -> None:
    for local_dir in local_dirs:
        if not local_dir.exists():
            continue
        for path in local_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix == ".json":
                _rewrite_json_file(path, mappings)
            elif path.suffix == ".jsonl":
                _rewrite_jsonl_file(path, mappings)
            elif path.suffix in {".yaml", ".yml"}:
                _rewrite_yaml_file(path, mappings)


def _upload_remote_config(volume, *, remote_rel_root: PurePosixPath, remote_cfg: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory(prefix="bridge-ai-modal-config-") as tmp:
        stage_root = Path(tmp)
        remote_config_path = stage_root / "remote_runtime_config.yaml"
        remote_config_path.write_text(yaml.safe_dump(remote_cfg, sort_keys=False), encoding="utf-8")
        with volume.batch_upload(force=True) as batch:
            batch.put_file(remote_config_path, str(remote_rel_root / "artifacts" / "remote_runtime_config.yaml"))


def _read_volume_json(volume, volume_path: PurePosixPath) -> Any | None:
    try:
        payload = b"".join(volume.read_file(str(volume_path)))
    except Exception:
        return None
    if not payload:
        return None
    return json.loads(payload.decode("utf-8"))


def _read_remote_checkpoint_progress(volume, *, remote_rel_root: PurePosixPath) -> dict[str, Any]:
    checkpoint_rel = remote_rel_root / "checkpoints"
    index_payload = _read_volume_json(volume, checkpoint_rel / "index.json")
    if isinstance(index_payload, list) and index_payload:
        last_record = index_payload[-1]
        last_iteration = int(last_record.get("iteration", -1))
        return {
            "checkpoint_dir": str(checkpoint_rel),
            "index_path": str(checkpoint_rel / "index.json"),
            "latest_snapshot": last_record.get("path"),
            "last_iteration": last_iteration,
            "current_step": last_iteration + 1 if last_iteration >= 0 else 0,
        }
    return {
        "checkpoint_dir": str(checkpoint_rel),
        "index_path": str(checkpoint_rel / "index.json"),
        "latest_snapshot": None,
        "last_iteration": None,
        "current_step": None,
    }


def _plan_pipeline_iterations(
    *,
    default_iterations: int,
    block_size: int,
    target_step: int | None,
    current_step: int | None,
) -> dict[str, Any]:
    planned_iterations = max(0, int(default_iterations))
    planning = {
        "planning_mode": "config_default",
        "default_pipeline_iterations": planned_iterations,
        "block_size": max(1, int(block_size)),
        "target_step": target_step,
        "current_step": current_step,
        "remaining_steps": None,
        "planned_pipeline_iterations": planned_iterations,
    }
    if target_step is None or current_step is None:
        return planning

    remaining_steps = max(0, int(target_step) - int(current_step))
    planning["remaining_steps"] = remaining_steps
    if remaining_steps == 0:
        planning["planning_mode"] = "already_at_target"
        planning["planned_pipeline_iterations"] = 0
        return planning

    if remaining_steps % planning["block_size"] != 0:
        raise ValueError(
            "modal.target_step must align with training.iterations when resuming from saved checkpoints: "
            f"target_step={target_step}, current_step={current_step}, block_size={planning['block_size']}"
        )

    planning["planning_mode"] = "target_step"
    planning["planned_pipeline_iterations"] = remaining_steps // planning["block_size"]
    return planning


def _volume_path_exists(volume, volume_path: PurePosixPath) -> bool:
    try:
        return bool(volume.listdir(str(volume_path)))
    except Exception:
        return False


def _build_remote_config(
    cfg: dict[str, Any],
    runtime: ModalRuntimeConfig,
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    remote_rel_root = PurePosixPath(runtime.remote_root.strip("/"))
    remote_abs_root = PurePosixPath(runtime.volume_mount_path) / remote_rel_root
    remote_paths = {
        "root": str(remote_abs_root),
        "checkpoint_dir": str(remote_abs_root / "checkpoints"),
        "replay_dir": str(remote_abs_root / "replays"),
        "artifacts_dir": str(remote_abs_root / "artifacts"),
        "manifest_path": str(remote_abs_root / "artifacts" / "manifest.json"),
        "remote_config_path": str(remote_abs_root / "artifacts" / "remote_runtime_config.yaml"),
    }

    storage = cfg.setdefault("storage", {})
    local_paths = {
        "checkpoint_dir": str(resolve_runtime_path(storage.get("checkpoint_dir", "checkpoints"))),
        "replay_dir": str(resolve_runtime_path(storage.get("replay_dir", "replays"))),
        "artifacts_dir": str(resolve_runtime_path(storage.get("artifacts_dir", "artifacts"))),
    }
    local_paths["manifest_path"] = str(resolve_runtime_path(storage.get("manifest_path", "artifacts/manifest.json")))

    remote_cfg = json.loads(json.dumps(cfg))
    remote_cfg.setdefault("storage", {})
    remote_cfg["storage"].update(
        {
            "checkpoint_dir": remote_paths["checkpoint_dir"],
            "replay_dir": remote_paths["replay_dir"],
            "artifacts_dir": remote_paths["artifacts_dir"],
            "manifest_path": remote_paths["manifest_path"],
        }
    )
    remote_cfg.setdefault("selfplay", {})
    remote_cfg["selfplay"]["output_path"] = str(PurePosixPath(remote_paths["replay_dir"]) / "latest.json")
    remote_cfg["selfplay"]["checkpoint"] = None
    remote_cfg.setdefault("training", {})
    remote_cfg["training"]["replay_path"] = str(PurePosixPath(remote_paths["replay_dir"]) / "latest.json")
    remote_cfg["training"]["init_checkpoint"] = None
    remote_cfg["training"]["ckpt_dir"] = remote_paths["checkpoint_dir"]
    remote_cfg["training"]["resume_latest"] = True
    if runtime.gpu:
        remote_cfg["training"]["device"] = "cuda"
    remote_cfg.setdefault("evaluation", {})
    remote_cfg["evaluation"]["checkpoint"] = str(PurePosixPath(remote_paths["checkpoint_dir"]) / "latest.pt")

    return remote_cfg, local_paths, remote_paths


def _bootstrap_volume(volume, runtime: ModalRuntimeConfig, remote_cfg: dict[str, Any], remote_paths: dict[str, str]) -> dict[str, str]:
    remote_rel_root = PurePosixPath(runtime.remote_root.strip("/"))
    remote_checkpoint_rel = remote_rel_root / "checkpoints"
    remote_artifacts_rel = remote_rel_root / "artifacts"
    remote_checkpoint_latest = remote_checkpoint_rel / "latest.pt"

    with tempfile.TemporaryDirectory(prefix="bridge-ai-modal-config-") as tmp:
        stage_root = Path(tmp)
        if runtime.force_bootstrap:
            try:
                volume.remove_file(str(remote_rel_root), recursive=True)
            except Exception:
                pass

        if not runtime.bootstrap_checkpoint:
            raise ValueError("modal.bootstrap_checkpoint must be set for continuation")

        if not runtime.force_bootstrap and _volume_path_exists(volume, remote_checkpoint_latest):
            return {
                "bootstrap_mode": "resume_existing",
            }

        stage_ckpt_dir = stage_root / "checkpoints"
        import_result = import_legacy_checkpoint(
            source_path=resolve_runtime_path(runtime.bootstrap_checkpoint),
            ckpt_dir=stage_ckpt_dir,
            iteration=runtime.bootstrap_iteration,
            snapshot_tag=runtime.bootstrap_snapshot_tag,
            overwrite=True,
        )
        stage_mappings = [(str(stage_ckpt_dir), remote_paths["checkpoint_dir"])]
        _rewrite_synced_files(local_dirs=[stage_ckpt_dir], mappings=stage_mappings)
        _rewrite_checkpoint_payloads(stage_ckpt_dir, stage_mappings)
        with volume.batch_upload(force=True) as batch:
            batch.put_directory(stage_ckpt_dir, str(remote_checkpoint_rel))
        return {
            "bootstrap_mode": "imported_legacy",
            "bootstrap_snapshot_path": import_result["snapshot_path"],
        }


def _sync_volume_prefix(volume, *, remote_prefix: PurePosixPath, local_dir: Path) -> int:
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        entries = volume.listdir(str(remote_prefix), recursive=True)
    except Exception:
        return 0

    files_synced = 0
    for entry in entries:
        entry_path = PurePosixPath(entry.path)
        relative = entry_path.relative_to(remote_prefix)
        target = local_dir / Path(str(relative))
        if FileEntryType is not None and entry.type == FileEntryType.DIRECTORY:
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            volume.read_file_into_fileobj(entry.path, handle)
        files_synced += 1
    return files_synced


def _build_modal_app(runtime: ModalRuntimeConfig):
    _require_modal()
    image = _build_modal_image(runtime)
    volume = modal.Volume.from_name(
        runtime.volume_name,
        create_if_missing=True,
        environment_name=runtime.environment_name,
    )
    app = modal.App(runtime.app_name)

    @app.function(
        image=image,
        volumes={runtime.volume_mount_path: volume},
        cpu=runtime.cpu,
        gpu=runtime.gpu,
        timeout=runtime.timeout_seconds,
        include_source=True,
        serialized=True,
        nonpreemptible=runtime.nonpreemptible,
        name="continue_checkpoint",
    )
    def continue_checkpoint(remote_config_path: str) -> dict[str, Any]:
        torch.set_num_threads(max(1, int(runtime.cpu)))
        torch.set_num_interop_threads(max(1, min(8, int(runtime.cpu))))
        volume.reload()
        cfg_path = Path(remote_config_path)
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        iteration_count = max(0, int(cfg.get("pipeline", {}).get("iterations", 1)))
        results: list[dict[str, Any]] = []
        for iteration in range(iteration_count):
            out = {"iteration": iteration}
            try:
                run_selfplay(config_path=str(cfg_path))
                out["selfplay_ok"] = True
            except Exception as exc:  # pragma: no cover
                out["selfplay_ok"] = False
                out["selfplay_error"] = str(exc)

            try:
                train(config_path=str(cfg_path))
                out["train_ok"] = True
            except Exception as exc:  # pragma: no cover
                out["train_ok"] = False
                out["train_error"] = str(exc)

            try:
                out["eval"] = run_eval(config_path=str(cfg_path))
                out["eval_ok"] = True
            except Exception as exc:  # pragma: no cover
                out["eval_ok"] = False
                out["eval_error"] = str(exc)
                out["eval"] = {}

            results.append(out)
            volume.commit()

        checkpoint_dir = cfg.get("storage", {}).get("checkpoint_dir", "")
        return {
            "results": results,
            "checkpoint_dir": checkpoint_dir,
            "latest_checkpoint": resolve_latest_checkpoint(checkpoint_dir),
            "latest_snapshot": resolve_latest_snapshot(checkpoint_dir),
            "manifest_path": cfg.get("storage", {}).get("manifest_path"),
            "artifacts_dir": cfg.get("storage", {}).get("artifacts_dir"),
            "replay_dir": cfg.get("storage", {}).get("replay_dir"),
        }

    return app, volume, continue_checkpoint


def _ensure_initial_checkpoint(local_config: dict[str, Any], config_path: str) -> str | None:
    league_cfg = local_config.get("league", {})
    if not league_cfg.get("enabled", False):
        return None
    initial_path = league_cfg.get("initial_checkpoint")
    if not initial_path:
        return None
    resolved_initial_path = resolve_runtime_path(initial_path)
    if resolved_initial_path.exists():
        return str(resolved_initial_path)

    model_cfg = local_config.get("model", {})
    created = create_initial_checkpoint(
        model_cfg=ModelConfig(**model_cfg),
        ckpt_dir=resolved_initial_path.parent,
        checkpoint_name=resolved_initial_path.name,
        seed=int(league_cfg.get("initial_seed", 0)),
        snapshot_tag=league_cfg.get("initial_snapshot_tag", "iter_000000"),
    )
    return str(resolve_runtime_path(created["latest_path"] or str(resolved_initial_path)))


def run_modal_continuation(config_path: str = "configs/default.yaml") -> dict[str, Any]:
    _require_modal()
    config_source = resolve_runtime_path(config_path)
    local_cfg = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    runtime = ModalRuntimeConfig(**local_cfg.get("modal", {}))
    remote_cfg, local_paths, remote_paths = _build_remote_config(local_cfg, runtime)
    remote_rel_root = PurePosixPath(runtime.remote_root.strip("/"))

    app, volume, continue_checkpoint = _build_modal_app(runtime)
    bootstrap = _bootstrap_volume(volume, runtime, remote_cfg, remote_paths)
    remote_progress = _read_remote_checkpoint_progress(volume, remote_rel_root=remote_rel_root)
    planning = _plan_pipeline_iterations(
        default_iterations=int(remote_cfg.get("pipeline", {}).get("iterations", 1)),
        block_size=int(remote_cfg.get("training", {}).get("iterations", 1)),
        target_step=runtime.target_step,
        current_step=remote_progress.get("current_step"),
    )
    remote_cfg.setdefault("pipeline", {})
    remote_cfg["pipeline"]["iterations"] = int(planning["planned_pipeline_iterations"])
    _upload_remote_config(volume, remote_rel_root=remote_rel_root, remote_cfg=remote_cfg)
    bootstrap["remote_config_path"] = str(remote_rel_root / "artifacts" / "remote_runtime_config.yaml")
    remote_config_path = remote_paths["remote_config_path"]
    try:
        with app.run(environment_name=runtime.environment_name, detach=True):
            remote_result = continue_checkpoint.remote(remote_config_path)
    except Exception as exc:
        remote_result = {"error": str(exc), "results": []}
    if not remote_result.get("results") and int(planning["planned_pipeline_iterations"]) > 0:
        remote_result["error"] = remote_result.get("error") or "remote_run_incomplete"

    sync_summary = {}
    if runtime.sync_back:
        if runtime.force_bootstrap:
            for local_dir in (
                Path(local_paths["checkpoint_dir"]),
                Path(local_paths["replay_dir"]),
                Path(local_paths["artifacts_dir"]),
            ):
                if local_dir.exists():
                    shutil.rmtree(local_dir)
        sync_summary = {
            "checkpoint_files": _sync_volume_prefix(
                volume,
                remote_prefix=PurePosixPath(runtime.remote_root.strip("/")) / "checkpoints",
                local_dir=Path(local_paths["checkpoint_dir"]),
            ),
            "replay_files": _sync_volume_prefix(
                volume,
                remote_prefix=PurePosixPath(runtime.remote_root.strip("/")) / "replays",
                local_dir=Path(local_paths["replay_dir"]),
            ),
            "artifact_files": _sync_volume_prefix(
                volume,
                remote_prefix=PurePosixPath(runtime.remote_root.strip("/")) / "artifacts",
                local_dir=Path(local_paths["artifacts_dir"]),
            ),
        }

        mappings = [
            (remote_paths["checkpoint_dir"], local_paths["checkpoint_dir"]),
            (remote_paths["replay_dir"], local_paths["replay_dir"]),
            (remote_paths["artifacts_dir"], local_paths["artifacts_dir"]),
        ]
        _rewrite_synced_files(
            local_dirs=[
                Path(local_paths["checkpoint_dir"]),
                Path(local_paths["replay_dir"]),
                Path(local_paths["artifacts_dir"]),
            ],
            mappings=mappings,
        )
        _rewrite_checkpoint_payloads(Path(local_paths["checkpoint_dir"]), mappings)

    manifest_issues = []
    manifest_path = Path(local_paths["manifest_path"])
    if manifest_path.exists():
        manifest_issues = validate_manifest(str(manifest_path))

    initial_checkpoint = _ensure_initial_checkpoint(local_cfg, str(config_source))
    league_result = None
    if local_cfg.get("league", {}).get("enabled", False) and remote_result.get("results") and not remote_result.get("error"):
        league_result = run_checkpoint_league(str(config_source))

    summary = {
        "config_path": str(config_source),
        "bootstrap": bootstrap,
        "remote_progress": remote_progress,
        "planning": planning,
        "remote_result": remote_result,
        "local_paths": local_paths,
        "remote_paths": remote_paths,
        "sync_summary": sync_summary,
        "manifest_issues": manifest_issues,
        "initial_checkpoint": initial_checkpoint,
        "league_result": league_result,
    }
    summary_path = Path(local_paths["artifacts_dir"]) / "modal_continue_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue a checkpoint remotely on Modal and sync artifacts back.")
    parser.add_argument("--config-path", "--config_path", default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = _parse_args()
    print(run_modal_continuation(config_path=args.config_path))


if __name__ == "__main__":
    main()

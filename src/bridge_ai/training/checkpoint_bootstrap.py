"""Helpers for importing legacy checkpoints and creating evaluation baselines."""

from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Any

import torch

from bridge_ai.models.monolithic_transformer import BridgeMonolithTransformer, ModelConfig
from bridge_ai.training.checkpoint_store import (
    CheckpointRecord,
    checkpoint_index_path,
    latest_checkpoint_path,
    load_checkpoint_index,
    save_checkpoint_bundle,
)


def _default_snapshot_tag(iteration: int) -> str:
    step = max(0, int(iteration) + 1)
    return f"iter_{step:06d}"


def _write_checkpoint_index(ckpt_dir: Path, records: list[CheckpointRecord]) -> None:
    path = checkpoint_index_path(ckpt_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [
                {
                    "path": record.path,
                    "iteration": int(record.iteration),
                    "created_at": float(record.created_at),
                    "run_id": record.run_id,
                    "parent_checkpoint": record.parent_checkpoint,
                    "source_checkpoint": record.source_checkpoint,
                }
                for record in records
            ],
            indent=2,
        ),
        encoding="utf-8",
    )


def import_legacy_checkpoint(
    *,
    source_path: str | Path,
    ckpt_dir: str | Path,
    checkpoint_name: str = "latest.pt",
    iteration: int | None = None,
    snapshot_tag: str | None = None,
    overwrite: bool = True,
) -> dict[str, str]:
    """Import an existing checkpoint into the immutable snapshot/index format."""
    source = Path(source_path)
    payload: dict[str, Any] = torch.load(source, map_location="cpu")

    imported_iteration = int(payload.get("iteration", -1) if iteration is None else iteration)
    tag = snapshot_tag or _default_snapshot_tag(imported_iteration)
    destination_root = Path(ckpt_dir)
    destination_root.mkdir(parents=True, exist_ok=True)

    snapshot_path = destination_root / f"{tag}.pt"
    latest_path = latest_checkpoint_path(destination_root, checkpoint_name=checkpoint_name)
    if not overwrite and (snapshot_path.exists() or latest_path.exists()):
        raise FileExistsError(f"refusing to overwrite imported checkpoint in {destination_root}")

    run_id = str(payload.get("run_id") or f"bootstrap_{int(time.time() * 1_000_000)}")
    saved_at = float(payload.get("saved_at") or time.time())
    source_checkpoint = str(source.resolve())
    parent_checkpoint = payload.get("parent_checkpoint")
    if isinstance(parent_checkpoint, Path):
        parent_checkpoint = str(parent_checkpoint)

    imported_payload = dict(payload)
    imported_payload.update(
        {
            "iteration": imported_iteration,
            "run_id": run_id,
            "saved_at": saved_at,
            "snapshot_path": str(snapshot_path),
            "source_checkpoint": source_checkpoint,
            "parent_checkpoint": parent_checkpoint,
            "imported_from": source_checkpoint,
        }
    )

    torch.save(imported_payload, snapshot_path)
    torch.save(imported_payload, latest_path)

    index = [record for record in load_checkpoint_index(destination_root) if record.path != str(snapshot_path)]
    index.append(
        CheckpointRecord(
            path=str(snapshot_path),
            iteration=imported_iteration,
            created_at=saved_at,
            run_id=run_id,
            parent_checkpoint=parent_checkpoint,
            source_checkpoint=source_checkpoint,
        )
    )
    index.sort(key=lambda record: (record.iteration, record.created_at, record.path))
    _write_checkpoint_index(destination_root, index)

    return {
        "latest_path": str(latest_path),
        "snapshot_path": str(snapshot_path),
        "source_path": source_checkpoint,
        "iteration": str(imported_iteration),
    }


def create_initial_checkpoint(
    *,
    model_cfg: ModelConfig,
    ckpt_dir: str | Path,
    checkpoint_name: str = "latest.pt",
    seed: int = 0,
    snapshot_tag: str = "iter_000000",
) -> dict[str, str | None]:
    """Create a deterministic fresh-initialized checkpoint for baseline evaluation."""
    torch.manual_seed(int(seed))
    model = BridgeMonolithTransformer(model_cfg)
    return save_checkpoint_bundle(
        model=model,
        model_cfg=model_cfg,
        ckpt_dir=ckpt_dir,
        checkpoint_name=checkpoint_name,
        iteration=-1,
        run_id=f"initial_{int(time.time() * 1_000_000)}",
        source_checkpoint=None,
        parent_checkpoint=None,
        save_snapshot=True,
        snapshot_tag=snapshot_tag,
    )

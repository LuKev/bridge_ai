"""Checkpoint lineage helpers for learner promotion and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time
from typing import Any, Dict, List

import torch


LATEST_CHECKPOINT_NAME = "latest.pt"


@dataclass
class CheckpointRecord:
    path: str
    iteration: int
    created_at: float
    run_id: str
    parent_checkpoint: str | None = None
    source_checkpoint: str | None = None


def checkpoint_index_path(ckpt_dir: str | Path) -> Path:
    return Path(ckpt_dir) / "index.json"


def load_checkpoint_index(ckpt_dir: str | Path) -> List[CheckpointRecord]:
    path = checkpoint_index_path(ckpt_dir)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [CheckpointRecord(**row) for row in payload]


def _write_checkpoint_index(ckpt_dir: str | Path, records: List[CheckpointRecord]) -> None:
    path = checkpoint_index_path(ckpt_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(record) for record in records], indent=2), encoding="utf-8")


def latest_checkpoint_path(ckpt_dir: str | Path, checkpoint_name: str = LATEST_CHECKPOINT_NAME) -> Path:
    return Path(ckpt_dir) / checkpoint_name


def resolve_latest_checkpoint(ckpt_dir: str | Path, checkpoint_name: str = LATEST_CHECKPOINT_NAME) -> str | None:
    latest = latest_checkpoint_path(ckpt_dir, checkpoint_name=checkpoint_name)
    if latest.exists():
        return str(latest)
    records = load_checkpoint_index(ckpt_dir)
    if records:
        return records[-1].path
    return None


def resolve_latest_snapshot(ckpt_dir: str | Path) -> str | None:
    records = load_checkpoint_index(ckpt_dir)
    if not records:
        return None
    return records[-1].path


def resolve_previous_snapshot(ckpt_dir: str | Path) -> str | None:
    records = load_checkpoint_index(ckpt_dir)
    if len(records) < 2:
        return None
    return records[-2].path


def resolve_anchor_snapshot(ckpt_dir: str | Path) -> str | None:
    records = load_checkpoint_index(ckpt_dir)
    if not records:
        return None
    return records[0].path


def load_checkpoint_payload(path: str | Path) -> Dict[str, Any]:
    return torch.load(Path(path), map_location="cpu")


def resolve_checkpoint_identity(path: str | Path, *, ckpt_dir: str | Path | None = None) -> str:
    checkpoint_path = Path(path)
    if checkpoint_path.exists():
        payload = load_checkpoint_payload(checkpoint_path)
        snapshot_path = payload.get("snapshot_path")
        if isinstance(snapshot_path, str) and snapshot_path:
            return snapshot_path
    if ckpt_dir is not None:
        latest_snapshot = resolve_latest_snapshot(ckpt_dir)
        if latest_snapshot:
            return latest_snapshot
    return str(checkpoint_path)


def save_checkpoint_bundle(
    *,
    model: torch.nn.Module,
    model_cfg,
    ckpt_dir: str | Path,
    checkpoint_name: str,
    iteration: int,
    run_id: str,
    source_checkpoint: str | None,
    parent_checkpoint: str | None,
    save_snapshot: bool,
    snapshot_tag: str | None = None,
) -> Dict[str, str | None]:
    ckpt_root = Path(ckpt_dir)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    snapshot_path: Path | None = None
    if save_snapshot:
        step = max(0, int(iteration) + 1)
        suffix = snapshot_tag if snapshot_tag else f"iter_{step:06d}"
        snapshot_path = ckpt_root / f"{suffix}.pt"

    latest_path = latest_checkpoint_path(ckpt_root, checkpoint_name=checkpoint_name)
    payload = {
        "state_dict": model.state_dict(),
        "config": getattr(model_cfg, "__dict__", {}),
        "iteration": int(iteration),
        "run_id": run_id,
        "source_checkpoint": source_checkpoint,
        "parent_checkpoint": parent_checkpoint,
        "snapshot_path": str(snapshot_path) if snapshot_path else None,
        "saved_at": time.time(),
    }
    torch.save(payload, latest_path)

    if snapshot_path is not None:
        torch.save(payload, snapshot_path)
        records = load_checkpoint_index(ckpt_root)
        records.append(
            CheckpointRecord(
                path=str(snapshot_path),
                iteration=int(iteration),
                created_at=float(payload["saved_at"]),
                run_id=run_id,
                parent_checkpoint=parent_checkpoint,
                source_checkpoint=source_checkpoint,
            )
        )
        _write_checkpoint_index(ckpt_root, records)

    return {
        "latest_path": str(latest_path),
        "snapshot_path": str(snapshot_path) if snapshot_path else None,
    }

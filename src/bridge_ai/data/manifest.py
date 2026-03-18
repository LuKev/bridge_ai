"""Artifact manifest helpers for experiment reproducibility."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from bridge_ai.common.runtime_paths import resolve_runtime_path


@dataclass
class ManifestEntry:
    run_type: str
    run_id: str
    start_time: float
    end_time: float
    config_path: str
    outputs: Dict[str, Any]
    run_signature: str | None = None
    config_snapshot: str | None = None
    metrics: Optional[Dict[str, float]] = None
    status: str = "ok"
    error: Optional[str] = None


def compute_config_signature(config_path: str, *, extra: Dict[str, Any] | None = None) -> str:
    """Build a deterministic hash describing this run configuration."""
    source = resolve_runtime_path(config_path)
    payload = {
        "config_path": str(source.resolve()),
        "config_text": source.read_text(encoding="utf-8"),
    }
    if extra:
        payload["extra"] = extra
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def write_config_snapshot(config_path: str, manifest_path: str, run_id: str) -> str:
    """Persist a frozen config copy for this run and return snapshot path."""
    source = resolve_runtime_path(config_path)
    destination = resolve_runtime_path(manifest_path).parent / "run_configs" / f"{run_id}.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return str(destination)


def append_manifest_entry(
    manifest_path: str,
    *,
    run_type: str,
    run_id: str,
    config_path: str,
    outputs: Dict[str, Any],
    run_signature: str | None = None,
    config_snapshot: str | None = None,
    metrics: Optional[Dict[str, float]] = None,
    status: str = "ok",
    error: Optional[str] = None,
) -> None:
    path = resolve_runtime_path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_manifest(path)

    now = time.time()
    start = existing[-1].end_time if existing else now
    entry = ManifestEntry(
        run_type=run_type,
        run_id=run_id,
        start_time=start,
        end_time=now,
        config_path=str(resolve_runtime_path(config_path)),
        run_signature=run_signature,
        config_snapshot=config_snapshot,
        outputs=outputs,
        metrics=metrics,
        status=status,
        error=error,
    )
    existing.append(entry)
    payload = [asdict(item) for item in existing]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest(path: Path | str) -> List[ManifestEntry]:
    path = resolve_runtime_path(path)
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [ManifestEntry(**row) for row in raw]


def validate_manifest_entry(entry: ManifestEntry) -> List[str]:
    issues: List[str] = []
    if not entry.run_signature:
        issues.append("missing_run_signature")
    if not entry.config_snapshot:
        issues.append("missing_config_snapshot")
    elif not Path(entry.config_snapshot).exists():
        issues.append(f"config_snapshot_missing:{entry.config_snapshot}")
    if not Path(entry.config_path).exists():
        issues.append(f"config_path_missing:{entry.config_path}")
    return issues


def validate_manifest(path: Path | str) -> List[Dict[str, Any]]:
    """Return per-entry reproducibility issues; empty means manifest passes checks."""
    entries = load_manifest(path)
    output = []
    for entry in entries:
        issues = validate_manifest_entry(entry)
        if issues:
            output.append({"run_id": entry.run_id, "issues": issues})
    return output

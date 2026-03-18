"""Replay shard helpers for self-play training windows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time
from typing import Any, Dict, List, Sequence

from bridge_ai.data.buffer import ReplayBuffer


@dataclass
class ReplayShardRecord:
    run_id: str
    path: str
    created_at: float
    items: int
    metadata: Dict[str, Any]


def _index_path(replay_dir: str | Path) -> Path:
    return Path(replay_dir) / "index.json"


def load_replay_index(replay_dir: str | Path) -> List[ReplayShardRecord]:
    path = _index_path(replay_dir)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [ReplayShardRecord(**row) for row in payload]


def _write_replay_index(replay_dir: str | Path, records: Sequence[ReplayShardRecord]) -> None:
    path = _index_path(replay_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(record) for record in records], indent=2),
        encoding="utf-8",
    )


def write_replay_shard(
    *,
    buffer: ReplayBuffer,
    replay_dir: str | Path,
    latest_path: str | Path,
    run_id: str,
    keep_last_shards: int = 0,
    metadata: Dict[str, Any] | None = None,
) -> ReplayShardRecord:
    replay_root = Path(replay_dir)
    replay_root.mkdir(parents=True, exist_ok=True)

    latest = Path(latest_path)
    buffer.save_json(latest)

    shard_path = replay_root / "shards" / f"{run_id}.json"
    buffer.save_json(shard_path)

    record = ReplayShardRecord(
        run_id=run_id,
        path=str(shard_path),
        created_at=time.time(),
        items=len(buffer),
        metadata=dict(metadata or {}),
    )
    existing = load_replay_index(replay_root)
    existing.append(record)

    if keep_last_shards > 0 and len(existing) > keep_last_shards:
        pruned = existing[:-keep_last_shards]
        existing = existing[-keep_last_shards:]
        for old in pruned:
            old_path = Path(old.path)
            if old_path.exists():
                old_path.unlink()

    _write_replay_index(replay_root, existing)
    return record


def load_replay_window(
    *,
    replay_dir: str | Path,
    latest_path: str | Path,
    max_shards: int = 1,
    max_items: int = 0,
) -> tuple[ReplayBuffer, List[ReplayShardRecord]]:
    records = load_replay_index(replay_dir)
    if records:
        if max_shards > 0:
            records = records[-max_shards:]
        buffer = ReplayBuffer()
        for record in records:
            shard_path = Path(record.path)
            if not shard_path.exists():
                continue
            shard = ReplayBuffer.load_json(shard_path)
            buffer.extend(shard._items)
    else:
        latest = Path(latest_path)
        if not latest.exists():
            raise FileNotFoundError(f"replay file does not exist: {latest}")
        buffer = ReplayBuffer.load_json(latest)
        records = [
            ReplayShardRecord(
                run_id="latest",
                path=str(latest),
                created_at=latest.stat().st_mtime,
                items=len(buffer),
                metadata={},
            )
        ]

    if max_items > 0 and len(buffer) > max_items:
        trimmed = ReplayBuffer(max_items=max_items)
        trimmed.extend(buffer._items[-max_items:])
        buffer = trimmed

    return buffer, records

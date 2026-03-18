"""Simple Elo-style rating persistence for checkpoint leagues."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, List


DEFAULT_ELO = 1500.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def score_from_diff(total_diff: float, *, draw_margin: float = 0.0) -> float:
    if abs(total_diff) <= draw_margin:
        return 0.5
    return 1.0 if total_diff > 0.0 else 0.0


def update_elo(
    rating_a: float,
    rating_b: float,
    *,
    actual_a: float,
    k_factor: float,
) -> tuple[float, float, float]:
    expected_a = expected_score(rating_a, rating_b)
    delta = k_factor * (actual_a - expected_a)
    return rating_a + delta, rating_b - delta, expected_a


def load_rating_table(path: str | Path) -> Dict[str, float]:
    source = Path(path)
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    return {str(key): float(value) for key, value in payload.items()}


def write_rating_table(path: str | Path, table: Dict[str, float]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(table, indent=2, sort_keys=True), encoding="utf-8")


def append_rating_history(path: str | Path, records: List[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

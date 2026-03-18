"""Benchmark suite definitions for fixed-board evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class BenchmarkSuite:
    name: str
    seed: int
    rounds: int


BENCHMARK_SUITES: dict[str, BenchmarkSuite] = {
    "quick": BenchmarkSuite(name="quick", seed=10_000, rounds=8),
    "gating": BenchmarkSuite(name="gating", seed=20_000, rounds=32),
    "ladder": BenchmarkSuite(name="ladder", seed=30_000, rounds=128),
}


def resolve_benchmark_suite(
    *,
    suite_name: str | None,
    rounds: int,
    seed: int,
    seed_sequence: Sequence[int] | None,
) -> tuple[str, List[int]]:
    if seed_sequence:
        return (suite_name or "explicit"), [int(value) for value in seed_sequence[:rounds]]

    if suite_name and suite_name in BENCHMARK_SUITES:
        suite = BENCHMARK_SUITES[suite_name]
        suite_rounds = rounds if rounds > 0 else suite.rounds
        return suite.name, [suite.seed + i for i in range(suite_rounds)]

    total = max(1, rounds)
    return (suite_name or "custom"), [seed + i for i in range(total)]

from __future__ import annotations

import random


def make_rng(seed: int | None = None) -> random.Random:
    return random.Random(seed)


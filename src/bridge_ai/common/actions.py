"""Unified action encoding for auction and play."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .types import ActionKind


PASS_ACTION = 0
DOUBLE_ACTION = 1
REDOUBLE_ACTION = 2
BID_BASE = 3
PLAY_BASE = 40
NO_OP_ACTION = 200

MAX_BID_CODE = 39
MAX_PLAY_CODE = 55

STRAIN_SYMBOLS = {
    0: "C",
    1: "D",
    2: "H",
    3: "S",
    4: "N",
}


def is_bid_code(code: int) -> bool:
    return BID_BASE <= code <= MAX_BID_CODE or code in (
        PASS_ACTION,
        DOUBLE_ACTION,
        REDOUBLE_ACTION,
    )


def is_play_code(code: int) -> bool:
    return PLAY_BASE <= code < PLAY_BASE + 52


def bid_code(level: int, strain: int) -> int:
    # level: 1..7, strain: 0..4 for C/D/H/S/N (0-4)
    return BID_BASE + (level - 1) * 5 + strain


def decode_bid_code(code: int) -> tuple[int, int] | None:
    if code == PASS_ACTION:
        return None
    if code == DOUBLE_ACTION:
        return (-1, -1)
    if code == REDOUBLE_ACTION:
        return (-2, -1)
    if not is_bid_code(code):
        raise ValueError(f"not a bidding action: {code}")
    if code >= BID_BASE:
        rank = (code - BID_BASE) // 5 + 1
        strain = (code - BID_BASE) % 5
        return rank, strain
    return None


def play_code_from_index(card_index: int) -> int:
    if not 0 <= card_index < 52:
        raise ValueError("card index must be in [0, 51]")
    return PLAY_BASE + card_index


def play_index_from_code(code: int) -> int:
    if not is_play_code(code):
        raise ValueError("not a card play code")
    return code - PLAY_BASE


def unified_action_space() -> List[int]:
    return list(range(PASS_ACTION, PLAY_BASE + 52))


def bid_to_string(code: int) -> str:
    if code == PASS_ACTION:
        return "PASS"
    if code == DOUBLE_ACTION:
        return "DOUBLE"
    if code == REDOUBLE_ACTION:
        return "REDOUBLE"
    level, strain = decode_bid_code(code)
    if level is None:
        raise ValueError(f"invalid bid code {code}")
    return f"{level}{STRAIN_SYMBOLS[strain]}"

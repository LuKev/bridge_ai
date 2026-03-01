"""Domain primitives for bridge state and action encoding."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Tuple


class Suit(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3


class Rank(IntEnum):
    C2 = 0
    C3 = 1
    C4 = 2
    C5 = 3
    C6 = 4
    C7 = 5
    C8 = 6
    C9 = 7
    CT = 8
    JT = 9
    QT = 10
    KT = 11
    AT = 12
    A = AT


class Seat(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    @property
    def partner(self) -> "Seat":
        return Seat((self + 2) % 4)

    def next(self) -> "Seat":
        return Seat((self + 1) % 4)


class Phase(str, Enum):
    AUCTION = "auction"
    LEAD = "lead"
    PLAY = "play"
    DEFENSE = "defense"


class Vulnerability(IntEnum):
    NONE = 0
    N_S = 1
    E_W = 2
    BOTH = 3


class ActionKind(IntEnum):
    BID = 0
    PLAY = 1


@dataclass(frozen=True)
class Card:
    suit: Suit
    rank: Rank

    def index(self) -> int:
        return int(self.suit) * 13 + int(self.rank)


@dataclass(frozen=True)
class Contract:
    level: int
    strain: int
    declarer: Seat
    doubled: int = 0


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    code: int

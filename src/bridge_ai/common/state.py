"""State containers shared between env, search, training, and UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .types import Card, Contract, Phase, Seat, Vulnerability


@dataclass(frozen=True)
class Trick:
    leader: Seat
    cards: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    winner: Optional[Seat] = None


@dataclass(frozen=True)
class BridgeState:
    hands: Dict[Seat, Tuple[Card, ...]]
    phase: Phase
    turn: int
    dealer: Seat
    current_player: Seat
    vulnerable: Vulnerability
    auction: Tuple[int, ...] = field(default_factory=tuple)
    contract: Optional[Contract] = None
    declarer: Optional[Seat] = None
    dummy: Optional[Seat] = None
    trick_leader: Optional[Seat] = None
    trick_number: int = 0
    tricks_won: Tuple[int, int] = (0, 0)
    current_trick: Tuple[Tuple[Seat, int], ...] = field(default_factory=tuple)
    played_cards: Tuple[Tuple[Seat, int], ...] = field(default_factory=tuple)
    done: bool = False
    result: Optional[float] = None

    def to_actor_dict(self, perspective: Seat) -> Dict:
        # Minimal privacy-aware public observation.
        visible_hands = {p: self.hands[p] if p == perspective else () for p in self.hands}
        return {
            "phase": self.phase.value,
            "turn": self.turn,
            "current_player": self.current_player,
            "dealer": self.dealer,
            "vulnerable": int(self.vulnerable),
            "auction": self.auction,
            "contract_level": -1 if self.contract is None else self.contract.level,
            "contract_strain": -1 if self.contract is None else self.contract.strain,
            "contract_doubled": 0 if self.contract is None else self.contract.doubled,
            "declarer": -1 if self.declarer is None else int(self.declarer),
            "dummy": -1 if self.dummy is None else int(self.dummy),
            "trick_number": self.trick_number,
            "tricks_north_south": self.tricks_won[0],
            "tricks_east_west": self.tricks_won[1],
            "done": self.done,
            "result": 0.0 if self.result is None else self.result,
            "visible_hands": {p.name: tuple(card.index() for card in cards) for p, cards in visible_hands.items()},
            "current_trick": list(self.current_trick),
            "played_cards": list(self.played_cards),
        }

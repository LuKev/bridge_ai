"""Card and deal utilities."""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Tuple

from .types import Card, Rank, Seat, Suit


def build_deck() -> List[Card]:
    return [Card(suit=suit, rank=rank) for suit in Suit for rank in Rank]


def index_to_card(index: int) -> Card:
    if not 0 <= index < 52:
        raise ValueError(f"invalid card index: {index}")
    suit = Suit(index // 13)
    rank = Rank(index % 13)
    return Card(suit=suit, rank=rank)


def shuffle_and_deal(seed: int | None = None) -> Dict[Seat, Tuple[Card, ...]]:
    deck = build_deck()
    rng = random.Random(seed)
    rng.shuffle(deck)
    return {seat: tuple(deck[i * 13 : (i + 1) * 13]) for i, seat in enumerate(Seat)}


RANK_SYMBOLS: Dict[Rank, str] = {
    Rank.C2: "2",
    Rank.C3: "3",
    Rank.C4: "4",
    Rank.C5: "5",
    Rank.C6: "6",
    Rank.C7: "7",
    Rank.C8: "8",
    Rank.C9: "9",
    Rank.CT: "T",
    Rank.JT: "J",
    Rank.QT: "Q",
    Rank.KT: "K",
    Rank.AT: "A",
    Rank.A: "A",
}


SUIT_SYMBOLS: Dict[Suit, str] = {
    Suit.CLUBS: "♣",
    Suit.DIAMONDS: "♦",
    Suit.HEARTS: "♥",
    Suit.SPADES: "♠",
}


def card_to_string(card: Card) -> str:
    return f"{RANK_SYMBOLS[card.rank]}{SUIT_SYMBOLS[card.suit]}"


def serialize_hand(hand: Iterable[Card]) -> Tuple[str, ...]:
    return tuple(card_to_string(card) for card in hand)


def hand_to_indices(hand: Iterable[Card]) -> Tuple[int, ...]:
    return tuple(card.index() for card in hand)


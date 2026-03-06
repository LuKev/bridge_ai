"""Constraint-aware posterior sampling for complete bridge deals."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Dict, List

import torch

from bridge_ai.common.cards import index_to_card
from bridge_ai.common.types import Seat


@dataclass
class SampledDeal:
    owners: List[int]
    valid: bool
    hands: Dict[Seat, tuple]


def sample_hidden_deal(example, owner_logits: torch.Tensor, *, seed: int = 0) -> SampledDeal:
    rng = Random(seed)
    logits = owner_logits.detach().cpu()
    owners = [-1] * 52
    remaining = {seat: 13 for seat in range(4)}
    own_set = set(int(card) for card in example.own_cards)
    acting_seat = int(example.acting_seat)

    for card_index in own_set:
        owners[card_index] = acting_seat
        remaining[acting_seat] -= 1

    unseen_cards = [idx for idx in range(52) if idx not in own_set]
    unseen_cards.sort(key=lambda idx: float(torch.softmax(logits[idx], dim=-1).max()), reverse=True)

    for card_index in unseen_cards:
        probs = torch.softmax(logits[card_index], dim=-1).tolist()
        masked: List[tuple[int, float]] = []
        for seat in range(4):
            if seat == acting_seat:
                continue
            if remaining[seat] <= 0:
                continue
            masked.append((seat, float(probs[seat])))
        if not masked:
            return SampledDeal(owners=owners, valid=False, hands={})
        total = sum(weight for _, weight in masked)
        if total <= 0.0:
            normalized = [(seat, 1.0 / len(masked)) for seat, _ in masked]
        else:
            normalized = [(seat, weight / total) for seat, weight in masked]
        draw = rng.random()
        cumulative = 0.0
        chosen = normalized[-1][0]
        for seat, weight in normalized:
            cumulative += weight
            if draw <= cumulative:
                chosen = seat
                break
        owners[card_index] = chosen
        remaining[chosen] -= 1

    if any(count != 0 for count in remaining.values()):
        return SampledDeal(owners=owners, valid=False, hands={})

    hands: Dict[Seat, tuple] = {}
    for seat in range(4):
        seat_cards = [index_to_card(idx) for idx, owner in enumerate(owners) if owner == seat]
        if len(seat_cards) != 13:
            return SampledDeal(owners=owners, valid=False, hands={})
        hands[Seat(seat)] = tuple(sorted(seat_cards, key=lambda card: card.index()))
    return SampledDeal(owners=owners, valid=True, hands=hands)

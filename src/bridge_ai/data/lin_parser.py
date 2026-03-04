"""Utilities for decoding and replaying BridgeBase LIN hand records."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, unquote_plus, urlparse
from html import unescape
import re

from bridge_ai.common.actions import (
    DOUBLE_ACTION,
    PASS_ACTION,
    REDOUBLE_ACTION,
    bid_code,
)
from bridge_ai.common.cards import index_to_card
from bridge_ai.common.state import BridgeState
from bridge_ai.common.types import Seat, Vulnerability, Phase
from bridge_ai.env.bridge_env import BridgeEnv


_SUIT_TO_CODE = {"C": 0, "D": 1, "H": 2, "S": 3}
_RANK_TO_CODE = {"2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6, "9": 7, "T": 8, "J": 9, "Q": 10, "K": 11, "A": 12}
_STRAIN_TO_CODE = {"C": 0, "D": 1, "H": 2, "S": 3, "N": 4}

_VULNERABILITY_MAP = {
    "O": Vulnerability.NONE,
    "N": Vulnerability.N_S,
    "E": Vulnerability.E_W,
    "B": Vulnerability.BOTH,
}

_DEALER_TO_SEAT = {"1": Seat.SOUTH, "2": Seat.WEST, "3": Seat.NORTH, "4": Seat.EAST}


def _decode_lin_payload(raw: str) -> str:
    """Normalize raw LIN or URL-encoded LIN into raw LIN payload text."""
    body = raw.strip()
    if "lin=" in body:
        match = re.search(r"lin=([^&]+)", body)
        if match:
            body = match.group(1)

    parsed = urlparse(body)
    if parsed.query:
        query = parse_qs(parsed.query)
        if query.get("lin"):
            body = query["lin"][0]

    return unescape(unquote_plus(body))


def _bid_token_to_action(token: str) -> int:
    token = token.strip().upper()
    if token in {"P", "PASS"}:
        return PASS_ACTION
    if token in {"X", "D", "DBL", "D1"}:
        return DOUBLE_ACTION
    if token in {"XX", "RD", "RR", "REDBL", "REDOUBLE"}:
        return REDOUBLE_ACTION

    cleaned = re.sub(r"[^0-9CDHSN]", "", token)
    if len(cleaned) != 2 or not cleaned[0].isdigit() or cleaned[1] not in _STRAIN_TO_CODE:
        raise ValueError(f"unexpected bid token {token!r}")

    level = int(cleaned[0])
    strain = _STRAIN_TO_CODE[cleaned[1]]
    return bid_code(level, strain)


def _card_token_to_code(token: str) -> int:
    token = token.strip().upper()
    if len(token) != 2:
        raise ValueError(f"unexpected card token {token!r}")

    suit = _SUIT_TO_CODE[token[0]]
    rank = _RANK_TO_CODE[token[1]]
    return 40 + 13 * suit + rank


def _parse_hand_text(text: str) -> Tuple:
    cards: List = []
    current_suit: Optional[int] = None
    for char in text:
        if char in _SUIT_TO_CODE:
            current_suit = _SUIT_TO_CODE[char]
            continue

        if char not in _RANK_TO_CODE:
            raise ValueError(f"unexpected card character {char!r}")
        if current_suit is None:
            raise ValueError(f"card token starts without suit: {text!r}")

        rank = _RANK_TO_CODE[char]
        cards.append(index_to_card(13 * current_suit + rank))

    return tuple(sorted(cards, key=lambda c: c.index()))


def _parse_lin_hands(raw: str):
    if not raw:
        raise ValueError("empty lin hand section")

    parts = raw.split(",")
    if not parts:
        raise ValueError(f"could not parse hand token {raw!r}")

    start_dealer = parts[0][0] if parts[0] and parts[0][0].isdigit() else None
    dealer = None
    if start_dealer is not None:
        if start_dealer not in _DEALER_TO_SEAT:
            raise ValueError(f"unknown dealer token {start_dealer!r} in md {raw!r}")
        dealer = _DEALER_TO_SEAT[start_dealer]
        parts[0] = parts[0][1:]

    # Keep declared hands only; some records omit the 4th hand.
    declared = parts[:4]
    while declared and declared[-1] == "":
        declared.pop()

    if len(declared) < 3:
        raise ValueError(f"not enough hands in md token: {raw!r}")

    order_seed = 0 if dealer is None else int(dealer)
    hand_order = [Seat((order_seed + i) % 4) for i in range(4)]

    hands: Dict[Seat, Tuple] = {}
    seen_indices = set()
    missing_seat = None
    for i, token in enumerate(declared):
        if i >= 4:
            break
        seat = hand_order[i]
        if token == "":
            if missing_seat is not None:
                raise ValueError(f"unexpected second missing hand in md token: {raw!r}")
            missing_seat = seat
            continue
        parsed = _parse_hand_text(token)
        duplicate = [card.index() for card in parsed if card.index() in seen_indices]
        if duplicate:
            raise ValueError(f"duplicate card index in md token {raw!r}: {duplicate}")
        hands[seat] = parsed
        seen_indices.update(card.index() for card in parsed)

    if missing_seat is not None and len(seen_indices) > 39:
        raise ValueError("md hand allocation has more than 39 cards; cannot infer missing hand")

    if len(declared) == 3 and missing_seat is None:
        missing_candidates = [seat for seat in hand_order if seat not in hands]
        if len(missing_candidates) != 1:
            raise ValueError(f"could not infer missing hand for md token {raw!r}")
        missing_seat = missing_candidates[0]

    if missing_seat is not None:
        missing_idx = tuple(i for i in range(52) if i not in seen_indices)
        if len(missing_idx) != 13:
            raise ValueError(f"could not infer missing hand for md token {raw!r}")
        hands[missing_seat] = tuple(index_to_card(idx) for idx in missing_idx)

    all_seen_seats = set(hands.keys())
    for seat in hand_order:
        if seat not in all_seen_seats:
            raise ValueError(f"md hand section missing seat {seat} in {raw!r}")

    return hands, dealer


@dataclass(frozen=True)
class ParsedLinRecord:
    source: str
    hands: Dict[Seat, Tuple]
    bids: Tuple[int, ...]
    plays: Tuple[int, ...]
    claim: Optional[int]
    dealer: Optional[Seat]
    vulnerability: Vulnerability
    fields: Dict[str, str]


def parse_lin_record(raw: str) -> ParsedLinRecord:
    lin = _decode_lin_payload(raw)
    fields: Dict[str, str] = {}
    bids: List[int] = []
    plays: List[int] = []
    claim: Optional[int] = None
    vulnerability = Vulnerability.NONE
    dealer: Optional[Seat] = None
    hands: Dict[Seat, Tuple] = {}

    parts = lin.split("|")
    for i in range(0, len(parts) - 1, 2):
        key = parts[i]
        value = parts[i + 1]
        if key == "sv":
            vulnerability = _VULNERABILITY_MAP.get(value.upper(), vulnerability)
        elif key == "md":
            hands, dealer = _parse_lin_hands(value)
        elif key == "mb":
            if value:
                bids.append(_bid_token_to_action(value))
        elif key == "pc":
            if value:
                plays.append(_card_token_to_code(value))
        elif key == "mc":
            try:
                claim = int(value)
            except ValueError:
                claim = None
        elif key:
            fields[key] = value

    if not hands:
        raise ValueError(f"no md hands in lin: {raw[:120]}")
    if not bids:
        raise ValueError(f"no auction sequence in lin: {raw[:120]}")

    if any(len(cards) > 13 for cards in hands.values()):
        raise ValueError("parsed hand has >13 cards")

    all_seen = sorted(card.index() for cards in hands.values() for card in cards)
    if len(all_seen) != len(set(all_seen)):
        raise ValueError("parsed lin hand allocation has duplicate cards")

    return ParsedLinRecord(
        source=raw,
        hands=hands,
        bids=tuple(bids),
        plays=tuple(plays),
        claim=claim,
        dealer=dealer,
        vulnerability=vulnerability,
        fields=fields,
    )


def parse_lin_records(raw_records: Iterable[str]) -> Tuple[ParsedLinRecord, ...]:
    return tuple(parse_lin_record(raw) for raw in raw_records)


def load_lin_records_from_path(path: str | Path) -> Tuple[ParsedLinRecord, ...]:
    record_path = Path(path)
    records = [line.strip() for line in record_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return parse_lin_records(records)


def run_lin_record(parsed: ParsedLinRecord, *, env: BridgeEnv | None = None) -> BridgeState:
    """Replay a parsed LIN record through BridgeEnv and validate every transition."""
    env = env or BridgeEnv()
    state = env.reset(hands=parsed.hands)
    state = replace(
        state,
        dealer=parsed.dealer if parsed.dealer is not None else Seat.NORTH,
        current_player=parsed.dealer if parsed.dealer is not None else Seat.NORTH,
        vulnerable=parsed.vulnerability,
    )
    env.set_state(state)

    for action_code in parsed.bids:
        legal = set(env.legal_actions())
        assert action_code in legal, f"illegal bid {action_code} in record"
        state, _, done, _ = env.step(action_code)
        if done:
            break

    for action_code in parsed.plays:
        if state.phase != Phase.PLAY:
            break
        legal = set(env.legal_actions())
        assert action_code in legal, f"illegal play {action_code} in record"
        state, _, done, _ = env.step(action_code)
        if done:
            break

    return state

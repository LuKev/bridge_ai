"""Dataset builders for bidding and hidden-hand belief training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from dataclasses import replace
from pathlib import Path
from random import Random
from typing import Any, Dict, List, Sequence
import json

from bridge_ai.common.types import Seat
from bridge_ai.data.bootstrap_records import SAMPLE_LIN_RECORDS
from bridge_ai.data.lin_parser import ParsedLinRecord, load_lin_records_from_path, parse_lin_records
from bridge_ai.data.tournament_bootstrap import (
    DEFAULT_EVENT_FILES,
    DEFAULT_RELEASE_URL,
    TournamentRoomRecord,
    build_source_manifest,
    load_tournament_room_records,
)
from bridge_ai.env.bridge_env import BridgeEnv
from bridge_ai.common.actions import play_code_from_index
from bridge_ai.common.state import BridgeState
from bridge_ai.common.types import Phase


@dataclass
class BeliefExample:
    record_id: str
    bid_index: int
    acting_seat: int
    dealer: int
    vulnerable: int
    phase: str
    trick_number: int
    played_count: int
    own_cards: List[int]
    auction_prefix: List[int]
    visible_dummy_cards: List[int]
    current_trick: List[List[int]]
    played_cards: List[List[int]]
    legal_actions: List[int]
    target_bid: int
    has_bid_target: bool
    card_owners: List[int]
    belief_target_mask: List[int]


def _current_owner_labels(state: BridgeState) -> List[int]:
    owners = [-1] * 52
    for seat, cards in state.hands.items():
        for card in cards:
            owners[card.index()] = int(seat)
    return owners


def _visible_dummy_cards(state: BridgeState) -> List[int]:
    if state.dummy is None:
        return []
    if len(state.played_cards) == 0:
        return []
    return sorted(card.index() for card in state.hands.get(state.dummy, tuple()))


def _belief_mask_for_state(state: BridgeState, acting_seat: Seat) -> List[int]:
    hidden = [0] * 52
    own_cards = {card.index() for card in state.hands.get(acting_seat, tuple())}
    dummy_cards = set(_visible_dummy_cards(state))
    public_cards = {card_idx for _, card_idx in state.played_cards}
    public_cards.update(card_idx for _, card_idx in state.current_trick)
    for seat, cards in state.hands.items():
        for card in cards:
            card_index = card.index()
            if card_index in own_cards:
                continue
            if card_index in dummy_cards:
                continue
            if card_index in public_cards:
                continue
            hidden[card_index] = 1
    return hidden


def _state_example(
    *,
    record_id: str,
    bid_index: int,
    state: BridgeState,
    legal_actions: List[int],
    target_bid: int,
    has_bid_target: bool,
) -> BeliefExample:
    acting_seat = state.current_player
    own_cards = sorted(card.index() for card in state.hands[acting_seat])
    return BeliefExample(
        record_id=record_id,
        bid_index=bid_index,
        acting_seat=int(acting_seat),
        dealer=int(state.dealer),
        vulnerable=int(state.vulnerable),
        phase=state.phase.value,
        trick_number=int(state.trick_number),
        played_count=len(state.played_cards),
        own_cards=own_cards,
        auction_prefix=[int(action) for action in state.auction],
        visible_dummy_cards=_visible_dummy_cards(state),
        current_trick=[[int(seat), int(card_idx)] for seat, card_idx in state.current_trick],
        played_cards=[[int(seat), int(card_idx)] for seat, card_idx in state.played_cards],
        legal_actions=legal_actions,
        target_bid=target_bid,
        has_bid_target=has_bid_target,
        card_owners=_current_owner_labels(state),
        belief_target_mask=_belief_mask_for_state(state, acting_seat),
    )


def _example_from_state(parsed: ParsedLinRecord, record_id: str, bid_index: int, env: BridgeEnv) -> BeliefExample:
    if env.state is None:
        raise RuntimeError("environment state must exist")
    return _state_example(
        record_id=record_id,
        bid_index=bid_index,
        state=env.state,
        legal_actions=[int(action) for action in env.legal_actions()],
        target_bid=int(parsed.bids[bid_index]),
        has_bid_target=True,
    )


def _play_examples_from_actions(
    *,
    record_id: str,
    env: BridgeEnv,
    plays: Sequence[int],
    bid_index_offset: int,
) -> List[BeliefExample]:
    examples: List[BeliefExample] = []
    for play_index, action_code in enumerate(plays):
        if env.state is None or env.state.done or env.state.phase != Phase.PLAY:
            break
        legal = [int(action) for action in env.legal_actions()]
        if action_code not in legal:
            break
        examples.append(
            _state_example(
                record_id=record_id,
                bid_index=bid_index_offset + play_index,
                state=env.state,
                legal_actions=legal,
                target_bid=-1,
                has_bid_target=False,
            )
        )
        next_state, _, done, _ = env.step(int(action_code))
        env.set_state(next_state)
        if done:
            break
    return examples


def _examples_from_structured_record(
    *,
    hands,
    bids,
    plays,
    dealer,
    vulnerability,
    record_id: str,
) -> List[BeliefExample]:
    env = BridgeEnv()
    state = env.reset(hands=hands)
    state = replace(
        state,
        dealer=dealer,
        current_player=dealer,
        vulnerable=vulnerability,
    )
    env.set_state(state)

    examples: List[BeliefExample] = []
    for bid_index, bid in enumerate(bids):
        if env.state is None or env.state.done:
            break
        legal_actions = env.legal_actions()
        if bid not in legal_actions:
            break
        examples.append(
            _state_example(
                record_id=record_id,
                bid_index=bid_index,
                state=env.state,
                legal_actions=[int(action) for action in env.legal_actions()],
                target_bid=int(bid),
                has_bid_target=True,
            )
        )
        next_state, _, done, _ = env.step(int(bid))
        env.set_state(next_state)
        if done:
            break
    examples.extend(
        _play_examples_from_actions(
            record_id=record_id,
            env=env,
            plays=plays,
            bid_index_offset=len(examples),
        )
    )
    return examples


def examples_from_record(parsed: ParsedLinRecord, record_id: str) -> List[BeliefExample]:
    dealer = parsed.dealer if parsed.dealer is not None else Seat.NORTH
    return _examples_from_structured_record(
        hands=parsed.hands,
        bids=parsed.bids,
        plays=parsed.plays,
        dealer=dealer,
        vulnerability=parsed.vulnerability,
        record_id=record_id,
    )


def examples_from_tournament_record(record: TournamentRoomRecord) -> List[BeliefExample]:
    return _examples_from_structured_record(
        hands=record.hands,
        bids=record.bids,
        plays=tuple(
            play_code_from_index((3 - "SHDC".index(card[0])) * 13 + "23456789TJQKA".index(card[1]))
            for trick in record.play_tricks
            for card in trick
            if card
        ),
        dealer=record.dealer,
        vulnerability=record.vulnerability,
        record_id=record.record_id,
    )


def _load_records(data_sources: Sequence[str], max_games: int) -> List[ParsedLinRecord]:
    records: List[ParsedLinRecord] = []
    for data_source in data_sources:
        source_path = Path(data_source)
        if not source_path.exists():
            continue
        loaded = load_lin_records_from_path(source_path)
        records.extend(list(loaded))
        if len(records) >= max_games:
            return records[:max_games]

    if len(records) < max_games:
        needed = max_games - len(records)
        records.extend(list(parse_lin_records(SAMPLE_LIN_RECORDS[:needed])))
    return records[:max_games]


def build_dataset(
    *,
    data_sources: Sequence[str],
    max_games: int,
    holdout_fraction: float,
    seed: int,
) -> Dict[str, Any]:
    records = _load_records(data_sources, max_games=max_games)
    rng = Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    holdout_count = max(1, int(round(len(indices) * holdout_fraction))) if len(indices) > 1 else 0
    holdout_ids = set(indices[:holdout_count])

    train_examples: List[BeliefExample] = []
    holdout_examples: List[BeliefExample] = []
    for index, record in enumerate(records):
        record_id = f"record_{index:04d}"
        examples = examples_from_record(record, record_id)
        if index in holdout_ids:
            holdout_examples.extend(examples)
        else:
            train_examples.extend(examples)

    return {
        "summary": {
            "num_records": len(records),
            "train_examples": len(train_examples),
            "holdout_examples": len(holdout_examples),
            "train_play_examples": sum(1 for example in train_examples if example.phase == Phase.PLAY.value),
            "holdout_play_examples": sum(1 for example in holdout_examples if example.phase == Phase.PLAY.value),
            "max_games": max_games,
            "holdout_fraction": holdout_fraction,
            "data_sources": list(data_sources),
            "used_embedded_records": len(records) > len([source for source in data_sources if Path(source).exists()]),
        },
        "train": [asdict(example) for example in train_examples],
        "holdout": [asdict(example) for example in holdout_examples],
    }


def build_tournament_dataset(
    *,
    archive_url: str = DEFAULT_RELEASE_URL,
    event_files: Sequence[str] = DEFAULT_EVENT_FILES,
    cache_dir: str | Path,
    extract_dir: str | Path | None,
    max_games: int,
    holdout_fraction: float,
    seed: int,
    records_per_event: int | None = None,
) -> Dict[str, Any]:
    records = load_tournament_room_records(
        archive_url=archive_url,
        event_files=event_files,
        cache_dir=cache_dir,
        extract_dir=extract_dir,
        max_records=max_games,
        records_per_event=records_per_event,
    )
    rng = Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    holdout_count = max(1, int(round(len(indices) * holdout_fraction))) if len(indices) > 1 else 0
    holdout_ids = set(indices[:holdout_count])

    train_examples: List[BeliefExample] = []
    holdout_examples: List[BeliefExample] = []
    for index, record in enumerate(records):
        examples = examples_from_tournament_record(record)
        if index in holdout_ids:
            holdout_examples.extend(examples)
        else:
            train_examples.extend(examples)

    return {
        "summary": {
            "num_records": len(records),
            "train_examples": len(train_examples),
            "holdout_examples": len(holdout_examples),
            "train_play_examples": sum(1 for example in train_examples if example.phase == Phase.PLAY.value),
            "holdout_play_examples": sum(1 for example in holdout_examples if example.phase == Phase.PLAY.value),
            "max_games": max_games,
            "holdout_fraction": holdout_fraction,
            "archive_url": archive_url,
            "event_files": list(event_files),
            "records_per_event": records_per_event,
            "used_embedded_records": False,
        },
        "source_manifest": build_source_manifest(
            archive_url=archive_url,
            event_files=event_files,
            records=records,
        ),
        "train": [asdict(example) for example in train_examples],
        "holdout": [asdict(example) for example in holdout_examples],
    }


def save_dataset(payload: Dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_dataset(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_examples(path: str | Path, split: str) -> List[BeliefExample]:
    payload = load_dataset(path)
    return [BeliefExample(**example) for example in payload.get(split, [])]

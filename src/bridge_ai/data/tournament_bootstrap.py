"""Reproducible tournament bootstrap dataset ingestion from public archives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import json
import tarfile
import tempfile
import urllib.request

from bridge_ai.common.actions import DOUBLE_ACTION, PASS_ACTION, REDOUBLE_ACTION, bid_code
from bridge_ai.common.cards import index_to_card
from bridge_ai.common.types import Seat, Vulnerability


DEFAULT_RELEASE_URL = "https://github.com/ureshvahalia/bridge_deals_db/releases/download/v1.0/bridge_deals.tar.gz"
DEFAULT_EVENT_FILES = (
    "JSON/WBF/2023/BermudaBowl2023.json",
    "JSON/WBF/2023/VeniceCup2023.json",
    "JSON/WBF/2023/dOrsiTrophy2023.json",
    "JSON/WBF/2023/WuhanCup2023.json",
)

_SEAT_MAP = {"N": Seat.NORTH, "E": Seat.EAST, "S": Seat.SOUTH, "W": Seat.WEST}
_VULNERABILITY_MAP = {
    "none": Vulnerability.NONE,
    "n-s": Vulnerability.N_S,
    "ns": Vulnerability.N_S,
    "n": Vulnerability.N_S,
    "e-w": Vulnerability.E_W,
    "ew": Vulnerability.E_W,
    "we": Vulnerability.E_W,
    "e": Vulnerability.E_W,
    "both": Vulnerability.BOTH,
    "all": Vulnerability.BOTH,
    "b": Vulnerability.BOTH,
}
_STRAIN_MAP = {"C": 0, "D": 1, "H": 2, "S": 3, "N": 4}
_SUIT_ORDER = "SHDC"
_RANK_ORDER = "23456789TJQKA"


@dataclass(frozen=True)
class TournamentRoomRecord:
    record_id: str
    source_file: str
    event_name: str
    room: str
    dealer: Seat
    vulnerability: Vulnerability
    hands: Dict[Seat, Tuple]
    bids: Tuple[int, ...]
    play_tricks: Tuple[Tuple[str | None, ...], ...]


def _download_archive(url: str, cache_dir: str | Path) -> Path:
    destination_dir = Path(cache_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    archive_path = destination_dir / Path(url).name
    if archive_path.exists():
        return archive_path
    with urllib.request.urlopen(url) as response:
        archive_path.write_bytes(response.read())
    return archive_path


def _extract_json_files(archive_path: Path, event_files: Sequence[str], extract_dir: str | Path) -> List[Path]:
    destination_dir = Path(extract_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    extracted: List[Path] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {member.name.lstrip("./"): member for member in archive.getmembers()}
        for rel_path in event_files:
            member = members.get(rel_path)
            if member is None:
                continue
            archive.extract(member, path=destination_dir)
            extracted.append(destination_dir / member.name)
    return extracted


def _call_to_action(token: str) -> int:
    cleaned = str(token).strip().upper()
    if cleaned in {"P", "PASS"}:
        return PASS_ACTION
    if cleaned in {"X", "D", "DBL"}:
        return DOUBLE_ACTION
    if cleaned in {"R", "XX", "RD", "REDBL", "REDOUBLE"}:
        return REDOUBLE_ACTION
    if len(cleaned) != 2 or not cleaned[0].isdigit() or cleaned[1] not in _STRAIN_MAP:
        raise ValueError(f"unexpected tournament call token: {token!r}")
    return bid_code(int(cleaned[0]), _STRAIN_MAP[cleaned[1]])


def _parse_hand_cards(suit_strings: Sequence[str]) -> Tuple:
    cards = []
    for suit_index, ranks in enumerate(suit_strings):
        if ranks in (None, "", "None"):
            continue
        for rank_char in str(ranks):
            rank_idx = _RANK_ORDER.index(rank_char)
            card_index = (3 - suit_index) * 13 + rank_idx
            cards.append(index_to_card(card_index))
    return tuple(sorted(cards, key=lambda card: card.index()))


def _parse_deal(deal: Dict[str, Any]) -> Tuple[Dict[Seat, Tuple], Seat, Vulnerability]:
    hands = {
        Seat.WEST: _parse_hand_cards(deal["W"]),
        Seat.NORTH: _parse_hand_cards(deal["N"]),
        Seat.EAST: _parse_hand_cards(deal["E"]),
        Seat.SOUTH: _parse_hand_cards(deal["S"]),
    }
    dealer = _SEAT_MAP[str(deal["Dealer"]).upper()]
    vulnerability = _VULNERABILITY_MAP[str(deal["Vulnerable"]).lower()]
    return hands, dealer, vulnerability


def _room_record_from_board(
    *,
    source_file: str,
    event_name: str,
    match_index: int,
    session_index: int,
    board_index: int,
    room_name: str,
    deal: Dict[str, Any],
    room_entry: Dict[str, Any],
) -> TournamentRoomRecord:
    hands, dealer, vulnerability = _parse_deal(deal)
    calls = tuple(_call_to_action(token) for token in room_entry.get("Auction", {}).get("Calls", []))
    play_entry = room_entry.get("Play") or {}
    if not isinstance(play_entry, dict):
        play_entry = {}
    play = play_entry.get("Tricks", [])
    play_tricks = tuple(tuple(card for card in trick) for trick in play)
    record_id = f"{Path(source_file).stem}:m{match_index:03d}:s{session_index:02d}:b{board_index:03d}:{room_name}"
    return TournamentRoomRecord(
        record_id=record_id,
        source_file=source_file,
        event_name=event_name,
        room=room_name,
        dealer=dealer,
        vulnerability=vulnerability,
        hands=hands,
        bids=calls,
        play_tricks=play_tricks,
    )


def load_tournament_room_records(
    *,
    archive_url: str = DEFAULT_RELEASE_URL,
    event_files: Sequence[str] = DEFAULT_EVENT_FILES,
    cache_dir: str | Path,
    extract_dir: str | Path | None = None,
    max_records: int = 1000,
    records_per_event: int | None = None,
    room_names: Sequence[str] = ("OR", "CR"),
) -> List[TournamentRoomRecord]:
    archive_path = _download_archive(archive_url, cache_dir=cache_dir)
    extract_root = Path(extract_dir) if extract_dir is not None else Path(tempfile.mkdtemp(prefix="bridge_ai_events_"))
    json_paths = _extract_json_files(archive_path, event_files=event_files, extract_dir=extract_root)

    output: List[TournamentRoomRecord] = []
    per_event_limit = records_per_event if records_per_event is not None else max_records
    for json_path in json_paths:
        event_payload = json.loads(json_path.read_text(encoding="utf-8"))
        event_name = str(event_payload.get("Event") or json_path.stem)
        event_records: List[TournamentRoomRecord] = []
        for match_index, match in enumerate(event_payload.get("Matches", [])):
            for session_index, session in enumerate(match.get("Sessions", [])):
                for board_index, board in enumerate(session.get("Boards", [])):
                    deal = board.get("Deal")
                    if not isinstance(deal, dict):
                        continue
                    for room_name in room_names:
                        room_entry = board.get(room_name)
                        if not isinstance(room_entry, dict):
                            continue
                        auction = room_entry.get("Auction") or {}
                        if not isinstance(auction, dict):
                            continue
                        calls = auction.get("Calls", [])
                        if not calls:
                            continue
                        event_records.append(
                            _room_record_from_board(
                                source_file=str(json_path.relative_to(extract_root)),
                                event_name=event_name,
                                match_index=match_index,
                                session_index=session_index,
                                board_index=board_index,
                                room_name=room_name,
                                deal=deal,
                                room_entry=room_entry,
                            )
                        )
                        if len(event_records) >= per_event_limit:
                            break
                    if len(event_records) >= per_event_limit:
                        break
                if len(event_records) >= per_event_limit:
                    break
            if len(event_records) >= per_event_limit:
                break
        output.extend(event_records)
        if len(output) >= max_records:
            break
    return output[:max_records]


def build_source_manifest(
    *,
    archive_url: str,
    event_files: Sequence[str],
    records: Iterable[TournamentRoomRecord],
) -> Dict[str, Any]:
    materialized = list(records)
    return {
        "archive_url": archive_url,
        "event_files": list(event_files),
        "num_records": len(materialized),
        "records": [
            {
                "record_id": record.record_id,
                "source_file": record.source_file,
                "event_name": record.event_name,
                "room": record.room,
            }
            for record in materialized
        ],
    }

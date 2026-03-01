"""Regression tests for bridge rule edge cases."""

from __future__ import annotations

from typing import Dict

from bridge_ai.common.cards import index_to_card
from bridge_ai.common.state import BridgeState
from bridge_ai.common.types import Contract, Phase, Seat, Vulnerability
from bridge_ai.env.bridge_env import BridgeEnv
from bridge_ai.common.actions import PASS_ACTION
from bridge_ai.data.manifest import validate_manifest, write_config_snapshot, append_manifest_entry, compute_config_signature


def _state_with_fields(**fields) -> BridgeState:
    defaults = dict(
        hands={Seat.NORTH: tuple(), Seat.EAST: tuple(), Seat.SOUTH: tuple(), Seat.WEST: tuple()},
        phase=Phase.AUCTION,
        turn=0,
        dealer=Seat.NORTH,
        current_player=Seat.NORTH,
        vulnerable=Vulnerability.NONE,
        auction=tuple(),
        contract=None,
        declarer=None,
        dummy=None,
        trick_leader=None,
        trick_number=0,
        tricks_won=(0, 0),
        current_trick=tuple(),
        played_cards=tuple(),
        done=False,
        result=None,
    )
    defaults.update(fields)
    return BridgeState(**defaults)


def test_pass_out_auction_results_zero_contract_and_terminal():
    env = BridgeEnv(seed=7)
    state = env.reset(seed=7)
    info: Dict[str, object] = {}
    for _ in range(4):
        state, reward, done, info = env.step(PASS_ACTION)
    assert done
    assert reward == 0.0
    assert state.phase.name == "DEFENSE"
    assert state.result == 0.0
    assert state.contract is None
    assert info["auction_closed"] is True


def test_follow_suit_restriction():
    env = BridgeEnv()
    state = _state_with_fields(
        phase=Phase.PLAY,
        turn=4,
        current_player=Seat.EAST,
        declarer=Seat.NORTH,
        dummy=Seat.SOUTH,
        contract=Contract(level=1, strain=3, declarer=Seat.NORTH),
        trick_leader=Seat.NORTH,
        current_trick=((Seat.NORTH, index_to_card(0).index()),),
        hands={
            Seat.EAST: (index_to_card(1), index_to_card(14), index_to_card(27)),
            Seat.NORTH: tuple(),
            Seat.SOUTH: tuple(),
            Seat.WEST: tuple(),
        },
    )
    env.set_state(state)
    legal = env.legal_actions()
    assert legal == (40 + 1,)


def test_trick_winner_prefers_trumps():
    env = BridgeEnv()
    contract = Contract(level=2, strain=3, declarer=Seat.NORTH)
    winner = env._trick_winner(
        trick_cards=(
            (Seat.NORTH, index_to_card(0).index()),
            (Seat.EAST, index_to_card(3).index()),
            (Seat.SOUTH, index_to_card(39).index()),
            (Seat.WEST, index_to_card(12).index()),
        ),
        contract=contract,
    )
    assert winner == Seat.SOUTH


def test_final_score_undoubled_make_and_fail():
    env = BridgeEnv()
    made_state = _state_with_fields(
        phase=Phase.DEFENSE,
        contract=Contract(level=3, strain=2, declarer=Seat.NORTH),
        declarer=Seat.NORTH,
        tricks_won=(11, 0),
        result=None,
        current_player=Seat.NORTH,
    )
    failed_state = _state_with_fields(
        phase=Phase.DEFENSE,
        contract=Contract(level=3, strain=0, declarer=Seat.NORTH),
        declarer=Seat.NORTH,
        tricks_won=(8, 0),
        result=None,
        current_player=Seat.NORTH,
    )
    assert env._final_score(made_state) == 200.0
    assert env._final_score(failed_state) == -50.0


def test_manifest_reproducibility_helpers(tmp_path):
    config_path = tmp_path / "cfg.yaml"
    manifest_path = tmp_path / "manifest.json"
    config_path.write_text(
        "selfplay:\n  num_episodes: 1\ntraining:\n  epochs: 1\n",
        encoding="utf-8",
    )

    run_id = "test-repro"
    signature = compute_config_signature(str(config_path), extra={"seed": 1})
    snapshot = write_config_snapshot(str(config_path), str(manifest_path), run_id)
    append_manifest_entry(
        str(manifest_path),
        run_type="selfplay",
        run_id=run_id,
        config_path=str(config_path),
        run_signature=signature,
        config_snapshot=snapshot,
        outputs={"ok": True},
    )
    issues = validate_manifest(str(manifest_path))
    assert not issues

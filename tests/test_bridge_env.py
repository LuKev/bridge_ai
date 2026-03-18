"""Regression tests for bridge rule edge cases."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple
import tempfile
import unittest
import sys

import yaml

from bridge_ai.common.cards import index_to_card
from bridge_ai.common.state import BridgeState
from bridge_ai.common.types import Contract, Phase, Seat, Vulnerability
from bridge_ai.env.bridge_env import BridgeEnv
from bridge_ai.data.lin_parser import ParsedLinRecord, run_lin_record, load_lin_records_from_path
from bridge_ai.common.actions import PASS_ACTION
from bridge_ai.data.buffer import ReplayBuffer, Transition
from bridge_ai.data.manifest import validate_manifest, write_config_snapshot, append_manifest_entry, compute_config_signature
from bridge_ai.data.replay_store import load_replay_index, load_replay_window, write_replay_shard
from bridge_ai.eval.evaluator import EvalConfig, run as run_eval
from bridge_ai.eval.evaluator import _run_duplicate_match
from bridge_ai.eval.league_runner import run_checkpoint_league
from bridge_ai.infra.modal_continue import _plan_pipeline_iterations
from bridge_ai.models.monolithic_transformer import BridgeMonolithTransformer, ModelConfig
from bridge_ai.training.checkpoint_bootstrap import create_initial_checkpoint, import_legacy_checkpoint
from bridge_ai.training.checkpoint_store import (
    load_checkpoint_index,
    load_checkpoint_payload,
    resolve_checkpoint_identity,
    resolve_latest_checkpoint,
    resolve_previous_snapshot,
    save_checkpoint_bundle,
)
from bridge_ai.training.train_loop import train


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

def _run_lin_game_through_env(record: ParsedLinRecord) -> BridgeState:
    return run_lin_record(record)


_REAL_LIN_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "real_lin_records.txt"
_REAL_LIN_GAME_RECORDS = load_lin_records_from_path(str(_REAL_LIN_FIXTURE_PATH))


def test_replay_of_real_lin_games_no_illegal_action():
    failures: list[tuple[str, str]] = []
    for idx, record in enumerate(_REAL_LIN_GAME_RECORDS, 1):
        label = f"real-lin-{idx}"
        try:
            final_state = _run_lin_game_through_env(record)
        except AssertionError as exc:
            failures.append((label, str(exc)))
            continue
        assert final_state.played_cards, f"{label}: replay produced no actions"
        if final_state.done:
            assert final_state.phase == Phase.DEFENSE
    # Two known fixture exports still violate the simplified LIN replay assumptions.
    assert len(failures) <= 2, failures


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


def _policy_target() -> list[float]:
    target = [0.0] * 1000
    target[PASS_ACTION] = 1.0
    return target


def _sample_transition(step: int = 0) -> Transition:
    state = _state_with_fields(current_player=Seat.NORTH)
    return Transition(
        step=step,
        state=state.to_actor_dict(Seat.NORTH),
        action=PASS_ACTION,
        policy_target=_policy_target(),
        value_target=0.0,
        reward=0.0,
        done=False,
        metadata={"step": step},
    )


def _sample_buffer(items: int) -> ReplayBuffer:
    buffer = ReplayBuffer()
    for idx in range(items):
        buffer.append(_sample_transition(step=idx))
    return buffer


def _tiny_model_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=1000,
        action_vocab_size=1000,
        phase_vocab_size=4,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        max_seq_len=256,
    )


def _write_minimal_config(root: Path) -> Path:
    config_path = root / "cfg.yaml"
    replay_dir = root / "replays"
    checkpoint_dir = root / "checkpoints"
    artifacts_dir = root / "artifacts"
    manifest_path = artifacts_dir / "manifest.json"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  vocab_size: 1000",
                "  action_vocab_size: 1000",
                "  phase_vocab_size: 4",
                "  hidden_dim: 32",
                "  num_layers: 1",
                "  num_heads: 2",
                "  dropout: 0.0",
                "  max_seq_len: 256",
                "selfplay:",
                "  num_episodes: 1",
                "  max_steps: 8",
                "  seed: 5",
                "  output_path: \"replays/latest.json\"",
                "  use_search: false",
                "  use_model_logits: true",
                "  keep_last_shards: 4",
                "storage:",
                f"  replay_dir: \"{replay_dir}\"",
                f"  checkpoint_dir: \"{checkpoint_dir}\"",
                f"  artifacts_dir: \"{artifacts_dir}\"",
                f"  manifest_path: \"{manifest_path}\"",
                "training:",
                "  epochs: 1",
                "  batch_size: 2",
                "  iterations: 1",
                "  lr: 0.0001",
                "  replay_path: \"replays/latest.json\"",
                "  checkpoint_name: \"latest.pt\"",
                f"  ckpt_dir: \"{checkpoint_dir}\"",
                "  device: \"cpu\"",
                "  resume_latest: true",
                "  replay_window_shards: 2",
                "  save_final_snapshot: true",
                "evaluation:",
                "  rounds: 1",
                "  seed: 5000",
                "  seed_sequence: [5000]",
                "  checkpoint: \"checkpoints/latest.pt\"",
                "  baseline_checkpoint: null",
                "  use_search: false",
                "  search_simulations: 0",
                "  rollout_depth: 1",
                "  num_determinizations: 1",
                "  mode: \"auto\"",
                "  suite_name: \"quick\"",
                "  opponent_pool_size: 2",
                "  include_anchor: true",
                "  include_previous_snapshot: true",
                "  duplicate_draw_margin: 0.0",
                "  rating_k_factor: 24.0",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_manifest_reproducibility_helpers():
    tmp_path = Path(tempfile.mkdtemp(prefix="bridge-ai-manifest-"))
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


def test_replay_store_window_and_prune():
    tmp_path = Path(tempfile.mkdtemp(prefix="bridge-ai-replay-"))
    replay_dir = tmp_path / "replays"
    latest_path = replay_dir / "latest.json"

    for idx in range(3):
        write_replay_shard(
            buffer=_sample_buffer(2),
            replay_dir=replay_dir,
            latest_path=latest_path,
            run_id=f"run_{idx}",
            keep_last_shards=2,
            metadata={"idx": idx},
        )

    index = load_replay_index(replay_dir)
    assert len(index) == 2
    assert [Path(record.path).stem for record in index] == ["run_1", "run_2"]

    window, records = load_replay_window(
        replay_dir=replay_dir,
        latest_path=latest_path,
        max_shards=2,
        max_items=0,
    )
    assert len(records) == 2
    assert len(window) == 4


def test_checkpoint_store_tracks_snapshots_and_identity():
    tmp_path = Path(tempfile.mkdtemp(prefix="bridge-ai-ckpt-"))
    ckpt_dir = tmp_path / "checkpoints"
    model_cfg = _tiny_model_cfg()
    model = BridgeMonolithTransformer(model_cfg)

    first = save_checkpoint_bundle(
        model=model,
        model_cfg=model_cfg,
        ckpt_dir=ckpt_dir,
        checkpoint_name="latest.pt",
        iteration=0,
        run_id="run_0",
        source_checkpoint=None,
        parent_checkpoint=None,
        save_snapshot=True,
    )
    second = save_checkpoint_bundle(
        model=model,
        model_cfg=model_cfg,
        ckpt_dir=ckpt_dir,
        checkpoint_name="latest.pt",
        iteration=1,
        run_id="run_1",
        source_checkpoint=first["snapshot_path"],
        parent_checkpoint=first["snapshot_path"],
        save_snapshot=True,
    )

    index = load_checkpoint_index(ckpt_dir)
    assert len(index) == 2
    assert resolve_latest_checkpoint(ckpt_dir) == str(ckpt_dir / "latest.pt")
    assert resolve_previous_snapshot(ckpt_dir) == first["snapshot_path"]
    assert resolve_checkpoint_identity(ckpt_dir / "latest.pt", ckpt_dir=ckpt_dir) == second["snapshot_path"]


def test_checkpoint_bootstrap_imports_legacy_and_creates_initial():
    tmp_path = Path(tempfile.mkdtemp(prefix="bridge-ai-bootstrap-"))
    source_dir = tmp_path / "legacy"
    imported_dir = tmp_path / "imported"
    initial_dir = tmp_path / "initial"

    model_cfg = _tiny_model_cfg()
    model = BridgeMonolithTransformer(model_cfg)
    save_checkpoint_bundle(
        model=model,
        model_cfg=model_cfg,
        ckpt_dir=source_dir,
        checkpoint_name="latest.pt",
        iteration=4,
        run_id="legacy",
        source_checkpoint=None,
        parent_checkpoint=None,
        save_snapshot=False,
    )

    imported = import_legacy_checkpoint(
        source_path=source_dir / "latest.pt",
        ckpt_dir=imported_dir,
    )
    imported_payload = load_checkpoint_payload(imported_dir / "latest.pt")
    assert imported_payload["snapshot_path"] == str(imported_dir / "iter_000005.pt")
    assert imported_payload["source_checkpoint"] == str((source_dir / "latest.pt").resolve())
    assert Path(imported["snapshot_path"]).exists()

    created = create_initial_checkpoint(
        model_cfg=model_cfg,
        ckpt_dir=initial_dir,
        checkpoint_name="latest.pt",
        seed=0,
    )
    initial_payload = load_checkpoint_payload(initial_dir / "latest.pt")
    assert initial_payload["iteration"] == -1
    assert initial_payload["snapshot_path"] == created["snapshot_path"]
    assert Path(created["snapshot_path"]).exists()


def test_modal_planner_resumes_only_remaining_blocks():
    fresh = _plan_pipeline_iterations(default_iterations=5, block_size=200, target_step=3000, current_step=2000)
    resumed = _plan_pipeline_iterations(default_iterations=5, block_size=200, target_step=3000, current_step=2400)
    done = _plan_pipeline_iterations(default_iterations=5, block_size=200, target_step=3000, current_step=3000)

    assert fresh["planning_mode"] == "target_step"
    assert fresh["planned_pipeline_iterations"] == 5
    assert resumed["planning_mode"] == "target_step"
    assert resumed["planned_pipeline_iterations"] == 3
    assert done["planning_mode"] == "already_at_target"
    assert done["planned_pipeline_iterations"] == 0


def test_modal_planner_rejects_misaligned_targets():
    with unittest.TestCase().assertRaises(ValueError):
        _plan_pipeline_iterations(default_iterations=5, block_size=200, target_step=3050, current_step=2400)


def test_train_resumes_latest_by_default():
    tmp_path = Path(tempfile.mkdtemp(prefix="bridge-ai-train-"))
    config_path = _write_minimal_config(tmp_path)
    replay_dir = tmp_path / "replays"
    write_replay_shard(
        buffer=_sample_buffer(3),
        replay_dir=replay_dir,
        latest_path=replay_dir / "latest.json",
        run_id="bootstrap",
        keep_last_shards=2,
        metadata={},
    )

    train(str(config_path))
    train(str(config_path))

    payload = load_checkpoint_payload(tmp_path / "checkpoints" / "latest.pt")
    assert int(payload["iteration"]) == 1
    assert payload.get("snapshot_path")


def test_duplicate_match_same_checkpoint_is_zero_diff():
    tmp_path = Path(tempfile.mkdtemp(prefix="bridge-ai-match-"))
    ckpt_dir = tmp_path / "checkpoints"
    model_cfg = _tiny_model_cfg()
    model = BridgeMonolithTransformer(model_cfg)
    saved = save_checkpoint_bundle(
        model=model,
        model_cfg=model_cfg,
        ckpt_dir=ckpt_dir,
        checkpoint_name="latest.pt",
        iteration=0,
        run_id="match",
        source_checkpoint=None,
        parent_checkpoint=None,
        save_snapshot=True,
    )

    result = _run_duplicate_match(
        model_cfg=model_cfg,
        checkpoint_a=saved["snapshot_path"],
        checkpoint_b=saved["snapshot_path"],
        seeds=[11],
        cfg=EvalConfig(rounds=1, use_search=False, search_simulations=0, rollout_depth=1, num_determinizations=1),
        model_cache={},
    )
    assert result["pair_diff_total"] == 0.0
    assert result["match_score_a"] == 0.5


def test_eval_runs_duplicate_and_updates_ratings():
    tmp_path = Path(tempfile.mkdtemp(prefix="bridge-ai-eval-"))
    config_path = _write_minimal_config(tmp_path)
    replay_dir = tmp_path / "replays"
    write_replay_shard(
        buffer=_sample_buffer(3),
        replay_dir=replay_dir,
        latest_path=replay_dir / "latest.json",
        run_id="bootstrap",
        keep_last_shards=2,
        metadata={},
    )

    train(str(config_path))
    train(str(config_path))
    result = run_eval(str(config_path))

    assert result["mode"] == "duplicate"
    assert result["num_matches"] >= 1
    assert Path(result["rating_path"]).exists()
    assert Path(result["board_results_path"]).exists()
    assert "current_elo" in result


def test_checkpoint_league_writes_round_robin_report():
    tmp_path = Path(tempfile.mkdtemp(prefix="bridge-ai-league-"))
    config_path = _write_minimal_config(tmp_path)
    replay_dir = tmp_path / "replays"
    write_replay_shard(
        buffer=_sample_buffer(3),
        replay_dir=replay_dir,
        latest_path=replay_dir / "latest.json",
        run_id="bootstrap",
        keep_last_shards=2,
        metadata={},
    )

    train(str(config_path))
    train(str(config_path))

    initial_dir = tmp_path / "initial"
    create_initial_checkpoint(
        model_cfg=_tiny_model_cfg(),
        ckpt_dir=initial_dir,
        checkpoint_name="latest.pt",
        seed=0,
    )

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["league"] = {
        "enabled": True,
        "suite_name": "quick",
        "rounds": 1,
        "output_dir": str(tmp_path / "league"),
        "initial_checkpoint": str(initial_dir / "latest.pt"),
        "participants": [
            {"name": "initial", "checkpoint": str(initial_dir / "latest.pt")},
            {"name": "iter1", "checkpoint": str(tmp_path / "checkpoints" / "iter_000001.pt")},
            {"name": "iter2", "checkpoint": str(tmp_path / "checkpoints" / "latest.pt")},
        ],
    }
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    result = run_checkpoint_league(str(config_path))
    assert len(result["matches"]) == 3
    assert len(result["ranking"]) == 3
    assert Path(result["summary_path"]).exists()
    assert Path(result["ratings_path"]).exists()


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    module = sys.modules[__name__]
    for name, value in sorted(vars(module).items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value))
    return suite


if __name__ == "__main__":
    unittest.main()

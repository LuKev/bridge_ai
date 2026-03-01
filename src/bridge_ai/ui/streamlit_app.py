"""Minimal Streamlit UI for inspecting replayed games and model decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import json

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None


SUIT = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}
RANK = {0: "2", 1: "3", 2: "4", 3: "5", 4: "6", 5: "7", 6: "8", 7: "9", 8: "T", 9: "J", 10: "Q", 11: "K", 12: "A"}



def _load_replays(path: str = "replays") -> Dict[str, Any]:
    files = list(Path(path).glob("*.json"))
    payloads: Dict[str, Any] = {}
    for f in sorted(files):
        payloads[f.name] = json.loads(f.read_text(encoding="utf-8"))
    return payloads


def _format_card(card_idx: int) -> str:
    suit = card_idx // 13
    rank = card_idx % 13
    return f"{RANK[rank]}{SUIT[suit]}"


def _format_action(action: int) -> str:
    if action in (0, 1, 2):
        if action == 0:
            return "PASS"
        if action == 1:
            return "DOUBLE"
        return "REDOUBLE"
    if action >= 40:
        return _format_card(action - 40)
    return f"ACTION_{action}"


def _decode_action(action: int) -> str:
    if action in (0, 1, 2):
        if action == 0:
            return "PASS"
        if action == 1:
            return "DOUBLE"
        return "REDOUBLE"
    if action >= 40:
        return f"CARD_{action - 40}"
    return f"ACTION_{action}"


def _action_probability(row: Dict[str, Any]) -> float:
    action = int(row.get("action", -1))
    policy = row.get("policy_target", [])
    if not policy or action < 0 or action >= len(policy):
        return 0.0
    return float(policy[action])


def _top_k_from_policy(row: Dict[str, Any], k: int = 5) -> list[str]:
    policy = row.get("policy_target", [])
    if not policy:
        return []
    ranked = sorted(enumerate(policy), key=lambda item: float(item[1]), reverse=True)
    return [f"{_decode_action(int(idx))}:{float(val):.3f}" for idx, val in ranked[:k]]


def _format_state_cards(cards: Tuple[Any, ...]) -> str:
    if not cards:
        return ""
    return ", ".join(_format_card(int(card_idx)) for card_idx in cards)


def _state_contract(state: Dict[str, Any]) -> str:
    level = state.get("contract_level")
    strain = state.get("contract_strain")
    if level is None or strain is None:
        return "passout"
    return f"{level}{['C', 'D', 'H', 'S', 'N'][int(strain)]}"


def _show_game_summary(trajectory: list[Dict[str, Any]]) -> None:
    if not st or not trajectory:
        return
    last_state = trajectory[-1].get("state", {})
    first_state = trajectory[0].get("state", {})
    visible = first_state.get("visible_hands", {})
    first_metadata = trajectory[0].get("metadata", {})

    st.subheader("Game summary")
    st.write(
        "Episode seats: "
        f"north={len(visible.get('NORTH', []))}, "
        f"east={len(visible.get('EAST', []))}, "
        f"south={len(visible.get('SOUTH', []))}, "
        f"west={len(visible.get('WEST', []))}"
    )
    st.write(f"Result: {last_state.get('result', 'unknown')}")
    st.write(f"Contract: {_state_contract(last_state)}")
    st.write(f"Auction size: {len(last_state.get('auction', []))}")
    st.write(f"Tricks (NS / EW): {last_state.get('tricks_north_south', 0)} / {last_state.get('tricks_east_west', 0)}")
    st.write(f"Seed: {first_metadata.get('episode_seed', 'unknown')}")
    st.write(f"Variant: {first_metadata.get('variant', 'default')}")
    st.write(f"Determinization count: {first_metadata.get('num_determinizations', 1)}")


def _build_replay_summary(name: str, trajectory: list[Dict[str, Any]]) -> Dict[str, Any]:
    if not trajectory:
        return {
            "name": name,
            "seed": "unknown",
            "variant": "default",
            "determinizations": "1",
            "result": "unknown",
            "contract": "passout",
        }

    first_metadata = trajectory[0].get("metadata", {})
    last_state = trajectory[-1].get("state", {})

    return {
        "name": name,
        "seed": str(first_metadata.get("episode_seed", "unknown")),
        "variant": str(first_metadata.get("variant", "default")),
        "determinizations": str(first_metadata.get("num_determinizations", 1)),
        "result": str(last_state.get("result", "unknown")),
        "contract": _state_contract(last_state),
    }


def _filter_replays(
    summaries: List[Dict[str, Any]],
    selected_seed: str,
    selected_variant: str,
    selected_determ: str,
) -> List[str]:
    names: List[str] = []
    for summary in summaries:
        if selected_seed != "any" and summary["seed"] != selected_seed:
            continue
        if selected_variant != "any" and summary["variant"] != selected_variant:
            continue
        if selected_determ != "any" and summary["determinizations"] != selected_determ:
            continue
        names.append(summary["name"])
    return names


def _show_deal_and_actions(trajectory: list[Dict[str, Any]]) -> None:
    if not st or not trajectory:
        return
    first_state = trajectory[0].get("state", {})
    visible_hands = first_state.get("visible_hands", {})
    with st.expander("Hand snapshots (visible to actor perspective)", expanded=True):
        cols = st.columns(4)
        for idx, seat in enumerate(["NORTH", "EAST", "SOUTH", "WEST"]):
            with cols[idx]:
                st.caption(f"{seat}")
                cards = visible_hands.get(seat, [])
                st.write(_format_state_cards(cards))


def _show_transitions(trajectory: Iterable[Dict[str, Any]]) -> None:
    if st is None:
        return
    for row in trajectory:
        st.json(row)



def run_ui() -> None:
    if st is None:
        raise RuntimeError("streamlit is not installed. Install optional ui dependencies.")

    st.set_page_config(page_title="Bridge AI Replay Viewer", layout="wide")
    st.title("Bridge AI Replay Viewer")
    st.caption("Inspect generated replays and model decision traces.")
    replays = _load_replays()

    if not replays:
        st.info("No replay files found under ./replays.")
        return

    replay_summaries = [
        _build_replay_summary(name, payload)
        for name, payload in replays.items()
        if isinstance(payload, list)
    ]

    seed_options = sorted({s["seed"] for s in replay_summaries})
    variant_options = sorted({s["variant"] for s in replay_summaries})
    determinization_options = sorted({s["determinizations"] for s in replay_summaries})

    st.sidebar.header("Replay filters")
    selected_seed = st.sidebar.selectbox("Seed", ["any"] + seed_options, index=0)
    selected_variant = st.sidebar.selectbox("Variant", ["any"] + variant_options, index=0)
    selected_determ = st.sidebar.selectbox("Determinization count", ["any"] + determinization_options, index=0)

    filtered_names = _filter_replays(replay_summaries, selected_seed, selected_variant, selected_determ)
    if not filtered_names:
        st.warning("No replays match the selected filters.")
        return

    current_name = st.selectbox("Replay file", filtered_names)
    trajectory = replays[current_name]
    if not trajectory:
        st.warning("Selected replay is empty.")
        return

    st.sidebar.header("Comparison controls")
    baseline_options = [name for name in filtered_names if name != current_name]
    baseline_name = st.sidebar.selectbox("Baseline replay (optional)", ["None"] + baseline_options, index=0)

    _show_game_summary(trajectory)
    _show_deal_and_actions(trajectory)

    if baseline_name != "None":
        baseline_trajectory = replays.get(baseline_name, [])
        if baseline_trajectory:
            baseline_last = baseline_trajectory[-1].get("state", {})
            baseline_seed = baseline_trajectory[0].get("metadata", {}).get("episode_seed", "unknown")
            current_last = trajectory[-1].get("state", {})
            with st.expander("Checkpoint comparison", expanded=True):
                cols = st.columns(3)
                cols[0].metric("Current contract", _state_contract(current_last))
                cols[1].metric("Baseline contract", _state_contract(baseline_last))
                cols[2].metric("Current seed", str(trajectory[0].get("metadata", {}).get("episode_seed", "unknown")))
                cols = st.columns(3)
                cols[0].metric("Baseline seed", str(baseline_seed))
                cols[1].metric("Current result", str(current_last.get("result", "unknown")))
                cols[2].metric("Baseline result", str(baseline_last.get("result", "unknown")))
        else:
            st.warning("Baseline replay exists but has no rows.")

    st.divider()
    st.subheader("Per-move diagnostics")
    k = st.slider("Top-k policy entries", min_value=1, max_value=10, value=5, step=1)

    rows = []
    for row in trajectory:
        state = row.get("state", {})
        metadata = row.get("metadata", {})
        rows.append(
            {
                "step": row.get("step", 0),
                "phase": state.get("phase", ""),
                "player": state.get("current_player", 0),
                "action": _format_action(row.get("action", 0)),
                "top_actions": ", ".join(_top_k_from_policy(row, k=k)),
                "action_prob": f"{_action_probability(row):.4f}",
                "value_target": row.get("value_target", 0.0),
                "reward": row.get("reward", 0.0),
                "done": row.get("done", False),
                "seed": metadata.get("episode_seed"),
                "variant": metadata.get("variant", "default"),
                "determinizations": metadata.get("num_determinizations", 1),
            }
        )

    st.dataframe(rows)
    with st.expander("Play-by-play diagnostics", expanded=False):
        _show_transitions(trajectory)


def main() -> None:  # pragma: no cover
    run_ui()


if __name__ == "__main__":
    main()

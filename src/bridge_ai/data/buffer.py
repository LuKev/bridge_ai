"""Replay and trajectory storage abstractions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Any, Dict, List, Sequence

from bridge_ai.common.types import Contract, Phase, Seat, Vulnerability
from bridge_ai.common.state import BridgeState


@dataclass
class Transition:
    step: int
    state: Dict[str, Any]
    action: int
    policy_target: Sequence[float]
    value_target: float
    reward: float
    done: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class ReplayBuffer:
    def __init__(self, max_items: int = 200_000):
        self.max_items = max_items
        self._items: List[Transition] = []

    def append(self, item: Transition) -> None:
        self._items.append(item)
        if len(self._items) > self.max_items:
            self._items.pop(0)

    def extend(self, items: Sequence[Transition]) -> None:
        for item in items:
            self.append(item)

    def sample(self, batch_size: int, *, seed: int | None = None) -> List[Transition]:
        if batch_size <= 0:
            return []
        if not self._items:
            return []
        if seed is not None:
            import random

            rng = random.Random(seed)
            return [self._items[i] for i in [rng.randrange(len(self._items)) for _ in range(batch_size)]]
        import random as _random

        return _random.sample(self._items, min(batch_size, len(self._items)))

    def __len__(self) -> int:  # pragma: no cover
        return len(self._items)

    def save_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = [item.to_dict() for item in self._items]
        with path.open("w", encoding="utf-8") as f:
            json.dump(serializable, f)

    @classmethod
    def load_json(cls, path: Path) -> "ReplayBuffer":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        buf = cls()
        for row in data:
            buf.append(
                Transition(
                    step=row["step"],
                    state=row["state"],
                    action=row["action"],
                    policy_target=row["policy_target"],
                    value_target=row["value_target"],
                    reward=row["reward"],
                    done=row["done"],
                    metadata=row.get("metadata", {}),
                )
            )
        return buf


def _to_json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_json_compatible(v) for v in value]
    if isinstance(value, (Seat, Vulnerability, int, float, str, bool)) or value is None:
        return value
    if isinstance(value, Phase):
        return str(value)
    if isinstance(value, Contract):
        return {
            "level": value.level,
            "strain": value.strain,
            "doubled": value.doubled,
            "declarer": int(value.declarer),
        }
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            pass
    try:
        return float(value)
    except Exception:
        pass
    return str(value)


def to_transition(
    step: int,
    state: BridgeState,
    action: int,
    policy_target,
    value_target: float,
    reward: float,
    done: bool,
    metadata: Dict[str, Any] | None = None,
) -> Transition:
    state_dict = state.to_actor_dict(state.current_player)
    if hasattr(policy_target, "__iter__"):
        policy_target = [_to_json_compatible(value) for value in list(policy_target)]
    return Transition(
        step=step,
        state=state_dict,
        action=action,
        policy_target=list(policy_target),
        value_target=float(value_target),
        reward=float(reward),
        done=done,
        metadata=_to_json_compatible(metadata or {}),
    )

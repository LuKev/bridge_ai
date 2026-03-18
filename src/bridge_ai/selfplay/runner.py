"""Self-play orchestration for trajectory generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import random
from typing import Dict, List, Optional, Sequence
import time

import torch
import yaml

from bridge_ai.common.state import BridgeState
from bridge_ai.data.buffer import ReplayBuffer, to_transition
from bridge_ai.data.replay_store import write_replay_shard
from bridge_ai.data.manifest import append_manifest_entry
from bridge_ai.data.manifest import compute_config_signature, write_config_snapshot
from bridge_ai.env.bridge_env import BridgeEnv
from bridge_ai.common.runtime_paths import resolve_runtime_path
from bridge_ai.models.monolithic_transformer import BridgeInputEncoder, BridgeMonolithTransformer, ModelConfig
from bridge_ai.search.ismcts import ISMCTS, ISMCTSConfig
from bridge_ai.training.checkpoint_store import resolve_latest_checkpoint


@dataclass
class SelfPlayConfig:
    num_episodes: int = 4
    max_steps: int = 200
    seed: int = 0
    seed_sequence: list[int] | None = None
    output_path: str = "replays/latest.json"
    determinization_count: int = 4
    num_determinizations: int | None = None
    search_simulations: int = 12
    rollout_depth: int = 16
    use_search: bool = True
    use_model_logits: bool = True
    checkpoint: str | None = None
    keep_last_shards: int = 8


def _build_model_from_cfg(cfg: Dict) -> BridgeMonolithTransformer:
    return BridgeMonolithTransformer(ModelConfig(**cfg.get("model", {})))


def _phase_id(phase: str) -> int:
    if phase == "auction":
        return 0
    if phase == "lead":
        return 1
    if phase == "play":
        return 2
    return 3


def _load_checkpoint_if_available(model: BridgeMonolithTransformer, path: str | None) -> None:
    if not path:
        return
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload.get("state_dict")
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict)


def _resolve_checkpoint(path: str | None, checkpoint_dir: str) -> str | None:
    if path:
        return str(resolve_runtime_path(path))
    return resolve_latest_checkpoint(checkpoint_dir)


def _seed_schedule(cfg: SelfPlayConfig) -> List[int]:
    if cfg.seed_sequence:
        base = list(cfg.seed_sequence)
        if len(base) >= cfg.num_episodes:
            return base[: cfg.num_episodes]
        base.extend([cfg.seed + i for i in range(cfg.num_episodes - len(base))])
        return base
    return [cfg.seed + i for i in range(cfg.num_episodes)]


def _to_distribution(model: BridgeMonolithTransformer, state: BridgeState, legal_actions: Sequence[int]) -> torch.Tensor:
    if not legal_actions:
        return torch.zeros(model.config.action_vocab_size, dtype=torch.float32)
    encoder = BridgeInputEncoder()
    token_ids = encoder.encode(state, perspective=state.current_player)
    legal_mask = encoder.action_mask(legal_actions, action_vocab_size=model.config.action_vocab_size).unsqueeze(0)
    phase = torch.tensor([_phase_id(state.phase.value)], dtype=torch.long)
    logits, _ = model(token_ids, phase, legal_action_mask=legal_mask)
    probs = torch.softmax(logits.squeeze(0), dim=-1)
    action_mask = torch.zeros_like(probs)
    action_mask[list(legal_actions)] = 1.0
    probs = probs * action_mask
    if torch.sum(probs).item() <= 0.0:
        probs = action_mask
    if torch.sum(probs).item() <= 0.0:
        return torch.zeros_like(action_mask)
    return probs / torch.sum(probs)


def _sample_policy_action(model: BridgeMonolithTransformer, state: BridgeState, legal_actions: Sequence[int]) -> int:
    probs = _to_distribution(model, state, legal_actions)
    if torch.sum(probs).item() <= 0.0:
        return int(random.choice(list(legal_actions)))
    return int(random.choices(list(legal_actions), weights=[float(probs[a]) for a in legal_actions])[0])


def generate_trajectories(model: BridgeMonolithTransformer, cfg: SelfPlayConfig, *, seeds: Optional[List[int]] = None) -> ReplayBuffer:
    env = BridgeEnv(seed=cfg.seed)
    search_cfg = ISMCTSConfig(
        num_determinizations=cfg.num_determinizations or cfg.determinization_count,
        num_simulations=cfg.search_simulations,
        rollout_depth=cfg.rollout_depth,
    )
    search = ISMCTS(search_cfg)
    buffer = ReplayBuffer()
    seeds = seeds if seeds is not None else _seed_schedule(cfg)

    for episode_idx, episode_seed in enumerate(seeds[: cfg.num_episodes]):
        state = env.reset(seed=episode_seed)
        done = False
        for t in range(cfg.max_steps):
            legal = env.legal_actions()
            if not legal:
                break

            if cfg.use_search and cfg.use_model_logits:
                token_ids = BridgeInputEncoder().encode(state, perspective=state.current_player)
                legal_mask = BridgeInputEncoder().action_mask(
                    legal,
                    action_vocab_size=model.config.action_vocab_size,
                ).unsqueeze(0)
                logits, _ = model(
                    token_ids,
                    torch.tensor([_phase_id(state.phase.value)]),
                    legal_action_mask=legal_mask,
                )
                result = search.select_action(
                    state,
                    policy_logits=logits,
                    legal_actions=legal,
                    model=model,
                )
                visit_distribution = result.visit_distribution
                action = int(result.action)
                value_target = float(result.value)
            else:
                action = _sample_policy_action(model, state, legal)
                visit_distribution = _to_distribution(model, state, legal)
                value_target = 0.0

            if action not in legal:
                action = int(random.choice(list(legal)))

            next_state, reward, done, info = env.step(action)
            transition = to_transition(
                step=t,
                state=state,
                action=action,
                policy_target=visit_distribution.to("cpu"),
                value_target=value_target,
                reward=reward,
                done=done,
                metadata={
                    "episode_index": episode_idx,
                    "episode_seed": episode_seed,
                    "num_determinizations": cfg.num_determinizations or cfg.determinization_count,
                    "phase": str(state.phase),
                    "trick": state.trick_number,
                    "terminal_info": info,
                },
            )
            buffer.append(transition)
            state = next_state
            if done:
                break

    return buffer


def run(config_path: str = "configs/default.yaml") -> ReplayBuffer:
    config_source = resolve_runtime_path(config_path)
    cfg_dict = yaml.safe_load(config_source.read_text(encoding="utf-8"))
    cfg = SelfPlayConfig(**cfg_dict.get("selfplay", {}))
    manifest_path = str(resolve_runtime_path(cfg_dict.get("storage", {}).get("manifest_path", "artifacts/manifest.json")))
    replay_dir = cfg_dict.get("storage", {}).get("replay_dir", "replays")
    checkpoint_dir = str(resolve_runtime_path(cfg_dict.get("storage", {}).get("checkpoint_dir", "checkpoints")))
    if cfg.output_path == "replays/latest.json":
        cfg.output_path = f"{replay_dir.rstrip('/')}/latest.json"
    output_path = resolve_runtime_path(cfg.output_path)
    replay_root = str(output_path.parent)
    model = _build_model_from_cfg(cfg_dict)
    resolved_checkpoint = _resolve_checkpoint(cfg.checkpoint, checkpoint_dir)
    _load_checkpoint_if_available(model, resolved_checkpoint)
    run_id = f"selfplay_{int(time.time() * 1_000_000)}"
    seeds = _seed_schedule(cfg)
    run_signature = compute_config_signature(
        str(config_source),
        extra={
            "seed": cfg.seed,
            "seed_sequence": cfg.seed_sequence,
            "num_episodes": cfg.num_episodes,
            "checkpoint": resolved_checkpoint,
        },
    )
    config_snapshot = write_config_snapshot(str(config_source), manifest_path, run_id)
    out = output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    buffer = generate_trajectories(model, cfg, seeds=seeds)
    shard_record = write_replay_shard(
        buffer=buffer,
        replay_dir=replay_root,
        latest_path=out,
        run_id=run_id,
        keep_last_shards=cfg.keep_last_shards,
        metadata={
            "checkpoint": resolved_checkpoint,
            "seed_sequence": seeds,
            "episodes": cfg.num_episodes,
        },
    )
    append_manifest_entry(
        manifest_path,
        run_type="selfplay",
        run_id=run_id,
        config_path=str(config_source),
        run_signature=run_signature,
        config_snapshot=config_snapshot,
        outputs={
            "replay_path": str(out),
            "replay_shard_path": shard_record.path,
            "steps": len(buffer),
            "episodes": cfg.num_episodes,
            "checkpoint": resolved_checkpoint,
        },
        metrics={
            "episodes": cfg.num_episodes,
            "steps": len(buffer),
            "seed_sequence": seeds,
        },
        status="ok",
    )
    return buffer


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run bridge self-play generation.")
    parser.add_argument("--config-path", "--config_path", default="configs/default.yaml")
    args = parser.parse_args()
    run(config_path=args.config_path)


if __name__ == "__main__":
    main()

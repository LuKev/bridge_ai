"""Imperfect-information MCTS-style policy selector with determinization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import random
import torch

from bridge_ai.common.cards import index_to_card
from bridge_ai.common.types import Card, Seat
from bridge_ai.common.state import BridgeState
from bridge_ai.env.bridge_env import BridgeEnv
from bridge_ai.models.monolithic_transformer import BridgeInputEncoder


@dataclass
class ISMCTSResult:
    action: int
    visit_distribution: torch.Tensor
    value: float


@dataclass
class ISMCTSConfig:
    num_simulations: int = 12
    num_determinizations: int = 4
    exploration: float = 1.0
    rollout_depth: int = 16
    seed: int = 0


class ISMCTS:
    """Pragmatic imperfect-information selector with root-level sampling.

    This is not a full UCT tree implementation. It is a deterministic-compatible
    baseline that performs root-action evaluations over determinization samples.
    """

    def __init__(self, config: ISMCTSConfig = ISMCTSConfig()):
        self.config = config
        self.rng = random.Random(config.seed)
        self.encoder = BridgeInputEncoder()

    @torch.no_grad()
    def select_action(
        self,
        state: BridgeState,
        policy_logits: torch.Tensor,
        legal_actions: Sequence[int],
        *,
        seed: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
    ) -> ISMCTSResult:
        if not legal_actions:
            return ISMCTSResult(action=0, visit_distribution=torch.zeros(1), value=0.0)

        if seed is not None:
            self.rng.seed(seed)

        if policy_logits.ndim == 1:
            prior = torch.softmax(policy_logits, dim=-1)
        else:
            prior = torch.softmax(policy_logits[0], dim=-1)
        legal_actions = tuple(legal_actions)

        legal_mask = torch.zeros_like(prior)
        legal_mask[list(legal_actions)] = 1.0
        prior = prior * legal_mask
        if torch.sum(prior).item() <= 0:
            prior = legal_mask / float(len(legal_actions))

        q = torch.zeros_like(prior)
        visits = torch.zeros_like(prior)

        for _ in range(self.config.num_simulations):
            score = q + self.config.exploration * prior / (visits + 1.0).sqrt()
            score[~legal_mask.bool()] = float("-inf")
            action = int(torch.argmax(score).item())
            value = self._estimate_action_value(state, action, model=model)
            visits[action] += 1.0
            q[action] += (value - q[action]) / float(visits[action])

        visit_total = visits.sum().item()
        if visit_total <= 0:
            visit_dist = torch.ones_like(visits) / float(len(visits))
            best = int(legal_actions[0])
            value = 0.0
        else:
            visit_dist = visits / visit_total
            max_visits = float(visits.max().item())
            best_candidates = [a for a in legal_actions if float(visits[a].item()) == max_visits]
            if len(best_candidates) > 1:
                best = int(self.rng.choice(best_candidates))
            else:
                best = int(torch.argmax(visits).item())
            value = float(q[best].item())
        if best not in legal_mask.bool().nonzero(as_tuple=True)[0].tolist():
            best = int(self.rng.choice(list(legal_actions)))

        return ISMCTSResult(action=best, visit_distribution=visit_dist, value=value)

    def _estimate_action_value(
        self,
        state: BridgeState,
        action: int,
        *,
        model: Optional[torch.nn.Module],
    ) -> float:
        values: list[float] = []
        for _ in range(self.config.num_determinizations):
            hands = self._sample_hidden_state(state)
            sampled_state = state
            if hands is not None:
                sampled_state = BridgeState(
                    hands=hands,
                    phase=state.phase,
                    turn=state.turn,
                    dealer=state.dealer,
                    current_player=state.current_player,
                    vulnerable=state.vulnerable,
                    auction=state.auction,
                    contract=state.contract,
                    declarer=state.declarer,
                    dummy=state.dummy,
                    trick_leader=state.trick_leader,
                    trick_number=state.trick_number,
                    tricks_won=state.tricks_won,
                    current_trick=state.current_trick,
                    played_cards=state.played_cards,
                    done=state.done,
                    result=state.result,
                )
            values.append(self._rollout(sampled_state, action, model=model))
        return float(sum(values) / max(1, len(values)))

    def _rollout(
        self,
        state: BridgeState,
        first_action: int,
        model: Optional[torch.nn.Module],
    ) -> float:
        env = BridgeEnv()
        env.set_state(state)
        _, reward, done, _ = env.step(first_action)
        total = float(reward)

        depth = 0
        while not done and depth < self.config.rollout_depth:
            depth += 1
            legal = env.legal_actions()
            if not legal:
                break

            if first_action not in legal:
                first_action = int(self.rng.choice(legal))

            if model is None:
                action = self.rng.choice(legal)
            else:
                action = self._model_action(env, legal, model)

            if int(action) == int(first_action):
                candidate = action
            else:
                candidate = first_action

            if candidate not in legal:
                candidate = int(self.rng.choice(legal))

            try:
                _, reward, done, _ = env.step(int(candidate))
            except ValueError:
                legal = env.legal_actions()
                if not legal:
                    break
                candidate = int(self.rng.choice(legal))
                _, reward, done, _ = env.step(candidate)
            total += float(reward)

            if env.state is not None and env.state.result is not None and done:
                return float(env.state.result)

        return total

    def _model_action(self, env: BridgeEnv, legal: Sequence[int], model: torch.nn.Module) -> int:
        if env.state is None:
            return int(self.rng.choice(legal))

        token_ids = self.encoder.encode(env.state, perspective=env.state.current_player)
        legal_mask = self.encoder.action_mask(
            legal,
            action_vocab_size=model.config.action_vocab_size,
        ).unsqueeze(0)
        phase_id = torch.tensor([self.encoder._phase_id(env.state.phase)], dtype=torch.long)
        logits, _ = model(token_ids, phase_id, legal_action_mask=legal_mask)
        probs = torch.softmax(logits.squeeze(0), dim=-1)
        legal_probs = probs[list(legal)]

        if torch.sum(legal_probs).item() <= 0:
            return int(self.rng.choice(legal))

        weights = legal_probs / torch.sum(legal_probs)
        return int(self.rng.choices(population=list(legal), weights=[float(w) for w in weights])[0])

    def _sample_hidden_state(self, state: BridgeState) -> Optional[Dict[Seat, Tuple[Card, ...]]]:
        if state.current_player is None:
            return None

        all_cards = [index_to_card(i) for i in range(52)]
        known_cards: set[Card] = set()

        if state.hands:
            for player, cards in state.hands.items():
                known_cards.update(cards)

        if state.phase is not None and state.contract is not None:
            if state.declarer is not None:
                known_cards.update(state.hands.get(state.declarer, tuple()))
            if state.dummy is not None:
                known_cards.update(state.hands.get(state.dummy, tuple()))

        for _, card_idx in state.current_trick:
            known_cards.add(index_to_card(card_idx))
        for _, card_idx in state.played_cards:
            known_cards.add(index_to_card(card_idx))

        hidden_pool = [card for card in all_cards if card not in known_cards]
        self.rng.shuffle(hidden_pool)

        output: Dict[Seat, Tuple[Card, ...]] = {}
        for seat in state.hands:
            if seat == state.current_player:
                output[seat] = state.hands[seat]
                continue

            if (
                state.phase is not None
                and state.contract is not None
                and state.declarer is not None
                and state.dummy is not None
                and seat in (state.declarer, state.dummy)
            ):
                output[seat] = state.hands[seat]
                continue

            need = len(state.hands.get(seat, tuple()))
            output[seat] = tuple(hidden_pool[:need])
            hidden_pool = hidden_pool[need:]

        return output

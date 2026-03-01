"""Single monolithic transformer with an integrated bridge-state encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn

from bridge_ai.common.cards import index_to_card
from bridge_ai.common.state import BridgeState
from bridge_ai.common.types import Card, Contract, Phase, Seat, Vulnerability


@dataclass
class ModelConfig:
    vocab_size: int | None = None
    action_vocab_size: int = 260
    phase_vocab_size: int = 4
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 256
    use_auxiliary_heads: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size is not None and self.action_vocab_size == 260:
            self.action_vocab_size = int(self.vocab_size)


class BridgeInputEncoder:
    """
    Tokenizes bridge state information into a fixed-length sequence.

    Token namespace:
    - 0: PAD
    - 1..4 phase tokens
    - 5..8 vulnerability and seat tokens
    - 9..60 hand cards (52 max)
    - 61..100 contract/history/control tokens
    """

    PAD = 0
    PHASE_OFFSET = 10
    VULN_OFFSET = 30
    METRIC_OFFSET = 40
    CARD_OFFSET = 100
    CALL_OFFSET = 360
    SEAT_OFFSET = 420
    HAND_OFFSET = 560

    def __init__(self, max_seq_len: int = 256):
        self.max_seq_len = max_seq_len

    def _add(self, seq: list[int], token: int) -> None:
        if token <= 0:
            token = self.PAD
        if len(seq) < self.max_seq_len:
            seq.append(token)

    def _card_token(self, seat: int, card_index: int) -> int:
        return self.CARD_OFFSET + seat * 52 + card_index

    def _call_token(self, code: int) -> int:
        return self.CALL_OFFSET + code

    def encode(
        self,
        state: BridgeState,
        *,
        perspective: Seat | None = None,
    ) -> torch.Tensor:
        if not isinstance(state, BridgeState):
            raise TypeError("BridgeInputEncoder.encode expects a BridgeState; call encode_dict() for dict payloads")
        return self.encode_from_parts(
            phase=state.phase,
            turn=state.turn,
            dealer=state.dealer,
            current_player=state.current_player,
            vulnerable=state.vulnerable,
            auction=state.auction,
            contract=state.contract,
            declarer=state.declarer,
            trick_number=state.trick_number,
            tricks_won=state.tricks_won,
            visible_hands=state.hands,
            current_trick=state.current_trick,
            played_cards=state.played_cards,
            perspective=perspective,
        )

    def encode_dict(
        self,
        state_dict: Dict,
        *,
        perspective: int | str | Seat | None = None,
    ) -> torch.Tensor:
        phase = state_dict.get("phase", Phase.AUCTION.value)
        if isinstance(phase, Phase):
            phase_enum = phase
        elif isinstance(phase, int):
            if phase == 0:
                phase_enum = Phase.AUCTION
            elif phase == 1:
                phase_enum = Phase.LEAD
            elif phase == 2:
                phase_enum = Phase.PLAY
            else:
                phase_enum = Phase.DEFENSE
        else:
            phase_enum = Phase(str(phase))

        current_player = state_dict.get("current_player", 0)
        if isinstance(current_player, Seat):
            current_player = int(current_player)

        dealer = state_dict.get("dealer", 0)
        if isinstance(dealer, Seat):
            dealer = int(dealer)

        if perspective is None:
            perspective = current_player
        if not isinstance(perspective, Seat):
            perspective = Seat(int(perspective))

        contract_level = state_dict.get("contract_level", -1)
        contract_strain = state_dict.get("contract_strain", -1)
        contract_doubled = state_dict.get("contract_doubled", 0)
        declarer = state_dict.get("declarer", -1)
        if isinstance(declarer, Seat):
            declarer = int(declarer)
        declarer_seat = None if int(declarer) < 0 else Seat(int(declarer))
        contract = (
            Contract(
                level=int(contract_level),
                strain=int(contract_strain),
                declarer=declarer_seat,
                doubled=int(contract_doubled),
            )
            if int(contract_level) >= 0 and int(contract_strain) >= 0 and int(declarer) >= 0
            else None
        )

        visible_hands: Dict[Seat, Tuple[Card, ...]] = {}
        for seat in Seat:
            raw = state_dict.get("visible_hands", {}).get(str(seat), [])
            if not raw:
                raw = state_dict.get("visible_hands", {}).get(int(seat), [])
            visible_hands[seat] = tuple(index_to_card(int(code)) for code in raw)

        return self.encode_from_parts(
            phase=phase_enum,
            turn=int(state_dict.get("turn", 0)),
            dealer=Seat(int(dealer)),
            current_player=Seat(int(current_player)),
            vulnerable=Vulnerability(int(state_dict.get("vulnerable", 0))),
            auction=tuple(int(a) for a in state_dict.get("auction", ())),
            contract=contract,
            declarer=declarer_seat,
            trick_number=int(state_dict.get("trick_number", 0)),
            tricks_won=(int(state_dict.get("tricks_north_south", 0)), int(state_dict.get("tricks_east_west", 0))),
            visible_hands=visible_hands,
            current_trick=tuple(
                (Seat(int(seat)), int(card_idx)) for seat, card_idx in state_dict.get("current_trick", ())
            ),
            played_cards=tuple((Seat(int(seat)), int(card_idx)) for seat, card_idx in state_dict.get("played_cards", ())),
            perspective=perspective,
        )

    def encode_from_parts(
        self,
        *,
        phase: Phase,
        turn: int,
        dealer: Seat,
        current_player: Seat,
        vulnerable: Vulnerability,
        auction: Tuple[int, ...],
        contract: Contract | None,
        declarer: Seat | None,
        trick_number: int,
        tricks_won: tuple[int, int],
        visible_hands: Dict[Seat, Tuple[Card, ...]],
        current_trick: Tuple[Tuple[Seat, int], ...],
        played_cards: Tuple[Tuple[Seat, int], ...],
        perspective: Seat,
    ) -> torch.Tensor:
        seq: list[int] = []
        self._add(seq, self.PHASE_OFFSET + self._phase_id(phase))
        self._add(seq, self.VULN_OFFSET + int(vulnerable))
        self._add(seq, self.METRIC_OFFSET + int(turn))
        self._add(seq, self.METRIC_OFFSET + 8 + int(dealer))
        self._add(seq, self.METRIC_OFFSET + 12 + int(current_player))
        self._add(seq, self.METRIC_OFFSET + 16 + trick_number)
        self._add(seq, self.METRIC_OFFSET + 20 + tricks_won[0])
        self._add(seq, self.METRIC_OFFSET + 40 + tricks_won[1])
        if contract is not None:
            self._add(seq, self.METRIC_OFFSET + 60 + min(contract.level, 7))
            self._add(seq, self.METRIC_OFFSET + 70 + int(contract.strain))
            self._add(seq, self.METRIC_OFFSET + 80 + int(contract.doubled))
        else:
            self._add(seq, self.METRIC_OFFSET + 90)

        # Visible private/public hands for the active perspective seat.
        pov = perspective
        for seat in Seat:
            if seat == pov:
                cards = visible_hands.get(seat, tuple())
            elif declarer is not None and seat in (declarer, Seat((int(declarer) + 2) % 4)) and phase in (Phase.PLAY, Phase.LEAD):
                cards = visible_hands.get(seat, tuple())
            else:
                cards = tuple()
            for card in cards:
                self._add(seq, self._card_token(int(seat), card.index()))

        # Auction encoding in temporal order.
        for call in auction[-24:]:
            self._add(seq, self._call_token(int(call)))

        # Current trick with explicit seat markers.
        for seat, card_idx in current_trick:
            self._add(seq, self.SEAT_OFFSET + int(seat))
            self._add(seq, self._card_token(int(seat), card_idx))

        # Recency-aware trick history (compressed to last 20 cards).
        for seat, card_idx in played_cards[-20:]:
            self._add(seq, self.SEAT_OFFSET + int(seat))
            self._add(seq, self._card_token(int(seat), card_idx))

        # Reveal ownership marker for public seats when available.
        if declarer is not None:
            self._add(seq, self.HAND_OFFSET + int(declarer))
            self._add(seq, self.HAND_OFFSET + int(declarer.partner))

        if len(seq) < self.max_seq_len:
            seq.extend([self.PAD] * (self.max_seq_len - len(seq)))
        return torch.tensor([seq[: self.max_seq_len]], dtype=torch.long)

    def action_mask(self, legal_actions: Sequence[int], *, action_vocab_size: int) -> torch.Tensor:
        mask = torch.zeros(action_vocab_size, dtype=torch.bool)
        for action in legal_actions:
            if 0 <= action < action_vocab_size:
                mask[int(action)] = True
        return mask

    @staticmethod
    def _phase_id(phase: Phase) -> int:
        return {
            Phase.AUCTION: 0,
            Phase.LEAD: 1,
            Phase.PLAY: 2,
            Phase.DEFENSE: 3,
        }.get(phase, 0)


class BridgeMonolithTransformer(nn.Module):
    """
    Shared transformer trunk + phase-aware policy/value heads.

    Inputs:
    - token_ids: [B, T]
    - phase_id: [B]
    - legal_action_mask: optional [B, A]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.PAD = BridgeInputEncoder.PAD
        self.config = config
        self.vocab_size = max(400 + 52 * 4 + 256, config.action_vocab_size + 200)
        self.token_embedding = nn.Embedding(self.vocab_size, config.hidden_dim)
        self.phase_embedding = nn.Embedding(config.phase_vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.policy_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.action_vocab_size),
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Tanh(),
        )
        self.use_auxiliary_heads = bool(config.use_auxiliary_heads)
        if self.use_auxiliary_heads:
            self.trick_share_head = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 2, 1),
                nn.Tanh(),
            )
            self.contract_level_head = nn.Sequential(
                nn.LayerNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(config.hidden_dim // 2, 8),
            )

    def forward(
        self,
        token_ids: torch.Tensor,
        phase_id: torch.Tensor,
        legal_action_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        batch, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device)
        pos_emb = self.position_embedding(positions).unsqueeze(0).expand(batch, -1, -1)
        x = self.token_embedding(token_ids) + pos_emb
        phase_emb = self.phase_embedding(phase_id).unsqueeze(1)
        x = x + phase_emb
        pad_mask = token_ids.eq(self.PAD)
        encoded = self.encoder(x, src_key_padding_mask=pad_mask)
        pooled = encoded[:, 0, :]
        logits = self.policy_head(pooled)
        if legal_action_mask is not None:
            # invalid actions should not be sampled by policy head
            logits = logits.masked_fill(~legal_action_mask.bool(), -1.0e9)
        value = self.value_head(pooled).squeeze(-1)
        if return_aux:
            if self.use_auxiliary_heads:
                trick_share = self.trick_share_head(pooled).squeeze(-1)
                contract_level = self.contract_level_head(pooled)
            else:
                trick_share = torch.zeros((batch,), device=token_ids.device, dtype=torch.float32)
                contract_level = torch.zeros((batch, 8), device=token_ids.device, dtype=torch.float32)
            return logits, value, {
                "trick_share": trick_share,
                "contract_level_logits": contract_level,
            }
        return logits, value

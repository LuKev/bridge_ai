"""Transformer for bidding prediction and hidden-hand belief inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


@dataclass
class BiddingBeliefConfig:
    action_vocab_size: int = 40
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    max_seq_len: int = 128
    owner_classes: int = 4


class BiddingBeliefEncoder:
    PAD = 0
    CLS = 1
    PHASE_OFFSET = 10
    SEAT_OFFSET = 20
    VULN_OFFSET = 40
    TRICK_OFFSET = 50
    CARD_OFFSET = 80
    DUMMY_CARD_OFFSET = 140
    CALL_OFFSET = 220
    PLAY_CARD_OFFSET = 320
    CURRENT_TRICK_CARD_OFFSET = 420
    PUBLIC_SEAT_OFFSET = 520

    def __init__(self, max_seq_len: int = 128):
        self.max_seq_len = max_seq_len
        self.vocab_size = self.PUBLIC_SEAT_OFFSET + 8

    def encode_example(self, example) -> torch.Tensor:
        seq = [
            self.CLS,
            self.PHASE_OFFSET + (0 if example.phase == "auction" else 1),
            self.SEAT_OFFSET + int(example.acting_seat),
            self.SEAT_OFFSET + 4 + int(example.dealer),
            self.VULN_OFFSET + int(example.vulnerable),
            self.TRICK_OFFSET + min(int(example.trick_number), 13),
            self.TRICK_OFFSET + 16 + min(int(example.played_count), 52),
        ]
        for card_index in sorted(example.own_cards):
            seq.append(self.CARD_OFFSET + int(card_index))
        for card_index in sorted(example.visible_dummy_cards):
            seq.append(self.DUMMY_CARD_OFFSET + int(card_index))
        for action in example.auction_prefix[-48:]:
            seq.append(self.CALL_OFFSET + int(action))
        for seat, card_index in example.current_trick[-8:]:
            seq.append(self.PUBLIC_SEAT_OFFSET + int(seat))
            seq.append(self.CURRENT_TRICK_CARD_OFFSET + int(card_index))
        for seat, card_index in example.played_cards[-32:]:
            seq.append(self.PUBLIC_SEAT_OFFSET + int(seat))
            seq.append(self.PLAY_CARD_OFFSET + int(card_index))
        if len(seq) < self.max_seq_len:
            seq.extend([self.PAD] * (self.max_seq_len - len(seq)))
        return torch.tensor(seq[: self.max_seq_len], dtype=torch.long)

    def action_mask(self, legal_actions: Sequence[int], action_vocab_size: int) -> torch.Tensor:
        mask = torch.zeros(action_vocab_size, dtype=torch.bool)
        for action in legal_actions:
            if 0 <= int(action) < action_vocab_size:
                mask[int(action)] = True
        return mask


class BiddingBeliefTransformer(nn.Module):
    def __init__(self, config: BiddingBeliefConfig):
        super().__init__()
        self.config = config
        self.encoder = BiddingBeliefEncoder(max_seq_len=config.max_seq_len)
        self.token_embedding = nn.Embedding(self.encoder.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.trunk = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.bid_head = nn.Linear(config.hidden_dim, config.action_vocab_size)
        self.owner_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 52 * config.owner_classes),
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        *,
        legal_action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(token_ids.shape[1], device=token_ids.device).unsqueeze(0)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        padding_mask = token_ids.eq(self.encoder.PAD)
        hidden = self.trunk(x, src_key_padding_mask=padding_mask)
        pooled = self.norm(hidden[:, 0, :])
        bid_logits = self.bid_head(pooled)
        if legal_action_mask is not None:
            bid_logits = bid_logits.masked_fill(~legal_action_mask, float("-inf"))
        owner_logits = self.owner_head(pooled).view(-1, 52, self.config.owner_classes)
        return bid_logits, owner_logits

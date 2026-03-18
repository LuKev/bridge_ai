"""Bridge environment with auction and trick-level play rules."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

from bridge_ai.common.actions import BID_BASE, DOUBLE_ACTION, MAX_BID_CODE, PASS_ACTION, REDOUBLE_ACTION, decode_bid_code, is_bid_code
from bridge_ai.common.cards import index_to_card, shuffle_and_deal
from bridge_ai.common.types import Card, Contract, Phase, Seat, Vulnerability
from bridge_ai.common.state import BridgeState


BidCall = Tuple[Seat, int]


class BridgeEnv:
    """Bridge environment for deterministic self-play and search harnesses."""

    MAX_TRICKS = 13
    STRAIN_TRICK_VALUE = {
        0: 20,
        1: 20,
        2: 30,
        3: 30,
        4: 30,
    }

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.state: BridgeState | None = None

    @staticmethod
    def _same_side(a: Seat, b: Seat) -> bool:
        return (a - b) % 2 == 0

    @staticmethod
    def _to_value(level: int, strain: int) -> int:
        return (level * 5) + strain

    @staticmethod
    def _phase_id(phase: Phase) -> int:
        return {Phase.AUCTION: 0, Phase.LEAD: 1, Phase.PLAY: 2, Phase.DEFENSE: 3}[phase]

    def reset(
        self,
        *,
        seed: int | None = None,
        hands: Dict[Seat, Tuple[Card, ...]] | None = None,
    ) -> BridgeState:
        if seed is not None:
            self.seed = seed
        if hands is None:
            hands = shuffle_and_deal(seed=self.seed)

        state = BridgeState(
            hands=hands,
            phase=Phase.AUCTION,
            turn=0,
            dealer=Seat.NORTH,
            current_player=Seat.NORTH,
            vulnerable=Vulnerability.NONE,
        )
        self.state = state
        return state

    def set_state(self, state: BridgeState) -> None:
        self.state = state

    def legal_actions(self) -> Tuple[int, ...]:
        if self.state is None or self.state.done:
            return tuple()

        if self.state.phase == Phase.AUCTION:
            return self._legal_bids()

        if self.state.phase in (Phase.LEAD, Phase.PLAY):
            return self._legal_plays()

        return tuple()

    def step(self, action_code: int) -> tuple[BridgeState, float, bool, dict]:
        if self.state is None:
            raise RuntimeError("environment not initialized")

        legal = set(self.legal_actions())
        if action_code not in legal:
            raise ValueError(f"illegal action {action_code} in phase {self.state.phase.value}")

        if self.state.phase == Phase.AUCTION:
            return self._step_bid(action_code)
        return self._step_play(action_code)

    def _legal_bids(self) -> Tuple[int, ...]:
        assert self.state is not None
        status = self._auction_status()

        # Pass is always legal in auction.
        legal: List[int] = [PASS_ACTION]

        # Bids must beat current contract and follow order.
        current_level, current_strain = status["current_bid"]
        if current_level is None:
            current_level = -1
            current_strain = -1

        for code in range(BID_BASE, MAX_BID_CODE + 1):
            level, strain = decode_bid_code(code)
            if self._to_value(level, strain) > self._to_value(current_level, current_strain):
                legal.append(code)

        if status["can_double"]:
            legal.append(DOUBLE_ACTION)
        if status["can_redouble"]:
            legal.append(REDOUBLE_ACTION)

        return tuple(sorted(set(legal)))

    def _legal_plays(self) -> Tuple[int, ...]:
        assert self.state is not None
        hand = self.state.hands.get(self.state.current_player, tuple())
        if not hand:
            return tuple()

        if not self.state.current_trick:
            return tuple(40 + card.index() for card in sorted(hand, key=lambda c: c.index()))

        led_suit = self.state.current_trick[0][1] // 13
        follow_cards = [card for card in hand if card.index() // 13 == led_suit]
        legal_cards = follow_cards if follow_cards else hand
        return tuple(40 + card.index() for card in sorted(legal_cards, key=lambda c: c.index()))

    def _step_bid(self, action_code: int) -> tuple[BridgeState, float, bool, dict]:
        assert self.state is not None
        state = self.state
        auction = state.auction + (action_code,)

        if not self._auction_done(auction):
            next_state = replace(
                state,
                turn=state.turn + 1,
                auction=auction,
                current_player=state.current_player.next(),
            )
            self.state = next_state
            return next_state, 0.0, False, {}

        contract = self._auction_to_contract(auction)
        next_player = state.current_player.next()
        if contract is None:
            next_state = replace(
                state,
                turn=state.turn + 1,
                auction=auction,
                phase=Phase.DEFENSE,
                done=True,
                result=0.0,
                current_player=next_player,
            )
            self.state = next_state
            return next_state, 0.0, True, {"contract": None, "auction_closed": True}

        lead = contract.declarer.next()
        next_state = replace(
            state,
            turn=state.turn + 1,
            auction=auction,
            contract=contract,
            declarer=contract.declarer,
            dummy=contract.declarer.partner,
            phase=Phase.PLAY,
            current_player=lead,
            trick_leader=lead,
            trick_number=0,
            tricks_won=(0, 0),
            current_trick=tuple(),
            played_cards=tuple(),
        )
        self.state = next_state
        return next_state, 0.0, False, {"contract": contract, "auction_closed": True}

    def _step_play(self, action_code: int) -> tuple[BridgeState, float, bool, dict]:
        assert self.state is not None
        state = self.state
        if state.contract is None:
            next_state = replace(state, done=True, result=0.0, turn=state.turn + 1)
            self.state = next_state
            return next_state, 0.0, True, {"terminal": "no_contract"}

        player = state.current_player
        hand = list(state.hands.get(player, tuple()))
        card_idx = action_code - 40
        if card_idx < 0 or card_idx >= 52:
            raise ValueError(f"invalid play action {action_code}")

        selected = next((card for card in hand if card.index() == card_idx), None)
        if selected is None:
            raise ValueError(f"card {card_idx} not in hand of {player}")

        hand.remove(selected)

        current_trick = state.current_trick + ((player, card_idx),)
        played = state.played_cards + ((player, card_idx),)
        hands = dict(state.hands)
        hands[player] = tuple(hand)

        if len(current_trick) < 4:
            next_state = replace(
                state,
                hands=hands,
                turn=state.turn + 1,
                current_player=player.next(),
                current_trick=current_trick,
                played_cards=played,
            )
            self.state = next_state
            return next_state, 0.0, False, {}

        assert state.contract is not None
        winner = self._trick_winner(current_trick, state.contract)
        tricks = list(state.tricks_won)
        if winner in (Seat.NORTH, Seat.SOUTH):
            tricks[0] += 1
        else:
            tricks[1] += 1

        trick_number = state.trick_number + 1
        next_state = replace(
            state,
            hands=hands,
            turn=state.turn + 1,
            trick_number=trick_number,
            tricks_won=tuple(tricks),
            current_player=winner,
            trick_leader=winner,
            current_trick=tuple(),
            played_cards=played,
            phase=Phase.PLAY,
        )

        if trick_number >= self.MAX_TRICKS:
            next_state = replace(
                next_state,
                done=True,
                phase=Phase.DEFENSE,
                result=self._final_score(next_state),
            )
            self.state = next_state
            return next_state, float(next_state.result or 0.0), True, {"winner": winner}

        self.state = next_state
        return next_state, 0.0, False, {"winner": winner}

    def _auction_status(self, auction: Tuple[int, ...] | None = None) -> dict[str, object]:
        assert self.state is not None
        calls = self._parse_auction(auction)
        current_bid: tuple[int | None, int | None] = (None, None)
        last_bid_seat: Seat | None = None
        last_bid_value: int | None = None
        last_code: int | None = None
        last_double_by: Seat | None = None
        doubled_state = 0

        for seat, code in calls:
            if code == PASS_ACTION:
                last_code = code
                continue

            if code == DOUBLE_ACTION:
                if last_bid_seat is not None and not self._same_side(seat, last_bid_seat) and doubled_state == 0:
                    doubled_state = 1
                    last_double_by = seat
                last_code = code
                continue

            if code == REDOUBLE_ACTION:
                if doubled_state == 1 and last_bid_seat is not None and self._same_side(seat, last_bid_seat):
                    doubled_state = 2
                    last_code = code
                continue

            # Bid override existing doubles.
            if code >= BID_BASE and code <= MAX_BID_CODE:
                level, strain = decode_bid_code(code)
                current_bid = (level, strain)
                last_bid_seat = seat
                last_bid_value = self._to_value(level, strain)
                last_code = code
                doubled_state = 0
                last_double_by = None

        if current_bid[0] is None:
            can_double = False
            can_redouble = False
        else:
            if self.state.current_player is None or last_bid_seat is None:
                can_double = False
                can_redouble = False
            else:
                can_double = not self._same_side(self.state.current_player, last_bid_seat) and last_code not in (DOUBLE_ACTION, REDOUBLE_ACTION)
                can_redouble = doubled_state == 1 and last_double_by is not None and self._same_side(self.state.current_player, last_bid_seat) and not self._same_side(
                    self.state.current_player,
                    last_double_by,
                )

        return {
            "current_bid": current_bid,
            "current_bid_seat": last_bid_seat,
            "current_bid_value": last_bid_value,
            "last_code": last_code,
            "can_double": can_double,
            "can_redouble": can_redouble,
            "doubled_state": doubled_state,
            "last_double_by": last_double_by,
        }

    def _auction_to_contract(self, auction: Tuple[int, ...]) -> Contract | None:
        status = self._auction_status(auction=auction)

        high_bid = status["current_bid"]
        if high_bid[0] is None or status["current_bid_seat"] is None:
            return None
        high_level, high_strain = high_bid
        doubled_state = status["doubled_state"]
        high_seat = status["current_bid_seat"]

        if high_seat is None or high_level is None or high_strain is None:
            return None

        declarer = self._discover_declarer(auction, high_strain, high_seat)
        return Contract(level=high_level, strain=high_strain, declarer=declarer, doubled=doubled_state)

    def _parse_auction(self, auction: Tuple[int, ...] | None = None) -> List[BidCall]:
        if self.state is None:
            return []
        source = self.state.auction if auction is None else auction
        out: List[BidCall] = []
        for i, code in enumerate(source):
            seat = Seat((self.state.dealer + i) % 4)
            out.append((seat, code))
            if i + 1 >= 128:
                break
        return out

    @staticmethod
    def _auction_done(auction: Tuple[int, ...]) -> bool:
        if len(auction) < 3:
            return False
        trailing = 0
        for action in reversed(auction):
            if action == PASS_ACTION:
                trailing += 1
            else:
                break

        # In a no-bid deal, auction ends after 4 consecutive passes.
        # Once a bid exists, ending is after 3 consecutive passes.
        if trailing < 3:
            return False

        has_bid = any(code not in (PASS_ACTION,) for code in auction)
        if not has_bid:
            return trailing >= 4
        return trailing >= 3

    def _discover_declarer(self, auction: Tuple[int, ...], strain: int, bidder: Seat) -> Seat:
        if self.state is None:
            return bidder
        for i, code in enumerate(auction):
            if not is_bid_code(code):
                continue
            decoded = decode_bid_code(code)
            if decoded is None:
                continue
            _, decoded_strain = decoded
            if decoded_strain != strain:
                continue
            seat = Seat((self.state.dealer + i) % 4)
            if self._same_side(seat, bidder):
                return seat
        return bidder

    def _trick_winner(self, trick_cards: Sequence[tuple[Seat, int]], contract: Contract) -> Seat:
        if not trick_cards:
            raise ValueError("cannot evaluate empty trick")

        led_card = index_to_card(trick_cards[0][1])
        led_suit = led_card.suit
        trump_suit = contract.strain if contract.strain < 4 else None
        best_seat = trick_cards[0][0]
        best_card = led_card

        def _beats(candidate: Card, incumbent: Card) -> bool:
            if trump_suit is not None:
                if candidate.suit == trump_suit and incumbent.suit != trump_suit:
                    return True
                if candidate.suit != trump_suit and incumbent.suit == trump_suit:
                    return False
                if candidate.suit == trump_suit and incumbent.suit == trump_suit:
                    return candidate.rank > incumbent.rank
            if candidate.suit == led_suit and incumbent.suit != led_suit:
                return True
            if candidate.suit != led_suit:
                return False
            if incumbent.suit != led_suit:
                return True
            return candidate.rank > incumbent.rank

        for seat, card_idx in trick_cards[1:]:
            card = index_to_card(card_idx)
            if _beats(card, best_card):
                best_seat, best_card = seat, card

        return best_seat

    def _final_score(self, state: BridgeState) -> float:
        if state.contract is None:
            return 0.0

        contract = state.contract
        required = 6 + contract.level
        made_by_declarer_side = state.tricks_won[0] if state.declarer in (Seat.NORTH, Seat.SOUTH) else state.tricks_won[1]
        made = made_by_declarer_side
        delta = made - required

        if delta >= 0:
            raw = self._positive_score(contract, delta, state.vulnerable)
        else:
            raw = -self._negative_score(contract, -delta, state.vulnerable)

        if state.declarer not in (Seat.NORTH, Seat.SOUTH):
            return -raw
        return raw

    def _positive_score(self, contract: Contract, over: int, vuln: Vulnerability) -> float:
        if contract.strain == 4:
            contract_pts = 40 + 30 * (contract.level - 1)
            overtrick_base = 30
        else:
            contract_pts = self.STRAIN_TRICK_VALUE[contract.strain] * contract.level
            overtrick_base = self.STRAIN_TRICK_VALUE[contract.strain]

        insult = 0
        if contract.doubled == 1:
            contract_pts *= 2
            overtrick = 100
            insult = 50
        elif contract.doubled == 2:
            contract_pts *= 4
            overtrick = 200
            insult = 100
        else:
            overtrick = overtrick_base

        score = contract_pts + over * overtrick + insult

        if contract_pts >= 100:
            game_bonus = 500 if vuln in (Vulnerability.N_S, Vulnerability.BOTH) else 300
        else:
            game_bonus = 50

        if contract.level == 7:
            game_bonus = 1000 if vuln in (Vulnerability.NONE, Vulnerability.E_W) else 1500
        elif contract.level == 6:
            game_bonus += 500 if vuln in (Vulnerability.N_S, Vulnerability.BOTH) else 500

        return float(score + game_bonus)

    def _negative_score(self, contract: Contract, down: int, vuln: Vulnerability) -> float:
        if contract.doubled == 0:
            return 50 * down

        if contract.doubled == 1:
            if vuln in (Vulnerability.N_S, Vulnerability.BOTH):
                return float(100 + (down - 1) * 100)
            return float(50 + (down - 1) * 100)

        if vuln in (Vulnerability.N_S, Vulnerability.BOTH):
            return float(200 + (down - 1) * 200)
        return float(100 + (down - 1) * 200)

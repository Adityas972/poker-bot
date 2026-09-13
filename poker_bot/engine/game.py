"""Heads-up No-Limit Texas Hold'em hand state machine.

Conventions
-----------
- Player 0 is the button/small blind, player 1 is the big blind. This
  matches the standard heads-up rule: the button posts the small blind
  and acts FIRST preflop, but acts LAST on every street after that
  (i.e. the big blind acts first postflop).
- Stacks reset to a fresh 100bb every hand (no persistent bankroll).
- No side pots are needed: with only 2 players, once either stack hits 0
  there are no more decisions possible for anyone, so the hand just runs
  out the board to showdown.
- Action abstraction: FOLD, CALL (check when nothing to call), BET_33 /
  BET_75 / BET_150 (bet or raise sized as a fraction of the pot *after*
  calling), ALLIN. Pot-fraction sizes are clamped to the legal min-raise
  and to the acting player's stack, and deduplicated when two fractions
  clamp to the same amount.
- Raises per street are capped at MAX_RAISES_PER_STREET. Real casino
  No-Limit rules have no such cap, but every practical poker-AI
  abstraction (and commercial solvers like PioSolver) caps raise depth
  per street -- without it, the betting tree is unbounded (a 100bb stack
  supports many successive pot-sized re-raises before an all-in), which
  makes exhaustive-action CFR traversal intractable regardless of card
  abstraction.
"""

from dataclasses import dataclass, field
from enum import IntEnum

from poker_bot.engine.card import Deck
from poker_bot.engine.evaluator import evaluate_hand

BET_FRACTIONS = [("BET_33", 0.33), ("BET_75", 0.75), ("BET_150", 1.5)]
MAX_RAISES_PER_STREET = 3


class Street(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


BOARD_SIZE_AT_STREET = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}


@dataclass
class HandState:
    hole: tuple
    full_board: list  # all 5 board cards, dealt upfront; revealed progressively
    stacks: list
    total_contrib: list
    street_contrib: list
    street: Street
    to_act: int
    num_actions_this_street: int
    last_raise_increment: int
    big_blind: int
    history: list = field(default_factory=list)
    folded_player: int = None
    showdown_done: bool = False
    aggressor_acted_this_street: bool = False
    raises_this_street: int = 0

    @property
    def board(self):
        return self.full_board[:BOARD_SIZE_AT_STREET[self.street]]

    @property
    def pot(self) -> int:
        return sum(self.total_contrib)

    @property
    def is_terminal(self) -> bool:
        return self.folded_player is not None or self.showdown_done

    def payouts(self) -> list:
        """Net chip change from each player's starting stack. Only valid
        when is_terminal is True."""
        if not self.is_terminal:
            raise ValueError("hand is not over yet")

        if self.folded_player is not None:
            winner = 1 - self.folded_player
            loser = self.folded_player
        else:
            result0 = evaluate_hand(list(self.hole[0]) + self.full_board)
            result1 = evaluate_hand(list(self.hole[1]) + self.full_board)
            if result0.sort_key == result1.sort_key:
                half = self.pot // 2
                return [half - self.total_contrib[0] + (self.pot % 2),
                        half - self.total_contrib[1]]
            winner, loser = (0, 1) if result0 > result1 else (1, 0)

        net = [0, 0]
        net[winner] = self.pot - self.total_contrib[winner]
        net[loser] = -self.total_contrib[loser]
        return net


def start_hand(rng=None, starting_stack_bb: int = 100, big_blind: int = 100) -> HandState:
    small_blind = big_blind // 2
    starting_stack = starting_stack_bb * big_blind

    deck = Deck(rng)
    hole = (tuple(deck.deal(2)), tuple(deck.deal(2)))
    board = deck.deal(5)

    stacks = [starting_stack - small_blind, starting_stack - big_blind]
    return HandState(
        hole=hole,
        full_board=board,
        stacks=stacks,
        total_contrib=[small_blind, big_blind],
        street_contrib=[small_blind, big_blind],
        street=Street.PREFLOP,
        to_act=0,
        num_actions_this_street=0,
        last_raise_increment=big_blind,
        big_blind=big_blind,
    )


def legal_actions(state: HandState):
    if state.is_terminal:
        return []

    player = state.to_act
    opponent = 1 - player
    to_call = state.street_contrib[opponent] - state.street_contrib[player]
    stack = state.stacks[player]
    pot = state.pot

    actions = []
    if to_call > 0:
        actions.append("FOLD")
    actions.append("CALL")  # covers check when to_call == 0

    call_amt = min(to_call, stack)
    remaining_after_call = stack - call_amt
    if remaining_after_call > 0 and state.raises_this_street < MAX_RAISES_PER_STREET:
        by_amount = {}
        for label, fraction in BET_FRACTIONS:
            pot_after_call = pot + call_amt
            raise_extra = round(fraction * pot_after_call)
            min_raise_extra = max(state.last_raise_increment, 1)
            total = call_amt + max(raise_extra, min_raise_extra)
            total = min(total, stack)
            if total > call_amt:
                by_amount.setdefault(total, label)
        by_amount[stack] = "ALLIN"
        actions.extend(label for _, label in sorted(by_amount.items()))

    return actions


def _action_amount(state: HandState, action: str) -> int:
    """Total chips the acting player commits this action (their new
    street_contrib), given the action label."""
    player = state.to_act
    opponent = 1 - player
    to_call = state.street_contrib[opponent] - state.street_contrib[player]
    stack = state.stacks[player]
    call_amt = min(to_call, stack)
    pot = state.pot

    if action == "CALL":
        return state.street_contrib[player] + call_amt
    if action == "ALLIN":
        return state.street_contrib[player] + stack

    fraction = dict(BET_FRACTIONS)[action]
    pot_after_call = pot + call_amt
    raise_extra = round(fraction * pot_after_call)
    min_raise_extra = max(state.last_raise_increment, 1)
    total = call_amt + max(raise_extra, min_raise_extra)
    total = min(total, stack)
    return state.street_contrib[player] + total


def _advance_street_or_showdown(state: HandState):
    if state.street == Street.RIVER:
        state.showdown_done = True
        return

    state.street = Street(state.street + 1)
    state.street_contrib = [0, 0]
    state.num_actions_this_street = 0
    state.last_raise_increment = state.big_blind
    state.aggressor_acted_this_street = False
    state.raises_this_street = 0
    state.to_act = 1  # big blind acts first on every street after preflop

    if any(s == 0 for s in state.stacks):
        # No chips left behind for at least one player: no more betting is
        # possible, so run out the remaining streets straight to showdown.
        while state.street != Street.RIVER:
            state.street = Street(state.street + 1)
        state.showdown_done = True


def apply_action(state: HandState, action: str) -> None:
    if action not in legal_actions(state):
        raise ValueError(f"illegal action {action!r} in state {state}")

    player = state.to_act
    opponent = 1 - player
    is_raise = action not in ("FOLD", "CALL")

    if action == "FOLD":
        state.folded_player = player
        state.history.append((state.street, player, action))
        return

    to_call_before = state.street_contrib[opponent] - state.street_contrib[player]
    is_first_action_this_street = state.num_actions_this_street == 0

    new_contrib = _action_amount(state, action)
    committed = new_contrib - state.street_contrib[player]

    state.stacks[player] -= committed
    state.street_contrib[player] = new_contrib
    state.total_contrib[player] += committed

    if is_raise:
        raise_increment = new_contrib - state.street_contrib[opponent]
        state.last_raise_increment = max(raise_increment, state.last_raise_increment)
        state.aggressor_acted_this_street = True
        state.raises_this_street += 1
    elif committed < to_call_before:
        # Short all-in call: refund the uncalled portion of the opponent's bet.
        refund = to_call_before - committed
        state.stacks[opponent] += refund
        state.street_contrib[opponent] -= refund
        state.total_contrib[opponent] -= refund

    state.history.append((state.street, player, action))
    state.num_actions_this_street += 1

    if is_raise:
        state.to_act = opponent
        return

    # action == "CALL": decide whether this closes the betting round.
    if to_call_before > 0:
        # Preflop-only special case: player 0's first action can be "calling"
        # just the SB/BB gap (a limp) with no real bet having occurred yet --
        # that must NOT close the round, since the big blind still gets their
        # option to raise. Any other call-facing-a-bet always closes.
        closes_round = state.aggressor_acted_this_street
    else:
        closes_round = not is_first_action_this_street

    if closes_round:
        _advance_street_or_showdown(state)
    else:
        state.to_act = opponent

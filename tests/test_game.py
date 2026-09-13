import random

from poker_bot.engine.card import Card
from poker_bot.engine.game import (
    HandState, Street, apply_action, legal_actions, start_hand,
)


def _fresh_state(hole0="Ah Ad", hole1="2c 3c", board="Kh Qd 9s 4h 2h",
                  starting_stack_bb=100, big_blind=100):
    def parse(text):
        return tuple(Card.from_str(t) for t in text.split())

    small_blind = big_blind // 2
    starting_stack = starting_stack_bb * big_blind
    return HandState(
        hole=(parse(hole0), parse(hole1)),
        full_board=list(parse(board)),
        stacks=[starting_stack - small_blind, starting_stack - big_blind],
        total_contrib=[small_blind, big_blind],
        street_contrib=[small_blind, big_blind],
        street=Street.PREFLOP,
        to_act=0,
        num_actions_this_street=0,
        last_raise_increment=big_blind,
        big_blind=big_blind,
    )


def test_blinds_posted_correctly():
    state = start_hand(rng=random.Random(0), starting_stack_bb=100, big_blind=100)
    assert state.stacks == [9950, 9900]
    assert state.total_contrib == [50, 100]
    assert state.to_act == 0  # button/SB acts first preflop


def test_fold_awards_pot_to_opponent():
    state = _fresh_state()
    apply_action(state, "FOLD")
    assert state.is_terminal
    net = state.payouts()
    assert net == [-50, 50]
    assert sum(net) == 0


def test_limp_gives_big_blind_the_option():
    state = _fresh_state()
    apply_action(state, "CALL")  # SB limps
    assert not state.is_terminal
    assert state.to_act == 1
    assert state.street == Street.PREFLOP  # round not over yet

    apply_action(state, "CALL")  # BB checks their option
    assert state.street == Street.FLOP
    assert state.to_act == 1  # BB acts first postflop
    assert state.street_contrib == [0, 0]


def test_check_check_every_street_reaches_showdown():
    state = _fresh_state(hole0="Ah Ad", hole1="2c 3c", board="Kh Qd 9s 4h 2h")
    apply_action(state, "CALL")  # preflop limp
    apply_action(state, "CALL")  # BB option check
    for _ in range(4):  # flop, turn, river: check, check (x3 streets = 6 actions)
        if state.is_terminal:
            break
        apply_action(state, "CALL")
        apply_action(state, "CALL")

    assert state.is_terminal
    assert state.showdown_done
    net = state.payouts()
    assert sum(net) == 0
    assert net[0] > 0  # pair of Aces beats King-high


def test_pot_grows_correctly_through_bet_and_call():
    state = _fresh_state()
    apply_action(state, "CALL")  # SB limp -> pot 200
    apply_action(state, "BET_75")  # BB bets 75% of pot (200) = 150
    assert state.pot == 350
    apply_action(state, "CALL")  # SB calls 150
    assert state.pot == 500
    assert state.street == Street.FLOP


def test_bet_sizing_matches_pot_fraction_formula():
    state = _fresh_state()
    apply_action(state, "CALL")  # pot now 200, to_call = 0 for BB
    actions = legal_actions(state)
    assert "BET_75" in actions
    apply_action(state, "BET_75")
    # pot_after_call (call_amt=0 since BB has nothing to call) = 200
    # raise_extra = round(0.75 * 200) = 150, added on top of BB's existing
    # 100-chip blind already counted in street_contrib -> total 250.
    assert state.street_contrib[1] == 250
    assert state.stacks[1] == (100 * 100 - 100) - 150


def test_min_raise_clamped_up_when_pot_fraction_too_small():
    # Right after blinds, pot is tiny (150), so a 33%-pot raise-on-top-of-call
    # would be far below the legal minimum raise (must be >= 1 BB more).
    state = _fresh_state()
    actions = legal_actions(state)
    assert "BET_33" in actions
    apply_action(state, "BET_33")
    # call_amt (SB calling the 50 gap) = 50; min raise increment = 100 (BB)
    # so total committed must be at least 50 (call) + 100 (min raise) = 150
    assert state.street_contrib[0] >= 150


def test_duplicate_clamped_bet_sizes_are_deduplicated():
    state = _fresh_state(starting_stack_bb=2)  # tiny stack forces clamping
    actions = legal_actions(state)
    # With only ~1.5bb behind, BET_33/75/150 likely all clamp to the same
    # all-in amount -- there must be no duplicate resulting bet sizes.
    seen_amounts = set()
    for action in actions:
        if action in ("FOLD", "CALL"):
            continue
        from poker_bot.engine.game import _action_amount
        amt = _action_amount(state, action)
        assert amt not in seen_amounts
        seen_amounts.add(amt)


def test_all_in_call_for_less_refunds_uncalled_portion():
    state = _fresh_state(starting_stack_bb=100, big_blind=100)
    # Give player 1 (BB) a short stack so they can't fully call an all-in.
    state.stacks[1] = 300
    apply_action(state, "ALLIN")  # SB shoves for 9950 total
    assert "CALL" in legal_actions(state)
    apply_action(state, "CALL")  # BB calls all-in for only 300 more

    assert state.stacks[1] == 0
    assert state.total_contrib[0] == state.total_contrib[1]  # excess refunded
    assert state.stacks[0] > 0  # SB got their uncalled excess back
    net = state.payouts()
    assert sum(net) == 0


def test_all_in_runs_out_board_with_no_more_betting():
    state = _fresh_state()
    state.stacks[1] = 300
    apply_action(state, "ALLIN")
    apply_action(state, "CALL")
    assert state.is_terminal
    assert state.showdown_done
    assert len(state.board) == 5

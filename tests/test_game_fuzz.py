"""Randomized stress test: play many full hands with random legal actions
and check invariants that must hold regardless of the specific action
sequence. Complements the targeted scenario tests in test_game.py."""

import random

from poker_bot.engine.game import apply_action, legal_actions, start_hand

NUM_HANDS = 2000
MAX_ACTIONS_PER_HAND = 40  # generous upper bound; a real hand needs far fewer


def test_random_hands_terminate_with_valid_zero_sum_payouts():
    rng = random.Random(12345)

    for hand_idx in range(NUM_HANDS):
        state = start_hand(rng=rng, starting_stack_bb=100, big_blind=100)
        starting_total = sum(state.stacks) + sum(state.total_contrib)

        actions_taken = 0
        while not state.is_terminal:
            actions = legal_actions(state)
            assert actions, f"no legal actions but hand not terminal (hand {hand_idx})"
            assert "CALL" in actions  # check/call must always be available

            action = rng.choice(actions)
            apply_action(state, action)
            actions_taken += 1
            assert actions_taken <= MAX_ACTIONS_PER_HAND, \
                f"hand {hand_idx} did not terminate within {MAX_ACTIONS_PER_HAND} actions"

            assert all(s >= 0 for s in state.stacks), f"negative stack in hand {hand_idx}"
            assert all(c >= 0 for c in state.total_contrib), f"negative contrib in hand {hand_idx}"

        net = state.payouts()
        assert sum(net) == 0, f"payouts not zero-sum in hand {hand_idx}: {net}"

        final_total = sum(state.stacks) + sum(state.total_contrib)
        assert final_total == starting_total, \
            f"chip conservation violated in hand {hand_idx}: {starting_total} -> {final_total}"

        # Nobody's net loss can exceed what they put in, and nobody's net
        # gain can exceed the pot.
        for player in (0, 1):
            assert net[player] >= -state.total_contrib[player] - 1
            assert net[player] <= state.pot

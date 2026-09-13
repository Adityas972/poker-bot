"""Monte Carlo hand-equity estimation.

Equity = P(win) + 0.5 * P(tie), estimated against a uniformly random
opponent hole-card holding and a random completion of the remaining
board, evaluated with our from-scratch hand evaluator.

This is the raw signal the card abstraction buckets on -- see
poker_bot/abstraction/buckets.py.
"""

import random

from poker_bot.engine.card import full_deck
from poker_bot.engine.evaluator import evaluate_hand


def estimate_equity(hole, board, rng: random.Random, num_samples: int = 200) -> float:
    """Estimate equity for `hole` (2 cards) given `board` (0, 3, 4, or 5
    known community cards), via Monte Carlo rollout of the rest of the deck.
    """
    used = set(hole) | set(board)
    remaining_deck = [c for c in full_deck() if c not in used]
    cards_needed = 2 + (5 - len(board))  # opponent's 2 hole cards + board runout

    wins = 0.0
    for _ in range(num_samples):
        sample = rng.sample(remaining_deck, cards_needed)
        opp_hole = sample[:2]
        full_board = list(board) + sample[2:]

        my_result = evaluate_hand(list(hole) + full_board)
        opp_result = evaluate_hand(opp_hole + full_board)

        if my_result.sort_key > opp_result.sort_key:
            wins += 1.0
        elif my_result.sort_key == opp_result.sort_key:
            wins += 0.5

    return wins / num_samples

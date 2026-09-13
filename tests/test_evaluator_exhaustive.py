"""Exhaustive check: evaluate_hand's category counts across all C(52,5) =
2,598,960 five-card hands must match the textbook frequency table exactly.
Slow (~8s) -- kept separate from the fast unit tests in test_evaluator.py.
"""

import itertools
from collections import Counter

import pytest

from poker_bot.engine.card import full_deck
from poker_bot.engine.evaluator import evaluate_hand

KNOWN_COUNTS = {
    "STRAIGHT_FLUSH": 40,
    "FOUR_OF_A_KIND": 624,
    "FULL_HOUSE": 3744,
    "FLUSH": 5108,
    "STRAIGHT": 10200,
    "THREE_OF_A_KIND": 54912,
    "TWO_PAIR": 123552,
    "PAIR": 1098240,
    "HIGH_CARD": 1302540,
}


@pytest.mark.slow
def test_all_five_card_hand_counts_match_textbook_table():
    deck = full_deck()
    counts = Counter()
    for combo in itertools.combinations(deck, 5):
        counts[evaluate_hand(list(combo)).category.name] += 1

    assert sum(counts.values()) == 2_598_960
    for category, expected in KNOWN_COUNTS.items():
        assert counts[category] == expected, f"{category}: got {counts[category]}, expected {expected}"

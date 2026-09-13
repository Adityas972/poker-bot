"""From-scratch 5-to-7 card poker hand evaluator.

Evaluates the best 5-card hand out of 5, 6, or 7 given cards (the 6/7-card
cases cover Texas Hold'em: 2 hole cards + up to 5 board cards). Aces play
both high (broadway, T-J-Q-K-A) and low (the wheel, A-2-3-4-5), per
standard Texas Hold'em rules.
"""

import itertools
from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum

from poker_bot.engine.card import Card

RANK_NAME = {
    2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven",
    8: "Eight", 9: "Nine", 10: "Ten", 11: "Jack", 12: "Queen", 13: "King",
    14: "Ace",
}
RANK_PLURAL = {
    2: "Twos", 3: "Threes", 4: "Fours", 5: "Fives", 6: "Sixes",
    7: "Sevens", 8: "Eights", 9: "Nines", 10: "Tens", 11: "Jacks",
    12: "Queens", 13: "Kings", 14: "Aces",
}


class HandCategory(IntEnum):
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8


@dataclass(order=True)
class HandResult:
    sort_key: tuple = field(compare=True)
    category: HandCategory = field(compare=False)
    description: str = field(compare=False)
    best_five: tuple = field(compare=False)


def _straight_high(unique_ranks_desc: list) -> int:
    """Highest card of a 5-card straight among these ranks, or None."""
    if len(unique_ranks_desc) != 5:
        return None
    if unique_ranks_desc[0] - unique_ranks_desc[4] == 4:
        return unique_ranks_desc[0]
    if unique_ranks_desc == [14, 5, 4, 3, 2]:
        return 5  # the wheel: A-2-3-4-5, plays as a 5-high straight
    return None


def _rank_five(cards) -> HandResult:
    ranks = sorted((c.rank for c in cards), reverse=True)
    is_flush = len({c.suit for c in cards}) == 1
    unique_ranks = sorted(set(ranks), reverse=True)
    straight_high = _straight_high(unique_ranks)

    if straight_high and is_flush:
        name = "Royal Flush" if straight_high == 14 else \
            f"Straight Flush, {RANK_NAME[straight_high]} High"
        return HandResult((HandCategory.STRAIGHT_FLUSH, straight_high),
                           HandCategory.STRAIGHT_FLUSH, name, tuple(cards))

    counts = Counter(ranks)
    groups = sorted(counts.items(), key=lambda kv: (-kv[1], -kv[0]))
    pattern = [count for _, count in groups]

    if pattern[0] == 4:
        quad, kicker = groups[0][0], groups[1][0]
        return HandResult((HandCategory.FOUR_OF_A_KIND, quad, kicker),
                           HandCategory.FOUR_OF_A_KIND,
                           f"Four of a Kind, {RANK_PLURAL[quad]}", tuple(cards))

    if pattern[0] == 3 and pattern[1] == 2:
        trip, pair = groups[0][0], groups[1][0]
        return HandResult((HandCategory.FULL_HOUSE, trip, pair),
                           HandCategory.FULL_HOUSE,
                           f"Full House, {RANK_PLURAL[trip]} full of {RANK_PLURAL[pair]}",
                           tuple(cards))

    if is_flush:
        return HandResult((HandCategory.FLUSH, *ranks), HandCategory.FLUSH,
                           f"Flush, {RANK_NAME[ranks[0]]} High", tuple(cards))

    if straight_high:
        return HandResult((HandCategory.STRAIGHT, straight_high), HandCategory.STRAIGHT,
                           f"Straight, {RANK_NAME[straight_high]} High", tuple(cards))

    if pattern[0] == 3:
        trip = groups[0][0]
        kickers = tuple(r for r, _ in groups[1:])
        return HandResult((HandCategory.THREE_OF_A_KIND, trip, *kickers),
                           HandCategory.THREE_OF_A_KIND,
                           f"Three of a Kind, {RANK_PLURAL[trip]}", tuple(cards))

    if pattern[0] == 2 and pattern[1] == 2:
        pair1, pair2 = groups[0][0], groups[1][0]
        kicker = groups[2][0]
        return HandResult((HandCategory.TWO_PAIR, pair1, pair2, kicker),
                           HandCategory.TWO_PAIR,
                           f"Two Pair, {RANK_PLURAL[pair1]} and {RANK_PLURAL[pair2]}",
                           tuple(cards))

    if pattern[0] == 2:
        pair = groups[0][0]
        kickers = tuple(r for r, _ in groups[1:])
        return HandResult((HandCategory.PAIR, pair, *kickers), HandCategory.PAIR,
                           f"Pair of {RANK_PLURAL[pair]}", tuple(cards))

    return HandResult((HandCategory.HIGH_CARD, *ranks), HandCategory.HIGH_CARD,
                       f"{RANK_NAME[ranks[0]]} High", tuple(cards))


def evaluate_hand(cards: list) -> HandResult:
    """Best possible 5-card hand out of 5-7 given cards."""
    if not 5 <= len(cards) <= 7:
        raise ValueError(f"evaluate_hand needs 5-7 cards, got {len(cards)}")

    best = None
    for combo in itertools.combinations(cards, 5):
        result = _rank_five(combo)
        if best is None or result > best:
            best = result
    return best

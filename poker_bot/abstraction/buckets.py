"""Hand-strength percentile bucketing: the card abstraction that makes
CFR on full Hold'em tractable.

There are just too many distinct (hole cards, board) combos in full
heads-up No-Limit for tabular CFR to keep one regret-table entry per
information set. The usual fix, going back to early poker-AI
abstraction work, is to swap exact cards for a coarse "hand strength
bucket" derived from Monte Carlo equity - hands with similar win
probability get treated as interchangeable and share a table entry.

Buckets are built by percentile rather than fixed equity ranges: sample
a lot of random hands per street, estimate their equity, and pick
cutoffs so each bucket holds an equal slice of the distribution.
Equity isn't uniformly spread out (most hands cluster in the middle),
so fixed-width bins would waste most of your buckets on equities that
barely ever come up.

The defaults below are set for fast iteration - seconds, not minutes.
For real training you'd want to bump up `num_calibration_samples` and
`equity_rollouts` for steadier bucket boundaries and equity estimates;
that's a tuning knob, not something that affects correctness.
"""

import bisect
import random

from poker_bot.abstraction.equity import estimate_equity
from poker_bot.engine.card import full_deck

STREET_BOARD_SIZES = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


def _street_name(board_len: int) -> str:
    for name, size in STREET_BOARD_SIZES.items():
        if size == board_len:
            return name
    raise ValueError(f"invalid board length {board_len}")


def _random_hole_and_board(street: str, rng: random.Random):
    deck = full_deck()
    rng.shuffle(deck)
    board_size = STREET_BOARD_SIZES[street]
    return tuple(deck[:2]), deck[2:2 + board_size]


def compute_percentile_thresholds(street: str, num_buckets: int = 10,
                                   num_calibration_samples: int = 300,
                                   equity_rollouts: int = 50,
                                   seed: int = None) -> list:
    """Sample random hands at this street, estimate their equities, and
    return the `num_buckets - 1` percentile cutoffs splitting the
    resulting distribution into equal-sized buckets."""
    rng = random.Random(seed)
    equities = []
    for _ in range(num_calibration_samples):
        hole, board = _random_hole_and_board(street, rng)
        equities.append(estimate_equity(hole, board, rng, equity_rollouts))
    equities.sort()

    thresholds = []
    for i in range(1, num_buckets):
        idx = min(len(equities) * i // num_buckets, len(equities) - 1)
        thresholds.append(equities[idx])
    return thresholds


def bucket_from_equity(equity: float, thresholds: list) -> int:
    return bisect.bisect_right(thresholds, equity)


class HandBucketer:
    """Precomputes per-street percentile thresholds, then maps any
    (hole, board) into a hand-strength bucket index in [0, num_buckets)."""

    def __init__(self, num_buckets: int = 10, num_calibration_samples: int = 300,
                 equity_rollouts: int = 50, seed: int = None):
        self.num_buckets = num_buckets
        self.equity_rollouts = equity_rollouts
        self.thresholds = {
            street: compute_percentile_thresholds(
                street, num_buckets, num_calibration_samples, equity_rollouts, seed)
            for street in STREET_BOARD_SIZES
        }
        self._equity_cache = {}
        self._rng = random.Random(seed)

    def bucket(self, hole, board) -> int:
        street = _street_name(len(board))
        cache_key = (street, frozenset(hole),
                     tuple(sorted(board, key=lambda c: (c.rank, c.suit))))
        if cache_key not in self._equity_cache:
            self._equity_cache[cache_key] = estimate_equity(
                hole, list(board), self._rng, self.equity_rollouts)
        equity = self._equity_cache[cache_key]
        return bucket_from_equity(equity, self.thresholds[street])

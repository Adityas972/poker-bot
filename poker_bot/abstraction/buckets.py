"""Hand-strength percentile bucketing: the card abstraction that makes
CFR on full Hold'em tractable.

Full heads-up No-Limit Hold'em has far too many distinct (hole cards,
board) combinations for tabular CFR to hold one regret table entry per
information set. The standard fix (used in early poker-AI abstraction
research) is to replace exact cards with a coarse "hand strength bucket"
computed from Monte Carlo equity: two hands with similar win probability
are treated as strategically interchangeable and share one CFR table
entry.

Bucketing is done by PERCENTILE, not by fixed equity ranges: we sample
many random hands per street, estimate each one's equity, and choose
cutoffs so every bucket holds an equal share of the strength
distribution. This matters because equity is not remotely uniformly
distributed (most hands cluster in the middle), so fixed-width equity
bins would waste most buckets on rarely-occurring equities.

Note on performance: the defaults below are tuned for fast iteration
(seconds, not minutes). Real CFR training in later stages should
recalibrate with larger `num_calibration_samples` / `equity_rollouts`
for more stable bucket boundaries and equity estimates -- this is a
tuning knob, not a correctness concern.
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

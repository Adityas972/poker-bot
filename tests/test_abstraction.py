import random

from poker_bot.abstraction.buckets import (
    HandBucketer, bucket_from_equity, compute_percentile_thresholds,
)
from poker_bot.abstraction.equity import estimate_equity
from poker_bot.engine.card import Card


def test_pocket_aces_beats_seven_two_offsuit_preflop():
    rng = random.Random(1)
    aa_equity = estimate_equity((Card.from_str("Ah"), Card.from_str("Ac")), [], rng, 400)
    rng = random.Random(1)
    weak_equity = estimate_equity((Card.from_str("7h"), Card.from_str("2c")), [], rng, 400)
    assert aa_equity > 0.8
    assert weak_equity < 0.45
    assert aa_equity > weak_equity


def test_made_nuts_on_river_has_near_certain_equity():
    rng = random.Random(2)
    # Board has a made flush already on it for our hand's suit; opponent
    # would need a very specific holding to beat quads.
    hole = (Card.from_str("Ah"), Card.from_str("Ad"))
    board = [Card.from_str("Ac"), Card.from_str("As"), Card.from_str("Kh"),
              Card.from_str("7d"), Card.from_str("2c")]
    equity = estimate_equity(hole, board, rng, 500)
    assert equity > 0.97  # only a better quad/straight-flush could beat this, impossible here


def test_percentile_thresholds_are_monotonically_increasing():
    thresholds = compute_percentile_thresholds("flop", num_buckets=8,
                                                 num_calibration_samples=150,
                                                 equity_rollouts=30, seed=3)
    assert thresholds == sorted(thresholds)
    assert len(thresholds) == 7


def test_bucket_from_equity_stays_in_range():
    thresholds = compute_percentile_thresholds("preflop", num_buckets=10,
                                                 num_calibration_samples=150,
                                                 equity_rollouts=30, seed=4)
    for equity in [0.0, 0.1, 0.5, 0.9, 1.0]:
        b = bucket_from_equity(equity, thresholds)
        assert 0 <= b < 10


def test_bucketer_assigns_strong_preflop_hand_to_higher_bucket_than_weak():
    bucketer = HandBucketer(num_buckets=10, num_calibration_samples=150,
                             equity_rollouts=30, seed=5)
    aa_bucket = bucketer.bucket((Card.from_str("Ah"), Card.from_str("Ac")), [])
    weak_bucket = bucketer.bucket((Card.from_str("7h"), Card.from_str("2c")), [])
    assert aa_bucket >= weak_bucket


def test_bucketer_caches_repeated_lookups():
    bucketer = HandBucketer(num_buckets=6, num_calibration_samples=100,
                             equity_rollouts=20, seed=6)
    hole = (Card.from_str("Kh"), Card.from_str("Kd"))
    first = bucketer.bucket(hole, [])
    second = bucketer.bucket(hole, [])
    assert first == second  # cached, not re-sampled
    assert len(bucketer._equity_cache) == 1

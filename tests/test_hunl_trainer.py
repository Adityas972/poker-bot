import numpy as np
import pytest

from poker_bot.abstraction.buckets import HandBucketer
from poker_bot.cfr.hunl_trainer import train
from poker_bot.engine.game import legal_actions, start_hand


def test_training_runs_and_produces_valid_average_strategies():
    bucketer = HandBucketer(num_buckets=4, num_calibration_samples=80,
                             equity_rollouts=15, seed=0)
    _, node_map = train(300, bucketer, seed=1)

    assert len(node_map) > 0
    for node in node_map.values():
        avg = node.get_average_strategy()
        assert avg.shape == (node.num_actions,)
        assert np.all(avg >= 0)
        assert abs(avg.sum() - 1.0) < 1e-9


@pytest.mark.slow
def test_strongest_preflop_bucket_folds_far_less_than_weakest():
    """Qualitative correctness check (no closed-form equilibrium exists
    for full HUNL): at the button's very first decision (preflop, no
    action yet), the strongest hand-strength bucket should learn to
    fold far less often than the weakest bucket. This is the same root
    info set for every hand, so it gets plenty of visits per bucket even
    at moderate iteration counts.

    Note: we deliberately do NOT assert that strong hands raise/bet more
    often in raw frequency terms. Real equilibrium strategies often have
    strong hands slowplay (just call) to keep weaker hands in the pot,
    while marginal hands get pushed towards a polarized raise-or-fold
    pattern -- so "aggression" is not a reliable monotonic signal here.
    Folding less with a strong hand is the theoretically robust
    invariant, and is what's checked below.
    """
    num_buckets = 4
    bucketer = HandBucketer(num_buckets=num_buckets, num_calibration_samples=150,
                             equity_rollouts=20, seed=2)
    _, node_map = train(3000, bucketer, seed=3)

    weak_key = "0:0:"
    strong_key = f"0:{num_buckets - 1}:"
    assert weak_key in node_map, "weakest preflop bucket never visited at the root"
    assert strong_key in node_map, "strongest preflop bucket never visited at the root"

    # The button faces the SB/BB gap at the root, so FOLD is a legal action
    # here -- get the actual action order rather than assuming it.
    fold_idx = legal_actions(start_hand()).index("FOLD")

    weak_fold_prob = node_map[weak_key].get_average_strategy()[fold_idx]
    strong_fold_prob = node_map[strong_key].get_average_strategy()[fold_idx]

    assert strong_fold_prob < weak_fold_prob
    assert strong_fold_prob < 0.1  # a top-bucket hand should almost never fold for free

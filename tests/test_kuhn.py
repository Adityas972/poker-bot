"""Regression tests: vanilla CFR on Kuhn Poker must converge to the known
closed-form Nash equilibrium (Kuhn, 1950). See poker_bot/games/kuhn.py for
the game definition and references.

The equilibrium is a one-parameter family (alpha in [0, 1/3]); most action
probabilities are pinned exactly (0 or 1) regardless of alpha, and we check
those directly. The alpha-dependent probabilities are checked for internal
consistency instead of against a fixed number.
"""

from poker_bot.games.kuhn import KUHN_EQUILIBRIUM_VALUE, train

ITERATIONS = 200_000
SEED = 1
TOLERANCE = 0.03


def _train():
    return train(ITERATIONS, seed=SEED)


def test_game_value_matches_theory():
    avg_value, _ = _train()
    assert abs(avg_value - KUHN_EQUILIBRIUM_VALUE) < TOLERANCE


def test_pinned_strategies_match_equilibrium():
    """Action probabilities the equilibrium fixes regardless of alpha."""
    _, node_map = _train()

    def prob_bet(info_set):
        return node_map[info_set].get_average_strategy()[1]

    # Player 0
    assert prob_bet("Jpb") < TOLERANCE          # J facing a bet: always fold
    assert prob_bet("Kpb") > 1 - TOLERANCE      # K facing a bet: always call
    assert prob_bet("Q") < TOLERANCE            # Q first to act: never bet

    # Player 1
    assert prob_bet("Jb") < TOLERANCE           # J facing a bet: always fold
    assert abs(prob_bet("Jp") - 1 / 3) < TOLERANCE   # checked to with J: bet 1/3
    assert abs(prob_bet("Qb") - 1 / 3) < TOLERANCE   # facing bet with Q: call 1/3
    assert prob_bet("Qp") < TOLERANCE           # checked to with Q: always check
    assert prob_bet("Kb") > 1 - TOLERANCE       # facing bet with K: always call
    assert prob_bet("Kp") > 1 - TOLERANCE       # checked to with K: always bet


def test_alpha_consistent_across_infosets():
    """J-bet-first probability (alpha) must be within its valid range, and
    the Qpb call probability must equal alpha + 1/3 (the known relationship
    between these two free-parameter action probabilities)."""
    _, node_map = _train()
    alpha = node_map["J"].get_average_strategy()[1]
    qpb_call = node_map["Qpb"].get_average_strategy()[1]

    assert -TOLERANCE < alpha < 1 / 3 + TOLERANCE
    assert abs(qpb_call - (alpha + 1 / 3)) < TOLERANCE

import random

import numpy as np
import pytest

from poker_bot.deep_cfr.features import FEATURE_DIM, encode_infoset, legal_action_mask
from poker_bot.deep_cfr.memory import ReservoirBuffer
from poker_bot.deep_cfr.trainer import (
    masked_regret_matching_strategy, policy_strategy, train,
)
from poker_bot.engine.card import Card
from poker_bot.engine.game import HandState, Street, legal_actions, start_hand


def _root_state(hole0: str, hole1: str, big_blind: int = 100, starting_stack_bb: int = 100):
    def parse(text):
        return tuple(Card.from_str(t) for t in text.split())

    small_blind = big_blind // 2
    starting_stack = starting_stack_bb * big_blind
    return HandState(
        hole=(parse(hole0), parse(hole1)),
        full_board=[],
        stacks=[starting_stack - small_blind, starting_stack - big_blind],
        total_contrib=[small_blind, big_blind],
        street_contrib=[small_blind, big_blind],
        street=Street.PREFLOP,
        to_act=0,
        num_actions_this_street=0,
        last_raise_increment=big_blind,
        big_blind=big_blind,
    )


def _facing_shove_state(hole0: str, hole1: str, big_blind: int = 100, starting_stack_bb: int = 100):
    """Player 0 having limped, then facing an all-in shove from player 1
    -- a real, reachable in-distribution state (player 0 always acts
    first preflop in this engine, so this models player 0 calling, then
    player 1 responding with a shove). Unlike the bare root (where the
    SB/BB gap is so cheap that even weak hands almost never fold,
    regardless of strength), this is a genuinely discriminating decision:
    continuing costs (almost) a full stack, so hand strength should
    visibly separate fold vs. call probability."""
    def parse(text):
        return tuple(Card.from_str(t) for t in text.split())

    starting_stack = starting_stack_bb * big_blind
    return HandState(
        hole=(parse(hole0), parse(hole1)),
        full_board=[],
        stacks=[starting_stack - big_blind, 0],
        total_contrib=[big_blind, starting_stack],
        street_contrib=[big_blind, starting_stack],
        street=Street.PREFLOP,
        to_act=0,
        num_actions_this_street=2,
        last_raise_increment=starting_stack - big_blind,
        big_blind=big_blind,
        raises_this_street=1,
        aggressor_acted_this_street=True,
        history=[(Street.PREFLOP, 0, "CALL"), (Street.PREFLOP, 1, "ALLIN")],
    )


def test_reservoir_buffer_respects_capacity():
    buf = ReservoirBuffer(capacity=10, seed=0)
    for i in range(1000):
        buf.add(i)
    assert len(buf) == 10
    assert buf.num_seen == 1000


def test_encode_infoset_has_expected_shape_and_range():
    state = start_hand(rng=random.Random(0))
    features = encode_infoset(state, 0, random.Random(1), equity_rollouts=20)
    assert features.shape == (FEATURE_DIM,)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))


def test_legal_action_mask_matches_legal_actions():
    state = start_hand(rng=random.Random(0))
    actions = legal_actions(state)
    mask = legal_action_mask(actions)
    assert mask.sum() == len(actions)


def test_masked_regret_matching_zeroes_illegal_actions():
    advantages = np.array([5.0, -1.0, 3.0, 2.0, 0.0, 1.0])
    mask = np.array([True, True, False, False, False, False])
    strategy = masked_regret_matching_strategy(advantages, mask)
    assert strategy[2:].sum() == 0.0
    assert abs(strategy.sum() - 1.0) < 1e-9
    assert strategy[0] > strategy[1]  # higher advantage -> higher probability


def test_training_runs_and_policy_strategy_is_valid_distribution():
    _, policy_net, adv_mem, strat_mem = train(
        num_cfr_iterations=1, traversals_per_player_per_iteration=15,
        advantage_train_steps=20, policy_train_steps=20,
        hidden_dim=16, equity_rollouts=10, seed=0)

    assert len(strat_mem) > 0
    state = start_hand(rng=random.Random(5))
    strategy = policy_strategy(policy_net, state, 0, random.Random(6), equity_rollouts=10)
    assert abs(sum(strategy.values()) - 1.0) < 1e-4
    assert all(p >= 0 for p in strategy.values())
    assert set(strategy.keys()) == set(legal_actions(state))


@pytest.mark.slow
def test_policy_network_folds_pocket_aces_less_than_seven_two_offsuit():
    """Qualitative correctness check, analogous to the tabular MCCFR test
    but exercising the neural path end-to-end. Note this does NOT test at
    the bare preflop root: facing just the cheap SB/BB gap, folding is
    almost never correct regardless of hand strength (real heads-up
    theory), so fold probability is ~0 for both strong and weak hands
    there and the comparison is degenerate. Instead we test facing an
    all-in shove, where continuing is expensive and hand strength should
    visibly separate fold vs. call probability."""
    _, policy_net, _, _ = train(
        num_cfr_iterations=3, traversals_per_player_per_iteration=60,
        advantage_train_steps=150, policy_train_steps=200,
        hidden_dim=32, equity_rollouts=30, seed=1)

    strong_state = _facing_shove_state("Ah Ac", "2c 3c")
    weak_state = _facing_shove_state("7h 2c", "3c 4c")
    rng = random.Random(7)

    strong_strategy = policy_strategy(policy_net, strong_state, 0, rng, equity_rollouts=200)
    weak_strategy = policy_strategy(policy_net, weak_state, 0, rng, equity_rollouts=200)

    assert strong_strategy["FOLD"] < weak_strategy["FOLD"]

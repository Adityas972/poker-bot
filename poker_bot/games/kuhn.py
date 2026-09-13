"""Kuhn Poker: the smallest nontrivial poker game, used to sanity-check CFR.

3 cards (J, Q, K, one each), 2 players, each antes 1 chip and gets 1
card, single betting round, player 0 first. Actions are 'p' (pass -
check, or fold if facing a bet) and 'b' (bet/call 1 chip). Terminal
histories: pp, bp, bb, pbp, pbb. Higher card wins at showdown.

It has a known closed-form Nash equilibrium (up to one free parameter),
and the game value for player 0 playing optimally is exactly -1/18.
Vanilla CFR self-play should converge to strategies close to that
equilibrium - which is exactly why this is a good first thing to run:
if the CFR code is wrong, this is where it'll show up, before trusting
it on anything bigger.
"""

import random

import numpy as np

from poker_bot.cfr.node import InfoSetNode

CARD_NAMES = ["J", "Q", "K"]
ACTIONS = ["p", "b"]
NUM_ACTIONS = len(ACTIONS)
KUHN_EQUILIBRIUM_VALUE = -1.0 / 18.0


def _terminal_utility(cards, history, player, opponent):
    """Utility to `player` if `history` is terminal, else None."""
    if len(history) <= 1:
        return None

    terminal_pass = history[-1] == "p"
    double_bet = history[-2:] == "bb"
    player_card_higher = cards[player] > cards[opponent]

    if terminal_pass:
        if history == "pp":
            return 1 if player_card_higher else -1
        return 1  # opponent folded after facing a bet
    if double_bet:
        return 2 if player_card_higher else -2
    return None


def cfr(cards, history, p0, p1, node_map):
    """One recursive CFR pass; returns expected utility to the player on
    move at `history`, under the current strategy profile."""
    plays = len(history)
    player = plays % 2
    opponent = 1 - player

    utility = _terminal_utility(cards, history, player, opponent)
    if utility is not None:
        return utility

    info_set = CARD_NAMES[cards[player]] + history
    node = node_map.setdefault(info_set, InfoSetNode(NUM_ACTIONS))

    reach_prob = p0 if player == 0 else p1
    strategy = node.get_strategy(reach_prob)

    action_utils = np.zeros(NUM_ACTIONS)
    for i, action in enumerate(ACTIONS):
        next_history = history + action
        if player == 0:
            action_utils[i] = -cfr(cards, next_history, p0 * strategy[i], p1, node_map)
        else:
            action_utils[i] = -cfr(cards, next_history, p0, p1 * strategy[i], node_map)

    node_util = float(np.dot(strategy, action_utils))

    opponent_reach = p1 if player == 0 else p0
    for i in range(NUM_ACTIONS):
        regret = action_utils[i] - node_util
        node.regret_sum[i] += opponent_reach * regret

    return node_util


def train(iterations: int, seed: int = None):
    """Run vanilla CFR self-play for `iterations` hands.

    Returns (average_game_value_for_player_0, node_map).
    """
    rng = random.Random(seed)
    node_map: dict[str, InfoSetNode] = {}
    cards = [0, 1, 2]
    total_util = 0.0

    for _ in range(iterations):
        rng.shuffle(cards)
        total_util += cfr(cards, "", 1.0, 1.0, node_map)

    return total_util / iterations, node_map


if __name__ == "__main__":
    ITERATIONS = 50_000
    avg_value, node_map = train(ITERATIONS, seed=42)

    print(f"Iterations: {ITERATIONS}")
    print(f"Average game value for player 0: {avg_value:.4f} "
          f"(theoretical: {KUHN_EQUILIBRIUM_VALUE:.4f})")
    print()
    print("Learned average strategies (P(pass/check-fold), P(bet/call)):")
    for key in sorted(node_map.keys()):
        strat = node_map[key].get_average_strategy()
        print(f"  {key:>4}: pass={strat[0]:.3f}  bet={strat[1]:.3f}")

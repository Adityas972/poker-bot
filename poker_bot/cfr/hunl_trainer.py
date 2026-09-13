"""External-sampling Monte Carlo CFR (MCCFR) self-play on abstracted
heads-up No-Limit Hold'em.

Plain vanilla CFR (exploring the whole game tree) doesn't work here.
Even with hand-strength bucketing and a 6-action bet abstraction,
No-Limit betting trees blow up combinatorially - a 100bb stack supports
a lot of successive pot-sized raises before someone's all-in, across 4
streets. It's the same reason real poker AI (Libratus, Pluribus) uses
Monte Carlo CFR instead of the vanilla version for anything past toy
games.

External sampling works around it like this: for one sampled deal, the
player whose turn it is to be "traversed" gets all their actions fully
explored (full regret computation, same as vanilla CFR), but the
opponent's actions just get sampled from their current strategy. That
collapses the opponent's raise-war branching down to one sampled path
per iteration. Which player is being traversed alternates every
iteration so both sides' regrets and average strategies keep updating.

Based on Lanctot et al.'s 2009 paper on Monte Carlo sampling for regret
minimization in extensive games.
"""

import copy
import random

from poker_bot.abstraction.buckets import HandBucketer
from poker_bot.cfr.node import InfoSetNode
from poker_bot.engine.game import apply_action, legal_actions, start_hand

ACTION_CODE = {"FOLD": "f", "CALL": "c", "BET_33": "3", "BET_75": "7",
               "BET_150": "5", "ALLIN": "a"}


def _info_set_key(state, player, bucketer) -> str:
    bucket = bucketer.bucket(state.hole[player], state.board)
    history = "".join(f"{street.value}{p}{ACTION_CODE[action]}"
                       for street, p, action in state.history)
    return f"{state.street.value}:{bucket}:{history}"


def _es_cfr(state, traverser: int, bucketer: HandBucketer, node_map: dict,
            rng: random.Random) -> float:
    """Returns the expected value of this subtree to `traverser`."""
    if state.is_terminal:
        return float(state.payouts()[traverser])

    player = state.to_act
    actions = legal_actions(state)
    info_set = _info_set_key(state, player, bucketer)
    node = node_map.setdefault(info_set, InfoSetNode(len(actions)))

    if player == traverser:
        strategy = node.current_strategy()
        action_utils = [0.0] * len(actions)
        for i, action in enumerate(actions):
            next_state = copy.deepcopy(state)
            apply_action(next_state, action)
            action_utils[i] = _es_cfr(next_state, traverser, bucketer, node_map, rng)

        node_value = sum(s * u for s, u in zip(strategy, action_utils))
        for i in range(len(actions)):
            node.regret_sum[i] += action_utils[i] - node_value
        return node_value

    # Opponent's node: sample one action from their current strategy,
    # and accumulate towards their average strategy at this visit.
    strategy = node.get_strategy(1.0)
    action_idx = rng.choices(range(len(actions)), weights=strategy, k=1)[0]
    next_state = copy.deepcopy(state)
    apply_action(next_state, actions[action_idx])
    return _es_cfr(next_state, traverser, bucketer, node_map, rng)


def train(iterations: int, bucketer: HandBucketer, seed: int = None,
          starting_stack_bb: int = 100, big_blind: int = 100):
    rng = random.Random(seed)
    node_map: dict = {}
    total_value = 0.0  # accumulated value to whoever was traverser that hand

    for i in range(iterations):
        state = start_hand(rng=rng, starting_stack_bb=starting_stack_bb, big_blind=big_blind)
        traverser = i % 2
        value = _es_cfr(state, traverser, bucketer, node_map, rng)
        total_value += value if traverser == 0 else -value

    return total_value / iterations, node_map


if __name__ == "__main__":
    import time

    bucketer = HandBucketer(num_buckets=6, num_calibration_samples=200,
                             equity_rollouts=40, seed=0)

    # Single growing run (not 3 independent from-scratch runs) so the
    # printed throughput reflects real achievable iterations/sec.
    node_map: dict = {}
    rng = random.Random(1)
    checkpoints = [200, 1000, 3000]
    total_value = 0.0
    done = 0
    t_start = time.time()
    for target in checkpoints:
        t0 = time.time()
        while done < target:
            state = start_hand(rng=rng, starting_stack_bb=100, big_blind=100)
            traverser = done % 2
            value = _es_cfr(state, traverser, bucketer, node_map, rng)
            total_value += value if traverser == 0 else -value
            done += 1
        elapsed = time.time() - t0
        print(f"{done:>6} iterations: {len(node_map):>7} info sets, "
              f"+{elapsed:>5.1f}s this batch ({done / (time.time() - t_start):.1f} hands/sec so far)")

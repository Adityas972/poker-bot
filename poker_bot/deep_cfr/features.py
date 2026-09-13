"""Fixed-size information-set feature encoding for Deep CFR.

Tabular CFR keys a table by an exact (street, hand bucket, history)
string. A neural network needs a fixed-size numeric vector instead, and
can work directly off continuous equity rather than needing hands
pre-sorted into discrete percentile buckets -- one of the practical
wins of moving to function approximation.

The action space is also fixed-size here (the 6 canonical actions),
with illegal actions at any given node handled by masking rather than
by the input changing shape.
"""

import numpy as np

from poker_bot.abstraction.equity import estimate_equity
from poker_bot.engine.game import MAX_RAISES_PER_STREET

CANONICAL_ACTIONS = ["FOLD", "CALL", "BET_33", "BET_75", "BET_150", "ALLIN"]
CANONICAL_INDEX = {action: i for i, action in enumerate(CANONICAL_ACTIONS)}
NUM_CANONICAL_ACTIONS = len(CANONICAL_ACTIONS)

NUM_STREETS = 4
# street one-hot + equity + pot_frac + to_call_frac + own_stack_frac +
# raises_frac + (self, opponent) action-count history over 6 actions each
FEATURE_DIM = NUM_STREETS + 5 + 2 * NUM_CANONICAL_ACTIONS


def encode_infoset(state, player: int, rng, equity_rollouts: int = 100,
                    precomputed_equity: float = None) -> np.ndarray:
    opponent = 1 - player
    starting_stack = state.stacks[player] + state.total_contrib[player]

    street_onehot = np.zeros(NUM_STREETS, dtype=np.float32)
    street_onehot[state.street.value] = 1.0

    if precomputed_equity is not None:
        equity = precomputed_equity
    else:
        equity = estimate_equity(state.hole[player], state.board, rng, equity_rollouts)

    pot = state.pot
    to_call = state.street_contrib[opponent] - state.street_contrib[player]
    pot_frac = pot / (2 * starting_stack)
    to_call_frac = (to_call / pot) if pot > 0 else 0.0
    own_stack_frac = state.stacks[player] / starting_stack
    raises_frac = state.raises_this_street / MAX_RAISES_PER_STREET

    self_counts = np.zeros(NUM_CANONICAL_ACTIONS, dtype=np.float32)
    opp_counts = np.zeros(NUM_CANONICAL_ACTIONS, dtype=np.float32)
    for _, actor, action in state.history:
        idx = CANONICAL_INDEX[action]
        (self_counts if actor == player else opp_counts)[idx] += 1.0
    self_counts /= 5.0
    opp_counts /= 5.0

    return np.concatenate([
        street_onehot,
        np.array([equity, pot_frac, to_call_frac, own_stack_frac, raises_frac], dtype=np.float32),
        self_counts, opp_counts,
    ]).astype(np.float32)


def legal_action_mask(legal_action_labels) -> np.ndarray:
    mask = np.zeros(NUM_CANONICAL_ACTIONS, dtype=bool)
    for label in legal_action_labels:
        mask[CANONICAL_INDEX[label]] = True
    return mask

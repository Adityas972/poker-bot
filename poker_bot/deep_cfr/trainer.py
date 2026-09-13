"""Deep CFR (Brown, Lerer, Gross & Sandholm, 2019) on heads-up No-Limit
Hold'em.

Same CFR math as poker_bot.cfr.hunl_trainer's external-sampling MCCFR --
the difference is entirely in *where the strategy comes from*. Tabular
MCCFR looks up (and updates) one regret-table row per exact information
set; that stops scaling once the number of information sets vastly
exceeds the number of hands we can afford to play (see the README's
Stage 4 notes -- 200k+ info sets after 3000 hands even with a coarse
abstraction). Deep CFR instead trains a neural network to predict
regret directly from a fixed-size feature vector, so information sets
that are never visited still get sensible predictions by
generalization from similar ones that were.

Algorithm (simplified Deep CFR, single-network-per-player variant):
  for t in 1..num_cfr_iterations:
      for traverser in (0, 1):
          run `traversals_per_player_per_iteration` external-sampling
          traversals, using the CURRENT advantage networks (both
          players') to pick strategies at every node. Traverser's own
          nodes: explore all actions, compute regret, store
          (features, regret_target, legal_mask, t) in that player's
          advantage memory. Opponent nodes: sample one action, store
          (features, strategy_target, legal_mask, t) in the shared
          strategy memory.
          retrain that player's advantage network from scratch on
          its (reservoir-sampled) advantage memory.
  train a policy network on the accumulated strategy memory -- this
  is the final approximate-average-strategy network.

Samples are weighted by iteration `t` (linear CFR weighting, as in the
paper) so later, better-informed iterations count more.
"""

import copy
import random

import numpy as np
import torch

from poker_bot.abstraction.equity import estimate_equity
from poker_bot.deep_cfr.features import (
    CANONICAL_INDEX, NUM_CANONICAL_ACTIONS, encode_infoset, legal_action_mask,
)
from poker_bot.deep_cfr.memory import ReservoirBuffer
from poker_bot.deep_cfr.networks import InfosetMLP
from poker_bot.engine.game import apply_action, legal_actions, start_hand


def masked_regret_matching_strategy(advantages: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """Regret matching restricted to legal actions; returns a length-6
    canonical-action probability vector, zero at illegal indices."""
    positive = np.clip(advantages, 0, None) * legal_mask
    total = positive.sum()
    if total > 0:
        return positive / total
    strategy = legal_mask.astype(np.float32)
    return strategy / strategy.sum()


def _strategy_for_state(net, features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        advantages = net(torch.from_numpy(features).unsqueeze(0)).squeeze(0).numpy()
    return masked_regret_matching_strategy(advantages, mask)


def _traverse(state, traverser: int, advantage_nets, advantage_memory,
              strategy_memory: ReservoirBuffer, cfr_iteration: int,
              rng: random.Random, equity_rollouts: int, equity_cache: dict) -> float:
    if state.is_terminal:
        return float(state.payouts()[traverser])

    player = state.to_act
    actions = legal_actions(state)
    mask = legal_action_mask(actions)

    equity_key = (player, len(state.board))
    if equity_key not in equity_cache:
        equity_cache[equity_key] = estimate_equity(state.hole[player], state.board, rng, equity_rollouts)
    features = encode_infoset(state, player, rng, precomputed_equity=equity_cache[equity_key])

    full_strategy = _strategy_for_state(advantage_nets[player], features, mask)
    strategy = np.array([full_strategy[CANONICAL_INDEX[a]] for a in actions])
    strategy = strategy / strategy.sum()

    if player == traverser:
        action_utils = np.zeros(len(actions))
        for i, action in enumerate(actions):
            next_state = copy.deepcopy(state)
            apply_action(next_state, action)
            action_utils[i] = _traverse(next_state, traverser, advantage_nets, advantage_memory,
                                         strategy_memory, cfr_iteration, rng, equity_rollouts, equity_cache)
        node_value = float(np.dot(strategy, action_utils))

        regret_target = np.zeros(NUM_CANONICAL_ACTIONS, dtype=np.float32)
        for i, action in enumerate(actions):
            regret_target[CANONICAL_INDEX[action]] = action_utils[i] - node_value
        advantage_memory[traverser].add((features, regret_target, mask.copy(), float(cfr_iteration)))
        return node_value

    strategy_target = np.zeros(NUM_CANONICAL_ACTIONS, dtype=np.float32)
    for i, action in enumerate(actions):
        strategy_target[CANONICAL_INDEX[action]] = strategy[i]
    strategy_memory.add((features, strategy_target, mask.copy(), float(cfr_iteration)))

    action_idx = rng.choices(range(len(actions)), weights=strategy, k=1)[0]
    next_state = copy.deepcopy(state)
    apply_action(next_state, actions[action_idx])
    return _traverse(next_state, traverser, advantage_nets, advantage_memory,
                      strategy_memory, cfr_iteration, rng, equity_rollouts, equity_cache)


def train_network(network, buffer: ReservoirBuffer, batch_size: int = 32,
                   num_steps: int = 200, lr: float = 1e-3) -> None:
    if len(buffer) == 0:
        return
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    for _ in range(num_steps):
        batch = buffer.sample(batch_size)
        features = torch.from_numpy(np.stack([b[0] for b in batch]))
        targets = torch.from_numpy(np.stack([b[1] for b in batch]))
        masks = torch.from_numpy(np.stack([b[2] for b in batch])).float()
        weights = torch.tensor([b[3] for b in batch], dtype=torch.float32)

        optimizer.zero_grad()
        preds = network(features)
        per_elem = (preds - targets) ** 2 * masks
        per_sample = per_elem.sum(dim=1) / masks.sum(dim=1).clamp(min=1)
        loss = (per_sample * weights).mean()
        loss.backward()
        optimizer.step()


def train(num_cfr_iterations: int = 5, traversals_per_player_per_iteration: int = 100,
          advantage_train_steps: int = 200, policy_train_steps: int = 300,
          advantage_buffer_capacity: int = 20000, strategy_buffer_capacity: int = 20000,
          hidden_dim: int = 64, equity_rollouts: int = 50, seed: int = None,
          starting_stack_bb: int = 100, big_blind: int = 100):
    rng = random.Random(seed)
    advantage_nets = [InfosetMLP(hidden_dim), InfosetMLP(hidden_dim)]
    advantage_memory = [ReservoirBuffer(advantage_buffer_capacity, seed),
                         ReservoirBuffer(advantage_buffer_capacity, seed)]
    strategy_memory = ReservoirBuffer(strategy_buffer_capacity, seed)

    for t in range(1, num_cfr_iterations + 1):
        for traverser in (0, 1):
            for _ in range(traversals_per_player_per_iteration):
                state = start_hand(rng=rng, starting_stack_bb=starting_stack_bb, big_blind=big_blind)
                equity_cache: dict = {}
                _traverse(state, traverser, advantage_nets, advantage_memory,
                          strategy_memory, t, rng, equity_rollouts, equity_cache)
            advantage_nets[traverser] = InfosetMLP(hidden_dim)
            train_network(advantage_nets[traverser], advantage_memory[traverser],
                          num_steps=advantage_train_steps)

    policy_net = InfosetMLP(hidden_dim)
    train_network(policy_net, strategy_memory, num_steps=policy_train_steps)

    return advantage_nets, policy_net, advantage_memory, strategy_memory


def policy_strategy(policy_net, state, player: int, rng: random.Random,
                     equity_rollouts: int = 100) -> dict:
    """The trained policy network's strategy at `state`, as a dict of
    {action_label: probability} over the actually-legal actions."""
    actions = legal_actions(state)
    mask = legal_action_mask(actions)
    features = encode_infoset(state, player, rng, equity_rollouts)
    with torch.no_grad():
        raw = policy_net(torch.from_numpy(features).unsqueeze(0)).squeeze(0).numpy()
    positive = np.clip(raw, 0, None) * mask
    total = positive.sum()
    full = positive / total if total > 0 else mask.astype(np.float32) / mask.sum()
    return {a: float(full[CANONICAL_INDEX[a]]) for a in actions}

"""Small MLPs used by Deep CFR.

Both the advantage network (predicts counterfactual regret per action)
and the policy network (predicts the average strategy) share the same
architecture: a fixed-size info-set feature vector in, one raw score
per canonical action out. What differs is only how the output is
interpreted and trained (see poker_bot/deep_cfr/trainer.py).
"""

import torch
import torch.nn as nn

from poker_bot.deep_cfr.features import FEATURE_DIM, NUM_CANONICAL_ACTIONS


class InfosetMLP(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, NUM_CANONICAL_ACTIONS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

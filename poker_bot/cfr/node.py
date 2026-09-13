import numpy as np


class InfoSetNode:
    """One information set's regret/strategy tables for vanilla CFR.

    Reused across games (Kuhn, Leduc, ...): a game only needs to supply
    the information-set key and the number of legal actions at that node.
    """

    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)

    def get_strategy(self, reach_prob: float) -> np.ndarray:
        """Regret matching: strategy proportional to positive regret."""
        positive_regrets = np.maximum(self.regret_sum, 0)
        total = positive_regrets.sum()
        if total > 0:
            strategy = positive_regrets / total
        else:
            strategy = np.full(self.num_actions, 1.0 / self.num_actions)
        self.strategy_sum += reach_prob * strategy
        return strategy

    def get_average_strategy(self) -> np.ndarray:
        total = self.strategy_sum.sum()
        if total > 0:
            return self.strategy_sum / total
        return np.full(self.num_actions, 1.0 / self.num_actions)

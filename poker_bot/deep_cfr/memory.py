"""Reservoir-sampling memory buffer, as used by Deep CFR (Brown et al.
2019) to keep a bounded, uniform-random sample of all (infoset,
target) pairs ever generated during training, without needing to store
the full (unboundedly large) history."""

import random


class ReservoirBuffer:
    def __init__(self, capacity: int, seed: int = None):
        self.capacity = capacity
        self.buffer = []
        self.num_seen = 0
        self._rng = random.Random(seed)

    def add(self, item) -> None:
        self.num_seen += 1
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            idx = self._rng.randint(0, self.num_seen - 1)
            if idx < self.capacity:
                self.buffer[idx] = item

    def sample(self, batch_size: int):
        batch_size = min(batch_size, len(self.buffer))
        return self._rng.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

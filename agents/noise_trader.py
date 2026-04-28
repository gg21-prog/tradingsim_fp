import numpy as np


class NoiseTrader:
    """Uninformed trader. Poisson arrivals, random side and size."""

    def __init__(self, lam=5, max_size=10, seed=None):
        self.lam = lam        # avg orders per step
        self.max_size = max_size
        self.rng = np.random.default_rng(seed)

    def get_orders(self):
        n = self.rng.poisson(self.lam)
        orders = []
        for _ in range(n):
            side = 'buy' if self.rng.random() < 0.5 else 'sell'
            size = int(self.rng.integers(1, self.max_size + 1))
            orders.append({'side': side, 'size': size})
        return orders

"""
Informed trader with a private price signal.

Based on Glosten-Milgrom (1985): an informed trader knows the true asset value
and trades when the quoted price diverges from it. Here we implement this as
look-ahead: the informed trader observes the next `signal_horizon` price steps
and trades directionally when the expected move exceeds a threshold.

Signal quality controlled by `noise_sigma` — at 0 the trader is omniscient,
at high values they are nearly random (degrading to noise trader behavior).
"""
import numpy as np


class InformedTrader:
    def __init__(
        self,
        signal_horizon=20,   # how many steps ahead they can see
        noise_sigma=0.001,   # Gaussian noise added to their signal
        trade_threshold=0.0005,  # min expected move to bother trading
        max_size=15,         # informed traders trade bigger
        lam=2,               # avg orders per step (less frequent, more directional)
        seed=None,
    ):
        self.signal_horizon = signal_horizon
        self.noise_sigma = noise_sigma
        self.trade_threshold = trade_threshold
        self.max_size = max_size
        self.lam = lam
        self.rng = np.random.default_rng(seed)

        self._future_prices = []   # set by env before each step

    def set_price_path(self, future_prices):
        """Environment calls this to give the informed trader their signal."""
        self._future_prices = future_prices

    def get_orders(self):
        if len(self._future_prices) == 0:
            return []

        # Private signal: expected price move over horizon (with noise)
        horizon = min(self.signal_horizon, len(self._future_prices))
        future_price = self._future_prices[horizon - 1]
        current_price = self._future_prices[0] if len(self._future_prices) > 1 else future_price

        # Add noise — informed traders aren't omniscient
        signal = (future_price - current_price) / current_price
        signal += self.rng.normal(0, self.noise_sigma)

        if abs(signal) < self.trade_threshold:
            return []

        # Trade in the direction of the signal
        side = 'buy' if signal > 0 else 'sell'
        n = self.rng.poisson(self.lam)
        n = max(1, n)  # always at least one order when signal is strong enough

        orders = []
        for _ in range(n):
            size = int(self.rng.integers(self.max_size // 2, self.max_size + 1))
            orders.append({'side': side, 'size': size, 'informed': True})
        return orders

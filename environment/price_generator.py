import numpy as np


class GBMPriceGenerator:
    """Geometric Brownian Motion price process. dS = mu*S*dt + sigma*S*dW"""

    def __init__(self, S0=100.0, mu=0.0, sigma=0.002, seed=None):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)
        self.price = S0

    def reset(self):
        self.price = self.S0
        return self.price

    def step(self):
        dW = self.rng.standard_normal()
        self.price *= np.exp((self.mu - 0.5 * self.sigma ** 2) + self.sigma * dW)
        return self.price

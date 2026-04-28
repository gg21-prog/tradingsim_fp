"""
Avellaneda-Stoikov closed-form market maker.
No learning — just the formula. Used as benchmark against SAC.

reservation price: r = mid - q * gamma * sigma^2 * (T - t)
optimal spread:    delta = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/kappa)
quotes at:         bid = r - delta/2,  ask = r + delta/2
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from environment.exchange import Exchange
from environment.price_generator import GBMPriceGenerator
from agents.noise_trader import NoiseTrader


class AvellanedaStoikov:
    def __init__(self, gamma=0.1, kappa=1.5, sigma=0.002, episode_length=390):
        self.gamma = gamma            # risk aversion
        self.kappa = kappa            # order arrival sensitivity
        self.sigma = sigma
        self.episode_length = episode_length

    def get_quotes(self, mid, inventory, steps_remaining):
        T_minus_t = steps_remaining / self.episode_length

        # Reservation price: shade toward zero inventory
        reservation = mid - inventory * self.gamma * (self.sigma ** 2) * T_minus_t

        # Optimal half-spread
        half_spread = (self.gamma * self.sigma ** 2 * T_minus_t / 2
                       + (1 / self.gamma) * np.log(1 + self.gamma / self.kappa))

        # Convert to offsets from mid (fractional)
        bid_price = reservation - half_spread
        ask_price = reservation + half_spread

        bid_offset = np.clip((mid - bid_price) / mid, 0.0, 0.01)
        ask_offset = np.clip((ask_price - mid) / mid, 0.0, 0.01)

        return bid_offset, ask_offset


def run_episode(seed=None):
    rng = np.random.default_rng(seed)
    sigma = 0.002
    episode_length = 390
    kappa = 1.5

    price_gen = GBMPriceGenerator(sigma=sigma, seed=seed)
    exchange = Exchange()
    noise_trader = NoiseTrader(seed=seed)
    agent = AvellanedaStoikov(sigma=sigma, kappa=kappa, episode_length=episode_length)

    mid = price_gen.reset()
    cash, inventory = 0.0, 0
    portfolio_value = 0.0
    history = []

    for step in range(episode_length):
        steps_remaining = episode_length - step
        bid_offset, ask_offset = agent.get_quotes(mid, inventory, steps_remaining)

        bid_price = mid * (1 - bid_offset)
        ask_price = mid * (1 + ask_offset)

        exchange.cancel_all_quotes()
        exchange.place_quote('bid', bid_price, 10_000)
        exchange.place_quote('ask', ask_price, 10_000)

        orders = noise_trader.get_orders()
        all_trades = exchange.match_orders(orders)

        # A-S fill probability
        p_fill_bid = np.exp(-kappa * (mid - bid_price))
        p_fill_ask = np.exp(-kappa * (ask_price - mid))

        for t in all_trades:
            p = p_fill_ask if t['side'] == 'buy' else p_fill_bid
            if rng.random() < p:
                if t['side'] == 'buy':
                    cash += t['price'] * t['size']
                    inventory -= t['size']
                else:
                    cash -= t['price'] * t['size']
                    inventory += t['size']

        mid = price_gen.step()
        portfolio_value = cash + inventory * mid

        history.append({
            'step':            step,
            'mid':             mid,
            'bid':             bid_price,
            'ask':             ask_price,
            'spread':          ask_price - bid_price,
            'inventory':       inventory,
            'portfolio_value': portfolio_value,
            'num_trades':      len([t for t in all_trades]),
        })

    return history


if __name__ == '__main__':
    pnls = []
    for i in range(100):
        h = run_episode(seed=i)
        pnls.append(h[-1]['portfolio_value'])
    print(f"A-S baseline over 100 episodes:")
    print(f"  Mean PnL:   {np.mean(pnls):.2f}")
    print(f"  Median PnL: {np.median(pnls):.2f}")
    print(f"  Win rate:   {sum(p > 0 for p in pnls)}%")
    print(f"  Std:        {np.std(pnls):.2f}")

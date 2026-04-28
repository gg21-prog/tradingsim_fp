"""
Phase 2 Gym environment: market maker faces a mix of noise traders AND
an informed trader with a private price signal.

Key differences from Phase 1:
  - 30% of order flow comes from an informed trader (configurable via informed_frac)
  - 8-dim observation: adds rolling imbalance and VPIN-like toxicity estimate
  - Reward unchanged — agent must figure out that widening spread protects against toxicity

The market maker cannot directly observe who is informed. It must infer toxicity
from behavioral signals (persistent directional flow, inventory drift) and adapt
its spread accordingly. This is the core adverse selection problem.

Observation (8-dim):
  0  inventory           (normalized by max_inventory)
  1  spread              (normalized by 0.05)
  2  price_change_zscore (z-scored by 3*sigma)
  3  flow_imbalance      (buy_vol - sell_vol at current step)
  4  trade_flow          (total volume normalized)
  5  time_remaining      (fraction of episode left)
  6  rolling_imbalance   (5-step buy-fraction — detects persistent directional flow)
  7  rolling_toxicity    (5-step |imbalance| — VPIN-like, high = likely informed flow)
"""
import numpy as np
import gymnasium as gym
from collections import deque

from environment.exchange import Exchange
from environment.price_generator import GBMPriceGenerator
from agents.noise_trader import NoiseTrader
from agents.informed_trader import InformedTrader


class MarketEnvPhase2(gym.Env):
    metadata = {'render_modes': []}

    def __init__(
        self,
        episode_length=390,
        max_inventory=100,
        sigma=0.002,
        kappa=1.5,
        informed_frac=0.3,    # fraction of steps that include informed trading
        signal_horizon=20,
    ):
        super().__init__()
        self.episode_length = episode_length
        self.max_inventory = max_inventory
        self.sigma = sigma
        self.kappa = kappa
        self.informed_frac = informed_frac
        self.signal_horizon = signal_horizon

        self.observation_space = gym.spaces.Box(
            low=-2.0, high=2.0, shape=(8,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=0.0, high=0.01, shape=(2,), dtype=np.float32
        )

        self.price_gen = GBMPriceGenerator(sigma=sigma)
        self.exchange = Exchange()
        self.noise_trader = NoiseTrader()
        self.informed_trader = InformedTrader(signal_horizon=signal_horizon)

        # Rolling window for toxicity features
        self._buy_history = deque(maxlen=5)
        self._sell_history = deque(maxlen=5)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.mid = self.price_gen.reset()
        self.exchange.reset()

        self.cash = 0.0
        self.inventory = 0
        self.step_count = 0
        self.portfolio_value = 0.0
        self.prev_mid = self.mid
        self.last_bid = self.mid * 0.999
        self.last_ask = self.mid * 1.001

        self._buy_history.clear()
        self._sell_history.clear()

        # Pre-generate the full price path so informed trader can look ahead
        self._price_path = self._generate_price_path()

        return self._obs(), {}

    def _generate_price_path(self):
        """Simulate the full episode price path for the informed trader's signal."""
        rng = np.random.default_rng(self.np_random.integers(0, 2**31))
        price = self.mid
        path = [price]
        for _ in range(self.episode_length + self.signal_horizon):
            dW = rng.standard_normal()
            price *= np.exp((- 0.5 * self.sigma ** 2) + self.sigma * dW)
            path.append(price)
        return path

    def step(self, action):
        bid_offset = float(np.clip(action[0], 0.0, 0.01))
        ask_offset = float(np.clip(action[1], 0.0, 0.01))

        bid_price = self.mid * (1.0 - bid_offset)
        ask_price = self.mid * (1.0 + ask_offset)
        self.last_bid = bid_price
        self.last_ask = ask_price

        self.exchange.cancel_all_quotes()
        self.exchange.place_quote('bid', bid_price, 10_000)
        self.exchange.place_quote('ask', ask_price, 10_000)

        # Fill probability (same A-S model as Phase 1)
        bid_offset_dollars = self.mid - bid_price
        ask_offset_dollars = ask_price - self.mid
        p_fill_bid = np.exp(-self.kappa * bid_offset_dollars)
        p_fill_ask = np.exp(-self.kappa * ask_offset_dollars)

        # Gather orders from both trader types
        orders = self.noise_trader.get_orders()

        # Informed trader participates based on informed_frac
        if self.np_random.random() < self.informed_frac:
            future_slice = self._price_path[self.step_count: self.step_count + self.signal_horizon + 1]
            self.informed_trader.set_price_path(future_slice)
            orders += self.informed_trader.get_orders()
        else:
            self.informed_trader.set_price_path([])

        all_trades = self.exchange.match_orders(orders)

        trades = []
        for t in all_trades:
            p = p_fill_ask if t['side'] == 'buy' else p_fill_bid
            if self.np_random.random() < p:
                trades.append(t)

        buy_vol = sell_vol = 0
        for t in trades:
            if t['side'] == 'buy':
                self.cash += t['price'] * t['size']
                self.inventory -= t['size']
                buy_vol += t['size']
            else:
                self.cash -= t['price'] * t['size']
                self.inventory += t['size']
                sell_vol += t['size']

        self._buy_history.append(buy_vol)
        self._sell_history.append(sell_vol)

        self.prev_mid = self.mid
        self.mid = self.price_gen.step()

        new_value = self.cash + self.inventory * self.mid
        reward = (new_value - self.portfolio_value) - 0.1 * abs(self.inventory)
        self.portfolio_value = new_value

        self.step_count += 1
        terminated = self.step_count >= self.episode_length

        obs = self._obs(buy_vol, sell_vol)
        info = {
            'portfolio_value': new_value,
            'inventory':       self.inventory,
            'spread':          ask_price - bid_price,
            'num_trades':      len(trades),
            'mid_price':       self.mid,
            'buy_vol':         buy_vol,
            'sell_vol':        sell_vol,
        }
        return obs, reward, terminated, False, info

    def _obs(self, buy_vol=0, sell_vol=0):
        total_vol = buy_vol + sell_vol
        imbalance = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0
        trade_flow = min(total_vol / 50.0, 1.0)

        price_change = (self.mid - self.prev_mid) / (3 * self.sigma * self.prev_mid)
        spread = (self.last_ask - self.last_bid) / self.mid
        time_remaining = (self.episode_length - self.step_count) / self.episode_length

        # Rolling toxicity features (VPIN-like)
        total_buy = sum(self._buy_history) + 1e-9
        total_sell = sum(self._sell_history) + 1e-9
        total_roll = total_buy + total_sell
        rolling_imbalance = (total_buy - total_sell) / total_roll   # signed: + = buy pressure
        rolling_toxicity = abs(total_buy - total_sell) / total_roll  # unsigned: high = toxic

        obs = np.array([
            np.clip(self.inventory / self.max_inventory, -1.0, 1.0),
            np.clip(spread / 0.05, 0.0, 1.0),
            np.clip(price_change, -1.0, 1.0),
            float(imbalance),
            float(trade_flow),
            float(time_remaining),
            float(np.clip(rolling_imbalance, -1.0, 1.0)),
            float(np.clip(rolling_toxicity,   0.0, 1.0)),
        ], dtype=np.float32)

        return obs

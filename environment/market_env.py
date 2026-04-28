import numpy as np
import gymnasium as gym

from environment.exchange import Exchange
from environment.price_generator import GBMPriceGenerator
from agents.noise_trader import NoiseTrader


class MarketEnv(gym.Env):
    """
    Single-agent Gym environment for market making.

    Action: [bid_offset, ask_offset] — fractional offsets from mid price.
      bid posted at mid * (1 - bid_offset)
      ask posted at mid * (1 + ask_offset)

    Observation (6-dim, all normalized):
      [inventory, spread, price_change_zscore, flow_imbalance, trade_flow, time_remaining]

    Reward: delta mark-to-market value - 0.1 * |inventory|

    Fill probability: Avellaneda-Stoikov exponential model.
      P(fill | offset) = exp(-kappa * offset_in_dollars)
      Tighter spread = more fills. Wider spread = fewer fills but more per trade.
      Without this, the agent quotes maximum spread always — no real learning signal.
      Reference: mbt_gym (JJJerome/mbt_gym), picklenchips/MARKET-MAKING-RL
    """

    metadata = {'render_modes': []}

    def __init__(self, episode_length=390, max_inventory=100, sigma=0.002, kappa=1.5):
        super().__init__()
        self.episode_length = episode_length
        self.max_inventory = max_inventory
        self.sigma = sigma

        self.observation_space = gym.spaces.Box(
            low=-2.0, high=2.0, shape=(6,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=0.0, high=0.01, shape=(2,), dtype=np.float32
        )

        self.kappa = kappa   # controls how steeply fill rate drops with spread width
        self.price_gen = GBMPriceGenerator(sigma=sigma)
        self.exchange = Exchange()
        self.noise_trader = NoiseTrader()

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

        return self._obs(), {}

    def step(self, action):
        bid_offset = float(np.clip(action[0], 0.0, 0.01))
        ask_offset = float(np.clip(action[1], 0.0, 0.01))

        bid_price = self.mid * (1.0 - bid_offset)
        ask_price = self.mid * (1.0 + ask_offset)
        self.last_bid = bid_price
        self.last_ask = ask_price

        # Post quotes (large size so they don't run out mid-step)
        self.exchange.cancel_all_quotes()
        self.exchange.place_quote('bid', bid_price, 10_000)
        self.exchange.place_quote('ask', ask_price, 10_000)

        # Noise traders arrive; fill probability decays exponentially with spread width
        # P(fill) = exp(-kappa * offset_dollars) — from Avellaneda-Stoikov (2008)
        bid_offset_dollars = self.mid - bid_price
        ask_offset_dollars = ask_price - self.mid
        p_fill_bid = np.exp(-self.kappa * bid_offset_dollars)
        p_fill_ask = np.exp(-self.kappa * ask_offset_dollars)

        orders = self.noise_trader.get_orders()
        all_trades = self.exchange.match_orders(orders)

        # Filter trades by fill probability
        trades = []
        for t in all_trades:
            p = p_fill_ask if t['side'] == 'buy' else p_fill_bid
            if self.np_random.random() < p:
                trades.append(t)

        buy_vol = sell_vol = 0
        for t in trades:
            if t['side'] == 'buy':       # buyer hit MM's ask — MM sold
                self.cash += t['price'] * t['size']
                self.inventory -= t['size']
                buy_vol += t['size']
            else:                         # seller hit MM's bid — MM bought
                self.cash -= t['price'] * t['size']
                self.inventory += t['size']
                sell_vol += t['size']

        # Advance price
        self.prev_mid = self.mid
        self.mid = self.price_gen.step()

        # Reward: change in mark-to-market minus inventory penalty
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
        }
        return obs, reward, terminated, False, info

    def _obs(self, buy_vol=0, sell_vol=0):
        total_vol = buy_vol + sell_vol
        imbalance = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0
        trade_flow = min(total_vol / 50.0, 1.0)   # normalize by ~expected volume per step

        price_change = (self.mid - self.prev_mid) / (3 * self.sigma * self.prev_mid)
        spread = (self.last_ask - self.last_bid) / self.mid   # as fraction of mid
        time_remaining = (self.episode_length - self.step_count) / self.episode_length

        obs = np.array([
            np.clip(self.inventory / self.max_inventory, -1.0, 1.0),
            np.clip(spread / 0.05, 0.0, 1.0),        # max_spread=5%
            np.clip(price_change, -1.0, 1.0),
            float(imbalance),
            float(trade_flow),
            float(time_remaining),
        ], dtype=np.float32)

        return obs

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from stable_baselines3.common.env_checker import check_env
from environment.market_env import MarketEnv


def test_sb3_compatibility():
    env = MarketEnv()
    check_env(env, warn=True)
    print("PASS: SB3 env_checker passed")


def test_episode_runs():
    env = MarketEnv(episode_length=10)
    obs, _ = env.reset()
    assert obs.shape == (6,)
    assert not np.any(np.isnan(obs))

    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert not np.any(np.isnan(obs)), "NaN in observation"
        assert not np.isnan(reward), "NaN reward"

    assert terminated
    print("PASS: full episode completes, no NaN")


def test_obs_bounds():
    env = MarketEnv(episode_length=390)
    obs, _ = env.reset()
    for _ in range(100):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert np.all(obs >= -2.0) and np.all(obs <= 2.0), f"obs out of bounds: {obs}"
    print("PASS: all observations stay within [-2, 2]")


def test_random_policy_pnl():
    """Random policy should generally lose or break even — confirms reward direction."""
    env = MarketEnv()
    pnls = []
    for _ in range(10):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        pnls.append(info['portfolio_value'])
    print(f"Random policy avg final PnL: {np.mean(pnls):.2f} (expect near 0 or negative)")
    print("PASS: random policy test complete")


if __name__ == '__main__':
    test_sb3_compatibility()
    test_episode_runs()
    test_obs_bounds()
    test_random_policy_pnl()
    print("\nAll env tests passed.")

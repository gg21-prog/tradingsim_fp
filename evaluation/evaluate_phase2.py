"""
Phase 2 evaluation: SAC vs A-S vs random in the informed-trader environment.
Adds VPIN-based toxicity detection benchmark on top of Phase 1 metrics.

Usage:
    python evaluation/evaluate_phase2.py --model models/phase2/<run-id>/best/best_model

Produces:
    1. Summary table   — PnL, spread, inventory, fill rate
    2. VPIN analysis   — spread vs toxicity signal correlation
    3. ROC AUC plot    — does the MM widen spread when flow is informed?
    4. PnL distribution — phase2 SAC vs baselines
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from environment.market_env_phase2 import MarketEnvPhase2
from baselines.avellaneda_stoikov import AvellanedaStoikov
from evaluation.vpin import compute_vpin, spread_vs_vpin_auc, plot_vpin_spread, plot_roc_comparison


def run_sac_episode(model, env):
    obs, _ = env.reset()
    done = False
    history = []
    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        history.append({
            'step':            step,
            'mid':             info['mid_price'],
            'bid':             env.last_bid,
            'ask':             env.last_ask,
            'spread':          info['spread'],
            'inventory':       info['inventory'],
            'portfolio_value': info['portfolio_value'],
            'num_trades':      info['num_trades'],
            'buy_vol':         info.get('buy_vol', 0),
            'sell_vol':        info.get('sell_vol', 0),
        })
        step += 1
    return history


def run_as_episode_phase2(env_config, seed=None):
    """Run A-S baseline in the Phase 2 environment."""
    env = MarketEnvPhase2(**env_config)
    obs, _ = env.reset(seed=seed)
    agent = AvellanedaStoikov(sigma=env_config.get('sigma', 0.002),
                              episode_length=env_config.get('episode_length', 390))
    done = False
    history = []
    step = 0
    while not done:
        steps_remaining = env.episode_length - env.step_count
        bid_offset, ask_offset = agent.get_quotes(env.mid, env.inventory, steps_remaining)
        action = np.array([bid_offset, ask_offset], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        history.append({
            'step':            step,
            'spread':          info['spread'],
            'inventory':       info['inventory'],
            'portfolio_value': info['portfolio_value'],
            'buy_vol':         info.get('buy_vol', 0),
            'sell_vol':        info.get('sell_vol', 0),
        })
        step += 1
    return history


def run_random_episode(env, seed=None):
    obs, _ = env.reset(seed=seed)
    done = False
    history = []
    step = 0
    while not done:
        action = env.action_space.sample()
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        history.append({
            'step':            step,
            'spread':          info['spread'],
            'inventory':       info['inventory'],
            'portfolio_value': info['portfolio_value'],
            'buy_vol':         info.get('buy_vol', 0),
            'sell_vol':        info.get('sell_vol', 0),
        })
        step += 1
    return history


def summarise(histories, label):
    pnls    = [h[-1]['portfolio_value'] for h in histories]
    spreads = [np.mean([s['spread'] for s in h]) for h in histories]
    inv     = [abs(h[-1]['inventory']) for h in histories]
    win     = sum(p > 0 for p in pnls)
    print(f"\n{label}")
    print(f"  Mean PnL:          {np.mean(pnls):>8.2f}")
    print(f"  Median PnL:        {np.median(pnls):>8.2f}")
    print(f"  Win rate:          {win}/{len(pnls)}")
    print(f"  Avg spread ($):    {np.mean(spreads):>8.4f}")
    print(f"  Avg closing |inv|: {np.mean(inv):>8.1f}")
    return np.array(pnls), np.array(spreads)


def plot_pnl_dist(sac_pnls, as_pnls, rand_pnls, out="evaluation/plots/phase2_pnl.png"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    lo = min(sac_pnls.min(), as_pnls.min(), rand_pnls.min())
    hi = max(sac_pnls.max(), as_pnls.max(), rand_pnls.max())
    bins = np.linspace(lo, hi, 40)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(rand_pnls, bins=bins, alpha=0.5, label='Random',              color='gray')
    ax.hist(as_pnls,   bins=bins, alpha=0.6, label='Avellaneda-Stoikov',  color='steelblue')
    ax.hist(sac_pnls,  bins=bins, alpha=0.7, label='SAC Phase 2',         color='#e8759a')
    ax.axvline(0, color='black', lw=1, ls='--')
    ax.set_xlabel('Episode PnL ($)')
    ax.set_ylabel('Count')
    ax.set_title('Phase 2 PnL: SAC vs A-S vs Random (with informed trader)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--n', type=int, default=100)
    args = parser.parse_args()

    env_config = dict(episode_length=390, max_inventory=100, sigma=0.002,
                      kappa=1.5, informed_frac=0.3, signal_horizon=20)

    print(f"Loading model: {args.model}")
    model = SAC.load(args.model)
    env   = MarketEnvPhase2(**env_config)

    print(f"\nRunning {args.n} episodes each...")
    sac_histories  = [run_sac_episode(model, env) for _ in range(args.n)]
    as_histories   = [run_as_episode_phase2(env_config, seed=i) for i in range(args.n)]
    rand_histories = [run_random_episode(env, seed=i) for i in range(args.n)]

    print("\n" + "=" * 50 + " RESULTS " + "=" * 50)
    sac_pnls,  _ = summarise(sac_histories,  "SAC Phase 2")
    as_pnls,   _ = summarise(as_histories,   "Avellaneda-Stoikov baseline")
    rand_pnls, _ = summarise(rand_histories, "Random policy")

    print("\n" + "=" * 50 + " VPIN ANALYSIS " + "=" * 50)
    sac_auc  = spread_vs_vpin_auc(sac_histories)
    as_auc   = spread_vs_vpin_auc(as_histories)
    rand_auc = spread_vs_vpin_auc(rand_histories)
    print(f"\n  VPIN ROC AUC (spread widening on toxic flow):")
    print(f"    SAC Phase 2:         {sac_auc:.4f}  {'PASS' if sac_auc > 0.60 else 'below target'}")
    print(f"    Avellaneda-Stoikov:  {as_auc:.4f}")
    print(f"    Random:              {rand_auc:.4f}  (baseline ~0.5)")

    print("\nGenerating plots...")
    plot_vpin_spread(sac_histories[0])
    plot_roc_comparison(sac_auc, as_auc, rand_auc)
    plot_pnl_dist(sac_pnls, as_pnls, rand_pnls)
    print("\nAll plots saved to evaluation/plots/")


if __name__ == '__main__':
    main()

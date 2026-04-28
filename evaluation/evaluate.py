"""
Market-focused evaluation. Run after training.

Usage:
    python evaluation/evaluate.py --model models/phase1/<run-id>/best/best_model

Produces:
    1. Summary table  — PnL, win rate, avg spread, fill rate vs A-S and random
    2. Episode plot   — price + MM quotes + inventory for one sample episode
    3. Spread heatmap — does spread widen with inventory? (key market maker behavior)
    4. PnL distribution — SAC vs A-S vs random across 100 episodes
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from stable_baselines3 import SAC
from environment.market_env import MarketEnv
from baselines.avellaneda_stoikov import run_episode as run_as_episode


# ── helpers ──────────────────────────────────────────────────────────────────

def run_sac_episode(model, env, deterministic=True):
    obs, _ = env.reset()
    done = False
    history = []
    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        bid_offset = float(np.clip(action[0], 0, 0.01))
        ask_offset = float(np.clip(action[1], 0, 0.01))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        history.append({
            'step':            step,
            'mid':             info['mid_price'] if 'mid_price' in info else env.mid,
            'bid':             env.last_bid,
            'ask':             env.last_ask,
            'spread':          info['spread'],
            'inventory':       info['inventory'],
            'portfolio_value': info['portfolio_value'],
            'num_trades':      info['num_trades'],
        })
        step += 1
    return history


def run_random_episode(env, seed=None):
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()
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
    return pnls, spreads


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_sample_episode(sac_history, out="evaluation/plots/episode.png"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    steps     = [h['step'] for h in sac_history]
    mid       = [h['mid'] for h in sac_history]
    bid       = [h['bid'] for h in sac_history]
    ask       = [h['ask'] for h in sac_history]
    inventory = [h['inventory'] for h in sac_history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1.plot(steps, mid, color='black', lw=1.2, label='mid price')
    ax1.plot(steps, bid, color='steelblue', lw=0.8, alpha=0.7, label='MM bid')
    ax1.plot(steps, ask, color='tomato',    lw=0.8, alpha=0.7, label='MM ask')
    ax1.fill_between(steps, bid, ask, alpha=0.08, color='gray')
    ax1.set_ylabel('Price ($)')
    ax1.legend(fontsize=9)
    ax1.set_title('Sample Episode — SAC Market Maker')

    ax2.fill_between(steps, inventory, 0,
                     where=[v >= 0 for v in inventory], alpha=0.5, color='steelblue', label='long')
    ax2.fill_between(steps, inventory, 0,
                     where=[v < 0 for v in inventory],  alpha=0.5, color='tomato',    label='short')
    ax2.axhline(0, color='black', lw=0.8, ls='--')
    ax2.set_ylabel('Inventory (shares)')
    ax2.set_xlabel('Step')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_spread_vs_inventory(sac_histories, out="evaluation/plots/spread_vs_inventory.png"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    inv, spread = [], []
    for h in sac_histories:
        for step in h:
            inv.append(step['inventory'])
            spread.append(step['spread'])

    inv    = np.array(inv)
    spread = np.array(spread)

    # Bin by inventory and show avg spread
    bins   = np.linspace(inv.min(), inv.max(), 20)
    labels = 0.5 * (bins[:-1] + bins[1:])
    means  = [spread[(inv >= bins[i]) & (inv < bins[i+1])].mean()
              for i in range(len(bins) - 1)]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(inv, spread, alpha=0.03, s=4, color='steelblue')
    ax.plot(labels, means, color='tomato', lw=2, label='avg spread per inventory bucket')
    ax.axvline(0, color='black', lw=0.8, ls='--')
    ax.set_xlabel('Inventory (shares)')
    ax.set_ylabel('Quoted Spread ($)')
    ax.set_title('Spread vs Inventory — does the MM adapt?')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_pnl_distribution(sac_pnls, as_pnls, rand_pnls, out="evaluation/plots/pnl_distribution.png"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.linspace(
        min(min(sac_pnls), min(as_pnls), min(rand_pnls)),
        max(max(sac_pnls), max(as_pnls), max(rand_pnls)),
        40
    )
    ax.hist(rand_pnls, bins=bins, alpha=0.5, label='Random',              color='gray')
    ax.hist(as_pnls,   bins=bins, alpha=0.6, label='Avellaneda-Stoikov',  color='steelblue')
    ax.hist(sac_pnls,  bins=bins, alpha=0.7, label='SAC (trained)',       color='tomato')
    ax.axvline(0, color='black', lw=1, ls='--')
    ax.set_xlabel('Episode PnL ($)')
    ax.set_ylabel('Count')
    ax.set_title('PnL Distribution — SAC vs A-S vs Random (100 episodes each)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='path to saved SAC model (no .zip)')
    parser.add_argument('--n',     type=int, default=100, help='episodes to evaluate')
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = SAC.load(args.model)
    env   = MarketEnv()

    print(f"\nRunning {args.n} episodes each for SAC, A-S, and random...")

    sac_histories  = [run_sac_episode(model, env) for _ in range(args.n)]
    as_histories   = [run_as_episode(seed=i)      for i in range(args.n)]
    rand_histories = [run_random_episode(env, seed=i) for i in range(args.n)]

    print("\n" + "=" * 45 + " RESULTS " + "=" * 45)
    sac_pnls,  _ = summarise(sac_histories,  "SAC (trained)")
    as_pnls,   _ = summarise(as_histories,   "Avellaneda-Stoikov baseline")
    rand_pnls, _ = summarise(rand_histories, "Random policy")

    print("\nGenerating plots...")
    plot_sample_episode(sac_histories[0])
    plot_spread_vs_inventory(sac_histories)
    plot_pnl_distribution(sac_pnls, as_pnls, rand_pnls)
    print("\nAll plots saved to evaluation/plots/")


if __name__ == '__main__':
    main()

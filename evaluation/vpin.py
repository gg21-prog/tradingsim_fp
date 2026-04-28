"""
VPIN (Volume-Synchronized Probability of Informed Trading) metric.
Easley, Lopez de Prado, O'Hara (2012).

VPIN = |buy_volume - sell_volume| / total_volume  (per bucket)

We use it as a ground-truth toxicity signal to benchmark whether the SAC
market maker implicitly detects informed flow by widening its spread.

Evaluation:
  - Compute VPIN per bucket from episode history
  - Extract MM's spread at each bucket
  - ROC AUC: can VPIN predict when MM widens spread?
  - A dumb MM (flat spread) scores ~0.5; a good adaptive MM scores >0.60
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def compute_vpin(history, bucket_size=50):
    """
    Compute VPIN from episode history.
    Returns per-bucket VPIN values and the corresponding step indices.
    """
    buy_acc = sell_acc = 0
    bucket_vpins = []
    bucket_steps = []
    bucket_start = 0

    for i, step in enumerate(history):
        buy_acc  += step.get('buy_vol', 0)
        sell_acc += step.get('sell_vol', 0)
        total = buy_acc + sell_acc

        if total >= bucket_size:
            vpin = abs(buy_acc - sell_acc) / total
            bucket_vpins.append(vpin)
            bucket_steps.append(i)
            buy_acc = sell_acc = 0
            bucket_start = i + 1

    return np.array(bucket_vpins), np.array(bucket_steps)


def spread_vs_vpin_auc(histories):
    """
    For each episode, compute VPIN buckets and correlate with MM spread.
    Returns ROC AUC of VPIN predicting above-median spread.
    """
    all_vpin = []
    all_spread_label = []

    for h in histories:
        vpins, steps = compute_vpin(h)
        if len(vpins) < 2:
            continue

        median_spread = np.median([h[i]['spread'] for i in steps])
        spreads = np.array([h[i]['spread'] for i in steps])

        label = (spreads > median_spread).astype(int)
        all_vpin.extend(vpins.tolist())
        all_spread_label.extend(label.tolist())

    if len(set(all_spread_label)) < 2:
        return 0.5

    return roc_auc_score(all_spread_label, all_vpin)


def plot_vpin_spread(history, out="evaluation/plots/vpin_spread.png"):
    """Plot VPIN and MM spread together for a single episode."""
    os.makedirs(os.path.dirname(out), exist_ok=True)

    vpins, steps = compute_vpin(history, bucket_size=30)
    all_steps = [h['step'] for h in history]
    all_spreads = [h['spread'] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    ax1.plot(all_steps, all_spreads, color='#e8759a', lw=1.2, label='MM Spread')
    ax1.set_ylabel('Spread ($)')
    ax1.set_title('Market Maker Spread vs VPIN Toxicity Signal')
    ax1.legend(fontsize=9)

    ax2.bar(steps, vpins, color='steelblue', alpha=0.7, label='VPIN (bucket)')
    ax2.axhline(np.mean(vpins), color='tomato', lw=1.5, ls='--', label=f'Mean VPIN={np.mean(vpins):.3f}')
    ax2.set_ylabel('VPIN')
    ax2.set_xlabel('Step')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def plot_roc_comparison(sac_auc, as_auc, rand_auc, out="evaluation/plots/vpin_roc.png"):
    """Bar chart comparing VPIN-detection AUC across policies."""
    os.makedirs(os.path.dirname(out), exist_ok=True)
    labels = ['Random', 'Avellaneda-Stoikov', 'SAC Phase 2']
    aucs   = [rand_auc, as_auc, sac_auc]
    colors = ['gray', 'steelblue', '#e8759a']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, aucs, color=colors, alpha=0.85, edgecolor='white')
    ax.axhline(0.5, color='black', lw=1, ls='--', label='Random baseline (0.5)')
    ax.axhline(0.6, color='tomato', lw=1, ls=':', label='Target AUC (0.6)')
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('ROC AUC (VPIN → spread widening)')
    ax.set_title('Implicit Toxicity Detection: does the MM widen spread when flow is toxic?')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")

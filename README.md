# trading-sim

**[Live demo: tradingsimfp.streamlit.app](https://tradingsimfp.streamlit.app/)**

A multi-agent trading simulation where a market maker learns to survive against informed traders using reinforcement learning.

The core problem is adverse selection: market makers earn money by quoting bid/ask spreads, but some traders have private information and systematically take the other side of good trades. The market maker cannot see who is informed. It has to figure it out from behavior.

## results

Two phases trained and evaluated over 100 episodes each.

**Phase 1 — noise traders only**

| Policy | Mean PnL | Win rate | Closing inventory |
|---|---|---|---|
| SAC (trained) | $2,512 | 100/100 | 20.9 shares |
| Avellaneda-Stoikov | $2,616 | 100/100 | 138.3 shares |
| Random | $2,109 | 99/100 | 160.6 shares |

**Phase 2 — 30% informed flow**

| Policy | Mean PnL | Win rate | Closing inventory |
|---|---|---|---|
| SAC (trained) | $3,061 | 100/100 | 20.7 shares |
| Avellaneda-Stoikov | $3,233 | 100/100 | 233.9 shares |
| Random | $2,741 | 99/100 | 292.2 shares |

VPIN ROC AUC (does the MM widen spread on toxic flow?): SAC 0.484, A-S 0.500, Random 0.507.

The AUC result is the interesting finding. SAC does not protect itself by widening spreads when flow is toxic, which is what the textbook (A-S) would prescribe. Instead it learned aggressive inventory management: closing inventory of ~21 shares vs 234 for A-S in the same environment. When informed traders drive inventory in one direction, SAC flattens its position quickly rather than quoting wider. Both strategies survive, but the RL agent found its own path without being told how.

## what's been built

**Phase 1** — SAC market maker vs noise traders.

**order book** (`environment/exchange.py`): limit order matching with price priority. Market maker posts quotes, incoming orders fill against the best available price.

**price process** (`environment/price_generator.py`): geometric Brownian motion, sigma=0.002 per step, 390 steps per episode (one trading day).

**noise traders** (`agents/noise_trader.py`): Poisson arrivals at lambda=5 orders per step, random side and size.

**gym environment** (`environment/market_env.py`): 6-dimensional observation (inventory, spread, price change z-score, order flow imbalance, trade flow, time remaining), action space of bid/ask offsets, reward is change in mark-to-market minus inventory penalty.

**Phase 2** — SAC market maker vs noise traders + informed trader.

**informed trader** (`agents/informed_trader.py`): looks ahead `signal_horizon=20` steps, trades directionally when expected price move exceeds a threshold. Inspired by Glosten-Milgrom (1985): the informed agent knows the true asset value and trades against mispriced quotes.

**Phase 2 environment** (`environment/market_env_phase2.py`): 30% of order flow is informed. Extended 8-dimensional observation adds rolling order flow imbalance and a VPIN-like toxicity estimate over a 5-step window.

**VPIN evaluation** (`evaluation/vpin.py`): computes Volume-synchronized Probability of Informed Trading per bucket, measures ROC AUC of whether the MM widens its spread when VPIN is elevated.

**dashboard** (`app.py`): Streamlit app with episode replay, spread vs inventory analysis, policy comparison histograms, and a Phase 2 toxicity tab with live VPIN visualization and benchmark.

## what comes next

Phase 3 makes both agents adaptive via self-play (RLlib + PettingZoo). The informed trader learns to hide — order splitting, mimicking noise volume, timing trades. The market maker adapts in response. The goal is observing whether an equilibrium emerges and what it looks like.

## running it

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt scikit-learn matplotlib streamlit tqdm rich

# train
python3 training/train_phase1.py
python3 training/train_phase2.py

# evaluate
python3 evaluation/evaluate.py --model models/phase1/<run-id>/best/best_model
python3 evaluation/evaluate_phase2.py --model models/phase2/<run-id>/best/best_model

# dashboard
streamlit run app.py
```

## stack

- PyTorch via Stable-Baselines3 (SAC)
- Weights and Biases for experiment tracking
- Gymnasium for the RL environment interface
- Streamlit dashboard

## references

Market mechanics from [Avellaneda & Stoikov (2008)](https://arxiv.org/abs/2003.01820). Price follows GBM, noise traders arrive via Poisson process, fill probability decays exponentially with spread width: `P(fill) = exp(-κδ)`.

Informed trader design from [Glosten & Milgrom (1985)](https://www.sciencedirect.com/science/article/pii/0304405X85900443): informed agents know the true asset value and trade against mispriced quotes.

VPIN metric from [Easley, Lopez de Prado, O'Hara (2012)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1695596): volume-synchronized probability of informed trading.

Overall RL framing from [Spooner et al. (2018)](https://arxiv.org/abs/1804.04216) and [Spooner & Savani (2020)](https://dl.acm.org/doi/10.5555/3491440.3492073). Environment structure references [JJJerome/mbt_gym](https://github.com/JJJerome/mbt_gym).

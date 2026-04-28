# Project Plan: Multi-Agent Trading Simulation (Learning-Focused)

## 1. The Concept

### The Problem
Market makers provide liquidity by continuously offering to buy and sell financial instruments. Their profit comes from the bid-ask spread—the difference between what they pay to buy and what they receive when selling.

However, market makers face a challenge: **toxic flow**. Some traders possess private information about future price movements (like advance knowledge of earnings reports). When these informed traders execute against market maker quotes, the market maker systematically loses money. This is called adverse selection.

The market maker must balance two competing objectives:
- Quote tight spreads to attract trading volume
- Quote wide spreads to protect against informed traders

The difficulty is that the market maker cannot directly observe whether any particular trader is informed.

### The Simulation
We build a simulated market environment with three types of participants:

**Market Maker**
- Provides continuous bid and ask quotes
- Receives all incoming orders
- Must learn to infer trader type from order characteristics
- Objective: Maximize profit from spreads while minimizing adverse selection losses

**Informed Trader**
- Receives private signals about future price direction
- Executes trades to profit from this information
- Objective: Maximize trading profit while avoiding detection

**Noise Traders**
- Trade randomly without information
- Provide baseline liquidity
- Represent retail investors

### The Learning Dynamic
Both the market maker and informed trader are reinforcement learning agents that improve through experience:

1. Initially, the market maker quotes naively (same spread for all traders)
2. The informed trader exploits predictable spreads
3. The market maker observes losses and learns to identify patterns associated with informed trading
4. The informed trader adapts by splitting orders, varying timing, or mimicking noise trader behavior
5. This co-evolution continues until reaching a stable equilibrium

### What We Measure
- Equilibrium spread width
- Market maker profitability
- Informed trader detection accuracy
- Comparison with classical finance benchmarks (VPIN metric)

---

## 2. Phased Development Approach

We follow a progressive complexity strategy: master the finance fundamentals first, then layer in RL.

### Phase 0 — Finance Primer (2-3 days, before any code)

**Goal:** Understand the problem domain before adding RL complexity. Do not skip this.

#### Study Resources
1. **Avellaneda & Stoikov (2008)** — "High-frequency trading in a limit order book"
   - Focus: Section 2 (reservation price), Section 3 (optimal spread formula)
   - Key formula: `spread* = γσ²(T-t) + (2/γ)ln(1 + γ/κ)`
2. **Spooner & Savani (2020)** — "Robust Market-Making via Adversarial Reinforcement Learning" (IJCAI)
   - This is the direct precedent — read the full paper
3. **Lilian Weng's SAC blog** — entropy-regularized RL intuition
4. **Easley et al. (2012)** — skim for VPIN formula and intuition

#### Concepts to nail before Phase 1
- **Inventory risk:** You quote bid=99, ask=101. A buy hits your ask → you're short 1 share at 101. Price drops to 95 → you lost $6. That's inventory risk.
- **Adverse selection:** Informed trader *knows* price will drop. Hits your ask at 101. Price falls to 95. You were selected adversely.
- **A-S formula:** Optimal spread widens with (a) volatility σ², (b) inventory size, (c) time remaining.
- **VPIN:** `VPIN = |V_buy - V_sell| / V_total` per volume bucket. High VPIN → likely informed flow.

#### Deliverable
- Implement `baselines/avellaneda_stoikov.py` — closed-form market maker, no RL
- Verify it earns positive PnL against random noise traders on a GBM price path
- You should be able to explain every variable in the A-S formula from memory

#### Questions to answer before moving on
- What is the A-S reservation price? Why does the optimal spread widen with inventory?

---

### Phase 1 — Single Agent Validation (Weeks 1-2)

**Framework: Stable-Baselines3** (not RLlib — easier to debug for single-agent, same SAC quality)

**Goal:** Prove the RL pipeline works before adding adversarial complexity.

#### Study Resources (before coding)
- SB3 docs: Custom Gym environment tutorial
- SAC paper (Haarnoja 2018) Section 3 — understand the entropy bonus and why it prevents policy collapse

#### Sub-tasks

**1.1 — Order Book (`environment/exchange.py`)**
```python
class Exchange:
    def __init__(self): self.bids = {}; self.asks = {}  # price → size
    def place_quote(self, side, price, size): ...
    def match_orders(self) -> list[Trade]: ...  # price-time priority
    def cancel_stale_quotes(self): ...
```
Manual test: place bid=99, ask=101, send buy order at 101 → confirm trade executes.

**1.2 — GBM Price Generator (`environment/price_generator.py`)**
```python
# dS = μS dt + σS dW
# Parameters: σ=0.002 per step (≈0.2%/min), μ=0
# 390 steps = 1 trading day
```

**1.3 — Noise Traders (`agents/noise_trader.py`)**
- Poisson arrivals: λ=5 orders/step
- Random side (50/50), random size (uniform 1-10 shares)
- No learning, no state

**1.4 — Gym Wrapper (`environment/market_env.py`)**
```python
class MarketEnv(gym.Env):
    observation_space = Box(shape=(8,))
    action_space = Box(low=0.0, high=0.01, shape=(2,))  # bid_offset, ask_offset
```
**State normalization is critical** — all values must be in [-1, 1] or [0, 1]:
```python
state = {
    'inventory':          inventory / max_inventory,         # max=100
    'spread':             spread / max_spread,               # max=0.05
    'mid_price_change':   price_change / (3 * sigma),        # z-score
    'bid_ask_imbalance':  imbalance,                         # already [-1, 1]
    'trade_flow':         flow / max_expected_volume,        # VPIN-like
    'time_to_close':      steps_remaining / total_steps,
}
```

**1.5 — Sanity check before training**
- Random policy → should lose money (confirms reward is working)
- A-S baseline → should earn money (confirms reward sign is correct)

**1.6 — Train SAC**
```python
from stable_baselines3 import SAC
model = SAC("MlpPolicy", env, verbose=1,
            learning_rate=3e-4, ent_coef="auto", batch_size=256)
model.learn(total_timesteps=200_000)
```
Log per episode: PnL, spread width, closing inventory, policy entropy.

**1.7 — Ablation: remove inventory penalty**
- Remove `- 0.1 * abs(inventory)` from reward
- Expected: MM accumulates huge inventory → end-of-day blowup
- Confirms the penalty is load-bearing

**1.8 — Validation**
- 3-line plot: SAC PnL vs. random baseline vs. A-S baseline
- Target: SAC ≥ A-S after 200k steps

#### Reward Function
```python
reward = (
    trading_pnl
    - 0.1 * abs(inventory)      # inventory risk
)
```

#### Common Pitfalls
| Symptom | Cause | Fix |
|---------|-------|-----|
| Reward is NaN | Price exploded (σ too high) | Cap σ=0.002 per step |
| Agent never quotes | Reward too sparse | Add small per-step reward for being quoted |
| Spread collapses to 0 | No lower bound on offsets | Clip `action_space.low = 0.0` |
| Policy entropy → 0 immediately | `ent_coef` fixed too low | Use `ent_coef="auto"` |
| PnL never improves | State not normalized | Check all state values land in [-2, 2] |

#### Success Criteria
- Episode PnL > 0 in 60%+ of episodes after 200k steps
- Spread varies with inventory (not constant)

#### Questions to answer before moving on
- What does the entropy term in SAC do? What breaks if you remove it?

---

### Phase 2 — Add Static Informed Trader (Weeks 3-4)

**Goal:** Introduce adverse selection and validate toxicity detection before making both agents adaptive.

#### Study Resources
- Glosten-Milgrom (1985) model — intuition on adverse selection math (3 pages)
- Kyle (1985) model — price impact, what lambda measures

#### Sub-tasks

**2.1 — Signal Generator**
```python
# 65% predictive of next N-step price direction
# signal_strength ∈ [-1, 1]: sign = direction, magnitude = confidence
```

**2.2 — Static Informed Trader (`agents/informed_trader.py` — heuristic)**
```python
# if signal > 0.3:  buy 20 shares spread over 3 steps
# if signal < -0.3: sell 20 shares spread over 3 steps
# otherwise:        behave like a noise trader
```

**2.3 — Extend MM state** with order history (last 10 trades):
```python
# Append to observation:
'recent_trade_sizes':       [float] * 10,  # normalized
'recent_trade_directions':  [float] * 10,  # +1 buy, -1 sell
'recent_mm_pnl_per_trade':  [float] * 10   # was each profitable?
```

**2.4 — Retrain MM** with new state space — reset weights entirely, don't fine-tune.

**2.5 — Post-hoc labeling** — after each episode, label each trade as "informed" or "noise":
```python
# MM does NOT see this label during training — used for evaluation only
```

**2.6 — Validation**
```python
# Does MM spread widening predict informed flow?
roc_auc_score(is_informed_label, spread_increase_after_trade)
# Target: AUC > 0.60
```

**2.7 — Implement VPIN** (`evaluation/compare_vpin.py`) and compare to MM's implicit detection

#### Reward Function
```python
reward = (
    trading_pnl
    - 0.1 * abs(inventory)
    - 0.5 * adverse_selection_loss   # losses specifically to informed trades
)
```

#### Common Pitfalls
| Symptom | Cause | Fix |
|---------|-------|-----|
| MM ignores informed trader | Signal too weak | Calibrate to 65% accuracy |
| MM stops quoting | Informed losses dominate reward | Scale adverse_selection weight down to 0.3 |
| MM worsens vs Phase 1 | New state dims not normalized | Normalize order history to [-1, 1] |
| AUC = 0.5 | MM state doesn't capture patterns | Ensure order history is in state |

#### Success Criteria
- ROC AUC > 0.60 for implicit toxicity detection
- MM widens spread by ≥20% after 3+ consecutive informed trades

#### Questions to answer before moving on
- Why can't the MM just ask if a trader is informed?
- What is VPIN actually measuring?

---

### Phase 3 — Adversarial Training (Weeks 5-8)

**Framework: Migrate to RLlib** (PettingZoo for multi-agent environment, Ray for self-play)

**Goal:** Both agents learn simultaneously through self-play.

#### Study Resources
- Ray RLlib multi-agent docs: `MultiAgentEnv` interface
- PettingZoo: AEC (Agent Environment Cycle) API
- Lilian Weng's MARL blog post
- PSRO (Policy Space Response Oracles) — self-play with policy pools

#### Sub-tasks

**3.1 — Convert to PettingZoo MultiAgent env (`environment/multi_agent_env.py`)**
```python
# Turn order: MM quotes → traders submit orders → exchange matches → rewards assigned
```

**3.2 — DDPG for Informed Trader**
```python
# Action: (order_size ∈ [0, 50], delay ∈ [0, 5] steps) — both continuous
# State: signal_strength, position, recent_spread, time_remaining
```
Why DDPG not SAC: informed trader wants *deterministic* exploitation — no need for entropy exploration once signal is clear.

**3.3 — Self-Play Loop (`training/self_play.py`)**
```python
opponent_pool = []

for iteration in range(2000):
    # Train MM against sampled opponents
    for ep in range(50):
        opp = random.choice(opponent_pool) if opponent_pool else static_informed_trader
        run_episode(mm_policy, opp)
    mm_policy.update()

    # Train informed trader against latest MM
    for ep in range(50):
        run_episode(mm_policy, informed_trader_policy)
    informed_trader_policy.update()

    # Pool management: add every 200 iters, cap at 20
    if iteration % 200 == 0:
        opponent_pool.append(copy(informed_trader_policy))
        if len(opponent_pool) > 20:
            opponent_pool.pop(0)

    log(avg_spread[-100:], informed_trader_win_rate[-100:])
```

**3.4 — Convergence check**
- "Converged" = avg spread stable within ±5% for 500 consecutive iterations

**3.5 — Emergence analysis** (the interesting part)
- Did informed trader learn order splitting? (plot order size over time per episode)
- Did informed trader learn to mimic noise traders? (compare timing distributions)
- Did MM learn a spread pattern correlated with true informed flow?

**3.6 — Baseline comparison (`evaluation/compare_baselines.py`)**
- Compare equilibrium vs. fixed A-S, VPIN-triggered rule, random MM
- Plot: PnL distribution across 500 episodes for each

**3.7 — Ablation: remove detection penalty from informed trader**
- Expected: informed trader trades aggressively → spread explodes

#### Common Pitfalls
| Symptom | Cause | Fix |
|---------|-------|-----|
| Training diverges | Alternating training unstable | Reduce LR 10x when switching agents |
| Informed trader spams orders | No rate limit | Cap at 5 orders/step |
| Spread → 0 or → ∞ | Reward not balanced | Clip rewards to [-10, 10] per step |
| Policies cycle forever | Opponent pool too small | Increase to 30; sample from last 10 only |
| MM stops quoting | Informed losses dominate | Rebalance: `0.5*pnl - 0.1*inv - 0.3*adv_sel` |

#### Success Criteria
- Neither agent dominates >70% of episodes (both win 30-70%)
- Equilibrium spread measurably wider than noise-only baseline

#### Questions to answer before moving on
- What does Nash equilibrium mean in this context?
- Did your agents find one? How do you know?

---

### Phase 4 — Extensions (Weeks 9-12)

**Pick one path:**

**Option A: Options Chain**
- Add European options with Black-Scholes Greeks (delta, gamma, vega)
- MM must delta-hedge in real time
- New state: option delta, portfolio delta, gamma, time-to-expiry
- New challenge: informed trader can trade options AND stock

**Option B: Multiple Market Makers**
- Add 2nd MM agent (also SAC)
- Both compete for order flow → tighter spreads
- New dynamic: race-to-bottom vs. emergent collusion
- Use independent learners

**Option C: Real Data Backtest**
- `yfinance` SPY 1-min bars (2020-2022)
- Replace GBM with historical mid-price feed
- Train on 2020-2021, test on 2022 (time split only — never random)

---

## 3. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SIMULATION LOOP                           │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Market     │      │   Informed   │      │   Noise      │
│   Maker      │      │   Trader     │      │   Traders    │
└──────┬───────┘      └──────┬───────┘      └──────┬───────┘
       │                     │                     │
       └─────────────────────┼─────────────────────┘
                             ▼
                      ┌──────────────┐
                      │  Order Book  │
                      │   (Exchange) │
                      └──────┬───────┘
                             │
                      ┌──────▼──────┐
                      │   Trades    │
                      │  Executed   │
                      └──────┬──────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
   ┌────────┐          ┌────────┐          ┌────────┐
   │  MM    │          │Informed│          │ Noise  │
   │Reward  │          │Reward  │          │Reward  │
   └────────┘          └────────┘          └────────┘
```

---

## 4. Framework & Algorithm Summary

| Phase | Framework | MM Algorithm | Informed Algo |
|-------|-----------|--------------|---------------|
| 0 | — | Avellaneda-Stoikov (closed-form) | — |
| 1 | Stable-Baselines3 | SAC | Fixed noise traders |
| 2 | Stable-Baselines3 | SAC (retrained) | Static heuristic |
| 3 | RLlib + PettingZoo | SAC | DDPG |
| 4 | RLlib | SAC (+ Greeks for options) | DDPG |

**Why SAC for MM:** Continuous action space, entropy bonus prevents quote collapse, good sample efficiency.
**Why DDPG for informed trader:** Deterministic exploitation is desired once signal is clear.
**Why SB3 first:** Dead-simple API, single-threaded debugging. Migrate to RLlib only when self-play pool requires it.

---

## 5. File Structure

```
trading-sim/
├── environment/
│   ├── exchange.py          # Order book, matching engine
│   ├── market_env.py        # SB3 Gym env (Phase 1-2)
│   ├── multi_agent_env.py   # PettingZoo env (Phase 3)
│   └── price_generator.py   # GBM + historical data feed
├── agents/
│   ├── market_maker.py      # SAC wrapper + state/action spec
│   ├── informed_trader.py   # DDPG wrapper + static heuristic
│   └── noise_trader.py      # Fixed random policy
├── baselines/
│   └── avellaneda_stoikov.py  # Phase 0 closed-form baseline
├── training/
│   ├── train_phase1.py      # SB3 single agent
│   ├── train_phase2.py      # SB3 + static informed trader
│   ├── self_play.py         # Phase 3 self-play loop
│   └── configs/             # Hyperparameter YAML files
├── evaluation/
│   ├── metrics.py           # PnL, spread, AUC, VPIN
│   ├── visualize.py         # Training curves, equilibrium plots
│   └── compare_baselines.py
└── tests/
    ├── test_exchange.py     # Unit tests for order book
    └── test_env.py          # Gym interface sanity checks
```

---

## 6. Related Work

- **Avellaneda & Stoikov (2008):** Foundational market making model
- **Spooner & Savani (2020):** Adversarial RL for market making (IJCAI) — direct precedent
- **Easley et al. (2012):** VPIN toxicity metric
- **ABIDES** (https://github.com/abides-sim/abides): Agent-based market simulation from JPMorgan AI Research
- **RLlib** (https://github.com/ray-project/ray): Multi-agent self-play framework

---

## 7. Verification

**After Phase 0:**
```bash
python baselines/avellaneda_stoikov.py  # Should print positive PnL
```

**After Phase 1:**
```bash
python training/train_phase1.py
# Training curve: PnL > 0 by ~100k steps
# Spread varies with inventory (not constant)
```

**After Phase 2:**
```bash
python evaluation/metrics.py --phase 2
# ROC AUC > 0.60 printed
# Spread timeline widens after informed trade clusters
```

**After Phase 3:**
```bash
python training/self_play.py
python evaluation/compare_baselines.py
# Equilibrium spread stabilizes
# Both agents non-trivially competitive
# Informed trader action log shows order splitting
```

---

## 8. What You Should Be Able to Explain After Each Phase

| Phase | Questions to answer without notes |
|-------|----------------------------------|
| 0 | What is the A-S reservation price? Why does spread widen with inventory? |
| 1 | What is the entropy term in SAC doing? What breaks if you remove it? |
| 2 | Why can't the MM just ask if a trader is informed? What is VPIN measuring? |
| 3 | What is Nash equilibrium in this context? Did your agents find one? How do you know? |

---

## 9. Success Criteria

| Phase | Criteria |
|-------|----------|
| 0 | A-S baseline earns positive PnL; can explain the formula from memory |
| 1 | SAC earns PnL > 0 in 60%+ of episodes; spread is adaptive |
| 2 | ROC AUC > 0.60; spread widens ≥20% after 3+ consecutive informed trades |
| 3 | Both agents win 30-70% of episodes; equilibrium spread stable within ±5% for 500 iters |
| 4 | Chosen extension validates learned behavior transfers to new setting |

---

## 10. Timeline

| Phase | Weeks | Focus | Key Deliverable |
|-------|-------|-------|-----------------|
| 0 | Pre-week 1 | Finance primer | A-S baseline + deep understanding |
| 1 | 1-2 | Single agent | Working MM against noise traders |
| 2 | 3-4 | Static adversary | MM learns toxicity detection |
| 3 | 5-8 | Adversarial training | Both agents adaptive, equilibrium emerges |
| 4 | 9-12 | Extension | Options / multi-MM / real data |

**Total:** 12 weeks + 2-3 day primer buffer

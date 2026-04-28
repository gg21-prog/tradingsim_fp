import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import glob
from collections import deque

from environment.market_env import MarketEnv
from environment.market_env_phase2 import MarketEnvPhase2
from baselines.avellaneda_stoikov import run_episode as run_as_episode, AvellanedaStoikov
from evaluation.vpin import compute_vpin, spread_vs_vpin_auc

# ── colours ───────────────────────────────────────────────────────────────────
C = {
    "bg":      "#ffffff",
    "surface": "#fdf0f5",
    "border":  "#f0c4d8",
    "pink1":   "#e8759a",   # strong — buttons, active tabs, chart lines
    "pink2":   "#f4a7be",   # mid — fills, secondary
    "pink3":   "#fce4ef",   # light — card bg
    "text":    "#2d1a24",   # near-black on white — always readable
    "sub":     "#7a4a5e",   # subtext — dark enough on white/pink3
    "grid":    "#f5dde8",
}

st.set_page_config(page_title="trading sim", layout="wide", page_icon="🌸")
st.markdown(f"""
<style>
* {{ box-sizing: border-box; }}
html, body, .stApp {{ background:{C["bg"]}; color:{C["text"]}; font-family:'Inter',sans-serif; }}
header[data-testid="stHeader"] {{ display:none !important; }}
#MainMenu, footer {{ display:none !important; }}
section[data-testid="stSidebar"] {{ background:{C["surface"]}; border-right:1px solid {C["border"]}; }}
section[data-testid="stSidebar"] * {{ color:{C["text"]} !important; }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap:6px; background:transparent; border-bottom:2px solid {C["border"]}; padding-bottom:0; }}
.stTabs [data-baseweb="tab"] {{ background:{C["pink3"]}; border-radius:6px 6px 0 0; padding:5px 16px;
    color:{C["sub"]} !important; font-weight:600; font-size:13px; border:1px solid {C["border"]}; border-bottom:none; }}
.stTabs [aria-selected="true"] {{ background:{C["pink1"]} !important; color:#fff !important; border-color:{C["pink1"]} !important; }}

/* metrics */
div[data-testid="metric-container"] {{
    background:{C["pink3"]}; border:1px solid {C["border"]}; border-radius:8px; padding:10px 14px; }}
div[data-testid="metric-container"] label {{ color:{C["sub"]} !important; font-size:11px !important; font-weight:600; }}
div[data-testid="metric-container"] [data-testid="metric-value"] {{ color:{C["text"]} !important; font-size:20px !important; font-weight:700; }}

/* buttons */
.stButton>button {{ background:{C["pink1"]}; color:#fff; border:none; border-radius:6px;
    padding:6px 18px; font-weight:600; font-size:13px; }}
.stButton>button:hover {{ background:{C["sub"]}; color:#fff; }}
.stButton>button:disabled {{ background:{C["pink3"]}; color:{C["border"]}; }}

/* inputs — light bg, dark text */
.stSelectbox label, .stSlider label, .stNumberInput label {{ color:{C["sub"]} !important; font-size:12px !important; font-weight:600; }}
.stSelectbox [data-baseweb="select"] > div,
.stSelectbox [data-baseweb="select"] div[class*="ValueContainer"],
.stSelectbox [data-baseweb="select"] div[class*="singleValue"],
.stSelectbox [data-baseweb="select"] input,
[data-baseweb="select"] span {{ background:{C["bg"]} !important; color:{C["text"]} !important; }}
[data-baseweb="select"] > div {{ background:{C["bg"]} !important; border-color:{C["border"]} !important; }}
input[type="number"], .stNumberInput input {{ background:{C["bg"]} !important; color:{C["text"]} !important; border-color:{C["border"]} !important; }}
[data-baseweb="popover"] [role="option"] {{ background:{C["bg"]} !important; color:{C["text"]} !important; }}
[data-baseweb="popover"] [role="option"]:hover {{ background:{C["pink3"]} !important; }}

/* tighten vertical spacing */
.block-container {{ padding-top:1.5rem; padding-bottom:1rem; }}
div[data-testid="stVerticalBlock"] > div {{ gap:0.5rem; }}

h1 {{ color:{C["pink1"]}; font-size:26px; font-weight:800; margin-bottom:2px; }}
h4 {{ color:{C["text"]}; font-size:14px; font-weight:700; margin:0 0 4px 0; }}
p, .stMarkdown p {{ color:{C["text"]}; font-size:13px; line-height:1.5; }}
</style>
""", unsafe_allow_html=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def ax_style(ax, title=""):
    ax.set_facecolor(C["surface"])
    ax.tick_params(colors=C["text"], labelsize=8)
    for s in ax.spines.values(): s.set_color(C["grid"])
    ax.grid(color=C["grid"], lw=0.6)
    if title: ax.set_title(title, color=C["text"], fontsize=10, fontweight="700", pad=8)

def note(text):
    st.markdown(f'<p style="color:{C["sub"]};font-size:12px;margin:0 0 10px 0">{text}</p>',
                unsafe_allow_html=True)

@st.cache_resource
def load_model(path):
    from stable_baselines3 import SAC
    return SAC.load(path)

def find_models(phase=1):
    p = glob.glob(f"models/phase{phase}/**/best_model.zip", recursive=True)
    p += glob.glob(f"models/phase{phase}/**/*final*.zip", recursive=True)
    return [x.replace(".zip","") for x in sorted(p)]

def run_sac(model, seed=None):
    env = MarketEnv()
    obs, _ = env.reset(seed=seed)
    done, h = False, []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        h.append({"mid": env.prev_mid, "bid": env.last_bid, "ask": env.last_ask,
                   "spread": info["spread"], "inventory": info["inventory"],
                   "pnl": info["portfolio_value"], "trades": info["num_trades"]})
    return h

def run_as(seed=None):
    h = run_as_episode(seed=seed)
    return [{"mid": s["mid"], "bid": s["bid"], "ask": s["ask"],
             "spread": s["spread"], "inventory": s["inventory"],
             "pnl": s["portfolio_value"], "trades": s["num_trades"]} for s in h]

def run_rand(seed=None):
    env = MarketEnv()
    obs, _ = env.reset(seed=seed)
    done, h = False, []
    while not done:
        obs, _, term, trunc, info = env.step(env.action_space.sample())
        done = term or trunc
        h.append({"mid": env.prev_mid, "bid": env.last_bid, "ask": env.last_ask,
                   "spread": info["spread"], "inventory": info["inventory"],
                   "pnl": info["portfolio_value"], "trades": info["num_trades"]})
    return h


@st.cache_resource
def load_model_p2(path):
    from stable_baselines3 import SAC
    return SAC.load(path)

def run_sac_p2(model, seed=None):
    env = MarketEnvPhase2()
    obs, _ = env.reset(seed=seed)
    done, h = False, []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        h.append({
            "mid": info["mid_price"], "bid": env.last_bid, "ask": env.last_ask,
            "spread": info["spread"], "inventory": info["inventory"],
            "pnl": info["portfolio_value"], "trades": info["num_trades"],
            "buy_vol": info.get("buy_vol", 0), "sell_vol": info.get("sell_vol", 0),
        })
    return h

def run_as_p2(seed=None):
    env = MarketEnvPhase2()
    obs, _ = env.reset(seed=seed)
    agent = AvellanedaStoikov(sigma=0.002, episode_length=390)
    done, h = False, []
    while not done:
        steps_rem = env.episode_length - env.step_count
        bid_off, ask_off = agent.get_quotes(env.mid, env.inventory, steps_rem)
        action = np.array([bid_off, ask_off], dtype=np.float32)
        obs, _, term, trunc, info = env.step(action)
        done = term or trunc
        h.append({
            "mid": info["mid_price"], "bid": env.last_bid, "ask": env.last_ask,
            "spread": info["spread"], "inventory": info["inventory"],
            "pnl": info["portfolio_value"], "trades": info["num_trades"],
            "buy_vol": info.get("buy_vol", 0), "sell_vol": info.get("sell_vol", 0),
        })
    return h

def run_rand_p2(seed=None):
    env = MarketEnvPhase2()
    obs, _ = env.reset(seed=seed)
    done, h = False, []
    while not done:
        obs, _, term, trunc, info = env.step(env.action_space.sample())
        done = term or trunc
        h.append({
            "mid": info["mid_price"], "bid": env.last_bid, "ask": env.last_ask,
            "spread": info["spread"], "inventory": info["inventory"],
            "pnl": info["portfolio_value"], "trades": info["num_trades"],
            "buy_vol": info.get("buy_vol", 0), "sell_vol": info.get("sell_vol", 0),
        })
    return h

# ── auto-load models ──────────────────────────────────────────────────────────
DEFAULT_P1 = "models/phase1/i83gc5gt/best/best_model"
if "model" not in st.session_state and os.path.exists(DEFAULT_P1 + ".zip"):
    st.session_state["model"] = load_model(DEFAULT_P1)

DEFAULT_P2 = "models/phase2/vp09ao3p/best/best_model"
if "model_p2" not in st.session_state and os.path.exists(DEFAULT_P2 + ".zip"):
    st.session_state["model_p2"] = load_model_p2(DEFAULT_P2)

model    = st.session_state.get("model", None)
model_p2 = st.session_state.get("model_p2", None)

# ── header ────────────────────────────────────────────────────────────────────
st.title("trading sim")
st.markdown(f'<p style="color:{C["sub"]};font-size:13px;margin-top:-8px">'
            'SAC market maker · adverse selection detection · Avellaneda-Stoikov benchmark</p>',
            unsafe_allow_html=True)

st.markdown(f'<hr style="border:none;border-top:1px solid {C["border"]};margin:8px 0 16px 0">',
            unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f'<p style="font-size:15px;font-weight:800;color:{C["pink1"]};margin-bottom:12px">controls</p>',
                unsafe_allow_html=True)

    paths = find_models()
    if paths:
        sel = st.selectbox("checkpoint", paths,
                           index=paths.index(DEFAULT_P1) if DEFAULT_P1 in paths else 0)
        if st.button("reload"):
            st.session_state["model"] = load_model(sel)
            model = st.session_state["model"]
            st.success("loaded")

    st.markdown("---")
    paths_p2 = find_models(phase=2)
    if paths_p2:
        st.markdown(f'<p style="font-size:12px;font-weight:700;color:{C["sub"]}">phase 2 checkpoint</p>',
                    unsafe_allow_html=True)
        sel_p2 = st.selectbox("p2 model", paths_p2,
                              index=paths_p2.index(DEFAULT_P2) if DEFAULT_P2 in paths_p2 else 0,
                              key="sel_p2")
        if st.button("load p2"):
            st.session_state["model_p2"] = load_model_p2(sel_p2)
            model_p2 = st.session_state["model_p2"]
            st.success("p2 loaded")
    else:
        pass

    st.markdown("---")
    n_ep = st.slider("comparison episodes", 20, 150, 50)
    seed = st.number_input("seed", value=42, step=1)

    st.markdown("---")
    st.markdown(f"""<div style="font-size:11px;color:{C["sub"]};line-height:1.8">
    <b>episode replay</b> — watch one trading day<br>
    <b>spread behaviour</b> — did agent learn inventory risk?<br>
    <b>policy comparison</b> — SAC vs A-S formula vs random<br>
    <b>phase 2 — informed flow</b> — adverse selection detection<br>
    <b>results</b> — evaluation across 100 episodes
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    p1_dot = f'<span style="color:{C["pink1"]}">●</span>' if model else f'<span style="color:{C["border"]}">○</span>'
    p2_dot = f'<span style="color:{C["pink1"]}">●</span>' if model_p2 else f'<span style="color:{C["border"]}">○</span>'
    st.markdown(f'<div style="font-size:11px;color:{C["sub"]};line-height:2">'
                f'{p1_dot} phase 1 model<br>{p2_dot} phase 2 model</div>',
                unsafe_allow_html=True)

# ── tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "📈  episode replay",
    "📊  spread behaviour",
    "⚖️  policy comparison",
    "🧠  phase 2 — informed flow",
    "📋  results",
])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with t1:
    note("one simulated trading day (390 steps). top: mid price with MM bid/ask quotes — the pink band is the spread, "
         "that's where profit comes from. bottom: inventory the MM is holding. right side: cumulative PnL.")

    b1, b2, b3 = st.columns(3)
    r_sac  = b1.button("run SAC",                disabled=model is None)
    r_as   = b2.button("run Avellaneda-Stoikov")
    r_rand = b3.button("run random")

    if   r_sac and model: ep, lbl, clr = run_sac(model, int(seed)),  "SAC (trained)",         C["pink1"]
    elif r_as:            ep, lbl, clr = run_as(int(seed)),           "Avellaneda-Stoikov",    C["pink2"]
    elif r_rand:          ep, lbl, clr = run_rand(int(seed)),         "random",                "#c9aab8"
    else:                 ep = None

    if ep:
        steps = list(range(len(ep)))
        mid, bid, ask = [s["mid"] for s in ep], [s["bid"] for s in ep], [s["ask"] for s in ep]
        inv, pnl      = [s["inventory"] for s in ep], [s["pnl"] for s in ep]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("final PnL",        f"${pnl[-1]:,.0f}")
        m2.metric("closing inventory", f"{inv[-1]} shares")
        m3.metric("avg spread",        f"${np.mean([s['spread'] for s in ep]):.4f}")
        m4.metric("total trades",      f"{sum(s['trades'] for s in ep)}")

        fig = plt.figure(figsize=(13, 6), facecolor=C["bg"])
        gs  = fig.add_gridspec(2, 2, width_ratios=[2.5, 1], hspace=0.35, wspace=0.3,
                               left=0.06, right=0.97, top=0.93, bottom=0.09)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax3 = fig.add_subplot(gs[:, 1])

        ax_style(ax1, f"{lbl} — price + quotes")
        ax1.plot(steps, mid, color=C["text"], lw=1.2, label="mid")
        ax1.plot(steps, bid, color=C["pink2"], lw=0.8, label="bid")
        ax1.plot(steps, ask, color=clr,        lw=0.8, label="ask")
        ax1.fill_between(steps, bid, ask, alpha=0.18, color=C["pink2"])
        ax1.set_ylabel("price ($)", color=C["sub"], fontsize=8)
        ax1.legend(fontsize=8, facecolor=C["bg"], edgecolor=C["grid"])

        ax_style(ax2)
        ax2.fill_between(steps, inv, 0, where=[v>=0 for v in inv], color=C["pink1"], alpha=0.6, label="long")
        ax2.fill_between(steps, inv, 0, where=[v<0  for v in inv], color=C["sub"],   alpha=0.6, label="short")
        ax2.axhline(0, color=C["text"], lw=0.6, ls="--", alpha=0.4)
        ax2.set_ylabel("inventory", color=C["sub"], fontsize=8)
        ax2.set_xlabel("step", color=C["sub"], fontsize=8)
        ax2.legend(fontsize=8, facecolor=C["bg"], edgecolor=C["grid"])

        ax_style(ax3, "cumulative PnL")
        ax3.plot(steps, pnl, color=clr, lw=1.4)
        ax3.fill_between(steps, pnl, 0, where=[p>=0 for p in pnl], color=C["pink2"], alpha=0.25)
        ax3.fill_between(steps, pnl, 0, where=[p<0  for p in pnl], color=C["sub"],   alpha=0.25)
        ax3.axhline(0, color=C["text"], lw=0.6, ls="--", alpha=0.4)
        ax3.set_ylabel("PnL ($)", color=C["sub"], fontsize=8)
        ax3.set_xlabel("step",    color=C["sub"], fontsize=8)

        st.pyplot(fig)
        plt.close()

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with t2:
    note("does the agent widen its spread when holding large inventory? "
         "a real market maker should — big positions = big risk. "
         "if the pink line tilts up at the extremes, the agent learned this without being told.")

    agents = {"Avellaneda-Stoikov": None}
    if model: agents = {"SAC (trained)": model, **agents}

    col1, col2 = st.columns([2, 1])
    chosen = col1.selectbox("agent", list(agents.keys()))
    n_sp   = col2.slider("episodes", 10, 80, 25, key="sp")

    if st.button("analyse"):
        with st.spinner("running..."):
            inv_all, spr_all = [], []
            m = agents[chosen]
            for i in range(n_sp):
                h = run_sac(m, i) if m else run_as(i)
                for s in h:
                    inv_all.append(s["inventory"])
                    spr_all.append(s["spread"])

        inv, spr = np.array(inv_all), np.array(spr_all)
        bins     = np.linspace(inv.min(), inv.max(), 25)
        mid_b    = 0.5 * (bins[:-1] + bins[1:])
        avgs     = [spr[(inv>=bins[i])&(inv<bins[i+1])].mean()
                    if ((inv>=bins[i])&(inv<bins[i+1])).sum()>0 else np.nan
                    for i in range(len(bins)-1)]

        fig, ax = plt.subplots(figsize=(10, 4), facecolor=C["bg"])
        ax_style(ax, f"spread vs inventory — {chosen}")
        ax.scatter(inv, spr, alpha=0.04, s=5, color=C["pink2"])
        ax.plot(mid_b, avgs, color=C["pink1"], lw=2.2, label="avg spread per inventory bucket")
        ax.axvline(0, color=C["text"], lw=0.7, ls="--", alpha=0.3)
        ax.set_xlabel("inventory (shares)", color=C["sub"], fontsize=8)
        ax.set_ylabel("spread ($)",         color=C["sub"], fontsize=8)
        ax.legend(fontsize=8, facecolor=C["bg"], edgecolor=C["grid"])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        corr = np.corrcoef(np.abs(inv), spr)[0, 1]
        r1, r2 = st.columns(2)
        r1.metric("|inventory| vs spread correlation", f"{corr:.3f}")
        verdict = "✅ learned inventory-dependent quoting" if corr > 0.10 else "⚠️ spread not adapting to inventory"
        r2.markdown(f'<div style="background:{C["pink3"]};border:1px solid {C["border"]};'
                    f'border-radius:8px;padding:12px;font-size:13px;font-weight:600;color:{C["text"]}'
                    f';margin-top:4px">{verdict}</div>', unsafe_allow_html=True)

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with t3:
    note("SAC (learned from scratch) vs Avellaneda-Stoikov (closed-form formula, the academic gold standard) "
         "vs random. if SAC's distribution is shifted right of A-S, it found something better than the formula.")

    if st.button(f"run {n_ep} episodes per policy"):
        bar = st.progress(0)
        as_p, rnd_p, sac_p = [], [], []
        for i in range(n_ep):
            as_p.append(run_as(i)[-1]["pnl"])
            rnd_p.append(run_rand(i)[-1]["pnl"])
            if model: sac_p.append(run_sac(model, i)[-1]["pnl"])
            bar.progress((i+1)/n_ep)
        bar.empty()

        table = [("A-S formula", as_p, C["pink2"]), ("random", rnd_p, "#c9aab8")]
        if model: table = [("SAC", sac_p, C["pink1"])] + table

        cols = st.columns(len(table))
        for col, (name, p, _) in zip(cols, table):
            col.metric(name, f"${np.mean(p):,.0f}", f"win {sum(x>0 for x in p)}/{n_ep}")

        all_v = sum([x for _,x,_ in table], [])
        bins  = np.linspace(min(all_v), max(all_v), 35)

        fig, ax = plt.subplots(figsize=(10, 4), facecolor=C["bg"])
        ax_style(ax, f"PnL distribution ({n_ep} episodes each)")
        for name, p, c in reversed(table):
            ax.hist(p, bins=bins, alpha=0.7, color=c, label=name, edgecolor=C["bg"], lw=0.3)
        ax.axvline(0, color=C["text"], lw=1, ls="--", alpha=0.4)
        ax.set_xlabel("episode PnL ($)", color=C["sub"], fontsize=8)
        ax.set_ylabel("count",           color=C["sub"], fontsize=8)
        ax.legend(fontsize=9, facecolor=C["bg"], edgecolor=C["grid"])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        if model:
            s, a, r = np.mean(sac_p), np.mean(as_p), np.mean(rnd_p)
            if   s > a: v = f"✅ SAC beats A-S by {((s-a)/abs(a)*100):.1f}% — learned beyond the formula"
            elif s > r: v = f"🟡 SAC between A-S and random — training worked, may need more steps"
            else:       v = "⚠️ SAC below A-S — try more training or tune the reward"
            st.markdown(f'<div style="background:{C["pink3"]};border-left:3px solid {C["pink1"]};'
                        f'border-radius:0 8px 8px 0;padding:10px 14px;font-size:13px;'
                        f'font-weight:600;color:{C["text"]};margin-top:8px">{v}</div>',
                        unsafe_allow_html=True)

# ── TAB 4 — Phase 2 ───────────────────────────────────────────────────────────
with t4:
    note(
        "phase 2 adds an informed trader who can see the future price direction. "
        "30% of order flow is now toxic. the market maker has to detect persistent "
        "directional pressure (measured by VPIN) and widen its spread to avoid being picked off. "
        "benchmark: ROC AUC > 0.60 means the MM is implicitly detecting informed flow."
    )

    # ── explainer cards ──────────────────────────────────────────────────────
    ec1, ec2, ec3 = st.columns(3)
    for col, title, body in [
        (ec1, "what is VPIN?",
         "Volume-synchronized Probability of Informed trading. "
         "When buys and sells are lopsided in a bucket, VPIN is high — "
         "that's a signal that informed traders are active."),
        (ec2, "what should the MM do?",
         "Widen its spread when VPIN is high. Tighter spread = free money for informed traders. "
         "A market maker that detects toxic flow and widens its spread is doing its job."),
        (ec3, "how do we measure it?",
         "ROC AUC: does VPIN predict when the MM widens its spread? "
         "Random quoting scores ~0.5. A smart adaptive MM should score >0.60. "
         "Higher = better adverse selection detection."),
    ]:
        col.markdown(f'<div style="background:{C["pink3"]};border:1px solid {C["border"]};'
                     f'border-radius:8px;padding:10px 12px;font-size:12px;color:{C["text"]}">'
                     f'<b style="color:{C["pink1"]}">{title}</b><br>{body}</div>',
                     unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── single episode: VPIN vs spread ───────────────────────────────────────
    st.markdown(f'<h4>episode view — spread vs toxicity signal</h4>', unsafe_allow_html=True)

    ep_col1, ep_col2, ep_col3 = st.columns(3)
    r_p2_sac  = ep_col1.button("run phase 2 SAC",  disabled=model_p2 is None, key="p2sac_ep")
    r_p2_as   = ep_col2.button("run A-S (p2 env)",  key="p2as_ep")
    r_p2_rand = ep_col3.button("run random (p2 env)", key="p2rand_ep")

    if   r_p2_sac and model_p2: ep2, lbl2 = run_sac_p2(model_p2, int(seed)), "SAC Phase 2"
    elif r_p2_as:               ep2, lbl2 = run_as_p2(int(seed)),            "Avellaneda-Stoikov"
    elif r_p2_rand:             ep2, lbl2 = run_rand_p2(int(seed)),          "Random"
    else:                       ep2 = None

    if ep2:
        # VPIN computation
        vpins, v_steps = compute_vpin(ep2, bucket_size=30)
        steps   = list(range(len(ep2)))
        spreads = [s["spread"] for s in ep2]
        inv     = [s["inventory"] for s in ep2]
        pnl     = [s["pnl"] for s in ep2]

        # buy/sell volume per step
        buy_vols  = [s.get("buy_vol", 0) for s in ep2]
        sell_vols = [s.get("sell_vol", 0) for s in ep2]

        mm1, mm2, mm3 = st.columns(3)
        mm1.metric("final PnL", f"${pnl[-1]:,.0f}")
        mm2.metric("avg VPIN",  f"{np.mean(vpins):.3f}" if len(vpins) else "n/a")
        mm3.metric("spread range", f"${min(spreads):.4f} – ${max(spreads):.4f}")

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), facecolor=C["bg"])
        fig.subplots_adjust(hspace=0.45, left=0.07, right=0.97, top=0.94, bottom=0.07)

        # Panel 1: spread colored by toxicity
        ax = axes[0]
        ax_style(ax, f"{lbl2} — spread over time (thicker when toxic)")
        ax.plot(steps, spreads, color=C["pink1"], lw=1.2, label="MM spread")
        if len(vpins):
            # highlight high-VPIN windows
            vpin_thresh = np.percentile(vpins, 60)
            for vi, vs in zip(vpins, v_steps):
                if vi > vpin_thresh:
                    ax.axvspan(max(0, vs-15), min(len(steps), vs+15),
                               alpha=0.12, color="steelblue")
        ax.set_ylabel("spread ($)", color=C["sub"], fontsize=8)
        ax.legend(fontsize=8, facecolor=C["bg"], edgecolor=C["grid"])

        # Panel 2: VPIN over time
        ax = axes[1]
        ax_style(ax, "VPIN — toxicity signal (blue shaded = high VPIN buckets)")
        if len(vpins):
            ax.bar(v_steps, vpins, color="steelblue", alpha=0.7, width=12, label="VPIN per bucket")
            ax.axhline(np.mean(vpins), color=C["pink1"], lw=1.5, ls="--",
                       label=f"mean VPIN={np.mean(vpins):.3f}")
        ax.set_ylabel("VPIN", color=C["sub"], fontsize=8)
        ax.legend(fontsize=8, facecolor=C["bg"], edgecolor=C["grid"])

        # Panel 3: buy vs sell volume (informed flow is lopsided)
        ax = axes[2]
        ax_style(ax, "order flow — buy vs sell volume per step")
        ax.bar(steps, buy_vols,  color=C["pink1"], alpha=0.7, label="buy vol",  width=1)
        ax.bar(steps, [-v for v in sell_vols], color=C["sub"], alpha=0.7, label="sell vol", width=1)
        ax.axhline(0, color=C["text"], lw=0.6, ls="--", alpha=0.4)
        ax.set_ylabel("volume (shares)", color=C["sub"], fontsize=8)
        ax.set_xlabel("step", color=C["sub"], fontsize=8)
        ax.legend(fontsize=8, facecolor=C["bg"], edgecolor=C["grid"])

        st.pyplot(fig)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<h4>benchmark: ROC AUC — does the MM detect informed flow?</h4>', unsafe_allow_html=True)

    n_bench = st.slider("episodes for benchmark", 10, 60, 20, key="bench_n")
    if st.button("run VPIN benchmark", key="bench_btn"):
        with st.spinner("running benchmark episodes..."):
            bar = st.progress(0)

            as_hists, rand_hists, sac_hists = [], [], []
            for i in range(n_bench):
                as_hists.append(run_as_p2(i))
                rand_hists.append(run_rand_p2(i))
                if model_p2:
                    sac_hists.append(run_sac_p2(model_p2, i))
                bar.progress((i+1)/n_bench)
            bar.empty()

        rand_auc = spread_vs_vpin_auc(rand_hists)
        as_auc   = spread_vs_vpin_auc(as_hists)
        sac_auc  = spread_vs_vpin_auc(sac_hists) if sac_hists else None

        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Random AUC",   f"{rand_auc:.3f}", "baseline ~0.5")
        bc2.metric("A-S AUC",      f"{as_auc:.3f}")
        bc3.metric("SAC p2 AUC",   f"{sac_auc:.3f}" if sac_auc else "not trained",
                   "target > 0.60" if sac_auc else "")

        # bar chart
        labels = ["Random", "A-S"]
        aucs   = [rand_auc, as_auc]
        colors = ["#c9aab8", C["pink2"]]
        if sac_auc is not None:
            labels.append("SAC Phase 2")
            aucs.append(sac_auc)
            colors.append(C["pink1"])

        fig, ax = plt.subplots(figsize=(8, 4), facecolor=C["bg"])
        ax_style(ax, "VPIN → Spread Widening: ROC AUC per policy")
        bars = ax.bar(labels, aucs, color=colors, alpha=0.85, edgecolor=C["bg"])
        ax.axhline(0.5, color=C["text"], lw=1, ls="--", alpha=0.5, label="random baseline (0.5)")
        ax.axhline(0.6, color="steelblue", lw=1, ls=":", alpha=0.7, label="target AUC (0.6)")
        for bar, val in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=11,
                    fontweight="bold", color=C["text"])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("ROC AUC", color=C["sub"], fontsize=8)
        ax.legend(fontsize=8, facecolor=C["bg"], edgecolor=C["grid"])
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # verdict
        if sac_auc is not None:
            if sac_auc >= 0.60:
                v = f"✅ AUC {sac_auc:.3f} — MM is detecting informed flow and widening spread adaptively"
            elif sac_auc > as_auc:
                v = f"🟡 AUC {sac_auc:.3f} — better than A-S but below 0.60 target, needs more training"
            else:
                v = f"⚠️ AUC {sac_auc:.3f} — MM not yet detecting toxic flow, inventory penalty may need tuning"
        else:
            if as_auc > 0.55:
                v = f"A-S AUC {as_auc:.3f} — static formula has some toxicity detection. Train SAC to beat it."
            else:
                v = f"A-S AUC {as_auc:.3f} — static formula is flat, no detection. Good baseline to beat."

        st.markdown(f'<div style="background:{C["pink3"]};border-left:3px solid {C["pink1"]};'
                    f'border-radius:0 8px 8px 0;padding:10px 14px;font-size:13px;'
                    f'font-weight:600;color:{C["text"]};margin-top:8px">{v}</div>',
                    unsafe_allow_html=True)

# ── TAB 5 — Results ───────────────────────────────────────────────────────────
with t5:
    note(
        "pre-computed evaluation results across 100 episodes each. "
        "re-run evaluation scripts to refresh these plots."
    )

    RESULTS = {
        "Phase 1 — PnL distribution":       "evaluation/plots/pnl_distribution.png",
        "Phase 1 — episode replay":          "evaluation/plots/episode.png",
        "Phase 1 — spread vs inventory":     "evaluation/plots/spread_vs_inventory.png",
        "Phase 2 — PnL distribution":        "evaluation/plots/phase2_pnl.png",
        "Phase 2 — VPIN vs spread":          "evaluation/plots/vpin_spread.png",
        "Phase 2 — VPIN ROC AUC":            "evaluation/plots/vpin_roc.png",
    }

    # Summary table
    st.markdown(f'<h4>headline numbers</h4>', unsafe_allow_html=True)
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown(f'<div style="background:{C["pink3"]};border:1px solid {C["border"]};border-radius:8px;padding:12px 16px">'
                    f'<b style="color:{C["pink1"]}">Phase 1 — noise traders only</b>'
                    f'<table style="width:100%;font-size:12px;color:{C["text"]};margin-top:8px;border-collapse:collapse">'
                    f'<tr><th style="text-align:left;color:{C["sub"]}">Policy</th><th style="color:{C["sub"]}">Mean PnL</th><th style="color:{C["sub"]}">Win rate</th><th style="color:{C["sub"]}">Closing |inv|</th></tr>'
                    f'<tr><td>SAC</td><td>$2,512</td><td>100/100</td><td>20.9</td></tr>'
                    f'<tr><td>A-S formula</td><td>$2,616</td><td>100/100</td><td>138.3</td></tr>'
                    f'<tr><td>Random</td><td>$2,109</td><td>99/100</td><td>160.6</td></tr>'
                    f'</table></div>', unsafe_allow_html=True)
    with rc2:
        st.markdown(f'<div style="background:{C["pink3"]};border:1px solid {C["border"]};border-radius:8px;padding:12px 16px">'
                    f'<b style="color:{C["pink1"]}">Phase 2 — 30% informed flow</b>'
                    f'<table style="width:100%;font-size:12px;color:{C["text"]};margin-top:8px;border-collapse:collapse">'
                    f'<tr><th style="text-align:left;color:{C["sub"]}">Policy</th><th style="color:{C["sub"]}">Mean PnL</th><th style="color:{C["sub"]}">Win rate</th><th style="color:{C["sub"]}">Closing |inv|</th></tr>'
                    f'<tr><td>SAC</td><td>$3,061</td><td>100/100</td><td>20.7</td></tr>'
                    f'<tr><td>A-S formula</td><td>$3,233</td><td>100/100</td><td>233.9</td></tr>'
                    f'<tr><td>Random</td><td>$2,741</td><td>99/100</td><td>292.2</td></tr>'
                    f'</table></div>', unsafe_allow_html=True)

    st.markdown(f'<div style="background:{C["pink3"]};border-left:3px solid {C["pink1"]};'
                f'border-radius:0 8px 8px 0;padding:10px 14px;font-size:13px;'
                f'color:{C["text"]};margin:12px 0">'
                f'<b>Key finding:</b> SAC achieves 91% lower closing inventory than A-S in Phase 2 (20.7 vs 233.9 shares). '
                f'It learned to flatten positions quickly under informed flow rather than widening spreads. '
                f'VPIN AUC of 0.484 confirms this — the agent is not doing textbook spread widening, '
                f'it found its own risk management strategy.'
                f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Plot grid
    st.markdown(f'<h4>evaluation plots</h4>', unsafe_allow_html=True)
    plot_items = list(RESULTS.items())
    for i in range(0, len(plot_items), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(plot_items):
                title, path = plot_items[i + j]
                col.markdown(f'<p style="font-size:11px;font-weight:700;color:{C["sub"]};margin-bottom:4px">{title}</p>',
                              unsafe_allow_html=True)
                if os.path.exists(path):
                    col.image(path, width='stretch')
                else:
                    col.markdown(f'<p style="font-size:11px;color:{C["border"]}">not generated yet</p>',
                                 unsafe_allow_html=True)

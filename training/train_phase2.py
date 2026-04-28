"""
Phase 2 training: SAC market maker vs. noise traders + informed trader.

The informed trader knows the future price direction. The market maker has to
learn to detect toxic flow (persistent directional pressure) and widen its
spread to protect against adverse selection.

New observation features (vs Phase 1):
  - rolling_imbalance: 5-step signed buy/sell pressure
  - rolling_toxicity:  5-step VPIN-like unsigned imbalance

Run: python training/train_phase2.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from environment.market_env_phase2 import MarketEnvPhase2


CONFIG = {
    "phase": 2,
    "total_timesteps": 300_000,
    "learning_rate": 3e-4,
    "ent_coef": "auto",
    "batch_size": 256,
    "buffer_size": 150_000,
    "learning_starts": 2_000,
    "episode_length": 390,
    "max_inventory": 100,
    "sigma": 0.002,
    "kappa": 1.5,
    "informed_frac": 0.3,
    "signal_horizon": 20,
}


class EpisodeLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self._ep_reward = 0.0
        self._ep_count = 0

    def _on_step(self):
        self._ep_reward += self.locals["rewards"][0]
        if self.locals["dones"][0]:
            self._ep_count += 1
            info = self.locals["infos"][0]
            wandb.log({
                "episode/reward":        self._ep_reward,
                "episode/portfolio_pnl": info.get("portfolio_value", 0),
                "episode/inventory":     info.get("inventory", 0),
                "episode/spread":        info.get("spread", 0),
                "episode/num_trades":    info.get("num_trades", 0),
                "episode/count":         self._ep_count,
            }, step=self.num_timesteps)
            self._ep_reward = 0.0
        return True


def main():
    run = wandb.init(
        project="trading-sim",
        name="phase2-sac-informed",
        config=CONFIG,
    )

    train_env = MarketEnvPhase2(
        episode_length=CONFIG["episode_length"],
        max_inventory=CONFIG["max_inventory"],
        sigma=CONFIG["sigma"],
        kappa=CONFIG["kappa"],
        informed_frac=CONFIG["informed_frac"],
        signal_horizon=CONFIG["signal_horizon"],
    )
    eval_env = MarketEnvPhase2(
        episode_length=CONFIG["episode_length"],
        max_inventory=CONFIG["max_inventory"],
        sigma=CONFIG["sigma"],
        kappa=CONFIG["kappa"],
        informed_frac=CONFIG["informed_frac"],
        signal_horizon=CONFIG["signal_horizon"],
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=0,
        learning_rate=CONFIG["learning_rate"],
        ent_coef=CONFIG["ent_coef"],
        batch_size=CONFIG["batch_size"],
        buffer_size=CONFIG["buffer_size"],
        learning_starts=CONFIG["learning_starts"],
    )

    os.makedirs(f"models/phase2/{run.id}", exist_ok=True)

    callbacks = [
        WandbCallback(
            gradient_save_freq=5_000,
            model_save_path=f"models/phase2/{run.id}",
            verbose=0,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=f"models/phase2/{run.id}/best",
            eval_freq=10_000,
            n_eval_episodes=20,
            deterministic=True,
            verbose=1,
        ),
        EpisodeLogger(),
    ]

    print(f"Run: {run.url}")
    print("Training SAC — Phase 2 (informed trader)")
    print("=" * 50)
    model.learn(
        total_timesteps=CONFIG["total_timesteps"],
        callback=callbacks,
        progress_bar=True,
    )

    model.save(f"models/phase2/{run.id}/sac_final")
    print(f"\nSaved to models/phase2/{run.id}/sac_final.zip")
    run.finish()


if __name__ == "__main__":
    main()

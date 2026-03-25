"""
PPO Training with 10 hyperparameter configurations for Brainiacs AI Tutor.
"""

import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import environment
import gymnasium

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'pg')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


class RewardLoggerCallback(BaseCallback):
    """Logs episode rewards and policy entropy during training."""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.entropy_values = []

    def _on_step(self):
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
        return True

    def _on_rollout_end(self):
        # Log entropy from the logger if available
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            try:
                entropy = self.model.logger.name_to_value.get('train/entropy_loss', None)
                if entropy is not None:
                    self.entropy_values.append(-entropy)  # SB3 logs negative entropy
            except Exception:
                pass


# 10 hyperparameter configurations
CONFIGS = [
    {'learning_rate': 3e-4, 'gamma': 0.99, 'n_steps': 1024, 'batch_size': 64, 'clip_range': 0.2, 'ent_coef': 0.01},
    {'learning_rate': 1e-4, 'gamma': 0.99, 'n_steps': 2048, 'batch_size': 128, 'clip_range': 0.2, 'ent_coef': 0.0},
    {'learning_rate': 1e-3, 'gamma': 0.95, 'n_steps': 512, 'batch_size': 32, 'clip_range': 0.3, 'ent_coef': 0.05},
    {'learning_rate': 3e-3, 'gamma': 0.9, 'n_steps': 256, 'batch_size': 64, 'clip_range': 0.1, 'ent_coef': 0.01},
    {'learning_rate': 1e-4, 'gamma': 0.95, 'n_steps': 1024, 'batch_size': 32, 'clip_range': 0.3, 'ent_coef': 0.0},
    {'learning_rate': 3e-4, 'gamma': 0.99, 'n_steps': 512, 'batch_size': 128, 'clip_range': 0.1, 'ent_coef': 0.05},
    {'learning_rate': 1e-3, 'gamma': 0.99, 'n_steps': 2048, 'batch_size': 64, 'clip_range': 0.2, 'ent_coef': 0.01},
    {'learning_rate': 3e-4, 'gamma': 0.9, 'n_steps': 256, 'batch_size': 32, 'clip_range': 0.3, 'ent_coef': 0.0},
    {'learning_rate': 1e-4, 'gamma': 0.99, 'n_steps': 1024, 'batch_size': 128, 'clip_range': 0.2, 'ent_coef': 0.05},
    {'learning_rate': 3e-3, 'gamma': 0.95, 'n_steps': 512, 'batch_size': 64, 'clip_range': 0.1, 'ent_coef': 0.01},
]

TOTAL_TIMESTEPS = 150000


def evaluate_model(model, n_episodes=20, seed_base=3000):
    """Evaluate a trained model over multiple episodes."""
    rewards = []
    for i in range(n_episodes):
        env = gymnasium.make('BrainiacsTutor-v0')
        obs, _ = env.reset(seed=seed_base + i)
        total_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        env.close()
    return np.mean(rewards), np.std(rewards)


def main():
    results = []
    best_mean_reward = -float('inf')
    best_model = None
    best_run = 0
    all_training_curves = {}
    all_entropy_curves = {}

    print("=" * 70)
    print("PPO Training - 10 Hyperparameter Configurations")
    print("=" * 70)

    for i, config in enumerate(CONFIGS):
        print(f"\n--- Run {i+1}/10 ---")
        print(f"Config: {config}")

        env = Monitor(gymnasium.make('BrainiacsTutor-v0'))

        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=config['learning_rate'],
            gamma=config['gamma'],
            n_steps=config['n_steps'],
            batch_size=config['batch_size'],
            clip_range=config['clip_range'],
            ent_coef=config['ent_coef'],
            verbose=0,
            seed=42 + i,
        )

        callback = RewardLoggerCallback()
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

        mean_reward, std_reward = evaluate_model(model)
        print(f"  Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        all_training_curves[f"run_{i+1}"] = callback.episode_rewards
        all_entropy_curves[f"run_{i+1}"] = callback.entropy_values

        result = {**config, 'mean_reward': mean_reward, 'std_reward': std_reward, 'run': i + 1}
        results.append(result)

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_model = model
            best_run = i + 1

        env.close()

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'ppo_results.csv'), index=False)
    print(f"\nResults saved to results/ppo_results.csv")

    # Save best model
    best_model.save(os.path.join(MODELS_DIR, 'ppo_best'))
    print(f"Best model (run {best_run}, reward={best_mean_reward:.2f}) saved to models/pg/ppo_best")

    # Save training curves
    max_len = max(len(v) for v in all_training_curves.values())
    curves_data = {}
    for k, v in all_training_curves.items():
        padded = v + [np.nan] * (max_len - len(v))
        curves_data[k] = padded
    curves_df = pd.DataFrame(curves_data)
    curves_df.to_csv(os.path.join(RESULTS_DIR, 'ppo_training_curves.csv'), index=False)

    # Save entropy curves
    if any(len(v) > 0 for v in all_entropy_curves.values()):
        max_len_e = max(len(v) for v in all_entropy_curves.values())
        entropy_data = {}
        for k, v in all_entropy_curves.items():
            padded = v + [np.nan] * (max_len_e - len(v))
            entropy_data[k] = padded
        entropy_df = pd.DataFrame(entropy_data)
        entropy_df.to_csv(os.path.join(RESULTS_DIR, 'ppo_entropy_curves.csv'), index=False)

    print("\n" + "=" * 70)
    print("PPO Training Complete!")
    print(f"Best configuration: Run {best_run}")
    print(f"Best mean reward: {best_mean_reward:.2f}")
    print("=" * 70)


if __name__ == '__main__':
    main()

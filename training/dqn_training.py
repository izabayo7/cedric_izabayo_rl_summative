"""
DQN Training with 10 hyperparameter configurations for Brainiacs AI Tutor.
"""

import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import environment  # registers the env

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'dqn')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


class RewardLoggerCallback(BaseCallback):
    """Logs episode rewards during training."""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self):
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])
        return True


# 10 hyperparameter configurations
CONFIGS = [
    {'learning_rate': 1e-4, 'gamma': 0.99, 'buffer_size': 50000, 'batch_size': 64, 'exploration_fraction': 0.2, 'exploration_final_eps': 0.05},
    {'learning_rate': 5e-4, 'gamma': 0.95, 'buffer_size': 10000, 'batch_size': 32, 'exploration_fraction': 0.1, 'exploration_final_eps': 0.01},
    {'learning_rate': 1e-3, 'gamma': 0.99, 'buffer_size': 50000, 'batch_size': 128, 'exploration_fraction': 0.3, 'exploration_final_eps': 0.1},
    {'learning_rate': 5e-3, 'gamma': 0.9, 'buffer_size': 5000, 'batch_size': 32, 'exploration_fraction': 0.2, 'exploration_final_eps': 0.05},
    {'learning_rate': 1e-2, 'gamma': 0.95, 'buffer_size': 10000, 'batch_size': 64, 'exploration_fraction': 0.1, 'exploration_final_eps': 0.01},
    {'learning_rate': 1e-4, 'gamma': 0.9, 'buffer_size': 10000, 'batch_size': 128, 'exploration_fraction': 0.3, 'exploration_final_eps': 0.1},
    {'learning_rate': 5e-4, 'gamma': 0.99, 'buffer_size': 5000, 'batch_size': 64, 'exploration_fraction': 0.2, 'exploration_final_eps': 0.01},
    {'learning_rate': 1e-3, 'gamma': 0.95, 'buffer_size': 50000, 'batch_size': 32, 'exploration_fraction': 0.1, 'exploration_final_eps': 0.05},
    {'learning_rate': 5e-4, 'gamma': 0.9, 'buffer_size': 10000, 'batch_size': 128, 'exploration_fraction': 0.3, 'exploration_final_eps': 0.05},
    {'learning_rate': 1e-3, 'gamma': 0.99, 'buffer_size': 5000, 'batch_size': 64, 'exploration_fraction': 0.2, 'exploration_final_eps': 0.1},
]

TOTAL_TIMESTEPS = 150000


def evaluate_model(model, n_episodes=20, seed_base=1000):
    """Evaluate a trained model over multiple episodes."""
    rewards = []
    for i in range(n_episodes):
        env = Monitor(gymnasium.make('BrainiacsTutor-v0'))
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


import gymnasium

def main():
    results = []
    best_mean_reward = -float('inf')
    best_model = None
    all_training_curves = {}

    print("=" * 70)
    print("DQN Training - 10 Hyperparameter Configurations")
    print("=" * 70)

    for i, config in enumerate(CONFIGS):
        print(f"\n--- Run {i+1}/10 ---")
        print(f"Config: {config}")

        env = Monitor(gymnasium.make('BrainiacsTutor-v0'))

        model = DQN(
            'MlpPolicy',
            env,
            learning_rate=config['learning_rate'],
            gamma=config['gamma'],
            buffer_size=config['buffer_size'],
            batch_size=config['batch_size'],
            exploration_fraction=config['exploration_fraction'],
            exploration_final_eps=config['exploration_final_eps'],
            verbose=0,
            seed=42 + i,
        )

        callback = RewardLoggerCallback()
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

        mean_reward, std_reward = evaluate_model(model)
        print(f"  Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        # Save training curve
        all_training_curves[f"run_{i+1}"] = callback.episode_rewards

        result = {**config, 'mean_reward': mean_reward, 'std_reward': std_reward, 'run': i + 1}
        results.append(result)

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_model = model
            best_run = i + 1

        env.close()

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'dqn_results.csv'), index=False)
    print(f"\nResults saved to results/dqn_results.csv")

    # Save best model
    best_model.save(os.path.join(MODELS_DIR, 'dqn_best'))
    print(f"Best model (run {best_run}, reward={best_mean_reward:.2f}) saved to models/dqn/dqn_best")

    # Save training curves
    max_len = max(len(v) for v in all_training_curves.values())
    curves_data = {}
    for k, v in all_training_curves.items():
        padded = v + [np.nan] * (max_len - len(v))
        curves_data[k] = padded
    curves_df = pd.DataFrame(curves_data)
    curves_df.to_csv(os.path.join(RESULTS_DIR, 'dqn_training_curves.csv'), index=False)

    print("\n" + "=" * 70)
    print("DQN Training Complete!")
    print(f"Best configuration: Run {best_run}")
    print(f"Best mean reward: {best_mean_reward:.2f}")
    print("=" * 70)


if __name__ == '__main__':
    main()

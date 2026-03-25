"""
REINFORCE (Monte Carlo Policy Gradient) Training - Manual PyTorch Implementation.
10 hyperparameter configurations for Brainiacs AI Tutor.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import environment
import gymnasium

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'pg')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


class PolicyNetwork(nn.Module):
    """Simple policy network with 2 hidden layers."""

    def __init__(self, obs_dim=13, action_dim=45, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs = self.forward(state_t)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


def compute_returns(rewards, gamma):
    """Compute discounted returns."""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)
    return returns


def train_reinforce(config, run_id, n_episodes=3000):
    """Train REINFORCE agent with given config."""
    env = gymnasium.make('BrainiacsTutor-v0')

    policy = PolicyNetwork(
        obs_dim=13, action_dim=45, hidden_size=config['hidden_size']
    )
    optimizer = optim.Adam(policy.parameters(), lr=config['learning_rate'])

    episode_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + run_id * 1000 + ep)
        log_probs = []
        rewards = []
        done = False

        while not done:
            action, log_prob = policy.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            done = terminated or truncated

        total_reward = sum(rewards)
        episode_rewards.append(total_reward)

        # Compute returns
        returns = compute_returns(rewards, config['gamma'])

        # Apply baseline (mean return) if configured
        if config['baseline']:
            returns = returns - returns.mean()

        # Normalize returns for stability
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)

        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()

        if (ep + 1) % 500 == 0:
            recent = episode_rewards[-100:]
            print(f"    Episode {ep+1}/{n_episodes} | "
                  f"Avg Reward (last 100): {np.mean(recent):.2f}")

    env.close()
    return policy, episode_rewards


def evaluate_policy(policy, n_episodes=20, seed_base=2000):
    """Evaluate trained policy."""
    rewards = []
    for i in range(n_episodes):
        env = gymnasium.make('BrainiacsTutor-v0')
        obs, _ = env.reset(seed=seed_base + i)
        total_reward = 0
        done = False
        while not done:
            state_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                probs = policy(state_t)
            action = torch.argmax(probs, dim=-1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        env.close()
    return np.mean(rewards), np.std(rewards)


# 10 hyperparameter configurations
CONFIGS = [
    {'learning_rate': 1e-3, 'gamma': 0.99, 'hidden_size': 128, 'baseline': True},
    {'learning_rate': 5e-4, 'gamma': 0.99, 'hidden_size': 128, 'baseline': True},
    {'learning_rate': 1e-4, 'gamma': 0.95, 'hidden_size': 256, 'baseline': True},
    {'learning_rate': 5e-3, 'gamma': 0.9, 'hidden_size': 64, 'baseline': False},
    {'learning_rate': 1e-3, 'gamma': 0.95, 'hidden_size': 256, 'baseline': False},
    {'learning_rate': 5e-4, 'gamma': 0.99, 'hidden_size': 64, 'baseline': True},
    {'learning_rate': 1e-4, 'gamma': 0.99, 'hidden_size': 128, 'baseline': False},
    {'learning_rate': 5e-3, 'gamma': 0.95, 'hidden_size': 128, 'baseline': True},
    {'learning_rate': 1e-3, 'gamma': 0.9, 'hidden_size': 256, 'baseline': True},
    {'learning_rate': 5e-4, 'gamma': 0.95, 'hidden_size': 64, 'baseline': False},
]

N_EPISODES = 3000


def main():
    results = []
    best_mean_reward = -float('inf')
    best_policy = None
    best_run = 0
    all_training_curves = {}

    print("=" * 70)
    print("REINFORCE Training - 10 Hyperparameter Configurations")
    print("=" * 70)

    for i, config in enumerate(CONFIGS):
        print(f"\n--- Run {i+1}/10 ---")
        print(f"Config: {config}")

        policy, episode_rewards = train_reinforce(config, i, N_EPISODES)
        mean_reward, std_reward = evaluate_policy(policy)

        print(f"  Eval Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        all_training_curves[f"run_{i+1}"] = episode_rewards

        result = {**config, 'mean_reward': mean_reward, 'std_reward': std_reward, 'run': i + 1}
        results.append(result)

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_policy = policy
            best_run = i + 1

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, 'reinforce_results.csv'), index=False)
    print(f"\nResults saved to results/reinforce_results.csv")

    # Save best model
    torch.save(best_policy.state_dict(), os.path.join(MODELS_DIR, 'reinforce_best.pt'))
    print(f"Best model (run {best_run}, reward={best_mean_reward:.2f}) saved to models/pg/reinforce_best.pt")

    # Save training curves
    max_len = max(len(v) for v in all_training_curves.values())
    curves_data = {}
    for k, v in all_training_curves.items():
        padded = v + [np.nan] * (max_len - len(v))
        curves_data[k] = padded
    curves_df = pd.DataFrame(curves_data)
    curves_df.to_csv(os.path.join(RESULTS_DIR, 'reinforce_training_curves.csv'), index=False)

    print("\n" + "=" * 70)
    print("REINFORCE Training Complete!")
    print(f"Best configuration: Run {best_run}")
    print(f"Best mean reward: {best_mean_reward:.2f}")
    print("=" * 70)


if __name__ == '__main__':
    main()

"""
Generate all required plots from training logs.
Saves plots to the plots/ directory.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import environment
import gymnasium

sns.set_style('whitegrid')
sns.set_palette('deep')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def smooth(data, window=50):
    """Rolling average smoothing."""
    if len(data) < window:
        return data
    return pd.Series(data).rolling(window=window, min_periods=1).mean().values


def find_best_run(results_csv):
    """Find the best run index from results CSV."""
    df = pd.read_csv(results_csv)
    best_idx = df['mean_reward'].idxmax()
    return int(df.loc[best_idx, 'run'])


def load_best_curve(method):
    """Load training curve for the best run of a method."""
    results_path = os.path.join(RESULTS_DIR, f'{method}_results.csv')
    curves_path = os.path.join(RESULTS_DIR, f'{method}_training_curves.csv')

    if not os.path.exists(results_path) or not os.path.exists(curves_path):
        return None

    best_run = find_best_run(results_path)
    curves = pd.read_csv(curves_path)
    col = f'run_{best_run}'
    if col in curves.columns:
        return curves[col].dropna().values
    return None


def plot_cumulative_rewards():
    """Plot 1: 1x3 subplot grid of cumulative rewards for best run of each method."""
    methods = ['dqn', 'reinforce', 'ppo']
    titles = ['DQN', 'REINFORCE', 'PPO']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Cumulative Rewards - Best Run per Method', fontsize=16, fontweight='bold')

    all_curves = {}
    for method in methods:
        curve = load_best_curve(method)
        if curve is not None:
            all_curves[method] = np.cumsum(curve)

    # Find global y-axis limits
    if all_curves:
        y_min = min(c.min() for c in all_curves.values())
        y_max = max(c.max() for c in all_curves.values())
        margin = (y_max - y_min) * 0.05
    else:
        y_min, y_max, margin = 0, 1, 0.1

    for idx, (method, title) in enumerate(zip(methods, titles)):
        ax = axes[idx]
        if method in all_curves:
            cumulative = all_curves[method]
            ax.plot(cumulative, linewidth=1.5, alpha=0.9)
            ax.fill_between(range(len(cumulative)), cumulative, alpha=0.15)
            ax.set_ylim(y_min - margin, y_max + margin)
        else:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Reward')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '1_cumulative_rewards.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 1_cumulative_rewards.png")


def plot_dqn_loss():
    """Plot 2: DQN training loss/reward curve over time."""
    curve = load_best_curve('dqn')
    if curve is None:
        print("  Skipped: DQN data not available")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(curve, alpha=0.3, color='steelblue', label='Episode Reward')
    smoothed = smooth(curve, window=50)
    ax.plot(smoothed, linewidth=2, color='navy', label='Smoothed (window=50)')
    ax.set_title('DQN Training Curve (Best Configuration)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '2_dqn_training_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 2_dqn_training_curve.png")


def plot_entropy_curves():
    """Plot 3: Policy entropy curves for PPO."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Policy Entropy During Training', fontsize=14, fontweight='bold')

    entropy_path = os.path.join(RESULTS_DIR, 'ppo_entropy_curves.csv')
    results_path = os.path.join(RESULTS_DIR, 'ppo_results.csv')

    if os.path.exists(entropy_path) and os.path.exists(results_path):
        best_run = find_best_run(results_path)
        entropy_df = pd.read_csv(entropy_path)
        col = f'run_{best_run}'
        if col in entropy_df.columns:
            data = entropy_df[col].dropna().values
            if len(data) > 0:
                ax.plot(data, linewidth=1.5, alpha=0.7, label='Entropy')
                if len(data) > 10:
                    smoothed = smooth(data, window=max(len(data) // 20, 5))
                    ax.plot(smoothed, linewidth=2, color='red', label='Smoothed')
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No entropy data', ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No entropy data', ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)

    ax.set_title('PPO Policy Entropy', fontsize=13, fontweight='bold')
    ax.set_xlabel('Update Step')
    ax.set_ylabel('Entropy')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '3_entropy_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 3_entropy_curves.png")


def plot_convergence_comparison():
    """Plot 4: All 3 methods on one plot with smoothed rewards."""
    methods = ['dqn', 'reinforce', 'ppo']
    labels = ['DQN', 'REINFORCE', 'PPO']
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    fig, ax = plt.subplots(figsize=(12, 7))

    for method, label, color in zip(methods, labels, colors):
        curve = load_best_curve(method)
        if curve is not None:
            smoothed = smooth(curve, window=50)
            ax.plot(smoothed, linewidth=2, label=label, color=color)

            # Mark convergence point (first time smoothed reward stays above threshold)
            if len(smoothed) > 100:
                threshold = np.percentile(smoothed[-50:], 25)
                for ep_idx in range(50, len(smoothed)):
                    if all(smoothed[ep_idx:min(ep_idx+50, len(smoothed))] > threshold * 0.9):
                        ax.axvline(x=ep_idx, color=color, linestyle='--', alpha=0.4)
                        ax.annotate(f'{label}: ep {ep_idx}',
                                    xy=(ep_idx, smoothed[ep_idx]),
                                    fontsize=8, color=color)
                        break

    ax.set_title('Convergence Comparison - All Methods (Smoothed, window=50)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward (Smoothed)')
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '4_convergence_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 4_convergence_comparison.png")


def plot_generalization_test():
    """Plot 5: Box plot comparing reward distributions across methods on unseen seeds."""
    methods_info = {
        'DQN': ('dqn', 'sb3', os.path.join(PROJECT_ROOT, 'models', 'dqn', 'dqn_best.zip')),
        'PPO': ('ppo', 'sb3', os.path.join(PROJECT_ROOT, 'models', 'pg', 'ppo_best.zip')),
        'REINFORCE': ('reinforce', 'torch', os.path.join(PROJECT_ROOT, 'models', 'pg', 'reinforce_best.pt')),
    }

    all_rewards = {}

    for name, (method, mtype, model_path) in methods_info.items():
        if not os.path.exists(model_path):
            print(f"  Skipping {name}: model not found at {model_path}")
            continue

        if mtype == 'sb3':
            if method == 'dqn':
                from stable_baselines3 import DQN
                model = DQN.load(model_path)
            elif method == 'ppo':
                from stable_baselines3 import PPO
                model = PPO.load(model_path)

            rewards = []
            for seed in range(5000, 5020):
                env = gymnasium.make('BrainiacsTutor-v0')
                obs, _ = env.reset(seed=seed)
                total_reward = 0
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    done = terminated or truncated
                rewards.append(total_reward)
                env.close()
            all_rewards[name] = rewards
        else:
            import torch
            from training.reinforce_training import PolicyNetwork
            # Read hidden_size from results CSV
            reinforce_results = pd.read_csv(os.path.join(RESULTS_DIR, 'reinforce_results.csv'))
            best_row = reinforce_results.loc[reinforce_results['mean_reward'].idxmax()]
            hidden_size = int(best_row['hidden_size'])
            policy = PolicyNetwork(obs_dim=13, action_dim=45, hidden_size=hidden_size)
            policy.load_state_dict(torch.load(model_path, weights_only=True))
            policy.eval()

            rewards = []
            for seed in range(5000, 5020):
                env = gymnasium.make('BrainiacsTutor-v0')
                obs, _ = env.reset(seed=seed)
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
            all_rewards[name] = rewards

    if not all_rewards:
        print("  Skipped: No models available for generalization test")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    data = []
    labels = []
    for name, rewards in all_rewards.items():
        data.append(rewards)
        labels.append(name)

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    for patch, color in zip(bp['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_title('Generalization Test - Reward Distribution on 20 Unseen Seeds',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Episode Reward')
    ax.set_xlabel('Method')

    # Add individual points
    for i, d in enumerate(data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.5, s=20, zorder=5)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '5_generalization_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 5_generalization_boxplot.png")


def main():
    print("=" * 70)
    print("Generating All Plots")
    print("=" * 70)

    print("\n1. Cumulative Rewards (subplots)...")
    plot_cumulative_rewards()

    print("\n2. DQN Training Curve...")
    plot_dqn_loss()

    print("\n3. Policy Entropy Curves (PPO)...")
    plot_entropy_curves()

    print("\n4. Convergence Comparison...")
    plot_convergence_comparison()

    print("\n5. Generalization Test (Box Plot)...")
    plot_generalization_test()

    print("\n" + "=" * 70)
    print(f"All plots saved to {PLOTS_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()

"""
Main entry point - Loads the best performing model and runs it
with pygame visualization.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import gymnasium
import environment  # registers BrainiacsTutor-v0
from environment.rendering import TutorRenderer

TOPICS = ['variable_scope', 'off_by_one', 'type_mismatch', 'loop_logic', 'function_params']
TOPIC_LABELS = ['Variable Scope', 'Off-by-One', 'Type Mismatch', 'Loop Logic', 'Function Params']
DIFFICULTIES = ['Easy', 'Medium', 'Hard']
SUPPORTS = ['No Support', 'Hint', 'Worked Example']

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def find_best_model():
    """Find the best model across all training methods by reading results CSVs."""
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    best_method = None
    best_reward = -float('inf')

    methods = {
        'dqn': os.path.join(results_dir, 'dqn_results.csv'),
        'ppo': os.path.join(results_dir, 'ppo_results.csv'),
        'reinforce': os.path.join(results_dir, 'reinforce_results.csv'),
    }

    for method, path in methods.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            best_row = df.loc[df['mean_reward'].idxmax()]
            if best_row['mean_reward'] > best_reward:
                best_reward = best_row['mean_reward']
                best_method = method

    return best_method, best_reward


def load_model(method):
    """Load the best model for the given method."""
    if method == 'dqn':
        from stable_baselines3 import DQN
        model_path = os.path.join(PROJECT_ROOT, 'models', 'dqn', 'dqn_best.zip')
        return DQN.load(model_path), 'sb3'
    elif method == 'ppo':
        from stable_baselines3 import PPO
        model_path = os.path.join(PROJECT_ROOT, 'models', 'pg', 'ppo_best.zip')
        return PPO.load(model_path), 'sb3'
    elif method == 'reinforce':
        import torch
        import pandas as pd_inner
        from training.reinforce_training import PolicyNetwork
        model_path = os.path.join(PROJECT_ROOT, 'models', 'pg', 'reinforce_best.pt')
        reinforce_results = pd_inner.read_csv(os.path.join(PROJECT_ROOT, 'results', 'reinforce_results.csv'))
        best_row = reinforce_results.loc[reinforce_results['mean_reward'].idxmax()]
        hidden_size = int(best_row['hidden_size'])
        policy = PolicyNetwork(obs_dim=13, action_dim=45, hidden_size=hidden_size)
        policy.load_state_dict(torch.load(model_path, weights_only=True))
        policy.eval()
        return policy, 'torch'
    else:
        raise ValueError(f"Unknown method: {method}")


def predict_action(model, model_type, obs):
    """Get action from model."""
    if model_type == 'sb3':
        action, _ = model.predict(obs, deterministic=True)
        return int(action)
    else:
        import torch
        state_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            probs = model(state_t)
        return torch.argmax(probs, dim=-1).item()


def main():
    print("=" * 70)
    print("Brainiacs AI - Adaptive Exercise Sequencer")
    print("Loading best trained model...")
    print("=" * 70)

    best_method, best_reward = find_best_model()

    if best_method is None:
        print("\nNo trained models found! Please run training first:")
        print("  python training/dqn_training.py")
        print("  python training/reinforce_training.py")
        print("  python training/ppo_training.py")
        sys.exit(1)

    print(f"\nBest method: {best_method.upper()} (mean reward: {best_reward:.2f})")

    model, model_type = load_model(best_method)
    print(f"Model loaded successfully.")

    env = gymnasium.make('BrainiacsTutor-v0')
    renderer = TutorRenderer()

    print("\nStarting tutoring session...")
    print("Close the pygame window to exit.\n")

    obs, info = env.reset(seed=0)
    renderer.episodes_completed = 0
    done = False

    print(f"Initial knowledge: {[f'{v:.2f}' for v in info['student_knowledge'].values()]}")
    print(f"Initial engagement: {info['engagement']:.2f}\n")

    while not done:
        action = predict_action(model, model_type, obs)

        topic_idx = action // 9
        difficulty = (action % 9) // 3
        support = action % 3

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        correct_str = "Yes" if info['last_correct'] else "No"
        knowledge_vals = [f"{v:.2f}" for v in info['student_knowledge'].values()]
        print(f"Step {info['step_count']:3d} | "
              f"Action: Topic={TOPIC_LABELS[topic_idx]}, "
              f"Diff={DIFFICULTIES[difficulty]}, "
              f"Support={SUPPORTS[support]} | "
              f"Correct: {correct_str} | "
              f"Knowledge: {knowledge_vals} | "
              f"Engagement: {info['engagement']:.2f} | "
              f"Reward: {reward:.1f}")

        running = renderer.render(info)
        if not running:
            break

    # Print summary
    knowledge = info['student_knowledge']
    if all(v >= 0.8 for v in knowledge.values()):
        outcome = "MASTERY ACHIEVED"
    elif info['engagement'] <= 0.0:
        outcome = "STUDENT DROPPED OUT"
    else:
        outcome = "SESSION TIMEOUT"

    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    print(f"  Method: {best_method.upper()}")
    print(f"  Outcome: {outcome}")
    print(f"  Steps: {info['step_count']}")
    print(f"  Total Reward: {info['total_reward']:.1f}")
    print(f"  Final Knowledge:")
    for topic, val in knowledge.items():
        idx = TOPICS.index(topic)
        status = "MASTERED" if val >= 0.8 else "in progress"
        print(f"    {TOPIC_LABELS[idx]:20s}: {val:.3f} ({status})")
    print(f"  Final Engagement: {info['engagement']:.2f}")
    print("=" * 70)

    # Keep window open until closed
    import pygame
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False

    renderer.close()
    env.close()


if __name__ == '__main__':
    main()

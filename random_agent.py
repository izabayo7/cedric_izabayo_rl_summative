"""
Random Agent Demo - Shows the Brainiacs AI Tutor environment
with random actions and pygame visualization. No trained model needed.
"""

import sys
import time
import pygame
import gymnasium
import environment  # registers BrainiacsTutor-v0
from environment.rendering import TutorRenderer

TOPICS = ['variable_scope', 'off_by_one', 'type_mismatch', 'loop_logic', 'function_params']
TOPIC_LABELS = ['Variable Scope', 'Off-by-One', 'Type Mismatch', 'Loop Logic', 'Function Params']
DIFFICULTIES = ['Easy', 'Medium', 'Hard']
SUPPORTS = ['No Support', 'Hint', 'Worked Example']

MAX_EPISODES = 5  # Run a fixed number of episodes then stop


def main():
    env = gymnasium.make('BrainiacsTutor-v0')
    renderer = TutorRenderer()

    print("=" * 70)
    print("Brainiacs AI - Random Agent Demo")
    print(f"Running {MAX_EPISODES} episodes with random actions.")
    print("Close the pygame window to exit early.")
    print("=" * 70)

    running = True

    for episode in range(1, MAX_EPISODES + 1):
        if not running:
            break

        obs, info = env.reset()
        renderer.episodes_completed = episode - 1
        done = False

        print(f"\n--- Episode {episode}/{MAX_EPISODES} ---")
        print(f"Initial knowledge: {[f'{v:.2f}' for v in info['student_knowledge'].values()]}")
        print(f"Initial engagement: {info['engagement']:.2f}")

        while not done and running:
            action = env.action_space.sample()

            topic_idx = action // 9
            difficulty = (action % 9) // 3
            support = action % 3

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Print step info
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

            # Render
            running = renderer.render(info)

        if running:
            renderer.episodes_completed = episode

            # Print episode summary
            knowledge = info['student_knowledge']
            if all(v >= 0.8 for v in knowledge.values()):
                outcome = "MASTERY ACHIEVED"
            elif info['engagement'] <= 0.0:
                outcome = "STUDENT DROPPED OUT"
            else:
                outcome = "SESSION TIMEOUT"

            print(f"\nEpisode {episode} Summary:")
            print(f"  Outcome: {outcome}")
            print(f"  Steps: {info['step_count']}")
            print(f"  Total Reward: {info['total_reward']:.1f}")
            print(f"  Final Knowledge: {[f'{v:.2f}' for v in knowledge.values()]}")
            print(f"  Final Engagement: {info['engagement']:.2f}")

            # Pause between episodes so user can see the result
            if episode < MAX_EPISODES:
                print(f"\n  Starting next episode in 3 seconds...")
                pause_start = time.time()
                while time.time() - pause_start < 3.0:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            break
                    if not running:
                        break
                    time.sleep(0.05)

    # Final message
    print("\n" + "=" * 70)
    print(f"Random Agent Demo Complete - {renderer.episodes_completed} episodes finished.")
    print("Close the pygame window to exit.")
    print("=" * 70)

    # Keep window open until user closes it
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
        time.sleep(0.05)

    renderer.close()
    env.close()
    print("\nDemo ended.")


if __name__ == '__main__':
    main()

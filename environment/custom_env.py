"""
Brainiacs AI Adaptive Exercise Sequencer - Custom Gymnasium Environment

The agent is an AI tutor deciding what coding exercise to give a simulated
beginner programmer. The student has internal knowledge levels across 5 topics
that change based on the exercises given.
"""

import gymnasium
from gymnasium import spaces
import numpy as np


TOPICS = ['variable_scope', 'off_by_one', 'type_mismatch', 'loop_logic', 'function_params']
DIFFICULTIES = ['easy', 'medium', 'hard']
SUPPORTS = ['no_support', 'hint', 'worked_example']


class BrainiacsTutorEnv(gymnasium.Env):
    """Custom environment for an AI tutor sequencing coding exercises."""

    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None, max_steps=200):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps

        # 13-dimensional continuous observation space (all normalized 0-1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(13,), dtype=np.float32
        )

        # 45 discrete actions: 5 topics x 3 difficulties x 3 support levels
        self.action_space = spaces.Discrete(45)

        self.student_knowledge = None
        self.engagement = None
        self.consecutive_correct = 0
        self.consecutive_incorrect = 0
        self.last_topic = 0
        self.last_difficulty = 0
        self.last_support = 0
        self.last_correct = False
        self.step_count = 0
        self.total_reward = 0.0
        self.topic_counts = None

    def _get_obs(self):
        knowledge_values = [self.student_knowledge[t] for t in TOPICS]
        avg_knowledge = np.mean(knowledge_values)
        knowledge_variance = np.var(knowledge_values)

        obs = np.array([
            self.student_knowledge['variable_scope'],
            self.student_knowledge['off_by_one'],
            self.student_knowledge['type_mismatch'],
            self.student_knowledge['loop_logic'],
            self.student_knowledge['function_params'],
            self.engagement,
            min(self.consecutive_correct / 10.0, 1.0),
            min(self.consecutive_incorrect / 10.0, 1.0),
            self.last_topic / 4.0,
            self.last_difficulty / 2.0,
            self.step_count / self.max_steps,
            avg_knowledge,
            knowledge_variance,
        ], dtype=np.float32)

        return np.clip(obs, 0.0, 1.0)

    def _get_info(self):
        return {
            'student_knowledge': dict(self.student_knowledge),
            'engagement': self.engagement,
            'last_topic': self.last_topic,
            'last_difficulty': self.last_difficulty,
            'last_support': self.last_support,
            'last_correct': self.last_correct,
            'step_count': self.step_count,
            'total_reward': self.total_reward,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Beginner student: knowledge between 0.05 and 0.25
        self.student_knowledge = {
            t: self.np_random.uniform(0.05, 0.25) for t in TOPICS
        }
        # Engagement between 0.7 and 0.9
        self.engagement = self.np_random.uniform(0.7, 0.9)
        self.consecutive_correct = 0
        self.consecutive_incorrect = 0
        self.last_topic = 0
        self.last_difficulty = 0
        self.last_support = 0
        self.last_correct = False
        self.step_count = 0
        self.total_reward = 0.0
        self.topic_counts = {t: 0 for t in TOPICS}

        return self._get_obs(), self._get_info()

    def step(self, action):
        # Decode action
        topic_idx = action // 9
        difficulty = (action % 9) // 3
        support = action % 3

        topic = TOPICS[topic_idx]
        self.last_topic = topic_idx
        self.last_difficulty = difficulty
        self.last_support = support
        self.topic_counts[topic] += 1

        # --- Simulated Student Model ---
        # Probability student gets the exercise correct
        # Base probability from knowledge, with a floor so beginners can still succeed
        base_prob = self.student_knowledge[topic]
        difficulty_modifier = {0: 1.5, 1: 1.0, 2: 0.5}[difficulty]
        # Support helps more: hints and worked examples make a real difference
        support_modifier = {0: 1.0, 1: 1.4, 2: 1.8}[support]
        # Beginners with easy+worked_example: 0.1*1.5*1.8=0.27, clipped min 0.20
        prob_correct = np.clip(
            base_prob * difficulty_modifier * support_modifier, 0.20, 0.95
        )

        correct = self.np_random.random() < prob_correct
        self.last_correct = correct

        if correct:
            # Learning happens — harder exercises teach more, support teaches less
            knowledge_gain = {0: 0.03, 1: 0.06, 2: 0.10}[difficulty]
            support_penalty = {0: 1.0, 1: 0.7, 2: 0.5}[support]
            self.student_knowledge[topic] += knowledge_gain * support_penalty
            self.student_knowledge[topic] = min(self.student_knowledge[topic], 1.0)
            self.consecutive_correct += 1
            self.consecutive_incorrect = 0
            # Success boosts engagement — more for streaks
            self.engagement += 0.02 + 0.005 * min(self.consecutive_correct, 5)
        else:
            # Frustration depends on how hard the exercise was relative to knowledge
            difficulty_level_value = {0: 0.3, 1: 0.6, 2: 0.9}[difficulty]
            gap = difficulty_level_value - self.student_knowledge[topic]
            # Support reduces frustration even on failure (student understands why)
            support_frustration_reduction = {0: 1.0, 1: 0.6, 2: 0.3}[support]
            frustration = max(0, gap) * 0.10 * support_frustration_reduction
            self.engagement -= frustration
            self.consecutive_incorrect += 1
            self.consecutive_correct = 0

        # Engagement natural decay (student gets tired over time)
        self.engagement -= 0.003
        self.engagement = np.clip(self.engagement, 0.0, 1.0)

        # Add small noise to student behavior (students aren't perfectly predictable)
        for t in TOPICS:
            self.student_knowledge[t] += self.np_random.normal(0, 0.005)
            self.student_knowledge[t] = np.clip(self.student_knowledge[t], 0.0, 1.0)

        # --- Reward calculation ---
        reward = 0.0
        difficulty_threshold = {0: 0.3, 1: 0.6, 2: 0.9}[difficulty]
        knowledge_values = list(self.student_knowledge.values())
        avg_knowledge = np.mean(knowledge_values)
        min_knowledge = min(knowledge_values)
        knowledge_variance = np.var(knowledge_values)

        if correct:
            if self.student_knowledge[topic] < 0.5:
                reward += 3.0  # Reward learning in weak areas more
            else:
                reward += 1.0

            # Zone of proximal development bonus
            if abs(self.student_knowledge[topic] - difficulty_threshold) < 0.3:
                reward += 1.5

            # Penalty for trivially easy exercises (no learning value)
            if difficulty == 0 and self.student_knowledge[topic] > 0.7:
                reward -= 1.0
        else:
            reward -= 0.5
            if difficulty == 2 and self.student_knowledge[topic] < 0.3:
                reward -= 2.0  # Gave a hard exercise to a clueless student

        # --- Balanced teaching incentives ---

        # Bonus for teaching the weakest topic
        weakest_topic = TOPICS[np.argmin(knowledge_values)]
        if topic == weakest_topic:
            reward += 2.0

        # Penalty for over-drilling one topic (>40% of exercises)
        total_actions = max(sum(self.topic_counts.values()), 1)
        topic_fraction = self.topic_counts[topic] / total_actions
        if topic_fraction > 0.4:
            reward -= 3.0

        # Variance penalty — penalize imbalanced knowledge
        reward -= knowledge_variance * 5.0

        # Reward raising the weakest topic (encourages balanced progress)
        reward += min_knowledge * 2.0

        # Engagement reward — keep student motivated
        reward += self.engagement * 0.5

        # Big penalty for student dropout
        if self.engagement <= 0.0:
            reward -= 10.0

        # Progressive mastery bonuses
        topics_above_05 = sum(1 for v in knowledge_values if v >= 0.5)
        topics_above_08 = sum(1 for v in knowledge_values if v >= 0.8)
        reward += topics_above_05 * 0.5
        reward += topics_above_08 * 1.5

        # Big bonus for full mastery (all topics >= 0.8)
        if all(v >= 0.8 for v in knowledge_values):
            reward += 20.0

        self.step_count += 1
        self.total_reward += reward

        # Terminal conditions
        terminated = all(v >= 0.8 for v in self.student_knowledge.values())
        truncated = self.engagement <= 0.0 or self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

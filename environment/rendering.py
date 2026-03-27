"""
Pygame visualization for the Brainiacs AI Tutor environment.
"""

import pygame
import math

TOPICS = ['variable_scope', 'off_by_one', 'type_mismatch', 'loop_logic', 'function_params']
TOPIC_LABELS = ['Variable Scope', 'Off-by-One', 'Type Mismatch', 'Loop Logic', 'Function Params']
DIFFICULTIES = ['Easy', 'Medium', 'Hard']
SUPPORTS = ['No Support', 'Hint', 'Worked Example']

# Color palette
BG_COLOR = (26, 26, 46)
PANEL_COLOR = (36, 36, 66)
PANEL_BORDER = (60, 60, 100)
TEXT_COLOR = (220, 220, 240)
TEXT_DIM = (140, 140, 170)
BAR_BG = (50, 50, 80)
RED = (231, 76, 60)
YELLOW = (241, 196, 15)
GREEN = (46, 204, 113)
BLUE = (52, 152, 219)
ORANGE = (230, 126, 34)
WHITE = (255, 255, 255)
DARK_GREEN = (39, 174, 96)
DARK_RED = (192, 57, 43)

DIFF_COLORS = {0: GREEN, 1: YELLOW, 2: RED}
SUPPORT_COLORS = {0: TEXT_DIM, 1: BLUE, 2: ORANGE}


def knowledge_color(value):
    if value < 0.3:
        return RED
    elif value < 0.6:
        return YELLOW
    else:
        return GREEN


class TutorRenderer:
    """Pygame renderer for the Brainiacs Tutor environment."""

    def __init__(self):
        pygame.init()
        self.width = 900
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Brainiacs AI - Adaptive Exercise Sequencer")
        self.clock = pygame.time.Clock()

        self.font_large = pygame.font.SysFont('arial', 24, bold=True)
        self.font_medium = pygame.font.SysFont('arial', 18)
        self.font_small = pygame.font.SysFont('arial', 14)
        self.font_icon = pygame.font.SysFont('arial', 28, bold=True)

        self.episodes_completed = 0

    def _draw_panel(self, x, y, w, h, title=None):
        rect = pygame.Rect(x, y, w, h)
        pygame.draw.rect(self.screen, PANEL_COLOR, rect, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, width=1, border_radius=8)
        if title:
            title_surf = self.font_medium.render(title, True, TEXT_DIM)
            self.screen.blit(title_surf, (x + 12, y + 8))

    def _draw_knowledge_bars(self, knowledge):
        self._draw_panel(20, 60, 420, 280, "Student Knowledge")
        bar_x = 40
        bar_w = 250
        bar_h = 28

        for i, topic in enumerate(TOPICS):
            y = 100 + i * 48
            value = knowledge.get(topic, 0.0)

            # Topic label
            label = self.font_small.render(TOPIC_LABELS[i], True, TEXT_COLOR)
            self.screen.blit(label, (bar_x, y - 2))

            # Background bar
            bg_rect = pygame.Rect(bar_x + 140, y, bar_w, bar_h)
            pygame.draw.rect(self.screen, BAR_BG, bg_rect, border_radius=4)

            # Filled bar
            fill_w = int(bar_w * value)
            if fill_w > 0:
                fill_rect = pygame.Rect(bar_x + 140, y, fill_w, bar_h)
                color = knowledge_color(value)
                pygame.draw.rect(self.screen, color, fill_rect, border_radius=4)

            # Value text
            val_text = self.font_small.render(f"{value:.2f}", True, WHITE)
            self.screen.blit(val_text, (bar_x + 140 + bar_w + 10, y + 4))

    def _draw_engagement_gauge(self, engagement):
        self._draw_panel(460, 60, 200, 180, "Engagement")
        cx = 560
        cy = 175
        radius = 60

        # Draw arc background
        start_angle = math.pi
        end_angle = 2 * math.pi
        steps = 40
        for i in range(steps):
            angle = start_angle + (end_angle - start_angle) * i / steps
            x1 = cx + radius * math.cos(angle)
            y1 = cy + radius * math.sin(angle)
            pygame.draw.circle(self.screen, BAR_BG, (int(x1), int(y1)), 5)

        # Draw filled arc
        filled_steps = int(steps * engagement)
        for i in range(filled_steps):
            angle = start_angle + (end_angle - start_angle) * i / steps
            x1 = cx + radius * math.cos(angle)
            y1 = cy + radius * math.sin(angle)
            frac = i / steps
            if frac < 0.3:
                color = RED
            elif frac < 0.6:
                color = YELLOW
            else:
                color = GREEN
            pygame.draw.circle(self.screen, color, (int(x1), int(y1)), 5)

        # Needle
        needle_angle = start_angle + (end_angle - start_angle) * engagement
        nx = cx + (radius - 15) * math.cos(needle_angle)
        ny = cy + (radius - 15) * math.sin(needle_angle)
        pygame.draw.line(self.screen, WHITE, (cx, cy), (int(nx), int(ny)), 2)
        pygame.draw.circle(self.screen, WHITE, (cx, cy), 4)

        # Value text
        val_text = self.font_large.render(f"{engagement:.2f}", True, TEXT_COLOR)
        val_rect = val_text.get_rect(center=(cx, cy + 25))
        self.screen.blit(val_text, val_rect)

    def _draw_exercise_info(self, info):
        self._draw_panel(460, 260, 420, 170, "Current Exercise")

        topic_idx = info.get('last_topic', 0)
        difficulty = info.get('last_difficulty', 0)
        support = info.get('last_support', 0)
        correct = info.get('last_correct', False)
        step = info.get('step_count', 0)

        x = 480
        y = 295

        # Topic
        topic_text = self.font_medium.render(f"Topic: {TOPIC_LABELS[topic_idx]}", True, TEXT_COLOR)
        self.screen.blit(topic_text, (x, y))

        # Difficulty
        diff_text = self.font_medium.render(f"Difficulty: ", True, TEXT_COLOR)
        self.screen.blit(diff_text, (x, y + 30))
        diff_val = self.font_medium.render(DIFFICULTIES[difficulty], True, DIFF_COLORS[difficulty])
        self.screen.blit(diff_val, (x + diff_text.get_width(), y + 30))

        # Support
        sup_text = self.font_medium.render(f"Support: ", True, TEXT_COLOR)
        self.screen.blit(sup_text, (x, y + 60))
        sup_val = self.font_medium.render(SUPPORTS[support], True, SUPPORT_COLORS[support])
        self.screen.blit(sup_val, (x + sup_text.get_width(), y + 60))

        # Result (only show after first step)
        if step > 0:
            if correct:
                result_text = self.font_icon.render("✓ Correct", True, GREEN)
            else:
                result_text = self.font_icon.render("✗ Incorrect", True, RED)
            self.screen.blit(result_text, (x, y + 95))

    def _draw_session_stats(self, info):
        self._draw_panel(680, 60, 200, 180, "Session Stats")
        x = 700
        y = 95

        step = info.get('step_count', 0)
        total_reward = info.get('total_reward', 0.0)

        step_text = self.font_medium.render(f"Step: {step}", True, TEXT_COLOR)
        self.screen.blit(step_text, (x, y))

        reward_color = GREEN if total_reward > 0 else RED
        reward_text = self.font_medium.render(f"Reward: {total_reward:.1f}", True, reward_color)
        self.screen.blit(reward_text, (x, y + 30))

        ep_text = self.font_medium.render(f"Episodes: {self.episodes_completed}", True, TEXT_COLOR)
        self.screen.blit(ep_text, (x, y + 60))

        # Mastery check
        knowledge = info.get('student_knowledge', {})
        if knowledge:
            mastered = sum(1 for v in knowledge.values() if v >= 0.8)
            mastery_text = self.font_medium.render(f"Mastered: {mastered}/5", True, TEXT_COLOR)
            self.screen.blit(mastery_text, (x, y + 90))

    def _draw_progress_bar(self, info):
        self._draw_panel(20, 540, 860, 45, None)
        progress = info.get('step_count', 0) / 200.0

        # Label
        label = self.font_small.render("Session Progress", True, TEXT_DIM)
        self.screen.blit(label, (30, 548))

        # Bar
        bar_x = 170
        bar_y = 552
        bar_w = 680
        bar_h = 20

        bg_rect = pygame.Rect(bar_x, bar_y, bar_w, bar_h)
        pygame.draw.rect(self.screen, BAR_BG, bg_rect, border_radius=4)

        fill_w = int(bar_w * min(progress, 1.0))
        if fill_w > 0:
            fill_rect = pygame.Rect(bar_x, bar_y, fill_w, bar_h)
            pygame.draw.rect(self.screen, BLUE, fill_rect, border_radius=4)

        pct_text = self.font_small.render(f"{progress * 100:.0f}%", True, WHITE)
        self.screen.blit(pct_text, (bar_x + bar_w + 8, bar_y + 2))

    def _draw_title(self):
        title = self.font_large.render("Brainiacs AI — Adaptive Exercise Sequencer", True, WHITE)
        self.screen.blit(title, (20, 18))

    def _draw_knowledge_summary(self, info):
        """Draw a compact knowledge summary below knowledge bars."""
        self._draw_panel(20, 360, 420, 160, "Learning Analytics")
        knowledge = info.get('student_knowledge', {})
        if not knowledge:
            return

        x = 40
        y = 395

        avg_k = sum(knowledge.values()) / len(knowledge)
        var_k = sum((v - avg_k) ** 2 for v in knowledge.values()) / len(knowledge)

        avg_text = self.font_medium.render(f"Avg Knowledge: {avg_k:.3f}", True, TEXT_COLOR)
        self.screen.blit(avg_text, (x, y))

        var_text = self.font_medium.render(f"Knowledge Variance: {var_k:.4f}", True, TEXT_COLOR)
        self.screen.blit(var_text, (x, y + 28))

        # Weakest and strongest
        weakest = min(knowledge, key=knowledge.get)
        strongest = max(knowledge, key=knowledge.get)
        w_idx = TOPICS.index(weakest)
        s_idx = TOPICS.index(strongest)

        weak_text = self.font_medium.render(f"Weakest: {TOPIC_LABELS[w_idx]} ({knowledge[weakest]:.2f})", True, ORANGE)
        self.screen.blit(weak_text, (x, y + 64))

        strong_text = self.font_medium.render(f"Strongest: {TOPIC_LABELS[s_idx]} ({knowledge[strongest]:.2f})", True, GREEN)
        self.screen.blit(strong_text, (x, y + 92))

    def render(self, info):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False

        self.screen.fill(BG_COLOR)

        self._draw_title()

        knowledge = info.get('student_knowledge', {})
        engagement = info.get('engagement', 0.5)

        self._draw_knowledge_bars(knowledge)
        self._draw_engagement_gauge(engagement)
        self._draw_exercise_info(info)
        self._draw_session_stats(info)
        self._draw_knowledge_summary(info)
        self._draw_progress_bar(info)

        pygame.display.flip()
        self.clock.tick(10)
        return True

    def close(self):
        pygame.quit()

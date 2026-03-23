from environment.custom_env import BrainiacsTutorEnv
from environment.rendering import TutorRenderer

import gymnasium
gymnasium.register(
    id='BrainiacsTutor-v0',
    entry_point='environment.custom_env:BrainiacsTutorEnv',
)

# Brainiacs AI: Adaptive Exercise Sequencer — RL Summative Assignment

**Student:** Cedric Izabayo

## Description

A reinforcement learning project where an AI tutor agent learns to optimally sequence coding exercises for a simulated beginner programmer. The agent decides what **topic**, **difficulty**, and **support level** to present next — maximizing the student's learning while preventing frustration and dropout.

This project is tied to the capstone project **"Brainiacs AI: An Adaptive AI Coding Tutor for Beginner Programmers."**

## Installation

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run Random Agent (no trained model needed)
```bash
python random_agent.py
```
Demonstrates the pygame visualization with an agent taking random actions.

### Train Models
```bash
python training/dqn_training.py
python training/reinforce_training.py
python training/ppo_training.py
```
Each script runs 10 hyperparameter configurations and saves the best model.

### Run Best Trained Model
```bash
python main.py
```
Loads the best model across all methods and runs it with pygame visualization.

### Generate Plots
```bash
python plot_results.py
```
Generates all comparison plots and saves them to `plots/`.

## Environment

### Observation Space (13 continuous dimensions, normalized 0-1)
| Index | Feature | Description |
|-------|---------|-------------|
| 0-4 | Knowledge levels | Mastery of 5 coding topics (variable scope, off-by-one errors, type mismatch, loop logic, function parameters) |
| 5 | Engagement | Student motivation level |
| 6 | Consecutive correct | Normalized streak of correct answers |
| 7 | Consecutive incorrect | Normalized streak of incorrect answers |
| 8 | Last topic | Previous topic given (normalized) |
| 9 | Last difficulty | Previous difficulty (normalized) |
| 10 | Session progress | Current step / max steps |
| 11 | Avg knowledge | Mean of all 5 knowledge values |
| 12 | Knowledge variance | Variance across topics |

### Action Space (45 discrete actions)
Combination of:
- **Topic** (5): variable_scope, off_by_one, type_mismatch, loop_logic, function_params
- **Difficulty** (3): easy, medium, hard
- **Support** (3): no_support, hint, worked_example

### Rewards
- Correct answer on weak topic: +3.0
- Correct answer on strong topic: +1.0
- Zone of proximal development bonus: +1.5
- Incorrect answer: -0.5
- Hard exercise for low-knowledge student: -2.0
- Imbalanced teaching penalty: -variance * 2.0
- Student dropout: -10.0
- Full mastery achieved: +20.0

### Terminal Conditions
- **Mastery**: All 5 topics >= 0.8
- **Dropout**: Engagement drops to 0
- **Timeout**: 200 steps reached

## Algorithms Implemented
1. **DQN** (Deep Q-Network) — via stable-baselines3
2. **REINFORCE** (Monte Carlo Policy Gradient) — manual PyTorch implementation
3. **PPO** (Proximal Policy Optimization) — via stable-baselines3

## Project Structure
```
cedric_izabayo_rl_summative/
├── environment/
│   ├── __init__.py
│   ├── custom_env.py          # Custom Gymnasium environment
│   └── rendering.py           # Pygame visualization
├── training/
│   ├── __init__.py
│   ├── dqn_training.py        # DQN training with 10 hyperparameter runs
│   ├── reinforce_training.py  # REINFORCE training with 10 hyperparameter runs
│   └── ppo_training.py        # PPO training with 10 hyperparameter runs
├── models/
│   ├── dqn/                   # Saved DQN models
│   └── pg/                    # Saved policy gradient models
├── main.py                    # Entry point: loads best model, runs pygame visualization
├── random_agent.py            # Shows agent taking RANDOM actions in pygame (no model)
├── plot_results.py            # Generates all required plots from training logs
├── requirements.txt
└── README.md
```

# Laser Maze RL Module

Reinforcement Learning training for the Laser Maze puzzle game.

## Quick Start

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Test the environment
python rl/test_env.py

# Train with default settings (curriculum learning, 1M steps)
python -m rl.train

# Train with specific settings
python -m rl.train --timesteps 500000 --num-pieces 2 --no-curriculum

# Monitor training with TensorBoard
tensorboard --logdir logs
```

## Architecture

### Environment (`rl/env.py`)

`LaserMazeEnv` implements the Gymnasium interface:

- **Observation Space**: `Box((13, 5, 5), float32)`
  - Channels 0-7: One-hot piece type
  - Channels 8-11: One-hot orientation
  - Channel 12: Fixed piece mask

- **Action Space**: `Discrete(476)` for 5x5 board
  - Action 0: Fire laser
  - Actions 1-25: Rotate piece at cell
  - Actions 26-50: Remove piece from cell
  - Remaining: Place actions (piece_type, cell, orientation)

- **Rewards**:
  - Step penalty: -0.01
  - Invalid action: -0.1
  - New target hit: +0.2
  - Checkpoint passed: +0.1
  - Puzzle solved: +1.0
  - Timeout: -0.3

### Vectorized Environments (`rl/vec_env.py`)

- `make_vec_env()`: Create parallel environments for faster training
- `CurriculumVecEnv`: Automatic difficulty progression

### Networks (`rl/networks.py`)

Three feature extractors optimized for the 5x5 grid:

1. **CNN** (default): Small convolutional network, good balance
2. **MLP**: Simple fully-connected, fastest inference
3. **Attention**: Transformer-based, captures piece relationships

### Training (`rl/train.py`)

PPO training with:
- Curriculum learning (1 piece → 4 pieces)
- TensorBoard logging
- Checkpoint saving
- Evaluation callbacks

## Configuration

Key hyperparameters in `train.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_envs` | 8 | Parallel environments |
| `total_timesteps` | 1M | Training duration |
| `learning_rate` | 3e-4 | Adam learning rate |
| `n_steps` | 2048 | Steps per rollout |
| `batch_size` | 64 | Minibatch size |
| `ent_coef` | 0.01 | Entropy bonus |
| `use_curriculum` | True | Progressive difficulty |

## Performance Tips

1. **GPU**: Update NVIDIA drivers for CUDA support
2. **Batch size**: Increase if GPU memory allows
3. **n_envs**: More environments = faster data collection
4. **Curriculum**: Start simple, let agent master basics first

## File Structure

```
rl/
├── __init__.py       # Module exports
├── env.py            # Gymnasium environment
├── vec_env.py        # Vectorized environments
├── networks.py       # Custom neural networks
├── train.py          # Training script
├── test_env.py       # Environment tests
└── README.md         # This file
```

## Using Trained Models

```python
from stable_baselines3 import PPO
from rl.env import LaserMazeEnv

# Load trained model
model = PPO.load("logs/ppo_TIMESTAMP/best/best_model")

# Create environment
env = LaserMazeEnv(puzzle_path="examples/beginner_1.json")
obs, info = env.reset()

# Run inference
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()

print(f"Solved: {info.get('solved', False)}")
```

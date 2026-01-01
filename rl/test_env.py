"""
Test script to verify the RL environment works correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def test_env_creation():
    """Test basic environment creation."""
    from rl.env import LaserMazeEnv

    env = LaserMazeEnv(random_puzzles=True, puzzle_difficulty="beginner")

    # Test reset
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Info: {info}")

    assert obs.shape == (13, 5, 5), f"Expected (13, 5, 5), got {obs.shape}"
    print("Environment creation: OK\n")


def test_action_space():
    """Test action encoding/decoding."""
    from rl.env import LaserMazeEnv

    env = LaserMazeEnv(random_puzzles=True)
    env.reset()

    print(f"Action space size: {env.action_space.n}")

    # Test action mask
    mask = env.get_action_mask()
    valid_count = mask.sum()
    print(f"Valid actions: {valid_count}")

    # Test step with random valid action
    valid_actions = np.where(mask == 1)[0]
    if len(valid_actions) > 0:
        action = valid_actions[0]
        obs, reward, term, trunc, info = env.step(action)
        print(f"Step result: reward={reward:.3f}, terminated={term}, truncated={trunc}")

    print("Action space: OK\n")


def test_episode_rollout():
    """Test running a full episode."""
    from rl.env import LaserMazeEnv

    env = LaserMazeEnv(
        random_puzzles=True,
        puzzle_difficulty="beginner",
        max_steps=50,
        dense_rewards=True,
    )

    obs, info = env.reset()
    total_reward = 0
    steps = 0

    print("Running episode...")
    while True:
        mask = env.get_action_mask()
        valid_actions = np.where(mask == 1)[0]

        if len(valid_actions) == 0:
            print("No valid actions!")
            break

        # Random valid action
        action = np.random.choice(valid_actions)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    print(f"Episode finished: steps={steps}, total_reward={total_reward:.3f}")
    print(f"Solved: {info.get('solved', False)}")
    print("Episode rollout: OK\n")


def test_vectorized_env():
    """Test vectorized environment."""
    from rl.vec_env import make_vec_env

    print("Creating vectorized environment...")
    vec_env = make_vec_env(
        n_envs=4,
        num_pieces=1,
        random_puzzles=True,
    )

    obs = vec_env.reset()
    print(f"Vectorized obs shape: {obs.shape}")

    # Random actions
    actions = np.random.randint(0, vec_env.action_space.n, size=4)
    obs, rewards, dones, infos = vec_env.step(actions)

    print(f"Rewards: {rewards}")
    print(f"Dones: {dones}")

    vec_env.close()
    print("Vectorized environment: OK\n")


def test_render():
    """Test environment rendering."""
    from rl.env import LaserMazeEnv

    env = LaserMazeEnv(
        random_puzzles=True,
        puzzle_difficulty="beginner",
        render_mode="ansi"
    )

    obs, info = env.reset()
    print("Board state:")
    output = env.render()
    print(output)

    print("Render: OK\n")


def test_ppo_integration():
    """Test PPO can be instantiated with our environment."""
    from stable_baselines3 import PPO
    from rl.env import LaserMazeEnv
    from rl.networks import LaserMazeCNN

    print("Creating PPO with custom network...")

    env = LaserMazeEnv(random_puzzles=True, puzzle_difficulty="beginner")

    policy_kwargs = {
        "features_extractor_class": LaserMazeCNN,
        "features_extractor_kwargs": {"features_dim": 64},
    }

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        n_steps=64,
        batch_size=32,
    )

    # Quick training test
    print("Running quick training (256 steps)...")
    model.learn(total_timesteps=256)

    print("PPO integration: OK\n")


def test_with_specific_puzzle():
    """Test with a specific puzzle file."""
    from rl.env import LaserMazeEnv

    puzzle_path = Path(__file__).parent.parent / "examples" / "beginner_1.json"

    if puzzle_path.exists():
        print(f"Testing with {puzzle_path}")
        env = LaserMazeEnv(puzzle_path=str(puzzle_path))

        obs, info = env.reset()
        print(f"Loaded puzzle: {info}")

        # Fire to see initial state
        fire_action = 0  # Fire is always action 0
        obs, reward, term, trunc, info = env.step(fire_action)
        print(f"After fire: targets_hit={info.get('targets_hit', 0)}")

        print("Specific puzzle: OK\n")
    else:
        print(f"Skipping: {puzzle_path} not found\n")


if __name__ == "__main__":
    print("=" * 50)
    print("RL Environment Tests")
    print("=" * 50 + "\n")

    test_env_creation()
    test_action_space()
    test_episode_rollout()
    test_vectorized_env()
    test_render()
    test_with_specific_puzzle()

    try:
        test_ppo_integration()
    except ImportError as e:
        print(f"Skipping PPO test: {e}")

    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)

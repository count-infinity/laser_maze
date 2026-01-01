"""
Evaluate a trained model on specific puzzles or random puzzles.

Usage:
    # Evaluate on a specific puzzle
    python -m rl.evaluate --model logs/ppo_xxx/best/best_model --puzzle examples/expert_47.json

    # Evaluate on random puzzles
    python -m rl.evaluate --model logs/ppo_xxx/best/best_model --num-episodes 100

    # Watch the agent solve puzzles
    python -m rl.evaluate --model logs/ppo_xxx/best/best_model --render
"""

import argparse
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np

from stable_baselines3 import PPO

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.env import LaserMazeEnv


def evaluate_on_puzzle(
    model: PPO,
    puzzle_path: str,
    render: bool = True,
    max_steps: int = 100,
) -> Dict:
    """
    Evaluate model on a specific puzzle.

    Returns:
        Dict with 'solved', 'steps', 'reward', 'actions'
    """
    env = LaserMazeEnv(
        puzzle_path=puzzle_path,
        max_steps=max_steps,
    )

    obs, info = env.reset()

    if render:
        print(f"\n{'='*50}")
        print(f"Puzzle: {puzzle_path}")
        print(f"{'='*50}")
        print("\nInitial board:")
        env.game.show()
        print(f"Pieces to place: {info.get('pieces_remaining', 0)}")
        print()

    actions_taken = []
    total_reward = 0
    step = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        decoded = env.game._decode_action(int(action))

        obs, reward, terminated, truncated, info = env.step(int(action))

        actions_taken.append({
            'step': step,
            'action': decoded,
            'reward': reward,
        })
        total_reward += reward
        step += 1

        if render and decoded.action_type.name != "FIRE":
            print(f"Step {step}: {decoded.action_type.name} ", end="")
            if decoded.piece_type:
                print(f"{decoded.piece_type} at ({decoded.row},{decoded.col}) ori={decoded.orientation}")
            else:
                print(f"at ({decoded.row},{decoded.col})")

        if terminated or truncated:
            break

    if render:
        print(f"\n{'='*50}")
        print("Final board:")
        env.game.show(with_laser=True)
        print(f"\nSolved: {info.get('solved', False)}")
        print(f"Steps: {step}")
        print(f"Total reward: {total_reward:.3f}")

    return {
        'solved': info.get('solved', False),
        'steps': step,
        'reward': total_reward,
        'actions': actions_taken,
        'puzzle': puzzle_path,
    }


def evaluate_random(
    model: PPO,
    num_episodes: int = 100,
    difficulty: str = "beginner",
    max_steps: int = 50,
    render: bool = False,
) -> Dict:
    """
    Evaluate model on random puzzles.

    Returns:
        Dict with aggregated statistics
    """
    env = LaserMazeEnv(
        random_puzzles=True,
        puzzle_difficulty=difficulty,
        max_steps=max_steps,
    )

    results = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        solved = info.get('solved', False)
        results.append({
            'solved': solved,
            'steps': steps,
            'reward': total_reward,
        })

        if render:
            status = "SOLVED" if solved else "FAILED"
            print(f"Episode {ep+1}: {status} in {steps} steps (reward: {total_reward:.2f})")

    # Compute statistics
    solved_count = sum(r['solved'] for r in results)
    avg_steps = np.mean([r['steps'] for r in results])
    avg_reward = np.mean([r['reward'] for r in results])

    stats = {
        'num_episodes': num_episodes,
        'difficulty': difficulty,
        'solved': solved_count,
        'success_rate': solved_count / num_episodes,
        'avg_steps': avg_steps,
        'avg_reward': avg_reward,
    }

    print(f"\n{'='*50}")
    print(f"Evaluation Results ({difficulty})")
    print(f"{'='*50}")
    print(f"Episodes: {num_episodes}")
    print(f"Solved: {solved_count}/{num_episodes} ({stats['success_rate']*100:.1f}%)")
    print(f"Avg steps: {avg_steps:.1f}")
    print(f"Avg reward: {avg_reward:.3f}")

    return stats


def evaluate_puzzle_set(
    model: PPO,
    puzzle_dir: str,
    pattern: str = "*.json",
    render: bool = False,
) -> Dict:
    """
    Evaluate model on all puzzles in a directory.
    """
    puzzle_path = Path(puzzle_dir)
    puzzles = sorted(puzzle_path.glob(pattern))

    if not puzzles:
        print(f"No puzzles found matching {puzzle_dir}/{pattern}")
        return {}

    results = []
    for puzzle in puzzles:
        result = evaluate_on_puzzle(model, str(puzzle), render=render)
        results.append(result)

    solved_count = sum(r['solved'] for r in results)
    print(f"\n{'='*50}")
    print(f"Summary: {puzzle_dir}")
    print(f"{'='*50}")
    print(f"Solved: {solved_count}/{len(results)} ({solved_count/len(results)*100:.1f}%)")

    # Show failed puzzles
    failed = [r['puzzle'] for r in results if not r['solved']]
    if failed:
        print(f"\nFailed puzzles:")
        for p in failed:
            print(f"  - {p}")

    return {
        'puzzles': len(results),
        'solved': solved_count,
        'success_rate': solved_count / len(results) if results else 0,
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Laser Maze model")

    parser.add_argument("--model", type=str, required=True,
                       help="Path to trained model (.zip file)")
    parser.add_argument("--puzzle", type=str, default=None,
                       help="Path to specific puzzle file")
    parser.add_argument("--puzzle-dir", type=str, default=None,
                       help="Directory of puzzles to evaluate")
    parser.add_argument("--num-episodes", type=int, default=100,
                       help="Number of random episodes (if no puzzle specified)")
    parser.add_argument("--difficulty", type=str, default="beginner",
                       choices=["beginner", "easy", "intermediate", "advanced", "expert"],
                       help="Difficulty for random puzzles")
    parser.add_argument("--max-steps", type=int, default=100,
                       help="Maximum steps per episode")
    parser.add_argument("--render", action="store_true",
                       help="Show puzzle solving process")

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    model = PPO.load(args.model)
    print("Model loaded.\n")

    if args.puzzle:
        # Evaluate on specific puzzle
        evaluate_on_puzzle(model, args.puzzle, render=True, max_steps=args.max_steps)

    elif args.puzzle_dir:
        # Evaluate on all puzzles in directory
        evaluate_puzzle_set(model, args.puzzle_dir, render=args.render)

    else:
        # Evaluate on random puzzles
        evaluate_random(
            model,
            num_episodes=args.num_episodes,
            difficulty=args.difficulty,
            max_steps=args.max_steps,
            render=args.render,
        )


if __name__ == "__main__":
    main()

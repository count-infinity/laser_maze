"""
Vectorized environment utilities for parallel training.

Provides efficient parallel simulation of multiple Laser Maze instances.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.env import LaserMazeEnv, MaskedLaserMazeEnv

# Map num_pieces to difficulty
PIECES_TO_DIFFICULTY = {
    1: "beginner",
    2: "intermediate",
    3: "advanced",
    4: "expert",
}


def make_env(
    puzzle_path: Optional[str] = None,
    puzzle_difficulty: str = "beginner",
    num_pieces: int = 1,
    random_puzzles: bool = True,
    max_steps: int = 50,
    dense_rewards: bool = True,
    use_action_mask: bool = False,
    rank: int = 0,
    seed: int = 0,
    log_dir: Optional[str] = None,
) -> Callable[[], LaserMazeEnv]:
    """
    Create a callable that returns a LaserMaze environment.

    Args:
        puzzle_path: Path to specific puzzle file
        puzzle_difficulty: Difficulty level
        num_pieces: Number of pieces to place
        random_puzzles: Generate random puzzles
        max_steps: Max steps per episode
        dense_rewards: Use reward shaping
        use_action_mask: Use MaskedLaserMazeEnv for action masking
        rank: Environment index (for seeding)
        seed: Base random seed
        log_dir: Directory for Monitor logs

    Returns:
        Callable that creates the environment
    """
    def _init() -> LaserMazeEnv:
        env_class = MaskedLaserMazeEnv if use_action_mask else LaserMazeEnv

        # Map num_pieces to appropriate difficulty
        difficulty = puzzle_difficulty
        if puzzle_difficulty == "beginner" and num_pieces > 1:
            difficulty = PIECES_TO_DIFFICULTY.get(num_pieces, puzzle_difficulty)

        env = env_class(
            puzzle_path=puzzle_path,
            puzzle_difficulty=difficulty,
            num_pieces=num_pieces,
            random_puzzles=random_puzzles,
            max_steps=max_steps,
            dense_rewards=dense_rewards,
        )

        # Set seed
        env.reset(seed=seed + rank)

        # Wrap with Monitor for logging
        if log_dir:
            env = Monitor(env, filename=f"{log_dir}/env_{rank}")

        return env

    return _init


def make_vec_env(
    n_envs: int = 4,
    puzzle_path: Optional[str] = None,
    puzzle_difficulty: str = "beginner",
    num_pieces: int = 1,
    random_puzzles: bool = True,
    max_steps: int = 50,
    dense_rewards: bool = True,
    use_action_mask: bool = False,
    seed: int = 0,
    log_dir: Optional[str] = None,
    use_subprocess: bool = False,
) -> DummyVecEnv:
    """
    Create vectorized environment for parallel training.

    Args:
        n_envs: Number of parallel environments
        puzzle_path: Path to specific puzzle file
        puzzle_difficulty: Difficulty level
        num_pieces: Number of pieces to place
        random_puzzles: Generate random puzzles
        max_steps: Max steps per episode
        dense_rewards: Use reward shaping
        use_action_mask: Use action masking
        seed: Base random seed
        log_dir: Directory for logs
        use_subprocess: Use SubprocVecEnv (True) or DummyVecEnv (False)

    Returns:
        Vectorized environment

    Notes:
        - DummyVecEnv: Single-threaded, good for debugging and when env is fast
        - SubprocVecEnv: Multi-process, better for CPU-heavy envs

        For Laser Maze, DummyVecEnv is often faster because:
        1. Game logic is simple and fast
        2. Subprocess overhead can outweigh parallelization benefit
        3. GPU batching handles the real parallelization
    """
    env_fns = [
        make_env(
            puzzle_path=puzzle_path,
            puzzle_difficulty=puzzle_difficulty,
            num_pieces=num_pieces,
            random_puzzles=random_puzzles,
            max_steps=max_steps,
            dense_rewards=dense_rewards,
            use_action_mask=use_action_mask,
            rank=i,
            seed=seed,
            log_dir=log_dir,
        )
        for i in range(n_envs)
    ]

    if use_subprocess:
        return SubprocVecEnv(env_fns)
    else:
        return DummyVecEnv(env_fns)


class CurriculumVecEnv:
    """
    Wrapper for curriculum learning with progressive difficulty.

    Starts with simple puzzles and increases difficulty as agent improves.

    Note: This updates the num_pieces parameter on the underlying environments
    rather than recreating them, so the model's env reference stays valid.
    """

    def __init__(
        self,
        n_envs: int = 4,
        initial_pieces: int = 1,
        max_pieces: int = 4,
        success_threshold: float = 0.7,
        window_size: int = 100,
        min_episodes_per_level: int = 200,
        seed: int = 0,
        log_dir: Optional[str] = None,
        use_action_mask: bool = False,
    ):
        """
        Initialize curriculum environment.

        Args:
            n_envs: Number of parallel environments
            initial_pieces: Starting number of pieces
            max_pieces: Maximum pieces to reach
            success_threshold: Success rate required to increase difficulty
            window_size: Episodes to consider for success rate
            min_episodes_per_level: Minimum episodes before allowing level increase
            seed: Random seed
            log_dir: Log directory
            use_action_mask: Use action masking for MaskablePPO
        """
        self.n_envs = n_envs
        self.current_pieces = initial_pieces
        self.max_pieces = max_pieces
        self.success_threshold = success_threshold
        self.window_size = window_size
        self.min_episodes_per_level = min_episodes_per_level
        self.seed = seed
        self.log_dir = log_dir
        self.use_action_mask = use_action_mask

        # Track success
        self._successes: List[bool] = []
        self._total_episodes = 0
        self._episodes_at_current_level = 0

        # Create initial env
        self._create_env()

    def _create_env(self):
        """Create environment with current difficulty."""
        self.vec_env = make_vec_env(
            n_envs=self.n_envs,
            num_pieces=self.current_pieces,
            random_puzzles=True,
            seed=self.seed,
            log_dir=self.log_dir,
            use_action_mask=self.use_action_mask,
        )

    def _update_difficulty(self):
        """Update difficulty on existing environments without recreating."""
        # Update num_pieces on each underlying environment
        for env in self.vec_env.envs:
            # Handle Monitor wrapper if present
            actual_env = env
            while hasattr(actual_env, 'env'):
                actual_env = actual_env.env
            actual_env.num_pieces = self.current_pieces
            # Map to appropriate difficulty string
            actual_env.puzzle_difficulty = PIECES_TO_DIFFICULTY.get(
                self.current_pieces, "expert"
            )

    @property
    def success_rate(self) -> float:
        """Get current success rate."""
        if len(self._successes) < 10:
            return 0.0
        recent = self._successes[-self.window_size:]
        return sum(recent) / len(recent)

    def record_episode(self, success: bool):
        """Record episode outcome and potentially increase difficulty."""
        self._successes.append(success)
        self._total_episodes += 1
        self._episodes_at_current_level += 1

        # Check if we should increase difficulty
        # Require BOTH:
        # 1. Minimum episodes at current level (prevents instant jumping)
        # 2. Full window of episodes to evaluate
        # 3. Success rate above threshold
        if (
            self.current_pieces < self.max_pieces
            and self._episodes_at_current_level >= self.min_episodes_per_level
            and len(self._successes) >= self.window_size
            and self.success_rate >= self.success_threshold
        ):
            self.current_pieces += 1
            self._successes.clear()  # Reset success tracking for new difficulty
            self._episodes_at_current_level = 0  # Reset episode counter
            self._update_difficulty()  # Update existing envs, don't recreate
            print(f"Curriculum: Increased to {self.current_pieces} pieces "
                  f"(after {self._total_episodes} total episodes)")

    def get_env(self):
        """Get current vectorized environment."""
        return self.vec_env

    @property
    def difficulty_level(self) -> int:
        """Get current difficulty (number of pieces)."""
        return self.current_pieces

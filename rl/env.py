"""
Gymnasium environment wrapper for Laser Maze.

Provides a standard Gym interface for RL training with:
- Efficient state encoding as tensors
- Dense reward shaping for faster learning
- Action masking for valid actions only
- Support for curriculum learning
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from game import LaserMaze, Action, ActionType, PIECE_ORIENTATIONS
from pieces import PIECE_CLASSES
from puzzle_generator import PuzzleGenerator, Difficulty
from file_io import board_from_dict
from challenge import Challenge


class LaserMazeEnv(gym.Env):
    """
    Gymnasium environment for Laser Maze puzzle solving.

    Observation Space:
        Box((13, 5, 5), float32) - Multi-channel grid representation
        - Channels 0-7: One-hot piece type
        - Channels 8-11: One-hot orientation
        - Channel 12: Fixed piece mask

    Action Space:
        Discrete(N) where N depends on board size and piece types
        - Action 0: Fire laser
        - Actions 1-25: Rotate piece at cell
        - Actions 26-50: Remove piece from cell
        - Remaining: Place actions (piece_type, cell, orientation)

    Rewards:
        - Step penalty: -0.01 (encourages efficiency)
        - Invalid action: -0.1
        - Target hit (incremental): +0.2 per new target
        - Checkpoint passed: +0.1 per checkpoint
        - Puzzle solved: +1.0
        - Failed (no pieces, not solved): -0.5
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 4}

    def __init__(
        self,
        puzzle_path: Optional[str] = None,
        puzzle_difficulty: str = "beginner",
        num_pieces: int = 1,
        random_puzzles: bool = True,
        max_steps: int = 50,
        dense_rewards: bool = True,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the environment.

        Args:
            puzzle_path: Path to specific puzzle file (overrides random generation)
            puzzle_difficulty: Difficulty for random puzzles ("beginner", "intermediate", "advanced", "expert")
            num_pieces: Number of pieces for random puzzles (1-4)
            random_puzzles: If True, generate new puzzle each reset
            max_steps: Maximum steps per episode
            dense_rewards: If True, use shaped rewards; if False, only terminal reward
            render_mode: "ansi" for text, "human" for console output
        """
        super().__init__()

        self.puzzle_path = puzzle_path
        self.puzzle_difficulty = puzzle_difficulty
        self.num_pieces = num_pieces
        self.random_puzzles = random_puzzles and puzzle_path is None
        self.max_steps = max_steps
        self.dense_rewards = dense_rewards
        self.render_mode = render_mode

        # Initialize game
        self.game = LaserMaze(board_size=5)
        self.generator = PuzzleGenerator() if self.random_puzzles else None

        # Define spaces
        self.board_size = 5
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(13, self.board_size, self.board_size),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(self.game.get_action_space_size())

        # Episode tracking
        self._step_count = 0
        self._prev_targets_hit = 0
        self._prev_checkpoints = 0
        self._episode_reward = 0.0

        # Precompute action mappings for efficiency
        self._valid_action_mask = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options (e.g., {"puzzle_path": "..."} to override)

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Handle options
        puzzle_path = self.puzzle_path
        if options and "puzzle_path" in options:
            puzzle_path = options["puzzle_path"]

        # Load or generate puzzle
        if puzzle_path:
            self.game.load(puzzle_path)
        elif self.random_puzzles and self.generator:
            # Map difficulty string to Difficulty enum
            difficulty_map = {
                "beginner": Difficulty.BEGINNER,
                "easy": Difficulty.EASY,
                "intermediate": Difficulty.MEDIUM,
                "advanced": Difficulty.HARD,
                "expert": Difficulty.EXPERT,
            }
            difficulty = difficulty_map.get(self.puzzle_difficulty, Difficulty.BEGINNER)

            # Generate random puzzle
            puzzle_data = self.generator.generate(difficulty=difficulty)

            if puzzle_data:
                # Load the generated puzzle from dict
                self.game.board = board_from_dict(puzzle_data)
                self.game.board_size = self.game.board.size
                self.game.initial_state = self.game.board.copy()

                # Set up challenge with available pieces
                if puzzle_data.get('available'):
                    challenge_data = {
                        'board': self.game.board,
                        'goal': puzzle_data.get('goal', {}),
                        'available': puzzle_data.get('available', [])
                    }
                    self.game.challenge = Challenge.from_dict(challenge_data)
                    self.game.available_pieces = [
                        p.copy() for p in self.game.challenge.available_pieces
                    ]
                else:
                    self.game.challenge = None
                    self.game.available_pieces = []

                self.game.placed_pieces = {}
                self.game.last_result = None
                self.game.last_solution = None
        else:
            # Reset existing puzzle
            self.game.reset()

        # Reset tracking
        self._step_count = 0
        self._prev_targets_hit = 0
        self._prev_checkpoints = 0
        self._episode_reward = 0.0

        # Get initial observation and info
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute action and return results.

        Args:
            action: Integer action ID

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self._step_count += 1

        # Decode and execute action
        decoded_action = self.game._decode_action(action)
        reward = self._calculate_reward(decoded_action)

        # Check termination conditions
        terminated = False
        truncated = False

        if self.game.last_solution and self.game.last_solution.solved:
            terminated = True
        elif self._step_count >= self.max_steps:
            truncated = True
            if self.dense_rewards:
                reward -= 0.3  # Timeout penalty
        elif len(self.game.available_pieces) == 0 and not self._is_valid_fire_action():
            # No pieces left and can't make progress
            # Only terminate if we've actually tried firing
            pass

        self._episode_reward += reward

        obs = self._get_obs()
        info = self._get_info()
        info['episode_reward'] = self._episode_reward

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, action: Action) -> float:
        """Calculate reward for an action with dense shaping."""
        reward = -0.01  # Step penalty

        if action.action_type == ActionType.PLACE:
            success = self.game.place(
                action.piece_type, action.row, action.col, action.orientation
            )
            if not success:
                reward = -0.1  # Invalid action
            elif self.dense_rewards:
                reward = 0.0  # Neutral for valid placement

        elif action.action_type == ActionType.ROTATE:
            success = self.game.rotate(action.row, action.col)
            if not success:
                reward = -0.1

        elif action.action_type == ActionType.REMOVE:
            success = self.game.remove(action.row, action.col)
            if not success:
                reward = -0.1

        elif action.action_type == ActionType.FIRE:
            result = self.game.fire()

            if self.dense_rewards:
                # Reward for new targets hit
                new_targets = len(result.targets_hit) - self._prev_targets_hit
                if new_targets > 0:
                    reward += 0.2 * new_targets
                self._prev_targets_hit = len(result.targets_hit)

                # Reward for checkpoints
                new_checkpoints = len(result.checkpoints_passed) - self._prev_checkpoints
                if new_checkpoints > 0:
                    reward += 0.1 * new_checkpoints
                self._prev_checkpoints = len(result.checkpoints_passed)

            # Solved bonus
            if self.game.last_solution and self.game.last_solution.solved:
                reward = 1.0  # Override with terminal reward

        return reward

    def _is_valid_fire_action(self) -> bool:
        """Check if firing is a valid action."""
        return self.game.board.laser_pos is not None or self.game.board.find_laser() is not None

    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        return self.game.get_state_tensor()

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary."""
        info = {
            'step_count': self._step_count,
            'pieces_remaining': len(self.game.available_pieces),
            'targets_hit': self._prev_targets_hit,
            'checkpoints_passed': self._prev_checkpoints,
        }

        if self.game.challenge:
            info['targets_required'] = self.game.challenge.required_targets
            info['checkpoints_required'] = len(self.game.challenge.required_checkpoints)

        if self.game.last_solution:
            info['solved'] = self.game.last_solution.solved

        return info

    def get_action_mask(self) -> np.ndarray:
        """
        Get mask of valid actions (1 = valid, 0 = invalid).

        Returns:
            Boolean array of shape (action_space.n,)
        """
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        valid_actions = self.game.get_valid_actions()

        for action in valid_actions:
            action_id = self.game.encode_action(action)
            mask[action_id] = 1

        return mask

    def render(self) -> Optional[str]:
        """Render the environment."""
        if self.render_mode == "ansi":
            return self.game.render(with_laser=self.game.last_result is not None)
        elif self.render_mode == "human":
            self.game.show(with_laser=self.game.last_result is not None)
            return None
        return None

    def close(self):
        """Clean up resources."""
        pass


class MaskedLaserMazeEnv(LaserMazeEnv):
    """
    LaserMaze environment with action masking support for SB3.

    Uses the gymnasium action mask wrapper pattern.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def action_masks(self) -> np.ndarray:
        """Return action mask for SB3's MaskablePPO."""
        return self.get_action_mask().astype(bool)

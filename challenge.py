"""
Challenge/goal tracking for Laser Maze game.

Manages puzzle challenges with fixed pieces, available pieces,
and win conditions.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set

from pieces import Piece, Direction, create_piece, Checkpoint
from board import Board
from laser import LaserResult


@dataclass
class Challenge:
    """
    Represents a puzzle challenge.

    Attributes:
        initial_board: Starting board state with fixed pieces
        available_pieces: Pieces that must be placed to solve
        required_targets: Number of targets that must be hit
        required_checkpoints: Checkpoint positions that must be passed
        target_positions: Optional set of specific target positions that must be hit.
                         If None, any target_mirror hits count toward required_targets.
    """
    initial_board: Board
    available_pieces: List[Piece] = field(default_factory=list)
    required_targets: int = 1
    required_checkpoints: Set[Tuple[int, int]] = field(default_factory=set)
    target_positions: Optional[Set[Tuple[int, int]]] = None

    def copy(self) -> 'Challenge':
        """Create a deep copy of the challenge."""
        return Challenge(
            initial_board=self.initial_board.copy(),
            available_pieces=[p.copy() for p in self.available_pieces],
            required_targets=self.required_targets,
            required_checkpoints=self.required_checkpoints.copy(),
            target_positions=self.target_positions.copy() if self.target_positions else None,
        )

    def check_solution(self, result: LaserResult,
                       placed_all: bool = True) -> 'SolutionResult':
        """
        Check if the laser result satisfies the challenge.

        Args:
            result: LaserResult from firing the laser
            placed_all: Whether all available pieces have been placed

        Returns:
            SolutionResult with success status and details
        """
        # Check targets based on whether we have explicit positions or just a count
        if self.target_positions is not None:
            # Explicit positions: must hit exactly those positions
            relevant_hits = result.targets_hit & self.target_positions
            targets_hit = len(relevant_hits)
            targets_ok = relevant_hits == self.target_positions
        else:
            # Legacy mode: any target_mirror hits count
            targets_hit = len(result.targets_hit)
            targets_ok = targets_hit >= self.required_targets

        checkpoints_hit = result.checkpoints_passed
        checkpoints_ok = self.required_checkpoints.issubset(checkpoints_hit)

        solved = targets_ok and checkpoints_ok and placed_all

        return SolutionResult(
            solved=solved,
            targets_hit=targets_hit,
            targets_required=self.required_targets,
            checkpoints_hit=len(checkpoints_hit),
            checkpoints_required=len(self.required_checkpoints),
            all_pieces_placed=placed_all,
        )

    @classmethod
    def from_dict(cls, data: dict) -> 'Challenge':
        """
        Create challenge from dictionary (loaded from JSON).

        Args:
            data: Dictionary with 'board', 'goal', 'available' keys

        Returns:
            Challenge instance
        """
        board = data['board']

        # Find checkpoints on board that are required
        required_checkpoints: Set[Tuple[int, int]] = set()
        for row in range(board.size):
            for col in range(board.size):
                piece = board.get_piece(row, col)
                if isinstance(piece, Checkpoint):
                    required_checkpoints.add((row, col))

        # Parse goal
        goal = data.get('goal', {})
        required_targets = goal.get('targets', 1)

        # Handle target positions (new format vs legacy)
        target_positions = None
        if 'target_positions' in goal and goal['target_positions']:
            target_positions = {tuple(pos) for pos in goal['target_positions']}
            required_targets = len(target_positions)

        # Override checkpoints if specified in goal
        if 'checkpoints' in goal and goal['checkpoints']:
            required_checkpoints = {tuple(cp) for cp in goal['checkpoints']}

        # Parse available pieces
        available = []
        for piece_data in data.get('available', []):
            orientation = piece_data.get('orientation')
            if orientation is None:
                orientation = 0  # Default, player must figure out correct orientation

            piece = create_piece(
                piece_type=piece_data['type'],
                orientation=orientation,
                fixed=False,
            )
            available.append(piece)

        return cls(
            initial_board=board,
            available_pieces=available,
            required_targets=required_targets,
            required_checkpoints=required_checkpoints,
            target_positions=target_positions,
        )

    def to_dict(self) -> dict:
        """Convert challenge to dictionary for serialization."""
        from file_io import board_to_dict, PIECE_TYPE_TO_NAME

        data = board_to_dict(self.initial_board)

        goal = {
            'checkpoints': [list(cp) for cp in self.required_checkpoints],
        }

        # Use target_positions if available, otherwise fall back to count
        if self.target_positions is not None:
            goal['target_positions'] = [list(pos) for pos in self.target_positions]
        else:
            goal['targets'] = self.required_targets

        data['goal'] = goal

        data['available'] = [
            {
                'type': PIECE_TYPE_TO_NAME[p.piece_type],
                'orientation': int(p.orientation),
            }
            for p in self.available_pieces
        ]

        return data


@dataclass
class SolutionResult:
    """Result of checking a solution against challenge requirements."""
    solved: bool
    targets_hit: int
    targets_required: int
    checkpoints_hit: int
    checkpoints_required: int
    all_pieces_placed: bool

    @property
    def targets_remaining(self) -> int:
        """Number of additional targets needed."""
        return max(0, self.targets_required - self.targets_hit)

    @property
    def checkpoints_remaining(self) -> int:
        """Number of additional checkpoints needed."""
        return max(0, self.checkpoints_required - self.checkpoints_hit)

    def __str__(self) -> str:
        if self.solved:
            return "SOLVED!"
        else:
            parts = []
            if self.targets_remaining > 0:
                parts.append(f"Need {self.targets_remaining} more target(s)")
            if self.checkpoints_remaining > 0:
                parts.append(f"Need {self.checkpoints_remaining} more checkpoint(s)")
            if not self.all_pieces_placed:
                parts.append("Not all pieces placed")
            return "Not solved: " + ", ".join(parts)

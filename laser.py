"""
Laser beam simulation for Laser Maze game.

Traces the path of the laser beam through the board, handling
reflections, splits, and target hits.
"""

from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional

from pieces import Direction, Piece, Laser, TargetMirror, Checkpoint
from board import Board


@dataclass
class BeamSegment:
    """Represents a segment of the laser beam path."""
    row: int
    col: int
    direction: Direction  # Direction beam is traveling when in this cell

    def __hash__(self):
        return hash((self.row, self.col, self.direction))

    def __eq__(self, other):
        if not isinstance(other, BeamSegment):
            return False
        return (self.row, self.col, self.direction) == (other.row, other.col, other.direction)


@dataclass
class LaserResult:
    """Result of firing the laser."""
    beam_path: List[BeamSegment] = field(default_factory=list)
    targets_hit: Set[Tuple[int, int]] = field(default_factory=set)
    checkpoints_passed: Set[Tuple[int, int]] = field(default_factory=set)
    terminated: bool = True  # True if beam exited grid or was absorbed

    @property
    def num_targets(self) -> int:
        """Number of targets hit."""
        return len(self.targets_hit)

    @property
    def num_checkpoints(self) -> int:
        """Number of checkpoints passed."""
        return len(self.checkpoints_passed)


def fire_laser(board: Board, max_steps: int = 1000) -> LaserResult:
    """
    Fire the laser and trace the beam path.

    Args:
        board: The game board
        max_steps: Maximum beam segments to prevent infinite loops

    Returns:
        LaserResult with beam path, targets hit, and checkpoints passed
    """
    result = LaserResult()

    # Find laser
    if board.laser_pos is None:
        laser_pos = board.find_laser()
        if laser_pos is None:
            return result
    else:
        laser_pos = board.laser_pos

    laser = board.get_piece(*laser_pos)
    if not isinstance(laser, Laser):
        return result

    # Start beam from laser position, traveling in laser direction
    initial_dir = laser.emit_direction()

    # Queue of (row, col, direction) to process (for beam splits)
    beam_queue: List[Tuple[int, int, Direction]] = []

    # Start from the cell the laser is in, beam exits in laser direction
    start_row, start_col = laser_pos
    dr, dc = initial_dir.delta
    next_row, next_col = start_row + dr, start_col + dc

    if board.is_valid_pos(next_row, next_col):
        beam_queue.append((next_row, next_col, initial_dir))

    # Track visited states to detect loops
    visited: Set[BeamSegment] = set()

    steps = 0
    while beam_queue and steps < max_steps:
        row, col, direction = beam_queue.pop(0)
        steps += 1

        segment = BeamSegment(row, col, direction)

        # Check for loop
        if segment in visited:
            continue
        visited.add(segment)
        result.beam_path.append(segment)

        # Get piece at this position
        piece = board.get_piece(row, col)

        if piece is None:
            # Empty cell - beam continues straight
            dr, dc = direction.delta
            next_row, next_col = row + dr, col + dc

            if board.is_valid_pos(next_row, next_col):
                beam_queue.append((next_row, next_col, direction))
            # else beam exits grid
        else:
            # Check if target is hit
            if isinstance(piece, TargetMirror):
                if piece.is_hit(direction):
                    result.targets_hit.add((row, col))

            # Check if checkpoint is passed
            if isinstance(piece, Checkpoint):
                result.checkpoints_passed.add((row, col))

            # Get outgoing directions from piece interaction
            outgoing_dirs = piece.interact(direction)

            for out_dir in outgoing_dirs:
                dr, dc = out_dir.delta
                next_row, next_col = row + dr, col + dc

                if board.is_valid_pos(next_row, next_col):
                    beam_queue.append((next_row, next_col, out_dir))

    # Check if we hit max steps (potential infinite loop)
    if steps >= max_steps:
        result.terminated = False

    return result


def get_beam_cells(result: LaserResult) -> Set[Tuple[int, int]]:
    """Get set of all cells the beam passes through."""
    return {(seg.row, seg.col) for seg in result.beam_path}


def get_beam_at_cell(result: LaserResult, row: int, col: int) -> List[Direction]:
    """Get all directions the beam travels through a specific cell."""
    directions = []
    for seg in result.beam_path:
        if seg.row == row and seg.col == col:
            directions.append(seg.direction)
    return directions

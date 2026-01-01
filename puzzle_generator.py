"""
Puzzle generator for Laser Maze.

Generates random solvable puzzles with varying difficulty levels.
Uses backward generation: place solution first, then verify it's solvable.
"""

import random
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from pieces import (
    Direction, PieceType, Piece, create_piece,
    Laser, Mirror, TargetMirror, BeamSplitter, DoubleMirror, Checkpoint
)
from board import Board
from laser import fire_laser, get_beam_cells, LaserResult
from game import LaserMaze, PIECE_ORIENTATIONS
from file_io import board_from_dict
from challenge import Challenge


def _can_reach_targets(board: Board, result: LaserResult,
                       remaining_pieces: List[Dict],
                       required_targets: int) -> bool:
    """
    Check if remaining pieces can possibly reach all required targets.

    Pruning logic:
    1. If we already hit enough targets, return True
    2. If no redirectors remain and targets aren't on current path, prune
    3. Check if targets are geometrically reachable

    Returns True if targets are still potentially reachable, False to prune.
    """
    # Already hit enough targets
    if result.num_targets >= required_targets:
        return True

    # Find unhit targets on the board
    unhit_targets = []
    for row in range(board.size):
        for col in range(board.size):
            piece = board.get_piece(row, col)
            if isinstance(piece, TargetMirror):
                if (row, col) not in result.targets_hit:
                    # Target's required incoming direction
                    required_dir = piece.orientation.opposite()
                    unhit_targets.append((row, col, required_dir))

    if not unhit_targets:
        # No targets on board yet, can't prune
        return True

    targets_needed = required_targets - result.num_targets
    if targets_needed > len(unhit_targets):
        # Not enough targets on board
        return False

    # Get current beam path
    beam_cells = set(get_beam_cells(result))
    if not beam_cells:
        return True

    # Count remaining pieces that can redirect the beam
    redirectors = 0  # Pieces that can change beam direction

    for piece_info in remaining_pieces:
        ptype = piece_info['type']
        if ptype in ('mirror', 'double_mirror', 'beam_splitter'):
            redirectors += 1

    # For each unhit target, check if it's reachable
    for target_row, target_col, required_dir in unhit_targets:
        target_reachable = False

        # Check if beam already passes through target cell from correct direction
        for seg in result.beam_path:
            if seg.row == target_row and seg.col == target_col:
                if seg.direction == required_dir:
                    # Beam hits target correctly - but wait, it's already marked as hit
                    # This shouldn't happen, so target is reachable
                    target_reachable = True
                    break
                else:
                    # Beam passes through but wrong direction
                    # Would need a redirector placed before this cell
                    if redirectors > 0:
                        target_reachable = True
                    break

        if target_reachable:
            continue

        # Target not on current beam path - check if reachable with redirectors
        if redirectors == 0:
            # No way to redirect beam to reach target
            return False

        # Check if target is aligned with any beam cell (same row or col)
        # If so, one redirect could potentially reach it
        aligned = False
        for seg in result.beam_path:
            if seg.row == target_row or seg.col == target_col:
                aligned = True
                break

        if not aligned and redirectors < 2:
            # Target not aligned with beam and less than 2 redirectors
            # Would need at least 2 redirects to reach
            return False

    return True


class Difficulty(IntEnum):
    """Puzzle difficulty levels."""
    BEGINNER = 1      # 1 piece to place, direct path
    EASY = 2          # 1-2 pieces, simple reflections
    MEDIUM = 3        # 2-3 pieces, may include beam splitter
    HARD = 4          # 3-4 pieces, multiple targets or checkpoints
    EXPERT = 5        # 4+ pieces, complex paths


@dataclass
class PuzzleConfig:
    """Configuration for puzzle generation."""
    difficulty: Difficulty
    min_pieces: int
    max_pieces: int
    piece_types: List[str]
    num_targets: int
    num_checkpoints: int

    @classmethod
    def for_difficulty(cls, difficulty: Difficulty) -> 'PuzzleConfig':
        """Get configuration for a difficulty level."""
        configs = {
            Difficulty.BEGINNER: cls(
                difficulty=Difficulty.BEGINNER,
                min_pieces=1, max_pieces=1,
                piece_types=['mirror'],
                num_targets=1, num_checkpoints=0
            ),
            Difficulty.EASY: cls(
                difficulty=Difficulty.EASY,
                min_pieces=1, max_pieces=2,
                piece_types=['mirror', 'double_mirror'],
                num_targets=1, num_checkpoints=0
            ),
            Difficulty.MEDIUM: cls(
                difficulty=Difficulty.MEDIUM,
                min_pieces=2, max_pieces=3,
                piece_types=['mirror', 'double_mirror', 'beam_splitter'],
                num_targets=1, num_checkpoints=0
            ),
            Difficulty.HARD: cls(
                difficulty=Difficulty.HARD,
                min_pieces=3, max_pieces=4,
                piece_types=['mirror', 'double_mirror', 'beam_splitter'],
                num_targets=2, num_checkpoints=0
            ),
            Difficulty.EXPERT: cls(
                difficulty=Difficulty.EXPERT,
                min_pieces=4, max_pieces=5,
                piece_types=['mirror', 'double_mirror', 'beam_splitter'],
                num_targets=2, num_checkpoints=1
            ),
        }
        return configs[difficulty]


class PuzzleGenerator:
    """
    Generates random solvable Laser Maze puzzles.

    Strategy:
    1. Place laser at random edge position
    2. Place target(s) at reachable positions
    3. Generate a valid solution path with mirrors
    4. Remove the placed mirrors (they become "available" pieces)
    5. Verify the puzzle is solvable
    """

    def __init__(self, board_size: int = 5, seed: Optional[int] = None):
        self.board_size = board_size
        if seed is not None:
            random.seed(seed)

    def generate(self, difficulty: Difficulty = Difficulty.BEGINNER,
                 max_attempts: int = 100) -> Optional[Dict[str, Any]]:
        """
        Generate a puzzle of the specified difficulty.

        Args:
            difficulty: Puzzle difficulty level
            max_attempts: Maximum generation attempts

        Returns:
            Puzzle dict or None if generation failed
        """
        config = PuzzleConfig.for_difficulty(difficulty)

        for attempt in range(max_attempts):
            puzzle = self._try_generate(config)
            if puzzle is not None:
                return puzzle

        return None

    def _try_generate(self, config: PuzzleConfig) -> Optional[Dict[str, Any]]:
        """Attempt to generate a single puzzle."""
        board = Board(size=self.board_size)

        # Step 1: Place laser at random edge
        laser_pos, laser_dir = self._place_laser(board)
        if laser_pos is None:
            return None

        # Step 2: Trace initial beam path to find reachable cells
        initial_result = fire_laser(board)
        beam_cells = get_beam_cells(initial_result)

        if not beam_cells:
            return None

        # Step 3: Build solution path with mirrors
        num_pieces = random.randint(config.min_pieces, config.max_pieces)
        solution_pieces = self._build_solution_path(
            board, laser_pos, laser_dir,
            num_pieces, config
        )

        if solution_pieces is None:
            return None

        # Step 4: Place target(s) at end of beam path
        targets_placed = self._place_targets(board, config.num_targets)
        if not targets_placed:
            return None

        # Step 5: Verify solution works
        result = fire_laser(board)
        if result.num_targets < config.num_targets:
            return None

        # Step 6: Place checkpoints if needed
        if config.num_checkpoints > 0:
            checkpoints = self._place_checkpoints(board, result, config.num_checkpoints)
            if len(checkpoints) < config.num_checkpoints:
                return None
        else:
            checkpoints = []

        # Step 7: Remove solution pieces to create the puzzle
        available_pieces = []
        for pos, piece_type, orientation in solution_pieces:
            board.remove_piece(pos[0], pos[1])
            available_pieces.append({
                'type': piece_type,
                'orientation': None  # Player chooses orientation
            })

        # Step 8: Verify puzzle is still solvable
        if not self._verify_solvable(board, available_pieces, config.num_targets, checkpoints):
            return None

        # Build puzzle dict
        return self._build_puzzle_dict(board, available_pieces, config.num_targets, checkpoints)

    def _place_laser(self, board: Board) -> Tuple[Optional[Tuple[int, int]], Optional[Direction]]:
        """Place laser at a random position with valid orientation.

        Laser can be placed anywhere on the board, but orientation must
        not immediately exit the board (e.g., laser on top edge can't face north).
        """
        # Generate all valid (position, direction) combinations
        valid_placements = []

        for row in range(self.board_size):
            for col in range(self.board_size):
                for direction in Direction:
                    # Check if beam would stay on board for at least one cell
                    dr, dc = direction.delta
                    next_row, next_col = row + dr, col + dc

                    if board.is_valid_pos(next_row, next_col):
                        valid_placements.append(((row, col), direction))

        random.shuffle(valid_placements)

        for pos, direction in valid_placements:
            laser = Laser(orientation=direction, fixed=True)
            if board.place_piece(pos[0], pos[1], laser):
                return pos, direction

        return None, None

    def _build_solution_path(self, board: Board, laser_pos: Tuple[int, int],
                             laser_dir: Direction, num_pieces: int,
                             config: PuzzleConfig) -> Optional[List[Tuple[Tuple[int, int], str, int]]]:
        """
        Build a solution path by placing mirrors along the beam.

        Returns list of (position, piece_type, orientation) tuples.
        """
        solution = []
        current_dir = laser_dir
        current_pos = laser_pos

        # Move one step from laser
        dr, dc = current_dir.delta
        current_pos = (current_pos[0] + dr, current_pos[1] + dc)

        pieces_placed = 0
        max_steps = self.board_size * 4  # Prevent infinite loops
        steps = 0

        while pieces_placed < num_pieces and steps < max_steps:
            steps += 1

            # Check if current position is valid and empty
            if not board.is_valid_pos(current_pos[0], current_pos[1]):
                break

            if board.get_piece(current_pos[0], current_pos[1]) is not None:
                # Move to next cell
                dr, dc = current_dir.delta
                current_pos = (current_pos[0] + dr, current_pos[1] + dc)
                continue

            # Decide whether to place a mirror here
            # Higher chance as we get closer to needing more pieces
            place_chance = 0.3 + (pieces_placed / num_pieces) * 0.4

            if random.random() < place_chance:
                # Choose piece type and orientation
                piece_type = random.choice(config.piece_types)

                # Determine orientation based on desired reflection
                new_dir = self._choose_reflection(current_dir, current_pos, board)
                orientation = self._get_mirror_orientation(current_dir, new_dir, piece_type)

                if orientation is not None:
                    piece = create_piece(piece_type, orientation, fixed=False)
                    if board.place_piece(current_pos[0], current_pos[1], piece):
                        solution.append((current_pos, piece_type, orientation))
                        pieces_placed += 1
                        current_dir = new_dir

            # Move to next cell
            dr, dc = current_dir.delta
            current_pos = (current_pos[0] + dr, current_pos[1] + dc)

        if pieces_placed < config.min_pieces:
            # Remove placed pieces and fail
            for pos, _, _ in solution:
                board.remove_piece(pos[0], pos[1])
            return None

        return solution

    def _choose_reflection(self, current_dir: Direction,
                          pos: Tuple[int, int], board: Board) -> Direction:
        """Choose a new direction after reflection."""
        # Prefer directions that keep beam on board longer
        possible = []

        for new_dir in Direction:
            if new_dir == current_dir or new_dir == current_dir.opposite():
                continue  # Skip same direction and 180° turn

            # Check how far beam can travel in this direction
            dr, dc = new_dir.delta
            test_pos = pos
            distance = 0
            while board.is_valid_pos(test_pos[0] + dr, test_pos[1] + dc):
                test_pos = (test_pos[0] + dr, test_pos[1] + dc)
                distance += 1
                if distance > 2:
                    break

            if distance > 0:
                possible.append((new_dir, distance))

        if not possible:
            # Just pick a perpendicular direction
            if current_dir in (Direction.NORTH, Direction.SOUTH):
                return random.choice([Direction.EAST, Direction.WEST])
            else:
                return random.choice([Direction.NORTH, Direction.SOUTH])

        # Weight by distance
        total = sum(d for _, d in possible)
        r = random.random() * total
        cumulative = 0
        for new_dir, distance in possible:
            cumulative += distance
            if r <= cumulative:
                return new_dir

        return possible[-1][0]

    def _get_mirror_orientation(self, incoming: Direction, outgoing: Direction,
                                piece_type: str) -> Optional[int]:
        """
        Get mirror orientation for desired reflection.

        For mirrors:
        - Orientation 0 (N/S): \\ diagonal - N↔W, E↔S
        - Orientation 1 (E/W): / diagonal - N↔E, S↔W
        """
        if piece_type in ('mirror', 'double_mirror', 'beam_splitter'):
            # Backslash diagonal: N↔W, E↔S, S↔E, W↔N
            backslash_map = {
                (Direction.NORTH, Direction.WEST): True,
                (Direction.WEST, Direction.NORTH): True,
                (Direction.EAST, Direction.SOUTH): True,
                (Direction.SOUTH, Direction.EAST): True,
            }

            # Forward slash diagonal: N↔E, E↔N, S↔W, W↔S
            slash_map = {
                (Direction.NORTH, Direction.EAST): True,
                (Direction.EAST, Direction.NORTH): True,
                (Direction.SOUTH, Direction.WEST): True,
                (Direction.WEST, Direction.SOUTH): True,
            }

            if (incoming, outgoing) in backslash_map:
                return 0  # N/S orientation = backslash
            elif (incoming, outgoing) in slash_map:
                return 1  # E/W orientation = forward slash

        return None

    def _place_targets(self, board: Board, num_targets: int) -> bool:
        """Place target mirrors at the end of beam paths."""
        result = fire_laser(board)
        beam_cells = list(get_beam_cells(result))

        if not beam_cells:
            return False

        # Find cells where beam exits the grid or could hit a target
        exit_cells = []
        for seg in result.beam_path:
            dr, dc = seg.direction.delta
            next_row, next_col = seg.row + dr, seg.col + dc

            # If next cell is off grid, this is a potential target location
            if not board.is_valid_pos(next_row, next_col):
                # Target should face opposite of beam direction
                target_dir = seg.direction.opposite()
                if board.get_piece(seg.row, seg.col) is None:
                    exit_cells.append(((seg.row, seg.col), target_dir))

        # Also consider cells at the end of beam path
        if result.beam_path:
            last_seg = result.beam_path[-1]
            if board.get_piece(last_seg.row, last_seg.col) is None:
                exit_cells.append(((last_seg.row, last_seg.col), last_seg.direction.opposite()))

        random.shuffle(exit_cells)

        targets_placed = 0
        for pos, target_dir in exit_cells:
            if targets_placed >= num_targets:
                break

            if board.get_piece(pos[0], pos[1]) is None:
                target = TargetMirror(orientation=target_dir, fixed=True)
                if board.place_piece(pos[0], pos[1], target):
                    targets_placed += 1

        return targets_placed >= num_targets

    def _place_checkpoints(self, board: Board, result,
                          num_checkpoints: int) -> List[Tuple[int, int]]:
        """Place checkpoints along the beam path."""
        beam_cells = list(get_beam_cells(result))
        checkpoints = []

        # Filter to empty cells along the path
        valid_cells = [
            (row, col) for row, col in beam_cells
            if board.get_piece(row, col) is None
        ]

        random.shuffle(valid_cells)

        for pos in valid_cells[:num_checkpoints]:
            checkpoint = Checkpoint(fixed=True)
            if board.place_piece(pos[0], pos[1], checkpoint):
                checkpoints.append(pos)

        return checkpoints

    def _verify_solvable(self, board: Board, available: List[Dict],
                        required_targets: int,
                        checkpoints: List[Tuple[int, int]],
                        require_unique: bool = True) -> bool:
        """
        Verify the puzzle is solvable, optionally checking for uniqueness.

        Args:
            require_unique: If True, verify exactly one solution exists (for 1-2 piece puzzles).
                           If False, just verify at least one solution exists.

        Uses brute-force search. For 3+ piece puzzles, only checks solvability (not uniqueness)
        because exhaustive search would be too slow.
        """
        if not available:
            # No pieces to place, check if already solved
            result = fire_laser(board)
            return result.num_targets >= required_targets

        # For puzzles with 3+ pieces, just verify solvability
        # Note: The pruning makes 3-piece uniqueness check fast (~50ms), but
        # most generated 3-piece puzzles have multiple solutions by design.
        # TODO: Improve puzzle generation to create more constrained layouts
        if len(available) >= 3:
            return self._has_solution(board, available, required_targets, checkpoints)

        # For 1-2 piece puzzles, verify exactly one solution if required
        if require_unique:
            solution_count = self._count_solutions(board, available, required_targets, checkpoints, max_count=2)
            return solution_count == 1
        else:
            return self._has_solution(board, available, required_targets, checkpoints)

    def _has_solution(self, board: Board, available: List[Dict],
                     required_targets: int,
                     checkpoints: List[Tuple[int, int]]) -> bool:
        """Quick check if at least one solution exists (stops at first solution)."""
        return self._search_first_solution(board, available, required_targets, checkpoints, 0)

    def _count_solutions(self, board: Board, available: List[Dict],
                        required_targets: int,
                        checkpoints: List[Tuple[int, int]],
                        max_count: int = 2) -> int:
        """
        Count solutions up to max_count.

        Args:
            max_count: Stop counting after finding this many solutions (optimization)

        Returns:
            Number of solutions found (capped at max_count)
        """
        solutions = []
        self._search_all_solutions(board, available, required_targets, checkpoints,
                                   0, [], solutions, max_count)
        return len(solutions)

    def _search_all_solutions(self, board: Board, available: List[Dict],
                              required_targets: int,
                              checkpoints: List[Tuple[int, int]],
                              depth: int,
                              current_placement: List[Tuple[int, int, int]],
                              solutions: List,
                              max_count: int) -> None:
        """Recursive search to find all solutions (up to max_count)."""
        if len(solutions) >= max_count:
            return  # Already found enough solutions

        if depth >= len(available):
            # All pieces placed, check solution
            result = fire_laser(board)
            if result.num_targets >= required_targets:
                # Check checkpoints
                for cp in checkpoints:
                    if cp not in result.checkpoints_passed:
                        return
                # Found a valid solution - record it
                solutions.append(list(current_placement))
            return

        # Fire laser to get current beam state
        result = fire_laser(board)

        # Goal-directed pruning: check if targets are still reachable
        remaining_pieces = available[depth:]
        if not _can_reach_targets(board, result, remaining_pieces, required_targets):
            return

        piece_info = available[depth]
        piece_type = piece_info['type']
        valid_orientations = PIECE_ORIENTATIONS.get(piece_type, [0, 1, 2, 3])

        # Get cells along the current beam path
        beam_cells = set(get_beam_cells(result))
        empty_cells = board.get_empty_cells()

        # Prioritize cells on the beam path
        prioritized_cells = [c for c in empty_cells if c in beam_cells]
        other_cells = [c for c in empty_cells if c not in beam_cells]

        for row, col in prioritized_cells + other_cells:
            for orientation in valid_orientations:
                piece = create_piece(piece_type, orientation, fixed=False)
                if board.place_piece(row, col, piece):
                    current_placement.append((row, col, orientation))
                    self._search_all_solutions(board, available, required_targets,
                                              checkpoints, depth + 1,
                                              current_placement, solutions, max_count)
                    current_placement.pop()
                    board.remove_piece(row, col)

                    if len(solutions) >= max_count:
                        return

    def _search_first_solution(self, board: Board, available: List[Dict],
                               required_targets: int,
                               checkpoints: List[Tuple[int, int]],
                               depth: int) -> bool:
        """Fast search that stops at first valid solution found."""
        if depth >= len(available):
            # All pieces placed, check solution
            result = fire_laser(board)
            if result.num_targets < required_targets:
                return False
            # Check checkpoints
            for cp in checkpoints:
                if cp not in result.checkpoints_passed:
                    return False
            return True

        # Fire laser to get current beam state
        result = fire_laser(board)

        # Goal-directed pruning: check if targets are still reachable
        remaining_pieces = available[depth:]
        if not _can_reach_targets(board, result, remaining_pieces, required_targets):
            return False

        piece_info = available[depth]
        piece_type = piece_info['type']
        valid_orientations = PIECE_ORIENTATIONS.get(piece_type, [0, 1, 2, 3])

        # Only search cells on the beam path for efficiency
        beam_cells = set(get_beam_cells(result))
        empty_cells = board.get_empty_cells()

        # Search beam cells first (more likely to contain solution)
        cells_to_search = [c for c in empty_cells if c in beam_cells]
        # Also include a few cells adjacent to beam for completeness
        cells_to_search.extend([c for c in empty_cells if c not in beam_cells][:5])

        for row, col in cells_to_search:
            for orientation in valid_orientations:
                piece = create_piece(piece_type, orientation, fixed=False)
                if board.place_piece(row, col, piece):
                    if self._search_first_solution(board, available, required_targets,
                                                   checkpoints, depth + 1):
                        board.remove_piece(row, col)
                        return True
                    board.remove_piece(row, col)

        return False

    def _build_puzzle_dict(self, board: Board, available: List[Dict],
                          num_targets: int,
                          checkpoints: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Build the puzzle dictionary for serialization."""
        pieces = []

        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = board.get_piece(row, col)
                if piece is not None:
                    piece_name = self._get_piece_name(piece)
                    pieces.append({
                        'type': piece_name,
                        'position': [row, col],
                        'orientation': int(piece.orientation),
                        'fixed': piece.fixed
                    })

        return {
            'grid_size': self.board_size,
            'pieces': pieces,
            'available': available,
            'goal': {
                'targets': num_targets,
                'checkpoints': [list(cp) for cp in checkpoints]
            }
        }

    def _get_piece_name(self, piece: Piece) -> str:
        """Get string name for a piece."""
        type_map = {
            PieceType.LASER: 'laser',
            PieceType.MIRROR: 'mirror',
            PieceType.TARGET_MIRROR: 'target_mirror',
            PieceType.BEAM_SPLITTER: 'beam_splitter',
            PieceType.DOUBLE_MIRROR: 'double_mirror',
            PieceType.CHECKPOINT: 'checkpoint',
            PieceType.CELL_BLOCKER: 'cell_blocker',
        }
        return type_map.get(piece.piece_type, 'unknown')

    def generate_batch(self, count: int, difficulty: Difficulty,
                      output_dir: str = 'examples/generated') -> List[str]:
        """
        Generate multiple puzzles and save to files.

        Returns list of generated file paths.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generated_files = []
        difficulty_name = difficulty.name.lower()

        for i in range(count):
            puzzle = self.generate(difficulty)
            if puzzle is not None:
                filename = f"{difficulty_name}_{i+1:03d}.json"
                filepath = output_path / filename

                with open(filepath, 'w') as f:
                    json.dump(puzzle, f, indent=2)

                generated_files.append(str(filepath))

        return generated_files


def demo_generator():
    """Demo the puzzle generator."""
    gen = PuzzleGenerator(seed=42)

    print("=== Puzzle Generator Demo ===\n")

    for difficulty in Difficulty:
        print(f"\n--- {difficulty.name} ---")
        puzzle = gen.generate(difficulty)

        if puzzle:
            print(f"Generated puzzle with {len(puzzle['available'])} pieces to place")
            print(f"Goal: {puzzle['goal']['targets']} target(s)")
            if puzzle['goal'].get('checkpoints'):
                print(f"      Checkpoints: {puzzle['goal']['checkpoints']}")

            # Load board from dict and set up game
            game = LaserMaze()
            game.board = board_from_dict(puzzle)
            game.board_size = game.board.size
            game.initial_state = game.board.copy()

            # Set up challenge with available pieces
            if puzzle.get('available'):
                # Build challenge data with board object
                challenge_data = {
                    'board': game.board,
                    'goal': puzzle.get('goal', {}),
                    'available': puzzle.get('available', [])
                }
                game.challenge = Challenge.from_dict(challenge_data)
                game.available_pieces = [p.copy() for p in game.challenge.available_pieces]

            game.show()

            # Fire laser to show current state
            result = game.fire()
            print(f"Current targets hit: {result.num_targets}")
        else:
            print("Failed to generate puzzle")


if __name__ == '__main__':
    demo_generator()

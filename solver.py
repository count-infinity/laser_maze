"""
Fast puzzle solver using DFS with aggressive pruning.

Key optimizations:
1. Place laser first (if available) - it's the root of the beam
2. Don't place laser where it fires off-grid immediately
3. Track beam state at each node - if beam dies with pieces left, prune
4. Don't place pieces where they can't interact with the beam
5. For rotatable fixed-position pieces, try rotations as part of search
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from copy import deepcopy

from pieces import Direction, Piece, create_piece, Laser, TargetMirror, Checkpoint
from board import Board
from laser import fire_laser, LaserResult, get_beam_cells
from game import PIECE_ORIENTATIONS
from file_io import PIECE_TYPE_TO_NAME


@dataclass
class SolverResult:
    """Result of solving a puzzle."""
    solved: bool
    solutions: List[Dict]  # List of solution configurations
    nodes_visited: int


def solve_puzzle(
    board: Board,
    available: List[Dict],
    required_targets: Set[Tuple[int, int]],
    required_checkpoints: List[Tuple[int, int]],
    max_solutions: int = 2,
    rotatable_positions: List[Tuple[int, int]] = None
) -> SolverResult:
    """
    Solve a puzzle using DFS with pruning.

    Args:
        board: Initial board state (will be modified during search)
        available: List of available pieces [{'type': 'laser'}, ...]
        required_targets: Set of (row, col) positions that must be hit
        required_checkpoints: List of (row, col) checkpoints to pass through
        max_solutions: Stop after finding this many solutions
        rotatable_positions: List of (row, col) for fixed-position pieces that can rotate

    Returns:
        SolverResult with solutions found
    """
    solutions = []
    nodes_visited = [0]  # Use list for mutable counter in nested function

    if rotatable_positions is None:
        rotatable_positions = []

    # Separate laser from other pieces (place laser first)
    laser_piece = None
    other_pieces = []
    for p in available:
        if p['type'] == 'laser':
            laser_piece = p
        else:
            other_pieces.append(p)

    # Reorder: laser first, then others
    ordered_available = []
    if laser_piece:
        ordered_available.append(laser_piece)
    ordered_available.extend(other_pieces)

    def check_solution(result: LaserResult) -> bool:
        """Check if current state is a valid solution."""
        if result.targets_hit != required_targets:
            return False
        for cp in required_checkpoints:
            if cp not in result.checkpoints_passed:
                return False
        return True

    def get_valid_laser_positions(board: Board) -> List[Tuple[int, int, int]]:
        """Get valid (row, col, orientation) for laser placement."""
        positions = []
        empty = board.get_empty_cells()

        for row, col in empty:
            for ori in [0, 1, 2, 3]:  # N, E, S, W
                # Check if laser would fire off-grid immediately
                dr, dc = Direction(ori).delta
                next_row, next_col = row + dr, col + dc
                if board.is_valid_pos(next_row, next_col):
                    positions.append((row, col, ori))

        return positions

    def get_beam_adjacent_cells(board: Board, result: LaserResult) -> Set[Tuple[int, int]]:
        """Get empty cells on or adjacent to the beam path."""
        beam_cells = get_beam_cells(result)
        adjacent = set()

        for row, col in beam_cells:
            # Add the beam cell itself if empty
            if board.get_piece(row, col) is None:
                adjacent.add((row, col))
            # Add adjacent cells
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if board.is_valid_pos(nr, nc) and board.get_piece(nr, nc) is None:
                    adjacent.add((nr, nc))

        return adjacent

    def dfs(depth: int, placements: List[Tuple], rotation_state: Dict[Tuple[int, int], int]):
        """
        DFS search for solutions.

        Args:
            depth: Current depth (index into ordered_available)
            placements: List of (piece_type, row, col, orientation) placed so far
            rotation_state: Current orientations of rotatable pieces
        """
        nonlocal solutions
        nodes_visited[0] += 1

        if len(solutions) >= max_solutions:
            return

        # Base case: all pieces placed
        if depth == len(ordered_available):
            # Try all rotation combinations for fixed-position pieces
            if rotatable_positions:
                search_rotations(placements, rotation_state, 0)
            else:
                result = fire_laser(board)
                if check_solution(result):
                    solutions.append({
                        'placements': list(placements),
                        'rotations': dict(rotation_state)
                    })
            return

        piece_info = ordered_available[depth]
        piece_type = piece_info['type']

        # Get current beam state
        result = fire_laser(board)

        # Special handling for laser (first piece typically)
        if piece_type == 'laser':
            for row, col, ori in get_valid_laser_positions(board):
                piece = create_piece('laser', ori)
                board.set_piece(row, col, piece)
                placements.append(('laser', row, col, ori))

                dfs(depth + 1, placements, rotation_state)

                placements.pop()
                board.remove_piece(row, col)

                if len(solutions) >= max_solutions:
                    return
        else:
            # For non-laser pieces, only consider cells on/near beam path
            # If no beam yet, we can't place non-laser pieces usefully
            beam_cells = get_beam_cells(result)
            if not beam_cells:
                # No beam - can't place pieces usefully
                return

            # Get cells where placing a piece could affect the beam
            candidate_cells = get_beam_adjacent_cells(board, result)

            # Also include cells on the beam path itself
            for cell in beam_cells:
                if board.get_piece(*cell) is None:
                    candidate_cells.add(cell)

            valid_orientations = PIECE_ORIENTATIONS.get(piece_type, [0, 1, 2, 3])

            for row, col in candidate_cells:
                for ori in valid_orientations:
                    piece = create_piece(piece_type, ori)
                    board.set_piece(row, col, piece)
                    placements.append((piece_type, row, col, ori))

                    # Pruning: check if beam still exists after placement
                    new_result = fire_laser(board)
                    beam_alive = len(new_result.beam_path) > 0

                    # Only continue if beam is still active or we've placed all pieces
                    if beam_alive or depth == len(ordered_available) - 1:
                        dfs(depth + 1, placements, rotation_state)

                    placements.pop()
                    board.remove_piece(row, col)

                    if len(solutions) >= max_solutions:
                        return

    def search_rotations(placements: List[Tuple], rotation_state: Dict, rot_idx: int):
        """Search through rotation combinations for fixed-position pieces."""
        nonlocal solutions

        if len(solutions) >= max_solutions:
            return

        if rot_idx == len(rotatable_positions):
            # All rotations set, check solution
            result = fire_laser(board)
            if check_solution(result):
                solutions.append({
                    'placements': list(placements),
                    'rotations': dict(rotation_state)
                })
            return

        row, col = rotatable_positions[rot_idx]
        piece = board.get_piece(row, col)
        if piece is None:
            search_rotations(placements, rotation_state, rot_idx + 1)
            return

        original_ori = piece.orientation

        # Get valid orientations for this piece type
        piece_type_name = PIECE_TYPE_TO_NAME.get(piece.piece_type, '')
        valid_orientations = PIECE_ORIENTATIONS.get(piece_type_name, [0, 1, 2, 3])

        for ori in valid_orientations:
            piece.orientation = Direction(ori)
            rotation_state[(row, col)] = ori
            search_rotations(placements, rotation_state, rot_idx + 1)

            if len(solutions) >= max_solutions:
                piece.orientation = original_ori
                return

        piece.orientation = original_ori

    # If there are rotatable pieces but no pieces to place, just search rotations
    if not ordered_available and rotatable_positions:
        search_rotations([], {}, 0)
    else:
        dfs(0, [], {})

    return SolverResult(
        solved=len(solutions) > 0,
        solutions=solutions,
        nodes_visited=nodes_visited[0]
    )


def solve_challenge_file(filepath: str, required_targets: Set[Tuple[int, int]] = None,
                         max_solutions: int = 2) -> SolverResult:
    """
    Solve a challenge from a JSON file.

    Args:
        filepath: Path to challenge JSON
        required_targets: Set of target positions that must be hit.
                         If None, uses target_positions from the JSON file.
        max_solutions: Maximum solutions to find

    Returns:
        SolverResult
    """
    from file_io import load_challenge

    challenge = load_challenge(filepath)
    board = challenge['board']
    available = challenge['available']
    goal = challenge['goal']

    checkpoints = [tuple(cp) for cp in goal.get('checkpoints', [])]

    # Use provided targets or get from goal spec
    if required_targets is None:
        target_positions = goal.get('target_positions')
        if target_positions:
            required_targets = set(target_positions)
        else:
            raise ValueError("No target_positions in goal and none provided. "
                           "Use 'target_positions' in JSON or pass required_targets.")

    # Find rotatable fixed-position pieces
    rotatable = []
    for row in range(board.size):
        for col in range(board.size):
            piece = board.get_piece(row, col)
            if piece and piece.fixed_position:
                rotatable.append((row, col))

    return solve_puzzle(
        board=board,
        available=available,
        required_targets=required_targets,
        required_checkpoints=checkpoints,
        max_solutions=max_solutions,
        rotatable_positions=rotatable
    )


if __name__ == '__main__':
    # Test with expert_47
    print("Solving expert_47.json...")
    print("(Target positions are now read from the JSON file)")
    print()

    result = solve_challenge_file('examples/expert_47.json', max_solutions=2)

    print(f"Solved: {result.solved}")
    print(f"Nodes visited: {result.nodes_visited}")
    print(f"Solutions found: {len(result.solutions)}")

    for i, sol in enumerate(result.solutions):
        print(f"\n=== Solution {i+1} ===")
        print("Placements:")
        for ptype, row, col, ori in sol['placements']:
            print(f"  {ptype} at ({row},{col}) orientation {ori} ({Direction(ori).name})")
        print("Rotations:")
        for (row, col), ori in sol['rotations'].items():
            print(f"  ({row},{col}): {Direction(ori).name}")

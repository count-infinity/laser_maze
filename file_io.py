"""
File I/O for Laser Maze game.

Handles loading and saving board states and challenges in JSON format.
"""

import json
from pathlib import Path
from typing import Optional, Union

from pieces import create_piece, Direction, PieceType
from board import Board


def save_board(board: Board, filepath: Union[str, Path]) -> None:
    """
    Save board state to a JSON file.

    Args:
        board: Board to save
        filepath: Path to output file
    """
    data = board_to_dict(board)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_board(filepath: Union[str, Path]) -> Board:
    """
    Load board state from a JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded Board instance
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    return board_from_dict(data)


def board_to_dict(board: Board) -> dict:
    """
    Convert board to dictionary for JSON serialization.

    Args:
        board: Board to convert

    Returns:
        Dictionary representation
    """
    pieces = []

    for row in range(board.size):
        for col in range(board.size):
            piece = board.get_piece(row, col)
            if piece is not None:
                piece_data = {
                    'type': PIECE_TYPE_TO_NAME[piece.piece_type],
                    'position': [row, col],
                    'orientation': int(piece.orientation),
                    'fixed': piece.fixed,
                }
                # Only include fixed_position if True (to keep JSON cleaner)
                if piece.fixed_position:
                    piece_data['fixed_position'] = True
                pieces.append(piece_data)

    return {
        'grid_size': board.size,
        'pieces': pieces,
    }


def board_from_dict(data: dict) -> Board:
    """
    Create board from dictionary.

    Args:
        data: Dictionary with board data

    Returns:
        Board instance
    """
    size = data.get('grid_size', 5)
    board = Board(size=size)

    for piece_data in data.get('pieces', []):
        piece = create_piece(
            piece_type=piece_data['type'],
            orientation=piece_data.get('orientation', 0),
            fixed=piece_data.get('fixed', False),
            fixed_position=piece_data.get('fixed_position', False),
        )
        row, col = piece_data['position']
        board.set_piece(row, col, piece)

    return board


def save_challenge(board: Board, goal: dict, available: list,
                   filepath: Union[str, Path]) -> None:
    """
    Save a complete challenge to JSON.

    Args:
        board: Initial board state with fixed pieces
        goal: Goal specification (targets, checkpoints)
        available: List of available pieces to place
        filepath: Output path
    """
    data = board_to_dict(board)
    data['goal'] = goal
    data['available'] = available

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_challenge(filepath: Union[str, Path]) -> dict:
    """
    Load a challenge from JSON file.

    Args:
        filepath: Path to challenge JSON

    Returns:
        Dictionary with 'board', 'goal', and 'available' keys

    Goal format supports two modes:
        - "targets": N (legacy) - hit N target_mirrors, any positions
        - "target_positions": [[r,c], ...] - hit specific target_mirror positions
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    board = board_from_dict(data)

    # Normalize goal format
    raw_goal = data.get('goal', {})
    goal = {
        'checkpoints': raw_goal.get('checkpoints', []),
    }

    # Handle target specification
    if 'target_positions' in raw_goal:
        # New format: explicit target positions
        goal['target_positions'] = [tuple(pos) for pos in raw_goal['target_positions']]
        goal['targets'] = len(goal['target_positions'])
    else:
        # Legacy format: just a count
        goal['targets'] = raw_goal.get('targets', 1)
        goal['target_positions'] = None  # Any target_mirrors count

    return {
        'board': board,
        'goal': goal,
        'available': data.get('available', []),
    }


def board_to_json(board: Board) -> str:
    """Convert board to JSON string."""
    return json.dumps(board_to_dict(board), indent=2)


def board_from_json(json_str: str) -> Board:
    """Create board from JSON string."""
    return board_from_dict(json.loads(json_str))


# Mapping between PieceType enum and string names
PIECE_TYPE_TO_NAME = {
    PieceType.LASER: 'laser',
    PieceType.MIRROR: 'mirror',
    PieceType.TARGET_MIRROR: 'target_mirror',
    PieceType.BEAM_SPLITTER: 'beam_splitter',
    PieceType.DOUBLE_MIRROR: 'double_mirror',
    PieceType.CHECKPOINT: 'checkpoint',
    PieceType.CELL_BLOCKER: 'cell_blocker',
}

NAME_TO_PIECE_TYPE = {v: k for k, v in PIECE_TYPE_TO_NAME.items()}

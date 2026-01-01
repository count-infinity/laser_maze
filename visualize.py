"""
ASCII visualization for Laser Maze game.

Renders the board state and laser beam path to text for terminal display.
Supports both Unicode box drawing and ASCII fallback for Windows console.
"""

import sys
from typing import Optional, List, Set, Tuple

from pieces import Direction, Piece, Laser, TargetMirror
from board import Board
from laser import LaserResult, get_beam_at_cell


def _supports_unicode() -> bool:
    """Check if the terminal supports Unicode output."""
    try:
        # Try to encode a box drawing character
        '┌'.encode(sys.stdout.encoding or 'utf-8')
        return True
    except (UnicodeEncodeError, LookupError):
        return False


# Detect Unicode support
USE_UNICODE = _supports_unicode()

if USE_UNICODE:
    # Unicode box drawing characters
    BOX_TL = '┌'
    BOX_TR = '┐'
    BOX_BL = '└'
    BOX_BR = '┘'
    BOX_H = '─'
    BOX_V = '│'
    BOX_T = '┬'
    BOX_B = '┴'
    BOX_L = '├'
    BOX_R = '┤'
    BOX_X = '┼'

    # Beam direction characters
    BEAM_CHARS = {
        frozenset([Direction.NORTH]): '│',
        frozenset([Direction.SOUTH]): '│',
        frozenset([Direction.EAST]): '─',
        frozenset([Direction.WEST]): '─',
        frozenset([Direction.NORTH, Direction.SOUTH]): '│',
        frozenset([Direction.EAST, Direction.WEST]): '─',
        frozenset([Direction.NORTH, Direction.EAST]): '└',
        frozenset([Direction.NORTH, Direction.WEST]): '┘',
        frozenset([Direction.SOUTH, Direction.EAST]): '┌',
        frozenset([Direction.SOUTH, Direction.WEST]): '┐',
        frozenset([Direction.NORTH, Direction.SOUTH, Direction.EAST]): '├',
        frozenset([Direction.NORTH, Direction.SOUTH, Direction.WEST]): '┤',
        frozenset([Direction.EAST, Direction.WEST, Direction.NORTH]): '┴',
        frozenset([Direction.EAST, Direction.WEST, Direction.SOUTH]): '┬',
        frozenset([Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]): '┼',
    }

    # Direction arrows
    DIR_ARROWS = {
        Direction.NORTH: '↑',
        Direction.EAST: '→',
        Direction.SOUTH: '↓',
        Direction.WEST: '←',
    }

    MIRROR_SLASH = '/'
    MIRROR_BACKSLASH = '\\'
else:
    # ASCII fallback
    BOX_TL = '+'
    BOX_TR = '+'
    BOX_BL = '+'
    BOX_BR = '+'
    BOX_H = '-'
    BOX_V = '|'
    BOX_T = '+'
    BOX_B = '+'
    BOX_L = '+'
    BOX_R = '+'
    BOX_X = '+'

    BEAM_CHARS = {
        frozenset([Direction.NORTH]): '|',
        frozenset([Direction.SOUTH]): '|',
        frozenset([Direction.EAST]): '-',
        frozenset([Direction.WEST]): '-',
        frozenset([Direction.NORTH, Direction.SOUTH]): '|',
        frozenset([Direction.EAST, Direction.WEST]): '-',
        frozenset([Direction.NORTH, Direction.EAST]): '+',
        frozenset([Direction.NORTH, Direction.WEST]): '+',
        frozenset([Direction.SOUTH, Direction.EAST]): '+',
        frozenset([Direction.SOUTH, Direction.WEST]): '+',
        frozenset([Direction.NORTH, Direction.SOUTH, Direction.EAST]): '+',
        frozenset([Direction.NORTH, Direction.SOUTH, Direction.WEST]): '+',
        frozenset([Direction.EAST, Direction.WEST, Direction.NORTH]): '+',
        frozenset([Direction.EAST, Direction.WEST, Direction.SOUTH]): '+',
        frozenset([Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]): '+',
    }

    DIR_ARROWS = {
        Direction.NORTH: '^',
        Direction.EAST: '>',
        Direction.SOUTH: 'v',
        Direction.WEST: '<',
    }

    MIRROR_SLASH = '/'
    MIRROR_BACKSLASH = '\\'


def get_beam_char(directions: List[Direction]) -> str:
    """Get the character to display for beam directions at a cell."""
    if not directions:
        return ' '
    key = frozenset(directions)
    return BEAM_CHARS.get(key, '*')


def get_piece_symbol(piece: Piece) -> str:
    """Get display symbol for a piece using current character set."""
    piece_type = type(piece).__name__
    orientation = piece.orientation

    if piece_type == 'Laser':
        return 'L' + DIR_ARROWS[orientation]
    elif piece_type == 'Mirror':
        if orientation in (Direction.NORTH, Direction.SOUTH):
            return 'M' + MIRROR_BACKSLASH
        else:
            return 'M' + MIRROR_SLASH
    elif piece_type == 'TargetMirror':
        return 'T' + DIR_ARROWS[orientation]
    elif piece_type == 'BeamSplitter':
        if orientation in (Direction.NORTH, Direction.SOUTH):
            return 'B' + MIRROR_BACKSLASH
        else:
            return 'B' + MIRROR_SLASH
    elif piece_type == 'DoubleMirror':
        if orientation in (Direction.NORTH, Direction.SOUTH):
            return 'D' + MIRROR_BACKSLASH
        else:
            return 'D' + MIRROR_SLASH
    elif piece_type == 'Checkpoint':
        # Show orientation: | for N/S passage, - for E/W passage
        if orientation in (Direction.NORTH, Direction.SOUTH):
            return 'C|'
        else:
            return 'C-'
    elif piece_type == 'CellBlocker':
        return 'XX'
    else:
        return '??'


def render_cell(piece: Optional[Piece], beam_dirs: List[Direction],
                is_target_hit: bool = False) -> str:
    """
    Render a single cell's contents.

    Args:
        piece: Piece in the cell, or None
        beam_dirs: Directions the beam travels through this cell
        is_target_hit: Whether a target in this cell was hit

    Returns:
        3-character string for the cell

    Piece state indicators:
        *XY - Fixed piece (cannot move, remove, or rotate)
        XY~ - Fixed-position piece (can rotate, but cannot move/remove)
         XY - Moveable piece (can move, remove, and rotate)
        T*! - Target hit by laser beam
    """
    if piece is None:
        if beam_dirs:
            beam_char = get_beam_char(beam_dirs)
            return f' {beam_char} '
        return '   '

    symbol = get_piece_symbol(piece)

    # Add hit indicator for targets (takes precedence over other formatting)
    if is_target_hit and isinstance(piece, TargetMirror):
        return 'T*!'  # T*! = target hit

    # Format based on piece state (cell width is 3 chars)
    if piece.fixed:
        # Fixed piece: show symbol with asterisk prefix
        if len(symbol) >= 2:
            return f'*{symbol[0]}{symbol[1]}'
        else:
            return f'*{symbol} '
    elif piece.fixed_position:
        # Fixed-position (rotatable): show with tilde suffix
        if len(symbol) >= 2:
            return f'{symbol[0]}{symbol[1]}~'  # Tilde indicates rotatable
        else:
            return f'{symbol}~ '
    else:
        # Moveable piece: space-padded
        if len(symbol) == 2:
            return f' {symbol}'
        elif len(symbol) == 1:
            return f' {symbol} '
        else:
            return symbol[:3]


def render_board(board: Board, laser_result: Optional[LaserResult] = None,
                 show_coords: bool = False) -> str:
    """
    Render the board as ASCII art.

    Args:
        board: The game board to render
        laser_result: Optional laser firing result to show beam path
        show_coords: Whether to show row/column coordinates

    Returns:
        Multi-line string representation of the board

    Piece state indicators:
        *XY - Fixed piece (cannot move, remove, or rotate)
        XY~ - Fixed-position piece (can rotate, but cannot move/remove)
         XY - Moveable piece (can move, remove, and rotate)
        T*! - Target hit by laser beam
    """
    size = board.size
    cell_width = 3
    lines = []

    # Get targets hit and beam info
    targets_hit: Set[Tuple[int, int]] = set()
    beam_at_cell: dict = {}

    if laser_result:
        targets_hit = laser_result.targets_hit
        for row in range(size):
            for col in range(size):
                dirs = get_beam_at_cell(laser_result, row, col)
                if dirs:
                    beam_at_cell[(row, col)] = dirs

    # Column headers
    if show_coords:
        header = '    '
        for col in range(size):
            header += f' {col}  '
        lines.append(header)

    # Top border
    top_border = BOX_TL
    for col in range(size):
        top_border += BOX_H * cell_width
        top_border += BOX_T if col < size - 1 else BOX_TR
    if show_coords:
        top_border = '   ' + top_border
    lines.append(top_border)

    # Rows
    for row in range(size):
        # Cell contents
        row_str = BOX_V
        for col in range(size):
            piece = board.get_piece(row, col)
            beam_dirs = beam_at_cell.get((row, col), [])
            is_hit = (row, col) in targets_hit

            cell = render_cell(piece, beam_dirs, is_hit)
            row_str += cell + BOX_V

        if show_coords:
            row_str = f' {row} ' + row_str
        lines.append(row_str)

        # Row separator or bottom border
        if row < size - 1:
            sep = BOX_L
            for col in range(size):
                sep += BOX_H * cell_width
                sep += BOX_X if col < size - 1 else BOX_R
        else:
            sep = BOX_BL
            for col in range(size):
                sep += BOX_H * cell_width
                sep += BOX_B if col < size - 1 else BOX_BR

        if show_coords:
            sep = '   ' + sep
        lines.append(sep)

    # Status line
    if laser_result:
        status = f'Targets hit: {laser_result.num_targets}'
        if laser_result.checkpoints_passed:
            status += f'  Checkpoints: {laser_result.num_checkpoints}'
        lines.append(status)

    return '\n'.join(lines)


def print_board(board: Board, laser_result: Optional[LaserResult] = None,
                show_coords: bool = False) -> None:
    """Print the board to stdout."""
    print(render_board(board, laser_result, show_coords))


def render_compact(board: Board) -> str:
    """
    Render a compact single-line representation of the board.
    Useful for logging or simple displays.
    """
    lines = []
    for row in range(board.size):
        row_str = ''
        for col in range(board.size):
            piece = board.get_piece(row, col)
            if piece is None:
                row_str += '.'
            else:
                type_chars = {
                    'Laser': 'L',
                    'Mirror': 'M',
                    'TargetMirror': 'T',
                    'BeamSplitter': 'B',
                    'DoubleMirror': 'D',
                    'Checkpoint': 'C',
                    'CellBlocker': 'X',
                }
                row_str += type_chars.get(type(piece).__name__, '?')
        lines.append(row_str)
    return '\n'.join(lines)

"""
Board state management for Laser Maze game.

The board is a 5x5 grid where each cell can contain one piece or be empty.
Coordinates are (row, col) with (0,0) at top-left.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import copy

from pieces import Piece, Direction, PieceType, Laser, create_piece


@dataclass
class Board:
    """
    Represents the game board state.

    Attributes:
        size: Grid dimensions (default 5x5)
        grid: 2D list of pieces (None for empty cells)
        laser_pos: Position of the laser emitter
        laser_dir: Direction the laser is pointing
    """
    size: int = 5
    grid: List[List[Optional[Piece]]] = field(default_factory=list)
    laser_pos: Optional[Tuple[int, int]] = None
    laser_dir: Optional[Direction] = None

    def __post_init__(self):
        """Initialize empty grid if not provided."""
        if not self.grid:
            self.grid = [[None for _ in range(self.size)] for _ in range(self.size)]

    def is_valid_pos(self, row: int, col: int) -> bool:
        """Check if position is within grid bounds."""
        return 0 <= row < self.size and 0 <= col < self.size

    def get_piece(self, row: int, col: int) -> Optional[Piece]:
        """Get piece at position, or None if empty or out of bounds."""
        if not self.is_valid_pos(row, col):
            return None
        return self.grid[row][col]

    def set_piece(self, row: int, col: int, piece: Optional[Piece]) -> bool:
        """
        Place a piece at position.

        Args:
            row: Row index
            col: Column index
            piece: Piece to place, or None to clear

        Returns:
            True if successful, False if position invalid
        """
        if not self.is_valid_pos(row, col):
            return False

        self.grid[row][col] = piece

        # Track laser position
        if isinstance(piece, Laser):
            self.laser_pos = (row, col)
            self.laser_dir = piece.orientation

        return True

    def place_piece(self, row: int, col: int, piece: Piece) -> bool:
        """
        Place a piece if the cell is empty.

        Args:
            row: Row index
            col: Column index
            piece: Piece to place

        Returns:
            True if successful, False if cell occupied or invalid
        """
        if not self.is_valid_pos(row, col):
            return False

        existing = self.grid[row][col]
        if existing is not None:
            return False

        return self.set_piece(row, col, piece)

    def remove_piece(self, row: int, col: int) -> Optional[Piece]:
        """
        Remove and return piece at position.

        Returns:
            The removed piece, or None if cell was empty/invalid
        """
        if not self.is_valid_pos(row, col):
            return None

        piece = self.grid[row][col]
        if piece is None:
            return None

        if not piece.can_remove:
            return None  # Cannot remove fixed or fixed_position pieces

        self.grid[row][col] = None

        # Clear laser tracking if removing laser
        if isinstance(piece, Laser) and self.laser_pos == (row, col):
            self.laser_pos = None
            self.laser_dir = None

        return piece

    def rotate_piece(self, row: int, col: int, clockwise: bool = True) -> bool:
        """
        Rotate piece at position.

        Args:
            row: Row index
            col: Column index
            clockwise: True for CW, False for CCW

        Returns:
            True if rotated, False if no piece or fixed
        """
        piece = self.get_piece(row, col)
        if piece is None or not piece.can_rotate:
            return False

        if clockwise:
            piece.rotate_cw()
        else:
            piece.rotate_ccw()

        # Update laser direction if rotating laser
        if isinstance(piece, Laser) and self.laser_pos == (row, col):
            self.laser_dir = piece.orientation

        return True

    def find_laser(self) -> Optional[Tuple[int, int]]:
        """Find the laser position on the board."""
        for row in range(self.size):
            for col in range(self.size):
                piece = self.grid[row][col]
                if isinstance(piece, Laser):
                    return (row, col)
        return None

    def get_all_pieces(self) -> Dict[Tuple[int, int], Piece]:
        """Return dict of all pieces on the board with their positions."""
        pieces = {}
        for row in range(self.size):
            for col in range(self.size):
                piece = self.grid[row][col]
                if piece is not None:
                    pieces[(row, col)] = piece
        return pieces

    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """Return list of empty cell positions."""
        empty = []
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row][col] is None:
                    empty.append((row, col))
        return empty

    def clear(self) -> None:
        """Remove all pieces from the board."""
        self.grid = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.laser_pos = None
        self.laser_dir = None

    def copy(self) -> 'Board':
        """Create a deep copy of the board."""
        new_board = Board(size=self.size)
        new_board.laser_pos = self.laser_pos
        new_board.laser_dir = self.laser_dir

        for row in range(self.size):
            for col in range(self.size):
                piece = self.grid[row][col]
                if piece is not None:
                    new_board.grid[row][col] = piece.copy()

        return new_board

    def to_dict(self) -> dict:
        """Convert board state to dictionary for serialization."""
        pieces = []
        for row in range(self.size):
            for col in range(self.size):
                piece = self.grid[row][col]
                if piece is not None:
                    # Find piece type name
                    type_name = None
                    for name, cls in PIECE_TYPE_NAMES.items():
                        if piece.piece_type == cls:
                            type_name = name
                            break

                    pieces.append({
                        'type': type_name,
                        'position': [row, col],
                        'orientation': int(piece.orientation),
                        'fixed': piece.fixed,
                    })

        return {
            'grid_size': self.size,
            'pieces': pieces,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Board':
        """Create board from dictionary."""
        size = data.get('grid_size', 5)
        board = cls(size=size)

        for piece_data in data.get('pieces', []):
            piece = create_piece(
                piece_type=piece_data['type'],
                orientation=piece_data.get('orientation', 0),
                fixed=piece_data.get('fixed', False),
            )
            row, col = piece_data['position']
            board.set_piece(row, col, piece)

        return board


# Mapping piece types to string names for serialization
PIECE_TYPE_NAMES = {
    'laser': PieceType.LASER,
    'mirror': PieceType.MIRROR,
    'target_mirror': PieceType.TARGET_MIRROR,
    'beam_splitter': PieceType.BEAM_SPLITTER,
    'double_mirror': PieceType.DOUBLE_MIRROR,
    'checkpoint': PieceType.CHECKPOINT,
    'cell_blocker': PieceType.CELL_BLOCKER,
}

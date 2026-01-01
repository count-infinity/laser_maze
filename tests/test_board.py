"""Unit tests for board.py - Board state management."""

import pytest
from board import Board
from pieces import Direction, Laser, Mirror, TargetMirror, create_piece


class TestBoardInit:
    """Tests for Board initialization."""

    def test_default_size(self):
        """Test default board size is 5x5."""
        board = Board()
        assert board.size == 5

    def test_custom_size(self):
        """Test custom board size."""
        board = Board(size=7)
        assert board.size == 7

    def test_grid_initialized_empty(self):
        """Test grid is initialized with None values."""
        board = Board()
        for row in range(board.size):
            for col in range(board.size):
                assert board.grid[row][col] is None

    def test_laser_tracking_initially_none(self):
        """Test laser position tracking starts as None."""
        board = Board()
        assert board.laser_pos is None
        assert board.laser_dir is None


class TestBoardValidPosition:
    """Tests for position validation."""

    def test_valid_positions(self):
        """Test valid positions return True."""
        board = Board(size=5)
        assert board.is_valid_pos(0, 0) is True
        assert board.is_valid_pos(4, 4) is True
        assert board.is_valid_pos(2, 3) is True

    def test_invalid_negative_positions(self):
        """Test negative positions return False."""
        board = Board(size=5)
        assert board.is_valid_pos(-1, 0) is False
        assert board.is_valid_pos(0, -1) is False

    def test_invalid_overflow_positions(self):
        """Test positions >= size return False."""
        board = Board(size=5)
        assert board.is_valid_pos(5, 0) is False
        assert board.is_valid_pos(0, 5) is False
        assert board.is_valid_pos(5, 5) is False


class TestBoardGetSetPiece:
    """Tests for getting and setting pieces."""

    def test_get_piece_empty_cell(self):
        """Test getting piece from empty cell returns None."""
        board = Board()
        assert board.get_piece(2, 2) is None

    def test_get_piece_invalid_position(self):
        """Test getting piece from invalid position returns None."""
        board = Board()
        assert board.get_piece(-1, 0) is None
        assert board.get_piece(10, 10) is None

    def test_set_piece(self):
        """Test setting a piece."""
        board = Board()
        mirror = Mirror(orientation=Direction.NORTH)
        result = board.set_piece(2, 2, mirror)
        assert result is True
        assert board.get_piece(2, 2) is mirror

    def test_set_piece_invalid_position(self):
        """Test setting piece at invalid position fails."""
        board = Board()
        mirror = Mirror()
        result = board.set_piece(-1, 0, mirror)
        assert result is False

    def test_set_piece_overwrites(self):
        """Test set_piece overwrites existing piece."""
        board = Board()
        m1 = Mirror(orientation=Direction.NORTH)
        m2 = Mirror(orientation=Direction.EAST)
        board.set_piece(2, 2, m1)
        board.set_piece(2, 2, m2)
        assert board.get_piece(2, 2) is m2

    def test_set_piece_none_clears(self):
        """Test setting None clears the cell."""
        board = Board()
        mirror = Mirror()
        board.set_piece(2, 2, mirror)
        board.set_piece(2, 2, None)
        assert board.get_piece(2, 2) is None


class TestBoardLaserTracking:
    """Tests for laser position tracking."""

    def test_laser_tracking_on_set(self):
        """Test laser position is tracked when set."""
        board = Board()
        laser = Laser(orientation=Direction.SOUTH)
        board.set_piece(0, 2, laser)
        assert board.laser_pos == (0, 2)
        assert board.laser_dir == Direction.SOUTH

    def test_laser_tracking_updates(self):
        """Test laser tracking updates when laser moves."""
        board = Board()
        laser = Laser(orientation=Direction.SOUTH)
        board.set_piece(0, 2, laser)
        board.set_piece(0, 2, None)  # Remove
        board.set_piece(1, 1, laser)  # Place elsewhere
        assert board.laser_pos == (1, 1)

    def test_find_laser(self):
        """Test finding laser on board."""
        board = Board()
        laser = Laser(orientation=Direction.EAST)
        board.set_piece(3, 1, laser)
        assert board.find_laser() == (3, 1)

    def test_find_laser_none(self):
        """Test find_laser returns None when no laser."""
        board = Board()
        board.set_piece(2, 2, Mirror())
        assert board.find_laser() is None


class TestBoardPlacePiece:
    """Tests for place_piece (non-overwriting placement)."""

    def test_place_piece_empty_cell(self):
        """Test placing piece on empty cell succeeds."""
        board = Board()
        mirror = Mirror()
        result = board.place_piece(2, 2, mirror)
        assert result is True
        assert board.get_piece(2, 2) is mirror

    def test_place_piece_occupied_cell(self):
        """Test placing piece on occupied cell fails."""
        board = Board()
        m1 = Mirror()
        m2 = Mirror()
        board.place_piece(2, 2, m1)
        result = board.place_piece(2, 2, m2)
        assert result is False
        assert board.get_piece(2, 2) is m1

    def test_place_piece_invalid_position(self):
        """Test placing piece at invalid position fails."""
        board = Board()
        result = board.place_piece(-1, 0, Mirror())
        assert result is False


class TestBoardRemovePiece:
    """Tests for remove_piece."""

    def test_remove_piece(self):
        """Test removing a piece returns it and clears cell."""
        board = Board()
        mirror = Mirror()
        board.set_piece(2, 2, mirror)
        removed = board.remove_piece(2, 2)
        assert removed is mirror
        assert board.get_piece(2, 2) is None

    def test_remove_piece_empty_cell(self):
        """Test removing from empty cell returns None."""
        board = Board()
        removed = board.remove_piece(2, 2)
        assert removed is None

    def test_remove_piece_invalid_position(self):
        """Test removing from invalid position returns None."""
        board = Board()
        removed = board.remove_piece(-1, 0)
        assert removed is None

    def test_remove_fixed_piece_fails(self):
        """Test removing fixed piece fails."""
        board = Board()
        mirror = Mirror(fixed=True)
        board.set_piece(2, 2, mirror)
        removed = board.remove_piece(2, 2)
        assert removed is None
        assert board.get_piece(2, 2) is mirror

    def test_remove_laser_clears_tracking(self):
        """Test removing laser clears laser tracking."""
        board = Board()
        laser = Laser(orientation=Direction.SOUTH)
        board.set_piece(0, 2, laser)
        board.remove_piece(0, 2)
        assert board.laser_pos is None
        assert board.laser_dir is None


class TestBoardRotatePiece:
    """Tests for rotate_piece."""

    def test_rotate_piece_cw(self):
        """Test rotating piece clockwise."""
        board = Board()
        mirror = Mirror(orientation=Direction.NORTH)
        board.set_piece(2, 2, mirror)
        result = board.rotate_piece(2, 2, clockwise=True)
        assert result is True
        assert board.get_piece(2, 2).orientation == Direction.EAST

    def test_rotate_piece_ccw(self):
        """Test rotating piece counter-clockwise."""
        board = Board()
        mirror = Mirror(orientation=Direction.NORTH)
        board.set_piece(2, 2, mirror)
        result = board.rotate_piece(2, 2, clockwise=False)
        assert result is True
        assert board.get_piece(2, 2).orientation == Direction.WEST

    def test_rotate_empty_cell(self):
        """Test rotating empty cell fails."""
        board = Board()
        result = board.rotate_piece(2, 2)
        assert result is False

    def test_rotate_fixed_piece_fails(self):
        """Test rotating fixed piece fails."""
        board = Board()
        mirror = Mirror(orientation=Direction.NORTH, fixed=True)
        board.set_piece(2, 2, mirror)
        result = board.rotate_piece(2, 2)
        assert result is False
        assert board.get_piece(2, 2).orientation == Direction.NORTH

    def test_rotate_laser_updates_tracking(self):
        """Test rotating laser updates direction tracking."""
        board = Board()
        laser = Laser(orientation=Direction.NORTH)
        board.set_piece(0, 2, laser)
        board.rotate_piece(0, 2)
        assert board.laser_dir == Direction.EAST


class TestBoardUtilityMethods:
    """Tests for utility methods."""

    def test_get_all_pieces(self):
        """Test getting all pieces on board."""
        board = Board()
        m1 = Mirror()
        m2 = Mirror()
        board.set_piece(0, 0, m1)
        board.set_piece(4, 4, m2)
        pieces = board.get_all_pieces()
        assert len(pieces) == 2
        assert (0, 0) in pieces
        assert (4, 4) in pieces
        assert pieces[(0, 0)] is m1
        assert pieces[(4, 4)] is m2

    def test_get_empty_cells(self):
        """Test getting empty cells."""
        board = Board(size=3)
        board.set_piece(0, 0, Mirror())
        board.set_piece(1, 1, Mirror())
        empty = board.get_empty_cells()
        assert len(empty) == 7  # 9 total - 2 occupied
        assert (0, 0) not in empty
        assert (1, 1) not in empty
        assert (0, 1) in empty

    def test_clear(self):
        """Test clearing the board."""
        board = Board()
        board.set_piece(0, 0, Mirror())
        board.set_piece(2, 2, Laser())
        board.clear()
        assert board.get_piece(0, 0) is None
        assert board.get_piece(2, 2) is None
        assert board.laser_pos is None
        assert board.laser_dir is None


class TestBoardCopy:
    """Tests for board copying."""

    def test_copy_preserves_pieces(self):
        """Test copy preserves all pieces."""
        board = Board()
        board.set_piece(0, 0, Mirror(orientation=Direction.EAST))
        board.set_piece(2, 2, Laser(orientation=Direction.SOUTH))
        copied = board.copy()
        assert copied.get_piece(0, 0).orientation == Direction.EAST
        assert copied.get_piece(2, 2).orientation == Direction.SOUTH

    def test_copy_is_independent(self):
        """Test copy is independent of original."""
        board = Board()
        board.set_piece(2, 2, Mirror(orientation=Direction.NORTH))
        copied = board.copy()
        copied.rotate_piece(2, 2)
        assert board.get_piece(2, 2).orientation == Direction.NORTH
        assert copied.get_piece(2, 2).orientation == Direction.EAST

    def test_copy_preserves_laser_tracking(self):
        """Test copy preserves laser tracking."""
        board = Board()
        board.set_piece(0, 2, Laser(orientation=Direction.SOUTH))
        copied = board.copy()
        assert copied.laser_pos == (0, 2)
        assert copied.laser_dir == Direction.SOUTH


class TestBoardSerialization:
    """Tests for board dict serialization."""

    def test_from_dict(self):
        """Test creating board from dict."""
        data = {
            'grid_size': 5,
            'pieces': [
                {'type': 'laser', 'position': [0, 2], 'orientation': 2, 'fixed': True},
                {'type': 'mirror', 'position': [2, 2], 'orientation': 1, 'fixed': False},
            ]
        }
        board = Board.from_dict(data)
        assert board.size == 5
        assert isinstance(board.get_piece(0, 2), Laser)
        assert board.get_piece(0, 2).orientation == Direction.SOUTH
        assert board.get_piece(0, 2).fixed is True
        assert isinstance(board.get_piece(2, 2), Mirror)

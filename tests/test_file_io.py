"""Unit tests for file_io.py - JSON serialization."""

import pytest
import json
import tempfile
import os
from pathlib import Path

from file_io import (
    save_board, load_board, board_to_dict, board_from_dict,
    save_challenge, load_challenge, board_to_json, board_from_json,
    PIECE_TYPE_TO_NAME, NAME_TO_PIECE_TYPE
)
from board import Board
from pieces import Direction, Laser, Mirror, TargetMirror, PieceType


class TestBoardToDict:
    """Tests for board_to_dict function."""

    def test_empty_board(self):
        """Test serializing empty board."""
        board = Board()
        data = board_to_dict(board)
        assert data['grid_size'] == 5
        assert data['pieces'] == []

    def test_board_with_pieces(self):
        """Test serializing board with pieces."""
        board = Board()
        board.set_piece(0, 2, Laser(orientation=Direction.SOUTH, fixed=True))
        board.set_piece(2, 2, Mirror(orientation=Direction.EAST))

        data = board_to_dict(board)
        assert len(data['pieces']) == 2

        laser_data = next(p for p in data['pieces'] if p['type'] == 'laser')
        assert laser_data['position'] == [0, 2]
        assert laser_data['orientation'] == 2  # SOUTH
        assert laser_data['fixed'] is True

        mirror_data = next(p for p in data['pieces'] if p['type'] == 'mirror')
        assert mirror_data['position'] == [2, 2]
        assert mirror_data['orientation'] == 1  # EAST
        assert mirror_data['fixed'] is False


class TestBoardFromDict:
    """Tests for board_from_dict function."""

    def test_empty_board(self):
        """Test deserializing empty board."""
        data = {'grid_size': 5, 'pieces': []}
        board = board_from_dict(data)
        assert board.size == 5
        assert board.get_all_pieces() == {}

    def test_board_with_pieces(self):
        """Test deserializing board with pieces."""
        data = {
            'grid_size': 5,
            'pieces': [
                {'type': 'laser', 'position': [0, 2], 'orientation': 2, 'fixed': True},
                {'type': 'mirror', 'position': [2, 2], 'orientation': 1, 'fixed': False},
            ]
        }
        board = board_from_dict(data)

        laser = board.get_piece(0, 2)
        assert isinstance(laser, Laser)
        assert laser.orientation == Direction.SOUTH
        assert laser.fixed is True

        mirror = board.get_piece(2, 2)
        assert isinstance(mirror, Mirror)
        assert mirror.orientation == Direction.EAST
        assert mirror.fixed is False

    def test_custom_grid_size(self):
        """Test deserializing board with custom size."""
        data = {'grid_size': 7, 'pieces': []}
        board = board_from_dict(data)
        assert board.size == 7

    def test_default_values(self):
        """Test default values for missing fields."""
        data = {
            'pieces': [
                {'type': 'mirror', 'position': [2, 2]}
            ]
        }
        board = board_from_dict(data)
        mirror = board.get_piece(2, 2)
        assert mirror.orientation == Direction.NORTH  # Default 0
        assert mirror.fixed is False  # Default


class TestBoardJsonString:
    """Tests for JSON string conversion."""

    def test_board_to_json(self):
        """Test board to JSON string."""
        board = Board()
        board.set_piece(2, 2, Mirror())
        json_str = board_to_json(board)
        assert isinstance(json_str, str)
        # Verify it's valid JSON
        data = json.loads(json_str)
        assert 'pieces' in data

    def test_board_from_json(self):
        """Test board from JSON string."""
        json_str = '{"grid_size": 5, "pieces": [{"type": "mirror", "position": [2, 2], "orientation": 0}]}'
        board = board_from_json(json_str)
        assert board.get_piece(2, 2) is not None


class TestSaveLoadBoard:
    """Tests for file save/load operations."""

    def test_save_and_load_board(self):
        """Test saving and loading a board."""
        board = Board()
        board.set_piece(0, 2, Laser(orientation=Direction.SOUTH))
        board.set_piece(2, 2, Mirror(orientation=Direction.EAST))
        board.set_piece(4, 4, TargetMirror(orientation=Direction.NORTH, fixed=True))

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            save_board(board, temp_path)
            loaded = load_board(temp_path)

            assert loaded.size == board.size
            assert isinstance(loaded.get_piece(0, 2), Laser)
            assert isinstance(loaded.get_piece(2, 2), Mirror)
            assert isinstance(loaded.get_piece(4, 4), TargetMirror)
            assert loaded.get_piece(4, 4).fixed is True
        finally:
            os.unlink(temp_path)

    def test_save_creates_valid_json(self):
        """Test saved file is valid JSON."""
        board = Board()
        board.set_piece(2, 2, Mirror())

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            save_board(board, temp_path)
            with open(temp_path, 'r') as f:
                data = json.load(f)
            assert 'grid_size' in data
            assert 'pieces' in data
        finally:
            os.unlink(temp_path)


class TestSaveLoadChallenge:
    """Tests for challenge save/load operations."""

    def test_save_and_load_challenge(self):
        """Test saving and loading a challenge."""
        board = Board()
        board.set_piece(0, 2, Laser(orientation=Direction.SOUTH, fixed=True))

        goal = {'targets': 2, 'checkpoints': [[1, 1]]}
        available = [
            {'type': 'mirror', 'orientation': 1},
            {'type': 'target_mirror', 'orientation': None},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            save_challenge(board, goal, available, temp_path)
            loaded = load_challenge(temp_path)

            assert isinstance(loaded['board'], Board)
            assert loaded['goal']['targets'] == 2
            assert len(loaded['available']) == 2
        finally:
            os.unlink(temp_path)

    def test_load_challenge_no_goal(self):
        """Test loading file without goal returns default."""
        data = {
            'grid_size': 5,
            'pieces': [
                {'type': 'laser', 'position': [0, 2], 'orientation': 2}
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            loaded = load_challenge(temp_path)
            assert loaded['goal']['targets'] == 1
            assert loaded['goal']['checkpoints'] == []
            assert loaded['goal']['target_positions'] is None  # Legacy mode
            assert loaded['available'] == []
        finally:
            os.unlink(temp_path)


class TestPieceTypeMappings:
    """Tests for piece type name mappings."""

    def test_all_piece_types_have_names(self):
        """Test all piece types have string names."""
        for piece_type in PieceType:
            if piece_type != PieceType.EMPTY:
                assert piece_type in PIECE_TYPE_TO_NAME

    def test_name_mapping_roundtrip(self):
        """Test name <-> type mapping is reversible."""
        for name, piece_type in NAME_TO_PIECE_TYPE.items():
            assert PIECE_TYPE_TO_NAME[piece_type] == name


class TestPathTypes:
    """Tests for different path types."""

    def test_save_with_string_path(self):
        """Test save with string path."""
        board = Board()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            save_board(board, temp_path)
            assert os.path.exists(temp_path)
        finally:
            os.unlink(temp_path)

    def test_save_with_path_object(self):
        """Test save with Path object."""
        board = Board()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)

        try:
            save_board(board, temp_path)
            assert temp_path.exists()
        finally:
            temp_path.unlink()

    def test_load_with_path_object(self):
        """Test load with Path object."""
        data = {'grid_size': 5, 'pieces': []}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = Path(f.name)

        try:
            board = load_board(temp_path)
            assert board.size == 5
        finally:
            temp_path.unlink()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_all_piece_types_serialize(self):
        """Test all piece types can be serialized."""
        from pieces import (
            Laser, Mirror, TargetMirror, BeamSplitter,
            DoubleMirror, Checkpoint, CellBlocker
        )

        board = Board(size=7)
        board.set_piece(0, 0, Laser())
        board.set_piece(0, 1, Mirror())
        board.set_piece(0, 2, TargetMirror())
        board.set_piece(0, 3, BeamSplitter())
        board.set_piece(0, 4, DoubleMirror())
        board.set_piece(0, 5, Checkpoint())
        board.set_piece(0, 6, CellBlocker())

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            save_board(board, temp_path)
            loaded = load_board(temp_path)
            assert len(loaded.get_all_pieces()) == 7
        finally:
            os.unlink(temp_path)

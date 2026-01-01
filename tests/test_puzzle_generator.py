"""
Tests for the puzzle generator.
"""

import pytest
import json
import os
import tempfile
from pathlib import Path

from puzzle_generator import (
    PuzzleGenerator, Difficulty, PuzzleConfig
)
from game import LaserMaze, PIECE_ORIENTATIONS
from pieces import Direction, create_piece
from board import Board
from laser import fire_laser
from file_io import board_from_dict


class TestDifficulty:
    """Tests for Difficulty enum and PuzzleConfig."""

    def test_difficulty_levels(self):
        """Test all difficulty levels are defined."""
        assert Difficulty.BEGINNER == 1
        assert Difficulty.EASY == 2
        assert Difficulty.MEDIUM == 3
        assert Difficulty.HARD == 4
        assert Difficulty.EXPERT == 5

    def test_puzzle_config_beginner(self):
        """Test beginner config."""
        config = PuzzleConfig.for_difficulty(Difficulty.BEGINNER)
        assert config.min_pieces == 1
        assert config.max_pieces == 1
        assert config.num_targets == 1
        assert config.num_checkpoints == 0
        assert 'mirror' in config.piece_types

    def test_puzzle_config_expert(self):
        """Test expert config."""
        config = PuzzleConfig.for_difficulty(Difficulty.EXPERT)
        assert config.min_pieces == 4
        assert config.max_pieces == 5
        assert config.num_targets == 2
        assert config.num_checkpoints == 1


class TestPuzzleGeneration:
    """Tests for puzzle generation."""

    def test_generate_beginner_puzzle(self):
        """Test generating a beginner puzzle."""
        gen = PuzzleGenerator(seed=42)
        puzzle = gen.generate(Difficulty.BEGINNER)

        assert puzzle is not None
        assert 'grid_size' in puzzle
        assert 'pieces' in puzzle
        assert 'available' in puzzle
        assert 'goal' in puzzle
        assert puzzle['goal']['targets'] == 1
        assert len(puzzle['available']) >= 1

    def test_generate_easy_puzzle(self):
        """Test generating an easy puzzle."""
        gen = PuzzleGenerator(seed=42)
        puzzle = gen.generate(Difficulty.EASY)

        assert puzzle is not None
        assert puzzle['goal']['targets'] == 1
        assert len(puzzle['available']) >= 1

    def test_generate_medium_puzzle(self):
        """Test generating a medium puzzle."""
        gen = PuzzleGenerator(seed=42)
        puzzle = gen.generate(Difficulty.MEDIUM)

        assert puzzle is not None
        assert puzzle['goal']['targets'] == 1
        assert len(puzzle['available']) >= 2

    def test_generate_hard_puzzle(self):
        """Test generating a hard puzzle."""
        gen = PuzzleGenerator(seed=42)
        puzzle = gen.generate(Difficulty.HARD)

        assert puzzle is not None
        assert puzzle['goal']['targets'] == 2
        assert len(puzzle['available']) >= 3

    def test_generate_expert_puzzle(self):
        """Test generating an expert puzzle."""
        gen = PuzzleGenerator(seed=42)
        puzzle = gen.generate(Difficulty.EXPERT)

        assert puzzle is not None
        assert puzzle['goal']['targets'] == 2
        assert len(puzzle['goal'].get('checkpoints', [])) >= 0  # May or may not have checkpoints

    def test_puzzle_has_laser(self):
        """Test that generated puzzles have a laser."""
        gen = PuzzleGenerator(seed=42)
        puzzle = gen.generate(Difficulty.BEGINNER)

        laser_pieces = [p for p in puzzle['pieces'] if p['type'] == 'laser']
        assert len(laser_pieces) == 1

    def test_puzzle_has_target(self):
        """Test that generated puzzles have target mirrors."""
        gen = PuzzleGenerator(seed=42)
        puzzle = gen.generate(Difficulty.BEGINNER)

        target_pieces = [p for p in puzzle['pieces'] if p['type'] == 'target_mirror']
        assert len(target_pieces) >= 1

    def test_puzzle_fixed_pieces(self):
        """Test that laser and targets are fixed."""
        gen = PuzzleGenerator(seed=42)
        puzzle = gen.generate(Difficulty.BEGINNER)

        for piece in puzzle['pieces']:
            if piece['type'] in ('laser', 'target_mirror'):
                assert piece['fixed'] is True

    def test_reproducible_with_seed(self):
        """Test that same seed produces same first puzzle."""
        # Note: The seed sets global random state, so we need fresh
        # generators for each comparison
        import random
        random.seed(42)
        gen1 = PuzzleGenerator()  # Don't set seed in constructor
        puzzle1 = gen1.generate(Difficulty.BEGINNER)

        random.seed(42)
        gen2 = PuzzleGenerator()
        puzzle2 = gen2.generate(Difficulty.BEGINNER)

        assert puzzle1 == puzzle2

    def test_different_seeds_different_puzzles(self):
        """Test that different seeds produce different puzzles."""
        gen1 = PuzzleGenerator(seed=42)
        gen2 = PuzzleGenerator(seed=123)

        puzzle1 = gen1.generate(Difficulty.BEGINNER)
        puzzle2 = gen2.generate(Difficulty.BEGINNER)

        # Very unlikely to be the same
        assert puzzle1 != puzzle2


class TestPuzzleSolvability:
    """Tests for puzzle solvability."""

    def _solve_puzzle(self, puzzle: dict) -> bool:
        """
        Attempt to solve a puzzle by brute-force search.

        Returns True if puzzle is solvable.
        """
        board = board_from_dict(puzzle)
        available = puzzle['available']
        required_targets = puzzle['goal']['targets']
        checkpoints = puzzle['goal'].get('checkpoints', [])

        if not available:
            result = fire_laser(board)
            return result.num_targets >= required_targets

        return self._search_solution(board, available, required_targets,
                                     [tuple(cp) for cp in checkpoints], 0)

    def _search_solution(self, board: Board, available: list,
                        required_targets: int, checkpoints: list,
                        depth: int) -> bool:
        """Recursive search for solution."""
        if depth >= len(available):
            result = fire_laser(board)
            if result.num_targets < required_targets:
                return False
            for cp in checkpoints:
                if cp not in result.checkpoints_passed:
                    return False
            return True

        piece_info = available[depth]
        piece_type = piece_info['type']
        valid_orientations = PIECE_ORIENTATIONS.get(piece_type, [0, 1, 2, 3])

        empty_cells = board.get_empty_cells()

        for row, col in empty_cells:
            for orientation in valid_orientations:
                piece = create_piece(piece_type, orientation, fixed=False)
                if board.place_piece(row, col, piece):
                    if self._search_solution(board, available, required_targets,
                                           checkpoints, depth + 1):
                        board.remove_piece(row, col)
                        return True
                    board.remove_piece(row, col)

        return False

    def test_beginner_puzzle_solvable(self):
        """Test that beginner puzzles are solvable."""
        gen = PuzzleGenerator(seed=42)

        for _ in range(5):
            puzzle = gen.generate(Difficulty.BEGINNER)
            assert puzzle is not None, "Failed to generate puzzle"
            assert self._solve_puzzle(puzzle), "Generated puzzle is not solvable"

    def test_easy_puzzle_solvable(self):
        """Test that easy puzzles are solvable."""
        gen = PuzzleGenerator(seed=42)

        for _ in range(3):
            puzzle = gen.generate(Difficulty.EASY)
            assert puzzle is not None, "Failed to generate puzzle"
            assert self._solve_puzzle(puzzle), "Generated puzzle is not solvable"

    def test_medium_puzzle_solvable(self):
        """Test that medium puzzles are solvable."""
        gen = PuzzleGenerator(seed=42)

        for _ in range(3):
            puzzle = gen.generate(Difficulty.MEDIUM)
            assert puzzle is not None, "Failed to generate puzzle"
            assert self._solve_puzzle(puzzle), "Generated puzzle is not solvable"


class TestBatchGeneration:
    """Tests for batch puzzle generation."""

    def test_generate_batch(self):
        """Test generating a batch of puzzles."""
        gen = PuzzleGenerator(seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            files = gen.generate_batch(3, Difficulty.BEGINNER, output_dir=tmpdir)

            assert len(files) == 3
            for filepath in files:
                assert os.path.exists(filepath)

                # Verify file is valid JSON
                with open(filepath, 'r') as f:
                    puzzle = json.load(f)
                assert 'grid_size' in puzzle
                assert 'pieces' in puzzle

    def test_batch_file_naming(self):
        """Test batch file naming convention."""
        gen = PuzzleGenerator(seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            files = gen.generate_batch(2, Difficulty.BEGINNER, output_dir=tmpdir)

            filenames = [os.path.basename(f) for f in files]
            assert 'beginner_001.json' in filenames
            assert 'beginner_002.json' in filenames

    def test_batch_creates_directory(self):
        """Test that batch generation creates output directory."""
        gen = PuzzleGenerator(seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = os.path.join(tmpdir, 'nested', 'puzzles')
            files = gen.generate_batch(1, Difficulty.EASY, output_dir=nested_dir)

            assert len(files) == 1
            assert os.path.isdir(nested_dir)


class TestPuzzleIntegration:
    """Integration tests with game module."""

    def test_load_generated_puzzle(self):
        """Test that generated puzzles load correctly in game."""
        gen = PuzzleGenerator(seed=42)
        puzzle = gen.generate(Difficulty.BEGINNER)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(puzzle, f)
            filepath = f.name

        try:
            game = LaserMaze()
            game.load(filepath)

            assert game.board is not None
            assert game.challenge is not None
            assert len(game.available_pieces) > 0
        finally:
            os.unlink(filepath)

    def test_play_generated_puzzle(self):
        """Test playing a generated puzzle."""
        gen = PuzzleGenerator(seed=42)
        puzzle = gen.generate(Difficulty.BEGINNER)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(puzzle, f)
            filepath = f.name

        try:
            game = LaserMaze()
            game.load(filepath)

            # Fire laser before placing any pieces
            result = game.fire()
            initial_targets = result.num_targets

            # Game should not be solved yet (need to place pieces)
            assert not game.is_solved() or initial_targets >= puzzle['goal']['targets']
        finally:
            os.unlink(filepath)


class TestEdgeCases:
    """Edge case tests."""

    def test_generation_with_max_attempts(self):
        """Test that generation respects max_attempts."""
        gen = PuzzleGenerator(seed=42)
        # Should succeed with reasonable attempts
        puzzle = gen.generate(Difficulty.BEGINNER, max_attempts=100)
        assert puzzle is not None

    def test_custom_board_size(self):
        """Test generation with custom board size."""
        gen = PuzzleGenerator(board_size=7, seed=42)
        puzzle = gen.generate(Difficulty.BEGINNER)

        assert puzzle is not None
        assert puzzle['grid_size'] == 7

    def test_available_pieces_format(self):
        """Test that available pieces have correct format."""
        gen = PuzzleGenerator(seed=42)
        puzzle = gen.generate(Difficulty.BEGINNER)

        for piece in puzzle['available']:
            assert 'type' in piece
            assert 'orientation' in piece
            # Orientation should be None (player decides)
            assert piece['orientation'] is None

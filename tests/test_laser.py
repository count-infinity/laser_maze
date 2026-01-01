"""Unit tests for laser.py - Beam simulation."""

import pytest
from board import Board
from pieces import Direction, Laser, Mirror, TargetMirror, BeamSplitter, Checkpoint, CellBlocker
from laser import fire_laser, LaserResult, BeamSegment, get_beam_cells, get_beam_at_cell


class TestBeamSegment:
    """Tests for BeamSegment dataclass."""

    def test_beam_segment_creation(self):
        """Test creating a beam segment."""
        seg = BeamSegment(row=2, col=3, direction=Direction.SOUTH)
        assert seg.row == 2
        assert seg.col == 3
        assert seg.direction == Direction.SOUTH

    def test_beam_segment_equality(self):
        """Test beam segment equality."""
        seg1 = BeamSegment(2, 3, Direction.SOUTH)
        seg2 = BeamSegment(2, 3, Direction.SOUTH)
        assert seg1 == seg2

    def test_beam_segment_hash(self):
        """Test beam segment hashing for use in sets."""
        seg1 = BeamSegment(2, 3, Direction.SOUTH)
        seg2 = BeamSegment(2, 3, Direction.SOUTH)
        seg_set = {seg1}
        assert seg2 in seg_set


class TestLaserResult:
    """Tests for LaserResult dataclass."""

    def test_empty_result(self):
        """Test empty laser result."""
        result = LaserResult()
        assert result.num_targets == 0
        assert result.num_checkpoints == 0
        assert result.terminated is True

    def test_result_with_targets(self):
        """Test result with targets hit."""
        result = LaserResult(
            targets_hit={(2, 3), (4, 4)},
            checkpoints_passed={(1, 1)},
        )
        assert result.num_targets == 2
        assert result.num_checkpoints == 1


class TestFireLaserBasic:
    """Basic tests for fire_laser function."""

    def test_no_laser_returns_empty(self):
        """Test firing with no laser returns empty result."""
        board = Board()
        board.set_piece(2, 2, Mirror())
        result = fire_laser(board)
        assert result.num_targets == 0
        assert len(result.beam_path) == 0

    def test_laser_beam_exits_grid(self):
        """Test beam exits grid and terminates."""
        board = Board()
        board.set_piece(0, 2, Laser(orientation=Direction.SOUTH))
        result = fire_laser(board)
        # Beam goes south from row 1 to row 4, then exits
        assert result.terminated is True
        assert len(result.beam_path) == 4  # rows 1, 2, 3, 4

    def test_laser_beam_straight_line(self):
        """Test beam travels in straight line."""
        board = Board()
        board.set_piece(0, 2, Laser(orientation=Direction.SOUTH))
        result = fire_laser(board)
        # All segments should be at col 2, traveling south
        for seg in result.beam_path:
            assert seg.col == 2
            assert seg.direction == Direction.SOUTH


class TestFireLaserReflection:
    """Tests for beam reflection with mirrors."""

    def test_single_mirror_reflection(self):
        """Test beam reflects off single mirror."""
        board = Board()
        board.set_piece(0, 2, Laser(orientation=Direction.SOUTH))
        # \ diagonal mirror - south beam reflects east
        board.set_piece(2, 2, Mirror(orientation=Direction.NORTH))
        result = fire_laser(board)

        # Verify beam path includes reflection
        beam_cells = get_beam_cells(result)
        assert (1, 2) in beam_cells  # Before mirror
        assert (2, 2) in beam_cells  # At mirror
        assert (2, 3) in beam_cells  # After reflection (east)

    def test_multiple_reflections(self):
        """Test beam reflects multiple times."""
        board = Board()
        board.set_piece(0, 2, Laser(orientation=Direction.SOUTH))
        # First mirror: south -> east (\ diagonal)
        board.set_piece(2, 2, Mirror(orientation=Direction.NORTH))
        # Second mirror: east -> north (/ diagonal: E->N)
        board.set_piece(2, 4, Mirror(orientation=Direction.EAST))
        result = fire_laser(board)

        beam_cells = get_beam_cells(result)
        assert (2, 3) in beam_cells  # Between mirrors
        assert (1, 4) in beam_cells  # After second reflection (going north)


class TestFireLaserTargetHit:
    """Tests for target hit detection."""

    def test_direct_target_hit(self):
        """Test beam directly hits target."""
        board = Board()
        board.set_piece(0, 2, Laser(orientation=Direction.SOUTH))
        # Target facing north (hit from south)
        board.set_piece(3, 2, TargetMirror(orientation=Direction.NORTH))
        result = fire_laser(board)

        assert result.num_targets == 1
        assert (3, 2) in result.targets_hit

    def test_reflected_target_hit(self):
        """Test beam hits target after reflection."""
        board = Board()
        board.set_piece(0, 2, Laser(orientation=Direction.SOUTH))
        # Mirror reflects south -> east
        board.set_piece(2, 2, Mirror(orientation=Direction.NORTH))
        # Target facing west (hit from east)
        board.set_piece(2, 4, TargetMirror(orientation=Direction.WEST))
        result = fire_laser(board)

        assert result.num_targets == 1
        assert (2, 4) in result.targets_hit

    def test_target_miss_wrong_direction(self):
        """Test beam misses target from wrong direction."""
        board = Board()
        board.set_piece(0, 2, Laser(orientation=Direction.SOUTH))
        # Target facing south (wrong direction - beam comes from north)
        board.set_piece(3, 2, TargetMirror(orientation=Direction.SOUTH))
        result = fire_laser(board)

        assert result.num_targets == 0
        # Beam should reflect off the mirror side


class TestFireLaserBeamSplitter:
    """Tests for beam splitter behavior."""

    def test_beam_splits_into_two(self):
        """Test beam splitter creates two beams."""
        board = Board()
        board.set_piece(0, 2, Laser(orientation=Direction.SOUTH))
        board.set_piece(2, 2, BeamSplitter(orientation=Direction.NORTH))  # \ diagonal
        result = fire_laser(board)

        # Check beam goes both south (through) and east (reflected)
        beam_cells = get_beam_cells(result)
        assert (3, 2) in beam_cells  # Continued south
        assert (2, 3) in beam_cells  # Reflected east

    def test_beam_splitter_hits_two_targets(self):
        """Test beam splitter can hit two targets."""
        board = Board()
        board.set_piece(0, 2, Laser(orientation=Direction.SOUTH))
        board.set_piece(2, 2, BeamSplitter(orientation=Direction.NORTH))
        # Target 1: south path
        board.set_piece(4, 2, TargetMirror(orientation=Direction.NORTH))
        # Target 2: east path
        board.set_piece(2, 4, TargetMirror(orientation=Direction.WEST))
        result = fire_laser(board)

        assert result.num_targets == 2


class TestFireLaserCheckpoint:
    """Tests for checkpoint detection."""

    def test_checkpoint_passed(self):
        """Test beam passing through checkpoint is detected."""
        board = Board()
        board.set_piece(0, 2, Laser(orientation=Direction.SOUTH))
        board.set_piece(2, 2, Checkpoint())
        result = fire_laser(board)

        assert result.num_checkpoints == 1
        assert (2, 2) in result.checkpoints_passed

    def test_multiple_checkpoints(self):
        """Test multiple checkpoints can be passed."""
        board = Board()
        board.set_piece(0, 2, Laser(orientation=Direction.SOUTH))
        board.set_piece(1, 2, Checkpoint())
        board.set_piece(3, 2, Checkpoint())
        result = fire_laser(board)

        assert result.num_checkpoints == 2

    def test_checkpoint_beam_continues(self):
        """Test beam continues through checkpoint."""
        board = Board()
        board.set_piece(0, 2, Laser(orientation=Direction.SOUTH))
        board.set_piece(2, 2, Checkpoint())
        board.set_piece(4, 2, TargetMirror(orientation=Direction.NORTH))
        result = fire_laser(board)

        assert result.num_checkpoints == 1
        assert result.num_targets == 1


class TestFireLaserCellBlocker:
    """Tests for cell blocker behavior."""

    def test_beam_passes_through_blocker(self):
        """Test beam passes through cell blocker."""
        board = Board()
        board.set_piece(0, 2, Laser(orientation=Direction.SOUTH))
        board.set_piece(2, 2, CellBlocker())
        result = fire_laser(board)

        beam_cells = get_beam_cells(result)
        assert (2, 2) in beam_cells  # Beam passes through
        assert (3, 2) in beam_cells  # Continues beyond


class TestFireLaserLoopDetection:
    """Tests for infinite loop detection."""

    def test_loop_detection(self):
        """Test infinite loop is detected and terminated."""
        board = Board()
        board.set_piece(0, 0, Laser(orientation=Direction.EAST))
        # Create a loop with mirrors
        board.set_piece(0, 2, Mirror(orientation=Direction.NORTH))  # E->S
        board.set_piece(2, 2, Mirror(orientation=Direction.EAST))   # S->W
        board.set_piece(2, 0, Mirror(orientation=Direction.NORTH))  # W->N
        # This would create a square loop
        result = fire_laser(board)

        # Should terminate without infinite loop
        assert result.terminated is True

    def test_max_steps_limit(self):
        """Test max steps prevents infinite processing."""
        board = Board()
        board.set_piece(0, 0, Laser(orientation=Direction.EAST))
        # Mirror bounces beam back and forth
        board.set_piece(0, 4, Mirror(orientation=Direction.NORTH))
        result = fire_laser(board, max_steps=10)

        # Should terminate after max steps
        assert len(result.beam_path) <= 10


class TestGetBeamHelpers:
    """Tests for helper functions."""

    def test_get_beam_cells(self):
        """Test getting all cells beam passes through."""
        result = LaserResult(
            beam_path=[
                BeamSegment(1, 2, Direction.SOUTH),
                BeamSegment(2, 2, Direction.SOUTH),
                BeamSegment(2, 3, Direction.EAST),
            ]
        )
        cells = get_beam_cells(result)
        assert cells == {(1, 2), (2, 2), (2, 3)}

    def test_get_beam_at_cell(self):
        """Test getting beam directions at specific cell."""
        result = LaserResult(
            beam_path=[
                BeamSegment(2, 2, Direction.SOUTH),
                BeamSegment(2, 2, Direction.EAST),  # Same cell, different direction
            ]
        )
        directions = get_beam_at_cell(result, 2, 2)
        assert Direction.SOUTH in directions
        assert Direction.EAST in directions

    def test_get_beam_at_cell_empty(self):
        """Test getting beam at cell with no beam."""
        result = LaserResult(
            beam_path=[BeamSegment(1, 1, Direction.SOUTH)]
        )
        directions = get_beam_at_cell(result, 2, 2)
        assert directions == []


class TestComplexScenarios:
    """Complex beam path scenarios."""

    def test_beam_absorbed_by_target(self):
        """Test beam is absorbed when hitting target."""
        board = Board()
        board.set_piece(0, 2, Laser(orientation=Direction.SOUTH))
        # Target facing north - beam hits target and is absorbed
        board.set_piece(2, 2, TargetMirror(orientation=Direction.NORTH))
        result = fire_laser(board)

        # Beam should be absorbed at target
        beam_cells = get_beam_cells(result)
        assert (2, 2) in beam_cells
        # Beam should not continue past the target
        assert (3, 2) not in beam_cells
        assert result.num_targets == 1

    def test_complex_path_with_splitter_and_targets(self):
        """Test complex path with splitter hitting multiple targets."""
        board = Board()
        # Laser at top
        board.set_piece(0, 2, Laser(orientation=Direction.SOUTH))
        # Beam splitter in middle (\ diagonal: S->S+E)
        board.set_piece(2, 2, BeamSplitter(orientation=Direction.NORTH))
        # Target 1: continues south
        board.set_piece(4, 2, TargetMirror(orientation=Direction.NORTH))
        # Mirror redirects east beam to south (\ diagonal: E->S)
        board.set_piece(2, 4, Mirror(orientation=Direction.NORTH))
        # Target 2: after reflection
        board.set_piece(4, 4, TargetMirror(orientation=Direction.NORTH))

        result = fire_laser(board)

        assert result.num_targets == 2
        assert (4, 2) in result.targets_hit
        assert (4, 4) in result.targets_hit

"""Unit tests for pieces.py - Direction enum and piece classes."""

import pytest
from pieces import (
    Direction, PieceType, Piece,
    Laser, Mirror, TargetMirror, BeamSplitter, DoubleMirror, Checkpoint, CellBlocker,
    create_piece, PIECE_CLASSES
)


class TestDirection:
    """Tests for Direction enum."""

    def test_direction_values(self):
        """Test direction integer values."""
        assert Direction.NORTH == 0
        assert Direction.EAST == 1
        assert Direction.SOUTH == 2
        assert Direction.WEST == 3

    def test_opposite(self):
        """Test opposite direction calculation."""
        assert Direction.NORTH.opposite() == Direction.SOUTH
        assert Direction.SOUTH.opposite() == Direction.NORTH
        assert Direction.EAST.opposite() == Direction.WEST
        assert Direction.WEST.opposite() == Direction.EAST

    def test_rotate_cw(self):
        """Test clockwise rotation."""
        assert Direction.NORTH.rotate_cw() == Direction.EAST
        assert Direction.EAST.rotate_cw() == Direction.SOUTH
        assert Direction.SOUTH.rotate_cw() == Direction.WEST
        assert Direction.WEST.rotate_cw() == Direction.NORTH

    def test_rotate_ccw(self):
        """Test counter-clockwise rotation."""
        assert Direction.NORTH.rotate_ccw() == Direction.WEST
        assert Direction.WEST.rotate_ccw() == Direction.SOUTH
        assert Direction.SOUTH.rotate_ccw() == Direction.EAST
        assert Direction.EAST.rotate_ccw() == Direction.NORTH

    def test_delta(self):
        """Test movement deltas (row, col)."""
        assert Direction.NORTH.delta == (-1, 0)
        assert Direction.SOUTH.delta == (1, 0)
        assert Direction.EAST.delta == (0, 1)
        assert Direction.WEST.delta == (0, -1)

    def test_full_rotation_cw(self):
        """Test that 4 CW rotations return to original."""
        d = Direction.NORTH
        for _ in range(4):
            d = d.rotate_cw()
        assert d == Direction.NORTH

    def test_full_rotation_ccw(self):
        """Test that 4 CCW rotations return to original."""
        d = Direction.EAST
        for _ in range(4):
            d = d.rotate_ccw()
        assert d == Direction.EAST


class TestLaser:
    """Tests for Laser piece."""

    def test_piece_type(self):
        """Test laser piece type."""
        laser = Laser()
        assert laser.piece_type == PieceType.LASER

    def test_emit_direction(self):
        """Test laser emits in orientation direction."""
        for d in Direction:
            laser = Laser(orientation=d)
            assert laser.emit_direction() == d

    def test_absorbs_all_beams(self):
        """Test laser absorbs any incoming beam."""
        laser = Laser(orientation=Direction.SOUTH)
        for d in Direction:
            assert laser.interact(d) == []

    def test_rotation(self):
        """Test laser rotation."""
        laser = Laser(orientation=Direction.NORTH)
        laser.rotate_cw()
        assert laser.orientation == Direction.EAST
        assert laser.emit_direction() == Direction.EAST

    def test_fixed_no_rotation(self):
        """Test fixed laser cannot rotate."""
        laser = Laser(orientation=Direction.NORTH, fixed=True)
        laser.rotate_cw()
        assert laser.orientation == Direction.NORTH


class TestMirror:
    """Tests for Mirror piece."""

    def test_piece_type(self):
        """Test mirror piece type."""
        mirror = Mirror()
        assert mirror.piece_type == PieceType.MIRROR

    def test_backslash_diagonal_reflections(self):
        r"""Test \ diagonal (NORTH/SOUTH orientation) reflections."""
        mirror = Mirror(orientation=Direction.NORTH)
        # \ diagonal: N->W, E->S, S->E, W->N
        assert mirror.interact(Direction.NORTH) == [Direction.WEST]
        assert mirror.interact(Direction.EAST) == [Direction.SOUTH]
        assert mirror.interact(Direction.SOUTH) == [Direction.EAST]
        assert mirror.interact(Direction.WEST) == [Direction.NORTH]

    def test_slash_diagonal_reflections(self):
        """Test / diagonal (EAST/WEST orientation) reflections."""
        mirror = Mirror(orientation=Direction.EAST)
        # / diagonal: N->E, E->N, S->W, W->S
        assert mirror.interact(Direction.NORTH) == [Direction.EAST]
        assert mirror.interact(Direction.EAST) == [Direction.NORTH]
        assert mirror.interact(Direction.SOUTH) == [Direction.WEST]
        assert mirror.interact(Direction.WEST) == [Direction.SOUTH]

    def test_north_south_same_diagonal(self):
        r"""Test NORTH and SOUTH give same \ diagonal."""
        m1 = Mirror(orientation=Direction.NORTH)
        m2 = Mirror(orientation=Direction.SOUTH)
        for d in Direction:
            assert m1.interact(d) == m2.interact(d)

    def test_east_west_same_diagonal(self):
        """Test EAST and WEST give same / diagonal."""
        m1 = Mirror(orientation=Direction.EAST)
        m2 = Mirror(orientation=Direction.WEST)
        for d in Direction:
            assert m1.interact(d) == m2.interact(d)


class TestTargetMirror:
    """Tests for TargetMirror piece."""

    def test_piece_type(self):
        """Test target mirror piece type."""
        tm = TargetMirror()
        assert tm.piece_type == PieceType.TARGET_MIRROR

    def test_target_hit_absorbs_beam(self):
        """Test beam hitting target side is absorbed."""
        # Target faces NORTH, beam from SOUTH hits the target
        tm = TargetMirror(orientation=Direction.NORTH)
        assert tm.interact(Direction.SOUTH) == []
        assert tm.is_hit(Direction.SOUTH) is True

    def test_target_miss_from_other_directions(self):
        """Test beam from wrong direction doesn't hit target."""
        tm = TargetMirror(orientation=Direction.NORTH)
        assert tm.is_hit(Direction.NORTH) is False
        assert tm.is_hit(Direction.EAST) is False
        assert tm.is_hit(Direction.WEST) is False

    def test_mirror_reflects_non_target_beams(self):
        """Test non-target-hitting beams are reflected or blocked."""
        # Target faces NORTH (so mirror is \ diagonal)
        # Using user's convention: "from X" = source, incoming = travel direction
        # Target NORTH: from W->S, from S->W, from E->blocked, from N->hit
        #   traveling E->S, traveling N->W, traveling W->blocked, traveling S->hit
        tm = TargetMirror(orientation=Direction.NORTH)
        # Traveling EAST (from west) -> SOUTH
        assert tm.interact(Direction.EAST) == [Direction.SOUTH]
        # Traveling NORTH (from south) -> WEST
        assert tm.interact(Direction.NORTH) == [Direction.WEST]
        # Traveling WEST (from east) -> blocked
        assert tm.interact(Direction.WEST) == []
        # Traveling SOUTH (from north) -> hits target (absorbed)
        assert tm.interact(Direction.SOUTH) == []

    def test_all_orientations_target_hit(self):
        """Test target hit from correct direction for all orientations."""
        test_cases = [
            (Direction.NORTH, Direction.SOUTH),  # faces N, hit from S
            (Direction.SOUTH, Direction.NORTH),  # faces S, hit from N
            (Direction.EAST, Direction.WEST),    # faces E, hit from W
            (Direction.WEST, Direction.EAST),    # faces W, hit from E
        ]
        for facing, hit_from in test_cases:
            tm = TargetMirror(orientation=facing)
            assert tm.is_hit(hit_from) is True, f"Target facing {facing} should be hit from {hit_from}"
            assert tm.interact(hit_from) == [], f"Target should absorb beam from {hit_from}"


class TestBeamSplitter:
    """Tests for BeamSplitter piece."""

    def test_piece_type(self):
        """Test beam splitter piece type."""
        bs = BeamSplitter()
        assert bs.piece_type == PieceType.BEAM_SPLITTER

    def test_splits_into_two_beams(self):
        """Test beam splitter always returns two directions."""
        bs = BeamSplitter(orientation=Direction.NORTH)
        for d in Direction:
            result = bs.interact(d)
            assert len(result) == 2, f"Should split into 2 beams, got {len(result)}"

    def test_one_passes_through(self):
        """Test one beam passes straight through."""
        bs = BeamSplitter(orientation=Direction.NORTH)
        for d in Direction:
            result = bs.interact(d)
            assert d in result, f"Incoming direction {d} should pass through"

    def test_backslash_diagonal_split(self):
        r"""Test \ diagonal beam splitter (NORTH orientation)."""
        bs = BeamSplitter(orientation=Direction.NORTH)
        # South beam: passes through (South) + reflects to East
        result = bs.interact(Direction.SOUTH)
        assert Direction.SOUTH in result
        assert Direction.EAST in result

    def test_slash_diagonal_split(self):
        """Test / diagonal beam splitter (EAST orientation)."""
        bs = BeamSplitter(orientation=Direction.EAST)
        # South beam: passes through (South) + reflects to West
        result = bs.interact(Direction.SOUTH)
        assert Direction.SOUTH in result
        assert Direction.WEST in result


class TestDoubleMirror:
    """Tests for DoubleMirror piece."""

    def test_piece_type(self):
        """Test double mirror piece type."""
        dm = DoubleMirror()
        assert dm.piece_type == PieceType.DOUBLE_MIRROR

    def test_reflects_like_mirror(self):
        """Test double mirror reflects same as regular mirror."""
        dm = DoubleMirror(orientation=Direction.NORTH)
        m = Mirror(orientation=Direction.NORTH)
        for d in Direction:
            assert dm.interact(d) == m.interact(d)

    def test_single_output(self):
        """Test double mirror returns single direction."""
        dm = DoubleMirror(orientation=Direction.EAST)
        for d in Direction:
            result = dm.interact(d)
            assert len(result) == 1


class TestCheckpoint:
    """Tests for Checkpoint piece (doorframe behavior)."""

    def test_piece_type(self):
        """Test checkpoint piece type."""
        cp = Checkpoint()
        assert cp.piece_type == PieceType.CHECKPOINT

    def test_ns_checkpoint_allows_ns_beams(self):
        """Test N/S oriented checkpoint allows N/S beams."""
        cp = Checkpoint(orientation=Direction.NORTH)
        assert cp.interact(Direction.NORTH) == [Direction.NORTH]
        assert cp.interact(Direction.SOUTH) == [Direction.SOUTH]

        cp_south = Checkpoint(orientation=Direction.SOUTH)
        assert cp_south.interact(Direction.NORTH) == [Direction.NORTH]
        assert cp_south.interact(Direction.SOUTH) == [Direction.SOUTH]

    def test_ns_checkpoint_blocks_ew_beams(self):
        """Test N/S oriented checkpoint blocks E/W beams."""
        cp = Checkpoint(orientation=Direction.NORTH)
        assert cp.interact(Direction.EAST) == []
        assert cp.interact(Direction.WEST) == []

    def test_ew_checkpoint_allows_ew_beams(self):
        """Test E/W oriented checkpoint allows E/W beams."""
        cp = Checkpoint(orientation=Direction.EAST)
        assert cp.interact(Direction.EAST) == [Direction.EAST]
        assert cp.interact(Direction.WEST) == [Direction.WEST]

        cp_west = Checkpoint(orientation=Direction.WEST)
        assert cp_west.interact(Direction.EAST) == [Direction.EAST]
        assert cp_west.interact(Direction.WEST) == [Direction.WEST]

    def test_ew_checkpoint_blocks_ns_beams(self):
        """Test E/W oriented checkpoint blocks N/S beams."""
        cp = Checkpoint(orientation=Direction.EAST)
        assert cp.interact(Direction.NORTH) == []
        assert cp.interact(Direction.SOUTH) == []

    def test_symbol_shows_orientation(self):
        """Test checkpoint symbol indicates passage direction."""
        cp_ns = Checkpoint(orientation=Direction.NORTH)
        assert cp_ns.symbol == 'C|'  # Vertical doorframe

        cp_ew = Checkpoint(orientation=Direction.EAST)
        assert cp_ew.symbol == 'C-'  # Horizontal doorframe


class TestCellBlocker:
    """Tests for CellBlocker piece."""

    def test_piece_type(self):
        """Test cell blocker piece type."""
        cb = CellBlocker()
        assert cb.piece_type == PieceType.CELL_BLOCKER

    def test_beam_passes_through(self):
        """Test beam passes through cell blocker (only blocks placement)."""
        cb = CellBlocker()
        for d in Direction:
            assert cb.interact(d) == [d]


class TestCreatePiece:
    """Tests for create_piece factory function."""

    def test_create_all_types(self):
        """Test creating all piece types."""
        type_class_map = {
            'laser': Laser,
            'mirror': Mirror,
            'target_mirror': TargetMirror,
            'beam_splitter': BeamSplitter,
            'double_mirror': DoubleMirror,
            'checkpoint': Checkpoint,
            'cell_blocker': CellBlocker,
        }
        for type_name, expected_class in type_class_map.items():
            piece = create_piece(type_name)
            assert isinstance(piece, expected_class)

    def test_create_with_orientation(self):
        """Test creating piece with specific orientation."""
        piece = create_piece('mirror', orientation=2)
        assert piece.orientation == Direction.SOUTH

    def test_create_with_fixed(self):
        """Test creating fixed piece."""
        piece = create_piece('laser', fixed=True)
        assert piece.fixed is True

    def test_invalid_type_raises(self):
        """Test invalid piece type raises ValueError."""
        with pytest.raises(ValueError):
            create_piece('invalid_type')


class TestPieceCopy:
    """Tests for piece copy functionality."""

    def test_copy_preserves_orientation(self):
        """Test copy preserves orientation."""
        original = Mirror(orientation=Direction.EAST)
        copied = original.copy()
        assert copied.orientation == original.orientation

    def test_copy_preserves_fixed(self):
        """Test copy preserves fixed status."""
        original = Laser(orientation=Direction.SOUTH, fixed=True)
        copied = original.copy()
        assert copied.fixed is True

    def test_copy_is_independent(self):
        """Test copy is independent of original."""
        original = Mirror(orientation=Direction.NORTH)
        copied = original.copy()
        copied.rotate_cw()
        assert original.orientation == Direction.NORTH
        assert copied.orientation == Direction.EAST

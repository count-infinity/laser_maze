"""Unit tests for game.py - LaserMaze API and RL interface."""

import pytest
import numpy as np
from game import LaserMaze, Action, ActionType
from pieces import Direction, PieceType


class TestLaserMazeInit:
    """Tests for LaserMaze initialization."""

    def test_default_init(self):
        """Test default initialization."""
        game = LaserMaze()
        assert game.board_size == 5
        assert game.board is not None
        assert game.challenge is None

    def test_custom_board_size(self):
        """Test custom board size."""
        game = LaserMaze(board_size=7)
        assert game.board_size == 7
        assert game.board.size == 7


class TestLaserMazePlace:
    """Tests for piece placement."""

    def test_place_piece(self):
        """Test placing a piece."""
        game = LaserMaze()
        result = game.place("mirror", 2, 2, orientation=1)
        assert result is True
        assert game.board.get_piece(2, 2) is not None

    def test_place_laser(self):
        """Test placing laser."""
        game = LaserMaze()
        result = game.place("laser", 0, 2, orientation=2)
        assert result is True
        assert game.board.laser_pos == (0, 2)

    def test_place_on_occupied_fails(self):
        """Test placing on occupied cell fails."""
        game = LaserMaze()
        game.place("mirror", 2, 2)
        result = game.place("laser", 2, 2)
        assert result is False

    def test_place_tracks_placed_pieces(self):
        """Test placed pieces are tracked."""
        game = LaserMaze()
        game.place("mirror", 2, 2)
        assert (2, 2) in game.placed_pieces


class TestLaserMazeRotate:
    """Tests for piece rotation."""

    def test_rotate_piece(self):
        """Test rotating a piece."""
        game = LaserMaze()
        game.place("mirror", 2, 2, orientation=0)
        result = game.rotate(2, 2)
        assert result is True
        assert game.board.get_piece(2, 2).orientation == Direction.EAST

    def test_rotate_empty_fails(self):
        """Test rotating empty cell fails."""
        game = LaserMaze()
        result = game.rotate(2, 2)
        assert result is False


class TestLaserMazeRemove:
    """Tests for piece removal."""

    def test_remove_piece(self):
        """Test removing a piece."""
        game = LaserMaze()
        game.place("mirror", 2, 2)
        result = game.remove(2, 2)
        assert result is True
        assert game.board.get_piece(2, 2) is None

    def test_remove_updates_tracking(self):
        """Test removal updates placed_pieces tracking."""
        game = LaserMaze()
        game.place("mirror", 2, 2)
        game.remove(2, 2)
        assert (2, 2) not in game.placed_pieces


class TestLaserMazeFire:
    """Tests for firing the laser."""

    def test_fire_returns_result(self):
        """Test fire returns LaserResult."""
        game = LaserMaze()
        game.place("laser", 0, 2, orientation=2)
        result = game.fire()
        assert result is not None
        assert hasattr(result, 'targets_hit')
        assert hasattr(result, 'beam_path')

    def test_fire_hits_target(self):
        """Test fire detects target hit."""
        game = LaserMaze()
        game.place("laser", 0, 2, orientation=2)
        game.place("target_mirror", 4, 2, orientation=0)  # Facing north
        result = game.fire()
        assert result.num_targets == 1

    def test_fire_stores_last_result(self):
        """Test fire stores result."""
        game = LaserMaze()
        game.place("laser", 0, 2, orientation=2)
        result = game.fire()
        assert game.last_result is result


class TestLaserMazeRender:
    """Tests for board rendering."""

    def test_render_returns_string(self):
        """Test render returns string."""
        game = LaserMaze()
        game.place("laser", 0, 2, orientation=2)
        output = game.render()
        assert isinstance(output, str)
        assert len(output) > 0

    def test_render_with_laser(self):
        """Test render with laser beam."""
        game = LaserMaze()
        game.place("laser", 0, 2, orientation=2)
        game.fire()
        output = game.render(with_laser=True)
        assert isinstance(output, str)


class TestLaserMazeRLInterface:
    """Tests for RL gym-style interface."""

    def test_get_state_tensor_shape(self):
        """Test state tensor shape."""
        game = LaserMaze()
        state = game.get_state_tensor()
        assert isinstance(state, np.ndarray)
        assert state.shape == (13, 5, 5)  # channels x height x width
        assert state.dtype == np.float32

    def test_state_tensor_empty_board(self):
        """Test state tensor for empty board."""
        game = LaserMaze()
        state = game.get_state_tensor()
        # Channel 0 should be all 1s (EMPTY)
        assert np.all(state[0] == 1.0)
        # Other piece type channels should be 0
        for i in range(1, 8):
            assert np.all(state[i] == 0.0)

    def test_state_tensor_with_piece(self):
        """Test state tensor with piece placed."""
        game = LaserMaze()
        game.place("laser", 0, 2, orientation=2)  # South
        state = game.get_state_tensor()

        # Laser channel (1) should have 1 at (0, 2)
        assert state[PieceType.LASER, 0, 2] == 1.0
        # Empty channel (0) should be 0 at (0, 2)
        assert state[0, 0, 2] == 0.0
        # Orientation channel for SOUTH (10 = 8 + 2)
        assert state[8 + Direction.SOUTH, 0, 2] == 1.0

    def test_get_valid_actions(self):
        """Test getting valid actions."""
        game = LaserMaze()
        game.place("laser", 0, 2, orientation=2)
        actions = game.get_valid_actions()
        assert isinstance(actions, list)
        assert len(actions) > 0

    def test_valid_actions_includes_fire(self):
        """Test valid actions includes fire when laser present."""
        game = LaserMaze()
        game.place("laser", 0, 2, orientation=2)
        actions = game.get_valid_actions()
        fire_actions = [a for a in actions if a.action_type == ActionType.FIRE]
        assert len(fire_actions) == 1

    def test_valid_actions_includes_rotate(self):
        """Test valid actions includes rotate for placed pieces."""
        game = LaserMaze()
        game.place("mirror", 2, 2)
        actions = game.get_valid_actions()
        rotate_actions = [a for a in actions if a.action_type == ActionType.ROTATE]
        assert any(a.row == 2 and a.col == 2 for a in rotate_actions)


class TestLaserMazeStep:
    """Tests for step function."""

    def test_step_fire_action(self):
        """Test step with fire action."""
        game = LaserMaze()
        game.place("laser", 0, 2, orientation=2)
        action = Action(ActionType.FIRE)
        state, reward, done, info = game.step(action)

        assert isinstance(state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_place_action(self):
        """Test step with place action."""
        game = LaserMaze()
        game.place("laser", 0, 2, orientation=2)
        action = Action(ActionType.PLACE, row=2, col=2, piece_type="mirror", orientation=0)
        state, reward, done, info = game.step(action)

        assert game.board.get_piece(2, 2) is not None
        assert info.get('valid_action') is True

    def test_step_invalid_action_penalty(self):
        """Test invalid action gives penalty."""
        game = LaserMaze()
        game.place("mirror", 2, 2)
        # Try to place on occupied cell
        action = Action(ActionType.PLACE, row=2, col=2, piece_type="laser", orientation=0)
        state, reward, done, info = game.step(action)

        assert reward < 0
        assert info.get('valid_action') is False

    def test_step_rotate_action(self):
        """Test step with rotate action."""
        game = LaserMaze()
        game.place("mirror", 2, 2, orientation=0)
        action = Action(ActionType.ROTATE, row=2, col=2)
        state, reward, done, info = game.step(action)

        assert game.board.get_piece(2, 2).orientation == Direction.EAST
        assert info.get('valid_action') is True

    def test_step_integer_action(self):
        """Test step with integer action ID."""
        game = LaserMaze()
        game.place("laser", 0, 2, orientation=2)
        # Action 0 is FIRE
        state, reward, done, info = game.step(0)
        assert info.get('valid_action') is True


class TestLaserMazeReset:
    """Tests for reset function."""

    def test_reset_returns_state(self):
        """Test reset returns state tensor."""
        game = LaserMaze()
        game.place("laser", 0, 2, orientation=2)
        state = game.reset()
        assert isinstance(state, np.ndarray)

    def test_reset_clears_board(self):
        """Test reset clears placed pieces."""
        game = LaserMaze()
        game.place("mirror", 2, 2)
        game.reset()
        # Without initial_state, board should be empty
        assert game.board.get_piece(2, 2) is None


class TestLaserMazeActionEncoding:
    """Tests for action encoding/decoding."""

    def test_encode_fire_action(self):
        """Test encoding fire action."""
        game = LaserMaze()
        action = Action(ActionType.FIRE)
        encoded = game.encode_action(action)
        assert encoded == 0

    def test_decode_fire_action(self):
        """Test decoding fire action."""
        game = LaserMaze()
        decoded = game._decode_action(0)
        assert decoded.action_type == ActionType.FIRE

    def test_encode_rotate_action(self):
        """Test encoding rotate action."""
        game = LaserMaze()
        action = Action(ActionType.ROTATE, row=2, col=3)
        encoded = game.encode_action(action)
        # Rotate starts at 1, cell index = 2*5 + 3 = 13
        assert encoded == 1 + 13

    def test_decode_rotate_action(self):
        """Test decoding rotate action."""
        game = LaserMaze()
        decoded = game._decode_action(14)  # 1 + 13
        assert decoded.action_type == ActionType.ROTATE
        assert decoded.row == 2
        assert decoded.col == 3

    def test_encode_decode_roundtrip(self):
        """Test encode/decode roundtrip."""
        game = LaserMaze()
        original = Action(ActionType.ROTATE, row=1, col=4)
        encoded = game.encode_action(original)
        decoded = game._decode_action(encoded)
        assert decoded.action_type == original.action_type
        assert decoded.row == original.row
        assert decoded.col == original.col


class TestAction:
    """Tests for Action dataclass."""

    def test_action_creation(self):
        """Test action creation."""
        action = Action(ActionType.PLACE, row=2, col=3, piece_type="mirror", orientation=1)
        assert action.action_type == ActionType.PLACE
        assert action.row == 2
        assert action.col == 3
        assert action.piece_type == "mirror"
        assert action.orientation == 1

    def test_action_hashable(self):
        """Test actions are hashable."""
        action = Action(ActionType.FIRE)
        action_set = {action}
        assert action in action_set


class TestReducedActionSpace:
    """Tests for reduced action space with symmetric piece orientations."""

    def test_mirror_only_two_orientations_in_valid_actions(self):
        """Test mirror placement only has 2 orientations."""
        from game import PIECE_ORIENTATIONS
        assert len(PIECE_ORIENTATIONS['mirror']) == 2
        assert PIECE_ORIENTATIONS['mirror'] == [0, 1]

    def test_beam_splitter_only_two_orientations(self):
        """Test beam splitter placement only has 2 orientations."""
        from game import PIECE_ORIENTATIONS
        assert len(PIECE_ORIENTATIONS['beam_splitter']) == 2
        assert PIECE_ORIENTATIONS['beam_splitter'] == [0, 1]

    def test_double_mirror_only_two_orientations(self):
        """Test double mirror placement only has 2 orientations."""
        from game import PIECE_ORIENTATIONS
        assert len(PIECE_ORIENTATIONS['double_mirror']) == 2
        assert PIECE_ORIENTATIONS['double_mirror'] == [0, 1]

    def test_checkpoint_two_orientations(self):
        """Test checkpoint placement has 2 orientations (doorframe behavior)."""
        from game import PIECE_ORIENTATIONS
        assert len(PIECE_ORIENTATIONS['checkpoint']) == 2
        assert PIECE_ORIENTATIONS['checkpoint'] == [0, 1]  # N/S passage, E/W passage

    def test_cell_blocker_only_one_orientation(self):
        """Test cell blocker placement only has 1 orientation."""
        from game import PIECE_ORIENTATIONS
        assert len(PIECE_ORIENTATIONS['cell_blocker']) == 1
        assert PIECE_ORIENTATIONS['cell_blocker'] == [0]

    def test_laser_has_four_orientations(self):
        """Test laser placement has all 4 orientations."""
        from game import PIECE_ORIENTATIONS
        assert len(PIECE_ORIENTATIONS['laser']) == 4
        assert PIECE_ORIENTATIONS['laser'] == [0, 1, 2, 3]

    def test_target_mirror_has_four_orientations(self):
        """Test target mirror placement has all 4 orientations."""
        from game import PIECE_ORIENTATIONS
        assert len(PIECE_ORIENTATIONS['target_mirror']) == 4
        assert PIECE_ORIENTATIONS['target_mirror'] == [0, 1, 2, 3]

    def test_action_space_size_reduced(self):
        """Test action space size reflects reduced orientations."""
        game = LaserMaze()
        size = game.get_action_space_size()
        # Fire: 1
        # Rotate: 25 (5x5)
        # Remove: 25 (5x5)
        # Place: laser (25*4) + mirror (25*2) + target_mirror (25*4) +
        #        beam_splitter (25*2) + double_mirror (25*2) +
        #        checkpoint (25*2) + cell_blocker (25*1)
        # = 100 + 50 + 100 + 50 + 50 + 50 + 25 = 425
        # Total: 1 + 25 + 25 + 425 = 476
        assert size == 476

    def test_encode_decode_place_mirror_roundtrip(self):
        """Test place mirror action encode/decode with reduced orientations."""
        game = LaserMaze()
        # Mirror with orientation 0 (backslash)
        action = Action(ActionType.PLACE, row=2, col=3, piece_type="mirror", orientation=0)
        encoded = game.encode_action(action)
        decoded = game._decode_action(encoded)
        assert decoded.action_type == ActionType.PLACE
        assert decoded.row == 2
        assert decoded.col == 3
        assert decoded.piece_type == "mirror"
        assert decoded.orientation == 0

        # Mirror with orientation 1 (forward slash)
        action = Action(ActionType.PLACE, row=2, col=3, piece_type="mirror", orientation=1)
        encoded = game.encode_action(action)
        decoded = game._decode_action(encoded)
        assert decoded.orientation == 1

    def test_encode_decode_place_laser_roundtrip(self):
        """Test place laser action encode/decode with all 4 orientations."""
        game = LaserMaze()
        for orientation in range(4):
            action = Action(ActionType.PLACE, row=1, col=2, piece_type="laser", orientation=orientation)
            encoded = game.encode_action(action)
            decoded = game._decode_action(encoded)
            assert decoded.action_type == ActionType.PLACE
            assert decoded.row == 1
            assert decoded.col == 2
            assert decoded.piece_type == "laser"
            assert decoded.orientation == orientation

    def test_encode_decode_place_checkpoint_roundtrip(self):
        """Test place checkpoint action encode/decode with only 1 orientation."""
        game = LaserMaze()
        action = Action(ActionType.PLACE, row=3, col=4, piece_type="checkpoint", orientation=0)
        encoded = game.encode_action(action)
        decoded = game._decode_action(encoded)
        assert decoded.action_type == ActionType.PLACE
        assert decoded.row == 3
        assert decoded.col == 4
        assert decoded.piece_type == "checkpoint"
        assert decoded.orientation == 0

    def test_state_tensor_canonical_orientation_mirror(self):
        """Test state tensor uses canonical orientation for symmetric pieces."""
        game = LaserMaze()
        # Place mirror with orientation SOUTH (2) - should map to 0
        game.place("mirror", 2, 2, orientation=2)
        state = game.get_state_tensor()

        # Mirror is PieceType 2
        assert state[PieceType.MIRROR, 2, 2] == 1.0
        # Orientation should be canonical 0 (not 2)
        assert state[8 + 0, 2, 2] == 1.0  # Canonical orientation 0
        assert state[8 + 2, 2, 2] == 0.0  # Not orientation 2

    def test_state_tensor_canonical_orientation_laser(self):
        """Test state tensor preserves orientation for asymmetric pieces."""
        game = LaserMaze()
        # Place laser with orientation SOUTH (2) - should stay as 2
        game.place("laser", 0, 2, orientation=2)
        state = game.get_state_tensor()

        # Laser is PieceType 1
        assert state[PieceType.LASER, 0, 2] == 1.0
        # Orientation should be 2 (unchanged)
        assert state[8 + 2, 0, 2] == 1.0  # Orientation 2
        assert state[8 + 0, 0, 2] == 0.0  # Not orientation 0


class TestPieceAvailability:
    """Tests for piece availability constraints."""

    def test_place_unavailable_piece_fails(self):
        """Test placing a piece not in available list fails."""
        game = LaserMaze()
        game.load("examples/beginner_challenge_01.json")
        # Laser is not in available list
        result = game.place("laser", 0, 0, orientation=0)
        assert result is False

    def test_place_available_piece_succeeds(self):
        """Test placing a piece in available list succeeds."""
        game = LaserMaze()
        game.load("examples/beginner_challenge_01.json")
        # Mirror is in available list
        result = game.place("mirror", 2, 2, orientation=0)
        assert result is True

    def test_place_exhausted_piece_fails(self):
        """Test placing piece after exhausting available fails."""
        game = LaserMaze()
        game.load("examples/beginner_challenge_01.json")
        # Place the only available mirror
        game.place("mirror", 2, 2, orientation=0)
        # Try to place another
        result = game.place("mirror", 3, 1, orientation=0)
        assert result is False

    def test_valid_actions_only_available_pieces(self):
        """Test get_valid_actions only includes available piece types."""
        game = LaserMaze()
        game.load("examples/beginner_challenge_01.json")
        actions = game.get_valid_actions()
        place_actions = [a for a in actions if a.action_type == ActionType.PLACE]
        piece_types = set(a.piece_type for a in place_actions)
        # Only mirror should be in actions (it's the only available piece)
        assert piece_types == {"mirror"}

    def test_valid_actions_count_matches_available(self):
        """Test place action count matches available pieces × cells × orientations."""
        game = LaserMaze()
        game.load("examples/beginner_challenge_01.json")
        actions = game.get_valid_actions()
        place_actions = [a for a in actions if a.action_type == ActionType.PLACE]
        # 23 empty cells (25 - laser - target) × 2 orientations = 46
        assert len(place_actions) == 46

    def test_no_challenge_allows_any_piece(self):
        """Test placing any piece works without challenge mode."""
        game = LaserMaze()
        # No challenge loaded, should allow any piece
        result = game.place("laser", 0, 0, orientation=0)
        assert result is True
        result = game.place("mirror", 1, 1, orientation=0)
        assert result is True

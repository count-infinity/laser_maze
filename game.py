"""
High-level game API for Laser Maze.

Provides both interactive play interface and RL gym-style interface.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import IntEnum
import numpy as np

from pieces import (
    Piece, Direction, PieceType, create_piece,
    Laser, Mirror, TargetMirror, BeamSplitter, DoubleMirror, Checkpoint, CellBlocker,
    PIECE_CLASSES
)
from board import Board
from laser import fire_laser, LaserResult
from challenge import Challenge, SolutionResult
from visualize import render_board, print_board
from file_io import load_board, save_board, load_challenge


# Define valid orientations per piece type for RL action space
# Symmetric pieces: Mirror, BeamSplitter, DoubleMirror, Checkpoint have only 2 functional orientations
# Orientation-independent: CellBlocker has only 1 (orientation doesn't matter)
# Asymmetric: Laser, TargetMirror have all 4 orientations
PIECE_ORIENTATIONS = {
    'laser': [0, 1, 2, 3],           # All 4 directions matter
    'mirror': [0, 1],                 # 0=\ diagonal, 1=/ diagonal
    'target_mirror': [0, 1, 2, 3],   # All 4 directions matter (target side)
    'beam_splitter': [0, 1],          # 0=\ diagonal, 1=/ diagonal
    'double_mirror': [0, 1],          # 0=\ diagonal, 1=/ diagonal
    'checkpoint': [0, 1],             # 0=N/S passage (|), 1=E/W passage (-)
    'cell_blocker': [0],              # Orientation doesn't affect behavior
}


class ActionType(IntEnum):
    """Types of actions in the game."""
    PLACE = 0
    ROTATE = 1
    REMOVE = 2
    FIRE = 3


@dataclass
class Action:
    """Represents a game action."""
    action_type: ActionType
    row: int = 0
    col: int = 0
    piece_type: Optional[str] = None
    orientation: int = 0

    def __hash__(self):
        return hash((self.action_type, self.row, self.col, self.piece_type, self.orientation))


@dataclass
class StepResult:
    """Result of taking a step in the RL environment."""
    state: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class LaserMaze:
    """
    Main game class providing both interactive and RL interfaces.

    Interactive usage:
        game = LaserMaze()
        game.load("puzzle.json")
        game.place("mirror", 2, 3, orientation=1)
        game.rotate(2, 3)
        result = game.fire()
        game.show()

    RL usage:
        game = LaserMaze()
        game.load("puzzle.json")
        state = game.get_state_tensor()
        actions = game.get_valid_actions()
        state, reward, done, info = game.step(action)
        game.reset()
    """

    def __init__(self, board_size: int = 5):
        """Initialize empty game."""
        self.board_size = board_size
        self.board: Board = Board(size=board_size)
        self.challenge: Optional[Challenge] = None
        self.initial_state: Optional[Board] = None
        self.available_pieces: List[Piece] = []
        self.placed_pieces: Dict[Tuple[int, int], Piece] = {}
        self.last_result: Optional[LaserResult] = None
        self.last_solution: Optional[SolutionResult] = None

        # RL tracking
        self._step_count = 0
        self._prev_targets_hit = 0

    # === File I/O ===

    def load(self, filepath: str) -> None:
        """
        Load a board or challenge from JSON file.

        Args:
            filepath: Path to JSON file
        """
        data = load_challenge(filepath)
        self.board = data['board']
        self.board_size = self.board.size
        self.initial_state = self.board.copy()

        # Check if this is a challenge (has goal/available)
        if data.get('available'):
            self.challenge = Challenge.from_dict(data)
            self.available_pieces = [p.copy() for p in self.challenge.available_pieces]
        else:
            self.challenge = None
            self.available_pieces = []

        self.placed_pieces = {}
        self.last_result = None
        self.last_solution = None
        self._step_count = 0
        self._prev_targets_hit = 0

    def save(self, filepath: str) -> None:
        """Save current board state to JSON file."""
        save_board(self.board, filepath)

    # === Interactive API ===

    def place(self, piece_type: str, row: int, col: int,
              orientation: int = 0) -> bool:
        """
        Place a piece on the board.

        Args:
            piece_type: Type name ('mirror', 'target_mirror', etc.)
            row: Row position
            col: Column position
            orientation: Direction (0-3)

        Returns:
            True if placed successfully, False if piece not available or cell occupied
        """
        # Check if piece type is available (in challenge mode)
        if self.challenge and piece_type not in self._get_available_piece_types():
            return False

        piece = create_piece(piece_type, orientation, fixed=False)
        success = self.board.place_piece(row, col, piece)

        if success:
            self.placed_pieces[(row, col)] = piece
            # Remove from available if in challenge mode
            self._consume_available_piece(piece_type)

        return success

    def rotate(self, row: int, col: int, clockwise: bool = True) -> bool:
        """
        Rotate piece at position.

        Args:
            row: Row position
            col: Column position
            clockwise: True for CW, False for CCW

        Returns:
            True if rotated successfully
        """
        return self.board.rotate_piece(row, col, clockwise)

    def remove(self, row: int, col: int) -> bool:
        """
        Remove piece from position.

        Args:
            row: Row position
            col: Column position

        Returns:
            True if removed successfully
        """
        piece = self.board.remove_piece(row, col)
        if piece is not None:
            if (row, col) in self.placed_pieces:
                del self.placed_pieces[(row, col)]
                # Return to available if in challenge mode
                self._return_available_piece(piece)
            return True
        return False

    def fire(self) -> LaserResult:
        """
        Fire the laser and get result.

        Returns:
            LaserResult with beam path and targets hit
        """
        self.last_result = fire_laser(self.board)

        if self.challenge:
            placed_all = len(self.available_pieces) == 0
            self.last_solution = self.challenge.check_solution(
                self.last_result, placed_all
            )

        return self.last_result

    def show(self, with_laser: bool = False, show_coords: bool = True) -> None:
        """
        Print the board to console.

        Args:
            with_laser: Whether to show laser beam path
            show_coords: Whether to show row/col coordinates
        """
        result = self.last_result if with_laser else None
        print_board(self.board, result, show_coords)

        if self.last_solution:
            print(self.last_solution)

    def render(self, with_laser: bool = False, show_coords: bool = True) -> str:
        """
        Get board as string.

        Args:
            with_laser: Whether to show laser beam path
            show_coords: Whether to show row/col coordinates

        Returns:
            String representation of board
        """
        result = self.last_result if with_laser else None
        return render_board(self.board, result, show_coords)

    def is_solved(self) -> bool:
        """Check if current state solves the challenge."""
        if self.last_solution is None:
            self.fire()
        return self.last_solution.solved if self.last_solution else False

    # === RL Interface ===

    def reset(self) -> np.ndarray:
        """
        Reset game to initial state.

        Returns:
            Initial state tensor
        """
        if self.initial_state:
            self.board = self.initial_state.copy()
        else:
            self.board = Board(size=self.board_size)

        if self.challenge:
            self.available_pieces = [p.copy() for p in self.challenge.available_pieces]
        else:
            self.available_pieces = []

        self.placed_pieces = {}
        self.last_result = None
        self.last_solution = None
        self._step_count = 0
        self._prev_targets_hit = 0

        return self.get_state_tensor()

    def step(self, action: Union[Action, int]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Take an action and return (state, reward, done, info).

        Args:
            action: Action to take (Action object or int action ID)

        Returns:
            Tuple of (state_tensor, reward, done, info_dict)
        """
        if isinstance(action, int):
            action = self._decode_action(action)

        self._step_count += 1
        reward = -0.01  # Small step penalty
        done = False
        info: Dict[str, Any] = {'valid_action': True}

        if action.action_type == ActionType.PLACE:
            success = self.place(
                action.piece_type,
                action.row,
                action.col,
                action.orientation
            )
            if not success:
                reward = -0.1  # Invalid action penalty
                info['valid_action'] = False

        elif action.action_type == ActionType.ROTATE:
            success = self.rotate(action.row, action.col)
            if not success:
                reward = -0.1
                info['valid_action'] = False

        elif action.action_type == ActionType.REMOVE:
            success = self.remove(action.row, action.col)
            if not success:
                reward = -0.1
                info['valid_action'] = False

        elif action.action_type == ActionType.FIRE:
            result = self.fire()
            info['laser_result'] = result

            # Reward for new targets hit
            new_targets = len(result.targets_hit) - self._prev_targets_hit
            if new_targets > 0:
                reward += 0.1 * new_targets
            self._prev_targets_hit = len(result.targets_hit)

            # Check if solved
            if self.last_solution and self.last_solution.solved:
                reward = 1.0
                done = True
                info['solved'] = True

        state = self.get_state_tensor()
        return state, reward, done, info

    def get_state_tensor(self) -> np.ndarray:
        """
        Get current state as tensor for RL.

        Returns:
            numpy array of shape (channels, board_size, board_size)

        Channels:
            0-7: One-hot piece type (EMPTY, LASER, MIRROR, etc.)
            8-11: One-hot orientation (N, E, S, W) - canonical for symmetric pieces
            12: Fixed piece mask

        Note:
            For symmetric pieces (Mirror, BeamSplitter, DoubleMirror), orientations
            are mapped to canonical form (0 or 1). For orientation-independent pieces
            (Checkpoint, CellBlocker), orientation is always 0.
        """
        num_piece_types = 8  # Including EMPTY
        num_orientations = 4
        num_channels = num_piece_types + num_orientations + 1

        state = np.zeros((num_channels, self.board_size, self.board_size), dtype=np.float32)

        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.board.get_piece(row, col)

                if piece is None:
                    state[0, row, col] = 1.0  # EMPTY channel
                else:
                    # Piece type one-hot
                    state[int(piece.piece_type), row, col] = 1.0

                    # Get canonical orientation for state representation
                    piece_type_name = self._get_piece_type_name(piece)
                    canonical_orientation = self._get_canonical_orientation(
                        piece_type_name, int(piece.orientation)
                    )

                    # Orientation one-hot
                    state[num_piece_types + canonical_orientation, row, col] = 1.0
                    # Fixed mask
                    if piece.fixed:
                        state[num_channels - 1, row, col] = 1.0

        return state

    def get_valid_actions(self) -> List[Action]:
        """
        Get list of valid actions from current state.

        Returns:
            List of valid Action objects

        Note:
            Orientations are reduced based on piece symmetry:
            - Symmetric pieces (Mirror, BeamSplitter, DoubleMirror): 2 orientations
            - Orientation-independent (Checkpoint, CellBlocker): 1 orientation
            - Asymmetric (Laser, TargetMirror): 4 orientations
        """
        actions = []

        # Fire action is always valid if there's a laser
        if self.board.laser_pos or self.board.find_laser():
            actions.append(Action(ActionType.FIRE))

        # Rotate actions for rotatable pieces (not fixed, but fixed_position is ok)
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.board.get_piece(row, col)
                if piece and piece.can_rotate:
                    actions.append(Action(ActionType.ROTATE, row, col))

        # Remove actions for removable pieces (not fixed and not fixed_position)
        for (row, col), piece in self.placed_pieces.items():
            if piece.can_remove:
                actions.append(Action(ActionType.REMOVE, row, col))

        # Place actions for available pieces on empty cells
        empty_cells = self.board.get_empty_cells()
        available_types = self._get_available_piece_types()

        for piece_type in available_types:
            valid_orientations = PIECE_ORIENTATIONS.get(piece_type, [0, 1, 2, 3])
            for row, col in empty_cells:
                for orientation in valid_orientations:
                    actions.append(Action(
                        ActionType.PLACE, row, col,
                        piece_type, orientation
                    ))

        return actions

    def get_action_space_size(self) -> int:
        """
        Get size of action space for discrete action encoding.

        Action encoding:
        - Fire: 1
        - Rotate: board_size^2
        - Remove: board_size^2
        - Place: sum of (board_size^2 * num_orientations) for each piece type

        Orientation counts per piece type:
        - laser: 4, target_mirror: 4
        - mirror: 2, beam_splitter: 2, double_mirror: 2
        - checkpoint: 1, cell_blocker: 1
        """
        board_cells = self.board_size ** 2

        # Calculate place actions with reduced orientations
        place_actions = 0
        for piece_type in PIECE_CLASSES.keys():
            num_orientations = len(PIECE_ORIENTATIONS.get(piece_type, [0, 1, 2, 3]))
            place_actions += board_cells * num_orientations

        rotate_actions = board_cells
        remove_actions = board_cells
        fire_action = 1

        return fire_action + rotate_actions + remove_actions + place_actions

    # === Helper Methods ===

    def _consume_available_piece(self, piece_type: str) -> None:
        """Remove one piece of given type from available list."""
        for i, piece in enumerate(self.available_pieces):
            type_name = self._get_piece_type_name(piece)
            if type_name == piece_type:
                self.available_pieces.pop(i)
                return

    def _return_available_piece(self, piece: Piece) -> None:
        """Return a piece to the available list."""
        self.available_pieces.append(piece.copy())

    def _get_piece_type_name(self, piece: Piece) -> str:
        """Get string name for a piece type."""
        for name, cls in PIECE_CLASSES.items():
            if isinstance(piece, cls):
                return name
        return 'unknown'

    def _get_available_piece_types(self) -> List[str]:
        """Get list of unique piece types available."""
        types = set()
        for piece in self.available_pieces:
            types.add(self._get_piece_type_name(piece))
        return list(types)

    def _get_canonical_orientation(self, piece_type: str, orientation: int) -> int:
        """
        Get canonical orientation for state representation.

        For symmetric pieces (Mirror, BeamSplitter, DoubleMirror):
            N/S (0/2) -> 0 (backslash diagonal)
            E/W (1/3) -> 1 (forward slash diagonal)

        For orientation-independent pieces (Checkpoint, CellBlocker):
            Any -> 0

        For asymmetric pieces (Laser, TargetMirror):
            Returns orientation unchanged (0-3)
        """
        valid_orientations = PIECE_ORIENTATIONS.get(piece_type, [0, 1, 2, 3])

        if len(valid_orientations) == 1:
            # Orientation-independent piece
            return 0
        elif len(valid_orientations) == 2:
            # Symmetric piece: map to canonical (0 or 1)
            # N(0)/S(2) -> 0 (backslash), E(1)/W(3) -> 1 (forward slash)
            return orientation % 2
        else:
            # Asymmetric piece: use full orientation
            return orientation

    def _decode_action(self, action_id: int) -> Action:
        """
        Decode integer action ID to Action object.

        Encoding scheme:
        - 0: Fire
        - 1 to board_size^2: Rotate at cell (id-1)
        - board_size^2+1 to 2*board_size^2: Remove at cell
        - Rest: Place actions with variable orientations per piece type
        """
        board_cells = self.board_size ** 2

        if action_id == 0:
            return Action(ActionType.FIRE)

        action_id -= 1

        if action_id < board_cells:
            row, col = divmod(action_id, self.board_size)
            return Action(ActionType.ROTATE, row, col)

        action_id -= board_cells

        if action_id < board_cells:
            row, col = divmod(action_id, self.board_size)
            return Action(ActionType.REMOVE, row, col)

        action_id -= board_cells

        # Place actions: iterate through piece types with their orientation counts
        for piece_type in PIECE_CLASSES.keys():
            num_orientations = len(PIECE_ORIENTATIONS.get(piece_type, [0, 1, 2, 3]))
            actions_for_this_type = board_cells * num_orientations

            if action_id < actions_for_this_type:
                cell_idx = action_id // num_orientations
                orientation_idx = action_id % num_orientations
                orientation = PIECE_ORIENTATIONS[piece_type][orientation_idx]
                row, col = divmod(cell_idx, self.board_size)
                return Action(ActionType.PLACE, row, col, piece_type, orientation)

            action_id -= actions_for_this_type

        # Should not reach here if action_id is valid
        raise ValueError(f"Invalid action_id: cannot decode")

    def encode_action(self, action: Action) -> int:
        """
        Encode Action object to integer ID.

        Returns:
            Integer action ID
        """
        board_cells = self.board_size ** 2

        if action.action_type == ActionType.FIRE:
            return 0

        if action.action_type == ActionType.ROTATE:
            return 1 + action.row * self.board_size + action.col

        if action.action_type == ActionType.REMOVE:
            return 1 + board_cells + action.row * self.board_size + action.col

        # Place - calculate offset based on piece types before this one
        base_offset = 1 + 2 * board_cells
        piece_types = list(PIECE_CLASSES.keys())

        for pt in piece_types:
            if pt == action.piece_type:
                break
            num_orientations = len(PIECE_ORIENTATIONS.get(pt, [0, 1, 2, 3]))
            base_offset += board_cells * num_orientations

        cell_idx = action.row * self.board_size + action.col
        valid_orientations = PIECE_ORIENTATIONS.get(action.piece_type, [0, 1, 2, 3])
        orientation_idx = valid_orientations.index(action.orientation)
        num_orientations = len(valid_orientations)

        return base_offset + cell_idx * num_orientations + orientation_idx

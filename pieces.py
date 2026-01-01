"""
Piece classes for Laser Maze game.

Each piece has an orientation (0-3 for N/E/S/W) and defines how it
interacts with incoming laser beams.

Direction encoding:
    0 = North (up, row decreases)
    1 = East (right, col increases)
    2 = South (down, row increases)
    3 = West (left, col decreases)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
from enum import IntEnum


class Direction(IntEnum):
    """Cardinal directions for beam travel and piece orientation."""
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    def opposite(self) -> 'Direction':
        """Return the opposite direction."""
        return Direction((self + 2) % 4)

    def rotate_cw(self) -> 'Direction':
        """Rotate 90 degrees clockwise."""
        return Direction((self + 1) % 4)

    def rotate_ccw(self) -> 'Direction':
        """Rotate 90 degrees counter-clockwise."""
        return Direction((self - 1) % 4)

    @property
    def delta(self) -> Tuple[int, int]:
        """Return (row_delta, col_delta) for moving in this direction."""
        deltas = {
            Direction.NORTH: (-1, 0),
            Direction.EAST: (0, 1),
            Direction.SOUTH: (1, 0),
            Direction.WEST: (0, -1),
        }
        return deltas[self]

    @property
    def symbol(self) -> str:
        """Arrow symbol for this direction."""
        symbols = {
            Direction.NORTH: '↑',
            Direction.EAST: '→',
            Direction.SOUTH: '↓',
            Direction.WEST: '←',
        }
        return symbols[self]


class PieceType(IntEnum):
    """Enumeration of piece types for tensor encoding."""
    EMPTY = 0
    LASER = 1
    MIRROR = 2
    TARGET_MIRROR = 3
    BEAM_SPLITTER = 4
    DOUBLE_MIRROR = 5
    CHECKPOINT = 6
    CELL_BLOCKER = 7


@dataclass
class Piece(ABC):
    """
    Base class for all game pieces.

    Attributes:
        orientation: Direction the piece is facing (0-3 for N/E/S/W)
        fixed: If True, piece cannot be moved, removed, OR rotated
        fixed_position: If True, piece cannot be moved or removed, but CAN be rotated
                       (used for puzzle pieces where player must find correct orientation)
    """
    orientation: Direction = Direction.NORTH
    fixed: bool = False
    fixed_position: bool = False

    @property
    @abstractmethod
    def piece_type(self) -> PieceType:
        """Return the piece type enum."""
        pass

    @property
    @abstractmethod
    def symbol(self) -> str:
        """Return a display symbol for this piece."""
        pass

    @abstractmethod
    def interact(self, incoming: Direction) -> List[Direction]:
        """
        Handle an incoming beam and return outgoing directions.

        Args:
            incoming: Direction the beam is traveling when it enters this cell

        Returns:
            List of directions the beam exits (empty if absorbed,
            multiple if split)
        """
        pass

    @property
    def can_rotate(self) -> bool:
        """Check if this piece can be rotated."""
        return not self.fixed  # fixed_position pieces CAN rotate

    @property
    def can_remove(self) -> bool:
        """Check if this piece can be removed from the board."""
        return not self.fixed and not self.fixed_position

    def rotate_cw(self) -> None:
        """Rotate piece 90 degrees clockwise."""
        if self.can_rotate:
            self.orientation = self.orientation.rotate_cw()

    def rotate_ccw(self) -> None:
        """Rotate piece 90 degrees counter-clockwise."""
        if self.can_rotate:
            self.orientation = self.orientation.rotate_ccw()

    def copy(self) -> 'Piece':
        """Create a copy of this piece."""
        return self.__class__(
            orientation=self.orientation,
            fixed=self.fixed,
            fixed_position=self.fixed_position
        )


@dataclass
class Laser(Piece):
    """
    Laser emitter. Emits a beam in the direction it's facing.
    Does not interact with incoming beams (absorbs them).
    """

    @property
    def piece_type(self) -> PieceType:
        return PieceType.LASER

    @property
    def symbol(self) -> str:
        return f'L{self.orientation.symbol}'

    def interact(self, incoming: Direction) -> List[Direction]:  # noqa: ARG002
        # Laser absorbs any beam that hits it
        return []

    def emit_direction(self) -> Direction:
        """Return the direction this laser emits its beam."""
        return self.orientation


@dataclass
class Mirror(Piece):
    """
    Single-sided diagonal mirror. Reflects beams 90 degrees.

    Orientation determines the diagonal:
    - NORTH/SOUTH (0/2): ╲ diagonal - reflects N↔E, S↔W
    - EAST/WEST (1/3): ╱ diagonal - reflects N↔W, S↔E
    """

    @property
    def piece_type(self) -> PieceType:
        return PieceType.MIRROR

    @property
    def symbol(self) -> str:
        if self.orientation in (Direction.NORTH, Direction.SOUTH):
            return 'M╲'
        else:
            return 'M╱'

    def interact(self, incoming: Direction) -> List[Direction]:
        # Determine which diagonal based on orientation
        is_backslash = self.orientation in (Direction.NORTH, Direction.SOUTH)

        if is_backslash:  # ╲ diagonal
            reflection_map = {
                Direction.NORTH: Direction.WEST,
                Direction.EAST: Direction.SOUTH,
                Direction.SOUTH: Direction.EAST,
                Direction.WEST: Direction.NORTH,
            }
        else:  # ╱ diagonal
            reflection_map = {
                Direction.NORTH: Direction.EAST,
                Direction.EAST: Direction.NORTH,
                Direction.SOUTH: Direction.WEST,
                Direction.WEST: Direction.SOUTH,
            }

        return [reflection_map[incoming]]


@dataclass
class TargetMirror(Piece):
    """
    Target with mirror backing. The target side (orientation direction)
    lights up when hit. The mirror has only 2 reflective edges.

    Physical layout (4 sides):
    - Target side: faces orientation direction, illuminates when hit, absorbs beam
    - Mirror side: 45° diagonal opposite the target, reflects beams 90°
    - Two reflective edges: where the mirror meets adjacent sides
    - Blocked side: opposite the mirror diagonal, non-reflective plastic

    Orientation -> Mirror diagonal -> Reflective edges -> Blocked side:
    - NORTH (target up):    \\ diagonal, reflects W<->S, blocks E
    - EAST (target right):  / diagonal, reflects N<->W, blocks S
    - SOUTH (target down):  \\ diagonal, reflects N<->E, blocks W
    - WEST (target left):   / diagonal, reflects S<->E, blocks N
    """

    @property
    def piece_type(self) -> PieceType:
        return PieceType.TARGET_MIRROR

    @property
    def symbol(self) -> str:
        return f'T{self.orientation.symbol}'

    def interact(self, incoming: Direction) -> List[Direction]:
        """
        Handle beam interaction. 'incoming' is the TRAVEL direction of the beam.

        E.g., incoming=SOUTH means beam is traveling south (coming from the north).

        Physical piece has 4 sides:
        - Target side (orientation): red side that illuminates when hit
        - Mirror diagonal (opposite target): 45° mirror with 2 reflective edges
        - Blocked side: non-reflective plastic backing, absorbs beam

        Using user's convention: "beam from X" means beam SOURCE is X (traveling opposite).

        Target EAST (mirror /): from N->W, from W->N, from S->blocked, from E->hit
        Target SOUTH (mirror \\): from N->E, from E->N, from W->blocked, from S->hit
        Target WEST (mirror /): from S->E, from E->S, from N->blocked, from W->hit
        Target NORTH (mirror \\): from W->S, from S->W, from E->blocked, from N->hit
        """
        # Beam traveling opposite to orientation hits the target
        # E.g., target faces EAST, beam traveling WEST (from east) hits it
        if incoming == self.orientation.opposite():
            return []  # Absorbed by target (but target is hit!)

        # For each orientation, one direction is blocked, two directions reflect
        if self.orientation == Direction.EAST:
            # from S -> blocked = traveling N -> blocked
            if incoming == Direction.NORTH:
                return []
            # from N -> W = traveling S -> W
            # from W -> N = traveling E -> N
            return [Direction.WEST if incoming == Direction.SOUTH else Direction.NORTH]

        elif self.orientation == Direction.SOUTH:
            # from W -> blocked = traveling E -> blocked
            if incoming == Direction.EAST:
                return []
            # from N -> E = traveling S -> E
            # from E -> N = traveling W -> N
            return [Direction.EAST if incoming == Direction.SOUTH else Direction.NORTH]

        elif self.orientation == Direction.WEST:
            # from N -> blocked = traveling S -> blocked
            if incoming == Direction.SOUTH:
                return []
            # from S -> E = traveling N -> E
            # from E -> S = traveling W -> S
            return [Direction.EAST if incoming == Direction.NORTH else Direction.SOUTH]

        else:  # NORTH
            # from E -> blocked = traveling W -> blocked
            if incoming == Direction.WEST:
                return []
            # from W -> S = traveling E -> S
            # from S -> W = traveling N -> W
            return [Direction.SOUTH if incoming == Direction.EAST else Direction.WEST]

    def is_hit(self, incoming: Direction) -> bool:
        """Check if this target would be hit by a beam traveling in 'incoming' direction."""
        # Target is hit when beam travels opposite to orientation (into the red side)
        # E.g., target faces EAST, beam traveling WEST (coming from east) hits it
        return incoming == self.orientation.opposite()


@dataclass
class BeamSplitter(Piece):
    """
    Beam splitter. Splits beam into two: one passes through,
    one reflects 90 degrees.

    Orientation determines the split diagonal (like mirror orientation).
    """

    @property
    def piece_type(self) -> PieceType:
        return PieceType.BEAM_SPLITTER

    @property
    def symbol(self) -> str:
        if self.orientation in (Direction.NORTH, Direction.SOUTH):
            return 'B╲'
        else:
            return 'B╱'

    def interact(self, incoming: Direction) -> List[Direction]:
        # One beam passes straight through
        pass_through = incoming

        # Other beam reflects (same logic as mirror)
        is_backslash = self.orientation in (Direction.NORTH, Direction.SOUTH)

        if is_backslash:  # ╲ diagonal
            reflection_map = {
                Direction.NORTH: Direction.WEST,
                Direction.EAST: Direction.SOUTH,
                Direction.SOUTH: Direction.EAST,
                Direction.WEST: Direction.NORTH,
            }
        else:  # ╱ diagonal
            reflection_map = {
                Direction.NORTH: Direction.EAST,
                Direction.EAST: Direction.NORTH,
                Direction.SOUTH: Direction.WEST,
                Direction.WEST: Direction.SOUTH,
            }

        reflected = reflection_map[incoming]

        return [pass_through, reflected]


@dataclass
class DoubleMirror(Piece):
    """
    Double-sided mirror. Both sides reflect like a mirror.

    Orientation determines the diagonal (same as Mirror).
    """

    @property
    def piece_type(self) -> PieceType:
        return PieceType.DOUBLE_MIRROR

    @property
    def symbol(self) -> str:
        if self.orientation in (Direction.NORTH, Direction.SOUTH):
            return 'D╲'
        else:
            return 'D╱'

    def interact(self, incoming: Direction) -> List[Direction]:
        # Same reflection logic as mirror
        is_backslash = self.orientation in (Direction.NORTH, Direction.SOUTH)

        if is_backslash:  # ╲ diagonal
            reflection_map = {
                Direction.NORTH: Direction.WEST,
                Direction.EAST: Direction.SOUTH,
                Direction.SOUTH: Direction.EAST,
                Direction.WEST: Direction.NORTH,
            }
        else:  # ╱ diagonal
            reflection_map = {
                Direction.NORTH: Direction.EAST,
                Direction.EAST: Direction.NORTH,
                Direction.SOUTH: Direction.WEST,
                Direction.WEST: Direction.SOUTH,
            }

        return [reflection_map[incoming]]


@dataclass
class Checkpoint(Piece):
    """
    Checkpoint that the laser must pass through.

    Acts like a doorframe - beams can only pass through when aligned
    with the checkpoint's orientation:
    - NORTH/SOUTH (0/2): Allows beams traveling N↔S (blocks E/W)
    - EAST/WEST (1/3): Allows beams traveling E↔W (blocks N/S)

    Perpendicular beams are blocked (absorbed).
    """

    @property
    def piece_type(self) -> PieceType:
        return PieceType.CHECKPOINT

    @property
    def symbol(self) -> str:
        # Show orientation: | for N/S passage, - for E/W passage
        if self.orientation in (Direction.NORTH, Direction.SOUTH):
            return 'C|'  # Vertical doorframe, N/S beams pass
        else:
            return 'C-'  # Horizontal doorframe, E/W beams pass

    def interact(self, incoming: Direction) -> List[Direction]:
        # Check if beam is aligned with checkpoint orientation
        # N/S orientation allows N/S beams, E/W orientation allows E/W beams
        is_ns_checkpoint = self.orientation in (Direction.NORTH, Direction.SOUTH)
        is_ns_beam = incoming in (Direction.NORTH, Direction.SOUTH)

        if is_ns_checkpoint == is_ns_beam:
            # Beam is aligned - passes through
            return [incoming]
        else:
            # Beam is perpendicular - blocked
            return []


@dataclass
class CellBlocker(Piece):
    """
    Blocks cell from having other pieces placed.
    Does NOT block laser beams - they pass through.
    """

    @property
    def piece_type(self) -> PieceType:
        return PieceType.CELL_BLOCKER

    @property
    def symbol(self) -> str:
        return 'X▪'

    def interact(self, incoming: Direction) -> List[Direction]:
        # Beam passes straight through (blocker only blocks placement)
        return [incoming]


# Mapping from string names to piece classes
PIECE_CLASSES = {
    'laser': Laser,
    'mirror': Mirror,
    'target_mirror': TargetMirror,
    'beam_splitter': BeamSplitter,
    'double_mirror': DoubleMirror,
    'checkpoint': Checkpoint,
    'cell_blocker': CellBlocker,
}


def create_piece(piece_type: str, orientation: int = 0, fixed: bool = False,
                 fixed_position: bool = False) -> Piece:
    """
    Factory function to create a piece by type name.

    Args:
        piece_type: String name of the piece type
        orientation: Direction as integer (0-3)
        fixed: Whether the piece cannot be moved or rotated
        fixed_position: Whether the piece cannot be moved but CAN be rotated

    Returns:
        A new Piece instance
    """
    if piece_type not in PIECE_CLASSES:
        raise ValueError(f"Unknown piece type: {piece_type}")

    return PIECE_CLASSES[piece_type](
        orientation=Direction(orientation),
        fixed=fixed,
        fixed_position=fixed_position
    )

"""
CLI entry point for Laser Maze game.

Provides interactive play and demo functionality.
"""

import argparse
import sys
from pathlib import Path

from game import LaserMaze
from pieces import Direction, create_piece
from board import Board
from laser import fire_laser
from visualize import print_board


def demo_simple():
    """Run a simple demonstration of the game."""
    print("=== Laser Maze Demo ===\n")

    game = LaserMaze()

    # Set up a simple puzzle
    print("Setting up board...")
    game.place("laser", 0, 2, orientation=Direction.SOUTH)
    game.place("mirror", 2, 2, orientation=Direction.NORTH)  # \ diagonal - south->east
    game.place("target_mirror", 2, 4, orientation=Direction.WEST)  # hit from east

    print("\nInitial board:")
    game.show(show_coords=True)

    print("\nFiring laser...")
    result = game.fire()

    print("\nBoard with laser beam:")
    game.show(with_laser=True, show_coords=True)

    print(f"\nTargets hit: {result.num_targets}")
    print(f"Beam path length: {len(result.beam_path)}")


def demo_beam_splitter():
    """Demo with beam splitter."""
    print("\n=== Beam Splitter Demo ===\n")

    game = LaserMaze()

    game.place("laser", 0, 2, orientation=Direction.SOUTH)
    game.place("beam_splitter", 2, 2, orientation=Direction.NORTH)  # \ - splits south->south+east
    game.place("target_mirror", 2, 4, orientation=Direction.WEST)  # hit from east
    game.place("target_mirror", 4, 2, orientation=Direction.NORTH)  # hit from south

    print("Board with beam splitter:")
    game.show(show_coords=True)

    print("\nFiring laser...")
    result = game.fire()

    print("\nBoard with laser beam:")
    game.show(with_laser=True, show_coords=True)

    print(f"\nTargets hit: {result.num_targets}")


def show_challenge_info(game: LaserMaze):
    """Display challenge information and available pieces."""
    if game.challenge:
        print(f"\n--- Challenge ---")
        print(f"Goal: Hit {game.challenge.required_targets} target(s)")
        if game.challenge.required_checkpoints:
            print(f"      Pass through checkpoints: {game.challenge.required_checkpoints}")

        if game.available_pieces:
            piece_counts = {}
            for piece in game.available_pieces:
                name = game._get_piece_type_name(piece)
                piece_counts[name] = piece_counts.get(name, 0) + 1
            pieces_str = ", ".join(f"{count}x {name}" for name, count in piece_counts.items())
            print(f"Available pieces: {pieces_str}")
        else:
            print("Available pieces: (none remaining)")
        print("-----------------")
    else:
        print("\n(Free play mode - no challenge loaded)")


def interactive_mode(filepath: str = None):
    """Run interactive game mode."""
    game = LaserMaze()

    if filepath:
        try:
            game.load(filepath)
            print(f"Loaded: {filepath}")
        except Exception as e:
            print(f"Error loading file: {e}")
            return

    print("\n=== Laser Maze Interactive Mode ===")
    print("Commands:")
    print("  place <type> <row> <col> [orientation]  - Place a piece")
    print("  rotate <row> <col>                      - Rotate piece clockwise")
    print("  remove <row> <col>                      - Remove piece")
    print("  fire                                    - Fire the laser")
    print("  show                                    - Show board")
    print("  beam                                    - Show board with laser")
    print("  info                                    - Show challenge info & available pieces")
    print("  save <file>                             - Save board to file")
    print("  load <file>                             - Load board from file")
    print("  reset                                   - Reset to initial state")
    print("  demo                                    - Run demos")
    print("  quit                                    - Exit")
    print("\nPiece types: laser, mirror, target_mirror, beam_splitter,")
    print("             double_mirror, checkpoint, cell_blocker")
    print("Orientations: 0=N, 1=E, 2=S, 3=W (for mirrors: 0=\\, 1=/)")

    show_challenge_info(game)
    print()
    game.show(show_coords=True)

    while True:
        try:
            cmd = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not cmd:
            continue

        parts = cmd.split()
        command = parts[0]

        try:
            if command == 'quit' or command == 'q':
                print("Goodbye!")
                break

            elif command == 'show' or command == 's':
                game.show(show_coords=True)

            elif command == 'beam' or command == 'b':
                if game.last_result is None:
                    game.fire()
                game.show(with_laser=True, show_coords=True)

            elif command == 'fire' or command == 'f':
                result = game.fire()
                game.show(with_laser=True, show_coords=True)
                if game.is_solved():
                    print("\n*** PUZZLE SOLVED! ***")

            elif command == 'info' or command == 'i':
                show_challenge_info(game)

            elif command == 'place' or command == 'p':
                if len(parts) < 4:
                    print("Usage: place <type> <row> <col> [orientation]")
                    continue
                piece_type = parts[1]
                row = int(parts[2])
                col = int(parts[3])
                orientation = int(parts[4]) if len(parts) > 4 else 0

                # Check if piece type is available before attempting
                if game.challenge and piece_type not in game._get_available_piece_types():
                    print(f"No {piece_type} available! Use 'info' to see available pieces.")
                    continue

                if game.place(piece_type, row, col, orientation):
                    print(f"Placed {piece_type} at ({row}, {col})")
                    if game.challenge and game.available_pieces:
                        remaining = len([p for p in game.available_pieces
                                       if game._get_piece_type_name(p) == piece_type])
                        if remaining > 0:
                            print(f"  ({remaining}x {piece_type} remaining)")
                    game.show(show_coords=True)
                else:
                    print("Cannot place piece there (cell occupied or invalid)")

            elif command == 'rotate' or command == 'r':
                if len(parts) < 3:
                    print("Usage: rotate <row> <col>")
                    continue
                row = int(parts[1])
                col = int(parts[2])

                if game.rotate(row, col):
                    print(f"Rotated piece at ({row}, {col})")
                    game.show(show_coords=True)
                else:
                    print("Cannot rotate (no piece or fixed)")

            elif command == 'remove' or command == 'rm':
                if len(parts) < 3:
                    print("Usage: remove <row> <col>")
                    continue
                row = int(parts[1])
                col = int(parts[2])

                if game.remove(row, col):
                    print(f"Removed piece at ({row}, {col})")
                    game.show(show_coords=True)
                else:
                    print("Cannot remove (no piece or fixed)")

            elif command == 'save':
                if len(parts) < 2:
                    print("Usage: save <filename>")
                    continue
                game.save(parts[1])
                print(f"Saved to {parts[1]}")

            elif command == 'load':
                if len(parts) < 2:
                    print("Usage: load <filename>")
                    continue
                game.load(parts[1])
                print(f"Loaded {parts[1]}")
                game.show(show_coords=True)

            elif command == 'reset':
                game.reset()
                print("Reset to initial state")
                show_challenge_info(game)
                game.show(show_coords=True)

            elif command == 'demo':
                demo_simple()
                demo_beam_splitter()

            elif command == 'help' or command == 'h' or command == '?':
                print("Commands: place, rotate, remove, fire, show, beam, info, save, load, reset, demo, quit")

            else:
                print(f"Unknown command: {command}")
                print("Type 'help' for available commands")

        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Laser Maze Game')
    parser.add_argument('file', nargs='?', help='JSON file to load')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run interactive mode')

    args = parser.parse_args()

    if args.demo:
        demo_simple()
        demo_beam_splitter()
    elif args.interactive or args.file:
        interactive_mode(args.file)
    else:
        # Default: show demo then interactive
        demo_simple()
        print("\n" + "=" * 50 + "\n")
        interactive_mode()


if __name__ == '__main__':
    main()

"""
Training script for Laser Maze RL agent.

Provides two training approaches:
1. PPO (standard) - Explores full action space, learns to avoid invalid actions
2. MaskablePPO - Only considers valid actions, more efficient learning

Both support:
- Curriculum learning
- TensorBoard logging
- Model checkpointing
- Hyperparameter configuration

Usage:
    # Standard PPO (original, for comparison)
    python -m rl.train --algorithm ppo

    # MaskablePPO (recommended, faster learning)
    python -m rl.train --algorithm maskable_ppo
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union

import torch
import numpy as np

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.logger import configure

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.env import LaserMazeEnv, MaskedLaserMazeEnv
from rl.vec_env import make_vec_env, CurriculumVecEnv
from rl.networks import get_feature_extractor, LaserMazeCNN


class CurriculumCallback(BaseCallback):
    """
    Callback for curriculum learning.

    Tracks success rate and increases difficulty when threshold is reached.
    """

    def __init__(
        self,
        curriculum_env: CurriculumVecEnv,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.curriculum_env = curriculum_env

    def _on_step(self) -> bool:
        # Check for episode completions
        for idx, done in enumerate(self.locals.get('dones', [])):
            if done:
                info = self.locals['infos'][idx]
                success = info.get('solved', False)
                self.curriculum_env.record_episode(success)

        return True


class SuccessRateCallback(BaseCallback):
    """
    Callback to track and log success rate.
    """

    def __init__(self, window_size: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window_size = window_size
        self._successes = []

    def _on_step(self) -> bool:
        for idx, done in enumerate(self.locals.get('dones', [])):
            if done:
                info = self.locals['infos'][idx]
                self._successes.append(info.get('solved', False))

                # Keep only recent episodes
                if len(self._successes) > self.window_size * 2:
                    self._successes = self._successes[-self.window_size:]

        # Log success rate periodically
        if self.n_calls % 1000 == 0 and len(self._successes) >= 10:
            recent = self._successes[-self.window_size:]
            success_rate = sum(recent) / len(recent)
            self.logger.record("rollout/success_rate", success_rate)

        return True


def train_ppo(
    # Environment settings
    n_envs: int = 8,
    num_pieces: int = 1,
    max_pieces: int = 4,
    max_steps: int = 50,
    random_puzzles: bool = True,

    # Training settings
    total_timesteps: int = 1_000_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,

    # Network settings
    feature_extractor: str = "cnn",
    features_dim: int = 128,
    net_arch: Optional[Dict[str, Any]] = None,

    # Curriculum settings
    use_curriculum: bool = True,
    success_threshold: float = 0.7,

    # Logging settings
    log_dir: str = "logs",
    save_freq: int = 10000,
    eval_freq: int = 5000,
    verbose: int = 1,

    # Resume settings
    resume_from: Optional[str] = None,

    # Device
    device: str = "auto",
) -> PPO:
    """
    Train a PPO agent on Laser Maze.

    Args:
        n_envs: Number of parallel environments
        num_pieces: Starting number of pieces (for curriculum)
        max_pieces: Maximum pieces for curriculum
        max_steps: Max steps per episode
        random_puzzles: Generate random puzzles
        total_timesteps: Total training timesteps
        learning_rate: Learning rate
        n_steps: Steps per environment per update
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        feature_extractor: Network architecture ("cnn", "mlp", "attention")
        features_dim: Feature dimension
        net_arch: Custom network architecture
        use_curriculum: Enable curriculum learning
        success_threshold: Success rate for curriculum progression
        log_dir: Directory for logs and checkpoints
        save_freq: Checkpoint frequency
        eval_freq: Evaluation frequency
        verbose: Verbosity level
        resume_from: Path to checkpoint to resume from
        device: Device to use ("auto", "cuda", "cpu")

    Returns:
        Trained PPO model
    """
    # Setup directories
    if resume_from:
        # When resuming, use parent directory of checkpoint
        checkpoint_path = Path(resume_from)
        run_dir = checkpoint_path.parent.parent
        print(f"Resuming from: {resume_from}")
        print(f"Continuing in: {run_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(log_dir) / f"ppo_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    # Create environment
    if use_curriculum:
        curriculum = CurriculumVecEnv(
            n_envs=n_envs,
            initial_pieces=num_pieces,
            max_pieces=max_pieces,
            success_threshold=success_threshold,
            log_dir=str(run_dir / "train"),
        )
        env = curriculum.get_env()
    else:
        env = make_vec_env(
            n_envs=n_envs,
            num_pieces=num_pieces,
            random_puzzles=random_puzzles,
            max_steps=max_steps,
            log_dir=str(run_dir / "train"),
        )
        curriculum = None

    # Create evaluation environment
    eval_env = make_vec_env(
        n_envs=4,
        num_pieces=num_pieces,
        random_puzzles=True,
        max_steps=max_steps,
    )

    # Policy kwargs
    policy_kwargs = {
        "features_extractor_class": get_feature_extractor(feature_extractor),
        "features_extractor_kwargs": {"features_dim": features_dim},
    }

    if net_arch is not None:
        policy_kwargs["net_arch"] = net_arch

    # Create or load model
    if resume_from:
        print(f"Loading model from checkpoint...")
        model = PPO.load(
            resume_from,
            env=env,
            device=device,
            tensorboard_log=str(run_dir),
        )
        print(f"Model loaded successfully!")
    else:
        model = PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            tensorboard_log=str(run_dir),
        )

    # Setup logger
    logger = configure(str(run_dir), ["stdout", "tensorboard"])
    model.set_logger(logger)

    # Create callbacks
    callbacks = [
        SuccessRateCallback(verbose=verbose),
        CheckpointCallback(
            save_freq=save_freq // n_envs,
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo_laser_maze",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "best"),
            log_path=str(run_dir / "eval"),
            eval_freq=eval_freq // n_envs,
            n_eval_episodes=20,
            deterministic=True,
        ),
    ]

    if curriculum:
        callbacks.append(CurriculumCallback(curriculum, verbose=verbose))

    callback = CallbackList(callbacks)

    # Train
    print(f"\nStarting training:")
    print(f"  - Environments: {n_envs}")
    print(f"  - Pieces: {num_pieces}" + (f" -> {max_pieces} (curriculum)" if use_curriculum else ""))
    print(f"  - Total timesteps: {total_timesteps:,}")
    print(f"  - Device: {model.device}")
    print(f"  - Log dir: {run_dir}")
    print()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    # Save final model
    model.save(run_dir / "final_model")
    print(f"\nTraining complete! Model saved to {run_dir / 'final_model'}")

    return model


def train_maskable_ppo(
    # Environment settings
    n_envs: int = 8,
    num_pieces: int = 1,
    max_pieces: int = 4,
    max_steps: int = 50,
    random_puzzles: bool = True,

    # Training settings
    total_timesteps: int = 1_000_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,

    # Network settings
    feature_extractor: str = "cnn",
    features_dim: int = 128,
    net_arch: Optional[Dict[str, Any]] = None,

    # Curriculum settings
    use_curriculum: bool = True,
    success_threshold: float = 0.7,

    # Logging settings
    log_dir: str = "logs",
    save_freq: int = 10000,
    eval_freq: int = 5000,
    verbose: int = 1,

    # Resume settings
    resume_from: Optional[str] = None,

    # Device
    device: str = "auto",
) -> MaskablePPO:
    """
    Train a MaskablePPO agent on Laser Maze.

    MaskablePPO only considers valid actions at each step, leading to
    much faster and more efficient learning compared to standard PPO.

    Args:
        (same as train_ppo)

    Returns:
        Trained MaskablePPO model
    """
    # Setup directories
    if resume_from:
        # When resuming, use parent directory of checkpoint
        checkpoint_path = Path(resume_from)
        run_dir = checkpoint_path.parent.parent
        print(f"Resuming from: {resume_from}")
        print(f"Continuing in: {run_dir}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(log_dir) / f"maskable_ppo_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

    # Create environment WITH action masking
    if use_curriculum:
        curriculum = CurriculumVecEnv(
            n_envs=n_envs,
            initial_pieces=num_pieces,
            max_pieces=max_pieces,
            success_threshold=success_threshold,
            log_dir=str(run_dir / "train"),
            use_action_mask=True,  # Enable masking
        )
        env = curriculum.get_env()
    else:
        env = make_vec_env(
            n_envs=n_envs,
            num_pieces=num_pieces,
            random_puzzles=random_puzzles,
            max_steps=max_steps,
            log_dir=str(run_dir / "train"),
            use_action_mask=True,  # Enable masking
        )
        curriculum = None

    # Create evaluation environment (also with masking)
    eval_env = make_vec_env(
        n_envs=4,
        num_pieces=num_pieces,
        random_puzzles=True,
        max_steps=max_steps,
        use_action_mask=True,
    )

    # Policy kwargs (needed for both new and resumed models)
    policy_kwargs = {
        "features_extractor_class": get_feature_extractor(feature_extractor),
        "features_extractor_kwargs": {"features_dim": features_dim},
    }

    if net_arch is not None:
        policy_kwargs["net_arch"] = net_arch

    # Create or load model
    if resume_from:
        print(f"Loading model from checkpoint...")
        model = MaskablePPO.load(
            resume_from,
            env=env,
            device=device,
            tensorboard_log=str(run_dir),
        )
        print(f"Model loaded successfully!")
    else:
        model = MaskablePPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            tensorboard_log=str(run_dir),
        )

    # Setup logger
    logger = configure(str(run_dir), ["stdout", "tensorboard"])
    model.set_logger(logger)

    # Create callbacks
    callbacks = [
        SuccessRateCallback(verbose=verbose),
        CheckpointCallback(
            save_freq=save_freq // n_envs,
            save_path=str(run_dir / "checkpoints"),
            name_prefix="maskable_ppo_laser_maze",
        ),
        # Note: EvalCallback for MaskablePPO needs special handling
        # We'll use a simpler approach
    ]

    if curriculum:
        callbacks.append(CurriculumCallback(curriculum, verbose=verbose))

    callback = CallbackList(callbacks)

    # Train
    print(f"\nStarting MaskablePPO training:")
    print(f"  - Algorithm: MaskablePPO (action masking enabled)")
    print(f"  - Environments: {n_envs}")
    print(f"  - Pieces: {num_pieces}" + (f" -> {max_pieces} (curriculum)" if use_curriculum else ""))
    print(f"  - Total timesteps: {total_timesteps:,}")
    print(f"  - Device: {model.device}")
    print(f"  - Log dir: {run_dir}")
    print()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    # Save final model
    model.save(run_dir / "final_model")
    print(f"\nTraining complete! Model saved to {run_dir / 'final_model'}")

    return model


def main():
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(
        description="Train RL agent on Laser Maze",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with MaskablePPO (recommended)
  python -m rl.train --algorithm maskable_ppo --timesteps 500000

  # Train with standard PPO (for comparison)
  python -m rl.train --algorithm ppo --timesteps 1000000

  # Quick test run
  python -m rl.train --algorithm maskable_ppo --timesteps 10000 --no-curriculum
"""
    )

    # Algorithm choice
    parser.add_argument("--algorithm", type=str, default="maskable_ppo",
                       choices=["ppo", "maskable_ppo"],
                       help="Algorithm: 'ppo' (standard) or 'maskable_ppo' (recommended)")

    # Environment
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--num-pieces", type=int, default=1, help="Starting number of pieces")
    parser.add_argument("--max-pieces", type=int, default=4, help="Max pieces for curriculum")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode")

    # Training
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")

    # Network
    parser.add_argument("--network", type=str, default="cnn",
                       choices=["cnn", "mlp", "attention"], help="Feature extractor")
    parser.add_argument("--features-dim", type=int, default=128, help="Feature dimension")

    # Curriculum
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    parser.add_argument("--success-threshold", type=float, default=0.7,
                       help="Success rate for curriculum progression")

    # Logging
    parser.add_argument("--log-dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--save-freq", type=int, default=10000, help="Checkpoint frequency")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Evaluation frequency")

    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint .zip file to resume training from")

    # Device
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"], help="Device to use")

    args = parser.parse_args()

    # Common kwargs for both algorithms
    train_kwargs = dict(
        n_envs=args.n_envs,
        num_pieces=args.num_pieces,
        max_pieces=args.max_pieces,
        max_steps=args.max_steps,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        feature_extractor=args.network,
        features_dim=args.features_dim,
        use_curriculum=not args.no_curriculum,
        success_threshold=args.success_threshold,
        log_dir=args.log_dir,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        device=args.device,
        resume_from=args.resume,
    )

    # Choose algorithm
    if args.algorithm == "maskable_ppo":
        print("=" * 50)
        print("Using MaskablePPO (action masking enabled)")
        print("=" * 50)
        train_maskable_ppo(**train_kwargs)
    else:
        print("=" * 50)
        print("Using standard PPO (no action masking)")
        print("=" * 50)
        train_ppo(**train_kwargs)


if __name__ == "__main__":
    main()

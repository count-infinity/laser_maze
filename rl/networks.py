"""
Custom neural network architectures for Laser Maze RL.

Provides efficient feature extractors optimized for the 5x5 grid.
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, List, Type


class LaserMazeCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for Laser Maze.

    Architecture designed for 5x5 grid with 13 channels:
    - Uses small kernels appropriate for grid size
    - Global context through final pooling
    - Efficient for the small state space
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]  # 13

        self.cnn = nn.Sequential(
            # Layer 1: 13 -> 32 channels, 3x3 conv
            nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            # Layer 2: 32 -> 64 channels, 3x3 conv
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # Layer 3: 64 -> 64 channels, 3x3 conv
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            # Global average pooling to capture full board context
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Linear layer to features_dim
        self.linear = nn.Sequential(
            nn.Linear(64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class LaserMazeAttention(BaseFeaturesExtractor):
    """
    Attention-based feature extractor for Laser Maze.

    Treats each cell as a token and uses self-attention to capture
    relationships between pieces on the board.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__(observation_space, features_dim)

        n_channels = observation_space.shape[0]  # 13
        grid_size = observation_space.shape[1]   # 5
        n_cells = grid_size * grid_size          # 25

        # Embed each cell
        self.cell_embed = nn.Linear(n_channels, features_dim)

        # Positional encoding for 5x5 grid
        self.pos_embed = nn.Parameter(torch.randn(1, n_cells, features_dim) * 0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=features_dim,
            nhead=n_heads,
            dim_feedforward=features_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # Reshape: (B, C, H, W) -> (B, H*W, C)
        x = observations.flatten(2).transpose(1, 2)

        # Embed cells
        x = self.cell_embed(x)

        # Add positional encoding
        x = x + self.pos_embed

        # Apply transformer
        x = self.transformer(x)

        # Global average over cells
        x = x.mean(dim=1)

        return self.output(x)


class LaserMazeMLP(BaseFeaturesExtractor):
    """
    Simple MLP feature extractor for Laser Maze.

    Flattens the grid and processes with fully connected layers.
    Fast and effective for small grids.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        hidden_dims: List[int] = [256, 256],
    ):
        super().__init__(observation_space, features_dim)

        input_dim = int(torch.prod(torch.tensor(observation_space.shape)))

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, features_dim))
        layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.flatten(start_dim=1)
        return self.mlp(x)


# Registry of available feature extractors
FEATURE_EXTRACTORS: Dict[str, Type[BaseFeaturesExtractor]] = {
    'cnn': LaserMazeCNN,
    'attention': LaserMazeAttention,
    'mlp': LaserMazeMLP,
}


def get_feature_extractor(name: str) -> Type[BaseFeaturesExtractor]:
    """Get feature extractor class by name."""
    if name not in FEATURE_EXTRACTORS:
        raise ValueError(f"Unknown feature extractor: {name}. Available: {list(FEATURE_EXTRACTORS.keys())}")
    return FEATURE_EXTRACTORS[name]

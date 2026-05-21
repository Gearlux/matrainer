import torch
import torch.nn as nn

from confluid import configurable


@configurable
class MNISTSimpleCNN(nn.Module):
    """
    Classic Simple CNN for MNIST classification.
    Expects [B, 1, 28, 28] input.
    """

    def __init__(self, num_classes: int = 10, input_channels: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 28x28 → MaxPool → 14x14 → MaxPool → 7x7
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expects [B, 1, 28, 28] input, pre-normalized to [0, 1]."""
        x = self.conv(x)
        return self.fc(x)  # type: ignore[no-any-return]

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
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expects [B, 1, 28, 28] input."""

        # Normalize: MNIST images are 0-255 in PIL/NumPy format
        if x.max() > 1.0:
            x = x / 255.0

        x = self.conv(x)
        return self.fc(x)  # type: ignore[no-any-return]

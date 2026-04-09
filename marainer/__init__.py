"""Marainer - AI Orchestration Engine."""

__version__ = "0.1.0"

import torch.optim
from confluid import register

# Register standard torch optimizers for confluid usage
register(torch.optim.Adam)
register(torch.optim.SGD)

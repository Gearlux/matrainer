"""Register the deferred third-party runtime members as confluid configurables.

These are the runtime members a standard Lightning training class defers and flows
itself — a ``torch`` optimizer (needs ``params=``), a ``DataLoader`` (needs
``dataset=``), and the ``pytorch_lightning.Trainer`` — plus the stock Lightning
loggers and matrainer's :class:`TrackerLogger`. They are NOT workspace classes, so
they carry no ``@configurable`` decorator; ``register`` here with ``lazy=True``
(optimizers/loader/trainer) or ``role="logger"`` (loggers) so a visual editor
generates the right node kind. Re-exports :class:`MetricPreset` via ``__all__`` so a
discovery ``__all__``-pass surfaces it. Import-only; entry-pointed under
``confluid.configurables`` (``matrainer-lightning``).
"""

from confluid import register
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader

from matrainer.lightning.trackers import TrackerLogger
from matrainer.torch.metrics import MetricPreset

# Optimizers — ``params`` is the runtime-injected slot.
register(Adam, group="optimizer", lazy=True)
register(AdamW, group="optimizer", lazy=True)
register(SGD, group="optimizer", lazy=True)

# DataLoader — ``dataset`` is the runtime-injected slot.
register(DataLoader, group="loader", lazy=True)

# The underlying Lightning Trainer — flowed after the callback / logger wiring.
register(Trainer, group="lightning", lazy=True)

# Stock Lightning loggers — eager object nodes (role="logger"), wired under lightning.logger:.
for _logger_cls in (TensorBoardLogger, CSVLogger, MLFlowLogger):
    register(_logger_cls, role="logger", group="logger")

# TrackerLogger — wrap a matrainer Tracker as a Lightning logger (also wired under lightning.logger:).
register(TrackerLogger, role="logger", group="logger")

__all__ = [
    "Adam",
    "AdamW",
    "SGD",
    "DataLoader",
    "Trainer",
    "TensorBoardLogger",
    "CSVLogger",
    "MLFlowLogger",
    "TrackerLogger",
    "MetricPreset",
]

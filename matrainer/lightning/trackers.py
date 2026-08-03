"""``TrackerLogger`` — expose a matrainer :class:`Tracker` as a ``pytorch_lightning`` logger.

The one thin bridge for the "let Lightning drive the tracker" style: wrap a matrainer
:class:`~matrainer.tracking.base.Tracker` so it can be passed to
``pytorch_lightning.Trainer(logger=...)`` (or wired under ``lightning.logger:`` in
YAML). Lightning then calls it like any logger — ``log_hyperparams`` /
``log_metrics`` / ``save`` / ``finalize`` — so the tracker receives per-step metrics
too, not just the provenance logged through the utility functions.

This is optional: a consumer that prefers stock ``pytorch_lightning.loggers.*`` for
metrics and matrainer only for provenance (git / config / system metrics / log file)
never needs it. The ``Tracker`` interface is the abstraction; this is a single
adapter, not a wrapper layer.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import pytorch_lightning as L
from confluid import register

from matrainer.tracking.base import Tracker


class TrackerLogger(L.loggers.Logger):
    """Adapt a matrainer :class:`Tracker` to the ``pytorch_lightning.loggers.Logger`` interface.

    Args:
        tracker: The matrainer tracker to expose as a Lightning logger.
    """

    def __init__(self, tracker: Tracker) -> None:
        super().__init__()
        self.tracker = tracker

    @property
    def name(self) -> str:
        return self.tracker.name

    @property
    def version(self) -> str:
        return self.tracker.version or ""

    @property
    def experiment(self) -> Any:
        return self.tracker.experiment

    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        mapping = dict(params) if isinstance(params, Mapping) else vars(params)
        self.tracker.log_hyperparams(mapping)

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        self.tracker.log_metrics(metrics, step)

    def save(self) -> None:
        self.tracker.save()

    def finalize(self, status: str = "success") -> None:
        self.tracker.finalize(status)


# Discoverable so a config can wire `!class:matrainer.lightning.TrackerLogger { tracker: ... }`
# under lightning.logger: as an eager object node (role="logger").
register(TrackerLogger, role="logger", group="logger")


__all__ = ["TrackerLogger"]

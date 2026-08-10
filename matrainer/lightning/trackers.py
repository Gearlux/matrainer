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

from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.tracking.base import Tracker


class TrackerLogger(L.loggers.Logger):
    """Adapt a matrainer :class:`Tracker` to the ``pytorch_lightning.loggers.Logger`` interface.

    Args:
        tracker: The matrainer tracker to expose as a Lightning logger.
        metric_names: How the metric keys Lightning emits are named on the way to the tracker —
            see :class:`~matrainer.metric_names.MetricNames`. A ``LightningModule`` that logs
            ``train/loss`` is already writing the canonical vocabulary, so the default translates
            nothing; the knob is what lets a run record in another engine's spelling.
    """

    def __init__(self, tracker: Tracker, metric_names: Optional[MetricNames] = None) -> None:
        super().__init__()
        self.tracker = tracker
        self.metric_names = as_metric_names(metric_names)

    @property
    def name(self) -> str:
        return self.tracker.name

    @property
    def version(self) -> str:
        return self.tracker.version or ""

    @property
    def save_dir(self) -> str:
        """The tracker's own output root — and the thing that places CHECKPOINTS.

        Lightning's base ``Logger.save_dir`` returns ``None``, and
        ``ModelCheckpoint.__resolve_ckpt_dir`` reads it FIRST, falling back to
        ``Trainer.default_root_dir`` (the process CWD) when it is ``None``. So an
        un-overridden adapter sent every checkpoint to
        ``<cwd>/<experiment_name>/<run_name>/checkpoints`` — a run-named directory
        written straight into whatever directory the run was launched from, which for
        a repo-rooted launch means the repository itself. Returning the tracker's
        ``save_dir`` (``./runs`` by default, on every backend) puts checkpoints under
        the SAME root the tracker already writes its metrics to, so one gitignored
        directory holds a whole run.

        This is a projection of an existing value, never a second source of truth:
        override the tracker's ``save_dir`` and the checkpoints follow it.
        """
        return self.tracker.save_dir

    @property
    def experiment(self) -> Any:
        return self.tracker.experiment

    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        mapping = dict(params) if isinstance(params, Mapping) else vars(params)
        self.tracker.log_hyperparams(mapping)

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        self.tracker.log_metrics(self.metric_names(metrics, source="lightning"), step)

    def save(self) -> None:
        self.tracker.save()

    def finalize(self, status: str = "success") -> None:
        self.tracker.finalize(status)


# Discoverable so a config can wire `!class:matrainer.lightning.TrackerLogger { tracker: ... }`
# under lightning.logger: as an eager object node (role="logger").
register(TrackerLogger, role="logger", group="logger")


__all__ = ["TrackerLogger"]

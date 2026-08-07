"""Lightning tracking utilities (``pip install matrainer[lightning]``).

matrainer does NOT own a Lightning training loop — a consumer's training class is a
plain ``pytorch_lightning.LightningModule`` (its own ``training_step`` /
``configure_optimizers`` / …). This subpackage only provides the *optional*
Lightning-specific tracking helpers a standard class can attach:

* :class:`~matrainer.lightning.callbacks.TrackingCallback` — log provenance (git /
  config / params) + capture the log file around a run.
* :class:`~matrainer.lightning.trackers.TrackerLogger` — expose a matrainer
  :class:`~matrainer.tracking.base.Tracker` as a ``pytorch_lightning`` logger.
* the system-metrics callbacks + :func:`~matrainer.lightning.callbacks.activate_for_lightning`,
  the progress bridge, and :func:`~matrainer.lightning.callbacks.build_checkpoint_callbacks`.
* metric-building helpers (:func:`~matrainer.torch.metrics.build_metric_collection`,
  :class:`~matrainer.torch.metrics.MetricPreset`, the JSON-safe coercion helpers).

Importing this subpackage requires ``pytorch_lightning`` (raised with a clear
``pip install matrainer[lightning]`` message when absent).
"""

from matrainer._imports import require

require("pytorch_lightning", "lightning", feature="matrainer.lightning")

from matrainer.lightning.callbacks import (  # noqa: E402
    LightningProgressCallback,
    MlflowSystemMetricsCallback,
    MPSMetricsCallback,
    TrackingCallback,
    activate_for_lightning,
    attach_lightning_progress,
    build_checkpoint_callbacks,
    checkpoint_dir,
    has_mlflow_logger,
)
from matrainer.lightning.module import LightningRunnable  # noqa: E402
from matrainer.lightning.trackers import TrackerLogger  # noqa: E402

# Re-exported from `matrainer.torch` since 2026-08-01, where they moved because they never
# needed Lightning — see that module's docstring. Kept here so this subpackage's public
# surface is unchanged for every existing consumer.
from matrainer.torch.metrics import (  # noqa: E402
    MetricPreset,
    MetricsSpec,
    build_metric_collection,
    coerce_metric_value,
    scalar_metrics,
    write_collection_sidecar,
)

__all__ = [
    "LightningRunnable",
    "TrackingCallback",
    "TrackerLogger",
    "LightningProgressCallback",
    "MPSMetricsCallback",
    "MlflowSystemMetricsCallback",
    "activate_for_lightning",
    "attach_lightning_progress",
    "build_checkpoint_callbacks",
    "checkpoint_dir",
    "has_mlflow_logger",
    "MetricPreset",
    "MetricsSpec",
    "build_metric_collection",
    "coerce_metric_value",
    "scalar_metrics",
    "write_collection_sidecar",
]

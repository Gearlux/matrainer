"""Lightning tracking utilities (``pip install marainer[lightning]``).

marainer does NOT own a Lightning training loop — a consumer's training class is a
plain ``pytorch_lightning.LightningModule`` (its own ``training_step`` /
``configure_optimizers`` / …). This subpackage only provides the *optional*
Lightning-specific tracking helpers a standard class can attach:

* :class:`~marainer.lightning.callbacks.TrackingCallback` — log provenance (git /
  config / params) + capture the log file around a run.
* :class:`~marainer.lightning.trackers.TrackerLogger` — expose a marainer
  :class:`~marainer.tracking.base.Tracker` as a ``pytorch_lightning`` logger.
* the system-metrics callbacks + :func:`~marainer.lightning.callbacks.activate_for_lightning`,
  the progress bridge, and :func:`~marainer.lightning.callbacks.build_checkpoint_callbacks`.
* metric-building helpers (:func:`~marainer.torch.metrics.build_metric_collection`,
  :class:`~marainer.torch.metrics.MetricPreset`, the JSON-safe coercion helpers).

Importing this subpackage requires ``pytorch_lightning`` (raised with a clear
``pip install marainer[lightning]`` message when absent).
"""

from marainer._imports import require

require("pytorch_lightning", "lightning", feature="marainer.lightning")

from marainer.lightning.callbacks import (  # noqa: E402
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
from marainer.lightning.trackers import TrackerLogger  # noqa: E402

# Re-exported from `marainer.torch` since 2026-08-01, where they moved because they never
# needed Lightning — see that module's docstring. Kept here so this subpackage's public
# surface is unchanged for every existing consumer.
from marainer.torch.metrics import (  # noqa: E402
    MetricPreset,
    build_metric_collection,
    coerce_metric_value,
    scalar_metrics,
)

__all__ = [
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
    "build_metric_collection",
    "coerce_metric_value",
    "scalar_metrics",
]

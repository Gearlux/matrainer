"""Torch-only, Lightning-free tracking helpers (``pip install marainer[torch]``).

What a torch training platform needs from a TRACKING library, without pulling
``pytorch_lightning`` in — so a non-Lightning platform (e.g. an Ultralytics trainer) can use it:

* :func:`~marainer.torch.checkpoint.load_lightning_checkpoint` — load weights from a
  Lightning / Ultralytics / plain-state-dict checkpoint (reading back what a tracked run wrote).
* the metric helpers (:func:`~marainer.torch.metrics.build_metric_collection`,
  :class:`~marainer.torch.metrics.MetricPreset`, the JSON-safe coercion helpers), moved here
  from ``marainer.lightning`` on 2026-08-01. They never imported ``pytorch_lightning`` — only
  confluid, torch and torchmetrics — so the old location described their first consumer rather
  than their dependencies, and a backend training on plain torch could not reach them without
  dragging Lightning in. ``marainer.lightning`` re-exports them all, unchanged.
* :func:`~marainer.torch.device.resolve_device` + the closed :data:`~marainer.torch.device.Device`
  literal — the ``"auto"`` accelerator choice (CUDA -> MPS -> CPU) a ``Trainer`` makes for you
  and a hand-written loop has to make itself.

Everything else that once lived here moved DOWN to recordstream (2026-07-29), because it was
data plumbing rather than tracking: ``ensure_record_dataset`` / ``RecordSource`` (a ``Stream``
normalizer), the prediction-output contracts + builders and ``ClassificationPredictionsSink``
(a model↔sink boundary, owned by the package that owns the sink), and the class-weight
statistic (``recordstream.inverse_frequency_weights`` — a property of the labels). The
framework-specific half of balancing — "does this loss take ``weight=``, and how do I inject
it?" — is a per-backend concern and lives in the consuming runnable as an overridable method.
``collate_fn_with_metadata`` was deleted outright: nothing used it.
"""

from marainer.torch.checkpoint import load_lightning_checkpoint
from marainer.torch.device import Device, resolve_device
from marainer.torch.metrics import MetricPreset, build_metric_collection, coerce_metric_value, scalar_metrics

__all__ = [
    "Device",
    "MetricPreset",
    "build_metric_collection",
    "coerce_metric_value",
    "load_lightning_checkpoint",
    "resolve_device",
    "scalar_metrics",
]

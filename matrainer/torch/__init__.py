"""Torch-only, Lightning-free tracking helpers (``pip install matrainer[torch]``).

What a torch training platform needs from a TRACKING library, without pulling
``pytorch_lightning`` in — so a non-Lightning platform (e.g. an Ultralytics trainer) can use it:

* :func:`~matrainer.torch.checkpoint.load_lightning_checkpoint` — load weights from a
  Lightning / Ultralytics / plain-state-dict checkpoint (reading back what a tracked run wrote).
* the metric helpers (:func:`~matrainer.torch.metrics.build_metric_collection`,
  :class:`~matrainer.torch.metrics.MetricPreset`, the JSON-safe coercion helpers), moved here
  from ``matrainer.lightning`` on 2026-08-01. They never imported ``pytorch_lightning`` — only
  confluid, torch and torchmetrics — so the old location described their first consumer rather
  than their dependencies, and a backend training on plain torch could not reach them without
  dragging Lightning in. ``matrainer.lightning`` re-exports them all, unchanged.
* :func:`~matrainer.torch.device.resolve_device` + the closed :data:`~matrainer.torch.device.Device`
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

import matrainer.torch.spellings  # noqa: F401 — registers the torch-family dump spellings (see that module)
from matrainer.torch.checkpoint import load_lightning_checkpoint
from matrainer.torch.device import Device, resolve_device
from matrainer.torch.loop import TorchLoopRunnable
from matrainer.torch.metrics import (
    MetricPreset,
    MetricsSpec,
    build_metric_collection,
    coerce_metric_value,
    scalar_metrics,
    write_collection_sidecar,
)
from matrainer.torch.runnable import TorchTrainerRunnable

__all__ = [
    "Device",
    "MetricPreset",
    "MetricsSpec",
    "TorchLoopRunnable",
    "TorchTrainerRunnable",
    "build_metric_collection",
    "coerce_metric_value",
    "load_lightning_checkpoint",
    "resolve_device",
    "scalar_metrics",
    "write_collection_sidecar",
]

"""Torch-only, Lightning-free training helpers (``pip install marainer[torch]``).

Reusable pieces a torch training platform needs but that carry no
``pytorch_lightning`` dependency, so a non-Lightning platform (e.g. an Ultralytics
trainer) can use them without pulling Lightning in:

* :func:`~marainer.torch.checkpoint.load_lightning_checkpoint` — load weights from a
  Lightning / Ultralytics / plain-state-dict checkpoint.
* :func:`~marainer.torch.data.ensure_record_dataset` /
  :func:`~marainer.torch.data.collate_fn_with_metadata` — dataset normalization + a
  metadata-preserving collate.
* :func:`~marainer.torch.balancing.apply_class_weights` and friends — inverse-frequency
  class weighting for a skewed label distribution.
* The prediction-output contracts (``ClassificationOutput`` / ``DetectionOutput`` /
  ``SegmentationOutput``) + their builders.
"""

from marainer.torch.balancing import apply_class_weights, inverse_frequency_weights, loss_accepts_weight
from marainer.torch.checkpoint import load_lightning_checkpoint
from marainer.torch.data import collate_fn_with_metadata, ensure_record_dataset
from marainer.torch.outputs import (
    ClassificationOutput,
    DetectionOutput,
    DetectionPredictions,
    SegmentationOutput,
    classification_output,
    segmentation_output,
)

__all__ = [
    "apply_class_weights",
    "inverse_frequency_weights",
    "loss_accepts_weight",
    "load_lightning_checkpoint",
    "collate_fn_with_metadata",
    "ensure_record_dataset",
    "ClassificationOutput",
    "DetectionOutput",
    "DetectionPredictions",
    "SegmentationOutput",
    "classification_output",
    "segmentation_output",
]

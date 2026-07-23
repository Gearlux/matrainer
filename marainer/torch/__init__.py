"""Torch-only, Lightning-free training helpers (``pip install marainer[torch]``).

Reusable pieces a torch training platform needs but that carry no
``pytorch_lightning`` dependency, so a non-Lightning platform (e.g. an Ultralytics
trainer) can use them without pulling Lightning in:

* :func:`~marainer.torch.checkpoint.load_lightning_checkpoint` — load weights from a
  Lightning / Ultralytics / plain-state-dict checkpoint.
* :func:`~marainer.torch.data.ensure_sample_dataset` /
  :func:`~marainer.torch.data.collate_fn_with_metadata` — dataset normalization + a
  metadata-preserving collate.
* The prediction-output contracts (``ClassificationOutput`` / ``DetectionOutput`` /
  ``SegmentationOutput``) + their builders.
"""

from marainer.torch.checkpoint import load_lightning_checkpoint
from marainer.torch.data import collate_fn_with_metadata, ensure_sample_dataset
from marainer.torch.outputs import (
    ClassificationOutput,
    DetectionOutput,
    DetectionPredictions,
    SegmentationOutput,
    classification_output,
    segmentation_output,
)

__all__ = [
    "load_lightning_checkpoint",
    "collate_fn_with_metadata",
    "ensure_sample_dataset",
    "ClassificationOutput",
    "DetectionOutput",
    "DetectionPredictions",
    "SegmentationOutput",
    "classification_output",
    "segmentation_output",
]

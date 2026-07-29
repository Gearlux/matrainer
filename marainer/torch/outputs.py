"""Torch BUILDERS for the prediction-output contracts.

The contracts themselves are framework-neutral and live in :mod:`marainer.outputs` — this
module is the torch half: the functions that turn raw logits into a full output dict, which
cannot be neutral because ``softmax`` / ``argmax`` are library calls.

The contracts are re-exported here (not redefined) so a torch consumer has ONE import for the
pair, and so existing ``from marainer.torch.outputs import DetectionOutput`` call sites keep
working. There is exactly one definition of each type.

* :class:`~marainer.outputs.ClassificationOutput` — per-record logits, softmax probabilities,
  argmax class index.
* :class:`~marainer.outputs.DetectionOutput` — per-image object detection (xyxy pixel boxes,
  scores, int64 labels).
* :class:`~marainer.outputs.SegmentationOutput` — per-pixel logits, softmax probabilities,
  argmax mask.

Detection outputs flow directly from their detector's interface — build the dict inline at the
call site; there is deliberately no ``detection_output`` builder to invent boxes from nothing.
"""

import torch
import torch.nn.functional as F
from torch import Tensor

from marainer.outputs import ClassificationOutput, DetectionOutput, DetectionPredictions, SegmentationOutput


def classification_output(logits: Tensor) -> ClassificationOutput[Tensor]:
    """Build a full :class:`~marainer.outputs.ClassificationOutput` from raw ``[B, C]`` logits.

    Example::

        def predict_step(self, batch, batch_idx):
            return classification_output(self(x))     # what a predictions sink reads
    """
    probs = F.softmax(logits, dim=-1)
    class_idx = torch.argmax(logits, dim=-1).to(torch.int64)
    return ClassificationOutput(logits=logits, probs=probs, class_idx=class_idx)


def segmentation_output(logits: Tensor) -> SegmentationOutput[Tensor]:
    """Build a full :class:`~marainer.outputs.SegmentationOutput` from raw ``[B, C, H, W]`` logits."""
    probs = F.softmax(logits, dim=1)
    mask = torch.argmax(logits, dim=1).to(torch.int64)
    return SegmentationOutput(logits=logits, probs=probs, mask=mask)


__all__ = [
    "ClassificationOutput",
    "DetectionOutput",
    "DetectionPredictions",
    "SegmentationOutput",
    "classification_output",
    "segmentation_output",
]

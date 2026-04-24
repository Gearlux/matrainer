"""Typed prediction-output contracts for Marainer models.

Every model's eval-mode ``forward`` should return one of these dicts — or a
task-specific superset of one. The keys *are* the documentation: downstream
metrics, sinks, visualizers, and evaluators inspect known keys instead of
guessing whether a tensor is logits, probabilities, or argmax'd class ids.

Three canonical families are defined:

* :class:`ClassificationOutput` — per-sample logits, softmax probabilities,
  and argmax class index.
* :class:`DetectionOutput` — per-image object detection predictions
  (xyxy pixel boxes, confidence scores, int64 class ids). Task-agnostic
  contract; implementations live in the ``raidar`` package (which depends
  on torchvision), but the schema itself is framework-level.
* :class:`SegmentationOutput` — per-pixel logits, softmax probabilities,
  and argmax mask.

The two helpers (:func:`classification_output`, :func:`segmentation_output`)
build the full dict from raw logits so model code can stay terse.

Detection outputs flow directly from their detector's interface — a helper
would add no value; build the dict inline at the call site.
"""

from typing import List, TypedDict

import torch
import torch.nn.functional as F
from torch import Tensor


class ClassificationOutput(TypedDict):
    """Per-sample classification prediction.

    Keys:
        logits: ``[B, C]`` float — raw pre-softmax scores.
        probs: ``[B, C]`` float — softmax over the last dim.
        class_idx: ``[B]`` int64 — argmax of ``logits`` along the last dim.
    """

    logits: Tensor
    probs: Tensor
    class_idx: Tensor


class DetectionOutput(TypedDict):
    """Per-image object detection prediction.

    Keys:
        boxes: ``[N, 4]`` float32 — xyxy in **absolute pixels**.
        scores: ``[N]`` float32 — confidence in ``[0, 1]``.
        labels: ``[N]`` int64 — class ids. Class 0 is conventionally reserved
            for background in torchvision-style detectors.
    """

    boxes: Tensor
    scores: Tensor
    labels: Tensor


class SegmentationOutput(TypedDict):
    """Per-pixel segmentation prediction.

    Keys:
        logits: ``[B, C, H, W]`` float — raw pre-softmax scores.
        probs: ``[B, C, H, W]`` float — softmax over the channel dim.
        mask: ``[B, H, W]`` int64 — argmax across channels.
    """

    logits: Tensor
    probs: Tensor
    mask: Tensor


DetectionPredictions = List[DetectionOutput]


def classification_output(logits: Tensor) -> ClassificationOutput:
    """Build a full :class:`ClassificationOutput` from raw ``[B, C]`` logits."""
    probs = F.softmax(logits, dim=-1)
    class_idx = torch.argmax(logits, dim=-1).to(torch.int64)
    return ClassificationOutput(logits=logits, probs=probs, class_idx=class_idx)


def segmentation_output(logits: Tensor) -> SegmentationOutput:
    """Build a full :class:`SegmentationOutput` from raw ``[B, C, H, W]`` logits."""
    probs = F.softmax(logits, dim=1)
    mask = torch.argmax(logits, dim=1).to(torch.int64)
    return SegmentationOutput(logits=logits, probs=probs, mask=mask)

"""Prediction-output contracts — FRAMEWORK-NEUTRAL typed dicts.

Every model's eval-mode ``forward`` should return one of these — or a task-specific superset.
The keys *are* the documentation: downstream metrics, sinks, visualizers, and evaluators
inspect known keys instead of guessing whether a tensor is logits, probabilities, or argmax'd
class ids. That guess is not hypothetical: two independently-written detector wrappers agree
that ``boxes`` is xyxy in absolute pixels only because :class:`DetectionOutput` says so.

**Generic in the array type.** Each contract is a generic ``TypedDict`` parameterized by the
array type it carries, so the SAME contract describes a torch run, a numpy/JAX run, or a
TensorFlow one::

    ClassificationOutput[Tensor]      # a torch model
    ClassificationOutput[np.ndarray]  # a numpy / TF / JAX model

The parameter is optional — a bare ``ClassificationOutput`` means "whatever array type", which
is what existing torch call sites already say.

This module is deliberately **dependency-free** (typing only): a consumer on a non-torch
framework must be able to declare its output contract without ``pip install marainer[torch]``.
The BUILDERS are necessarily per-framework — ``softmax`` and ``argmax`` are library calls, not
type declarations — and live in the framework subpackage
(:mod:`marainer.torch.outputs` today). Same split as
:mod:`marainer.checkpoints` (neutral policies) vs ``marainer.lightning`` (the translation).
"""

from typing import Generic, List, TypedDict, TypeVar

#: The array type a contract carries — ``torch.Tensor``, ``np.ndarray``, a TF/JAX array.
ArrayT = TypeVar("ArrayT")

__all__ = [
    "ArrayT",
    "ClassificationOutput",
    "DetectionOutput",
    "DetectionPredictions",
    "SegmentationOutput",
]


class ClassificationOutput(TypedDict, Generic[ArrayT]):
    """Per-record classification prediction.

    Keys:
        logits: ``[B, C]`` float — raw pre-softmax scores.
        probs: ``[B, C]`` float — softmax over the last dim.
        class_idx: ``[B]`` int64 — argmax of ``logits`` along the last dim.
    """

    logits: ArrayT
    probs: ArrayT
    class_idx: ArrayT


class DetectionOutput(TypedDict, Generic[ArrayT]):
    """Per-image object detection prediction.

    Keys:
        boxes: ``[N, 4]`` float32 — xyxy in **absolute pixels**.
        scores: ``[N]`` float32 — confidence in ``[0, 1]``.
        labels: ``[N]`` int64 — class ids. Class 0 is conventionally reserved
            for background in torchvision-style detectors.
    """

    boxes: ArrayT
    scores: ArrayT
    labels: ArrayT


class SegmentationOutput(TypedDict, Generic[ArrayT]):
    """Per-pixel segmentation prediction.

    Keys:
        logits: ``[B, C, H, W]`` float — raw pre-softmax scores.
        probs: ``[B, C, H, W]`` float — softmax over the channel dim.
        mask: ``[B, H, W]`` int64 — argmax across channels.
    """

    logits: ArrayT
    probs: ArrayT
    mask: ArrayT


#: A detector's per-image results — one :class:`DetectionOutput` per image in the batch.
DetectionPredictions = List[DetectionOutput]

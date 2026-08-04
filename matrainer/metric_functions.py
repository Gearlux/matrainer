"""Metrics an engine that owns its loop must accumulate, not average — the confusion-matrix set.

The companion to :mod:`matrainer.metric_slots`: that module decides WHICH metrics a split
measures, this one computes the ones a hand-written loop cannot get right by averaging.

**Why this exists at all.** A backend whose engine owns the loop (Keras, Lightning, a
``transformers.Trainer``) gets stateful metrics for free — the engine accumulates across batches
and reduces once at the end. A backend whose loop WE own (bare torch, TensorFlow, JAX, MLX) sees
one batch at a time, and the obvious move is to score each batch and average. For accuracy that
is very nearly right. For a macro-averaged F1 it is not:

    acc≈0.9, 10 classes, batch 32:  global macro-F1 = 0.9182   batch-averaged = 0.8849
    acc≈0.3, 10 classes, batch 32:  global macro-F1 = 0.3486   batch-averaged = 0.3032

Three to five points, always downward, and worse the more classes a batch is missing — because a
class absent from a batch scores 0 for that batch and drags the mean. In a cross-framework
comparison that is not noise, it is a systematic slander of whichever engines happen to own their
own loop, which is the one thing such a comparison must not do.

So the loop accumulates a **confusion matrix** — one ``C×C`` integer array, the size of which
does not grow with the dataset — and derives the metric from it once per split. That is what
``torchmetrics`` does internally, which is why a number from here is comparable with one from
there.

**Why matrainer and not a training project.** Same argument as
:mod:`matrainer.metric_slots`: this is the metrics vocabulary, three engines need the identical
arithmetic, and the alternative is the same twenty lines copied into each. It sits at the TOP
LEVEL rather than under :mod:`matrainer.torch` because that subpackage requires torch and the
callers here are deliberately framework-free — numpy is the only import, and every engine that
needs this already produces arrays numpy can read.

Typical use — one pass over a split, then one reduction::

    matrix = zero_confusion(num_classes)
    for logits, targets in batches:
        accumulate_confusion(matrix, logits, targets)
    print(macro_f1(matrix))
"""

from typing import Any

import numpy as np

__all__ = ["accumulate_confusion", "macro_f1", "zero_confusion"]


def zero_confusion(num_classes: int) -> np.ndarray:
    """An empty ``num_classes × num_classes`` accumulator: rows are TRUTH, columns PREDICTION."""
    if num_classes < 1:
        raise ValueError(f"num_classes must be at least 1, got {num_classes}.")
    return np.zeros((num_classes, num_classes), dtype=np.int64)


def accumulate_confusion(matrix: np.ndarray, logits: Any, targets: Any) -> np.ndarray:
    """Add one batch to ``matrix`` IN PLACE, and return it for chaining.

    ``logits`` is scored by ``argmax`` over its last axis, so raw logits and probabilities give
    the same answer and no backend has to say which it produces. ``targets`` are int class ids.

    Counting with :func:`numpy.bincount` on the flattened ``truth * C + prediction`` index rather
    than :func:`numpy.add.at`: the latter is the obvious spelling and roughly an order of
    magnitude slower, which is measurable when this runs once per training batch.
    """
    classes = int(matrix.shape[0])
    predicted = np.asarray(logits).argmax(axis=-1).reshape(-1).astype(np.int64)
    actual = np.asarray(targets).reshape(-1).astype(np.int64)
    if predicted.shape != actual.shape:
        raise ValueError(f"logits scored {predicted.shape[0]} rows but targets hold {actual.shape[0]}.")
    # Out-of-range ids would silently wrap into another class's counts, so they are an error —
    # the usual cause is a confusion matrix sized from a stale `num_classes`.
    if actual.size and (int(actual.max()) >= classes or int(actual.min()) < 0):
        raise ValueError(f"target ids must lie in [0, {classes}), got [{int(actual.min())}, {int(actual.max())}].")
    matrix += np.bincount(actual * classes + predicted, minlength=classes * classes).reshape(classes, classes)
    return matrix


def macro_f1(matrix: np.ndarray) -> float:
    """Macro-averaged F1 from a confusion matrix — the unweighted mean of the per-class F1s.

    Written as ``2·TP / (predicted + actual)`` rather than the textbook
    ``2·precision·recall / (precision + recall)``: the two are algebraically the same and this
    form has ONE division, so a class that never appears and is never predicted needs no
    special-casing beyond the zero guard.

    A class with no support and no predictions contributes **0**, matching
    ``sklearn.metrics.f1_score(..., average="macro", zero_division=0)`` and
    ``torchmetrics.classification.MulticlassF1Score(average="macro")`` — which is what makes a
    number from a hand-written loop comparable with the torch backends' torchmetrics one.
    """
    true_positives = np.diag(matrix).astype(np.float64)
    denominator = matrix.sum(axis=0).astype(np.float64) + matrix.sum(axis=1).astype(np.float64)
    per_class = np.divide(2.0 * true_positives, denominator, out=np.zeros_like(true_positives), where=denominator > 0)
    return float(per_class.mean())

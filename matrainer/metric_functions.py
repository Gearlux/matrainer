"""Metrics an engine that owns its loop must accumulate, not average.

TWO families live here, and they answer the same question for different tasks: the
CONFUSION-MATRIX set (classification and per-pixel) and the SQUARED-ERROR set (image restoration).

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

**The squared-error family is the same argument for a different metric, with a different risk
profile worth stating precisely.** PSNR is ``10 * log10(range^2 / mse)`` — a LOGARITHM of a mean —
so the mean of per-batch PSNRs is not the PSNR of the split. Unlike macro-F1, the error is not
always material, and pretending otherwise would be the wrong lesson:

    64 real images, uniform sigma 0.1:   accumulated 20.485 dB, batch-averaged 20.485 (+0.000)
    the same images, sigma 0.02 .. 0.30: accumulated 16.609 dB, batch-averaged 20.114 (**+3.505**)

So on a fixed-degradation run the two agree to three decimals and the accumulator buys nothing
visible; the moment the per-batch error VARIES — mixed noise levels, a model that does well on some
images and badly on others, a ragged final batch — Jensen's inequality bites and the average reads
several dB high, always optimistically. Since the accumulator costs two floats and one pass either
way, it is what every loop uses: the alternative is a metric that is correct until the day the data
stops being uniform, and then silently flattering.

Typical use — one pass over a split, then one reduction::

    matrix = zero_confusion(num_classes)
    for logits, targets in batches:
        accumulate_confusion(matrix, logits, targets)
    print(macro_f1(matrix))

    total = zero_squared_error()
    for restored, clean in batches:
        accumulate_squared_error(total, restored, clean)
    print(psnr(total, data_range=1.0))
"""

from typing import Any, Optional

import numpy as np

__all__ = [
    "accumulate_confusion",
    "accumulate_squared_error",
    "iou_per_class",
    "macro_f1",
    "mean_absolute_error",
    "mean_iou",
    "mean_squared_error",
    "overall_accuracy",
    "psnr",
    "zero_confusion",
    "zero_squared_error",
]


def zero_confusion(num_classes: int) -> np.ndarray:
    """An empty ``num_classes × num_classes`` accumulator: rows are TRUTH, columns PREDICTION."""
    if num_classes < 1:
        raise ValueError(f"num_classes must be at least 1, got {num_classes}.")
    return np.zeros((num_classes, num_classes), dtype=np.int64)


def accumulate_confusion(
    matrix: np.ndarray, logits: Any, targets: Any, ignore_index: Optional[int] = None
) -> np.ndarray:
    """Add one batch to ``matrix`` IN PLACE, and return it for chaining.

    ``logits`` is scored by ``argmax`` over its last axis, so raw logits and probabilities give
    the same answer and no backend has to say which it produces. ``targets`` are int class ids.

    ``ignore_index`` drops the positions whose TARGET is that value — a per-pixel task's void
    class, a padded token — before anything is counted, so they affect no cell of the matrix and
    therefore no metric derived from it. Dropping them is what makes the result agree with
    ``torchmetrics``' own ``ignore_index=``; leaving them in would score the model on positions it
    was told to ignore, and the direction of that error depends on how much of each input is void.

    The drop happens BEFORE the range check on purpose: the usual sentinels (``-100``, ``255``)
    are deliberately outside ``[0, C)``, so checking first would reject every batch that has one.

    Counting with :func:`numpy.bincount` on the flattened ``truth * C + prediction`` index rather
    than :func:`numpy.add.at`: the latter is the obvious spelling and roughly an order of
    magnitude slower, which is measurable when this runs once per training batch.
    """
    classes = int(matrix.shape[0])
    predicted = np.asarray(logits).argmax(axis=-1).reshape(-1).astype(np.int64)
    actual = np.asarray(targets).reshape(-1).astype(np.int64)
    if predicted.shape != actual.shape:
        raise ValueError(f"logits scored {predicted.shape[0]} rows but targets hold {actual.shape[0]}.")
    if ignore_index is not None:
        keep = actual != int(ignore_index)
        predicted, actual = predicted[keep], actual[keep]
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


def overall_accuracy(matrix: np.ndarray) -> float:
    """The fraction of counted positions the model got right — ``trace / total``.

    Micro-averaged accuracy, which is what a per-pixel task calls PIXEL accuracy and what
    ``torchmetrics.classification.MulticlassAccuracy(average="micro")`` reports. It is the one
    confusion-matrix metric a hand-written loop could also get right by averaging — but only when
    every batch holds the same number of counted positions, which stops being true the moment an
    ``ignore_index`` drops a varying number of them or the last batch is short. Deriving it here
    costs nothing and removes the caveat.

    An empty matrix scores ``0.0`` rather than dividing by zero, matching :func:`macro_f1`.
    """
    total = float(matrix.sum())
    return float(np.trace(matrix)) / total if total > 0 else 0.0


def iou_per_class(matrix: np.ndarray) -> np.ndarray:
    """Per-class intersection-over-union — ``TP / (TP + FP + FN)``, one value per class.

    A class that appears in NEITHER the targets nor the predictions scores ``nan``, not ``0``: it
    was never given a chance to be wrong, so a zero would be a judgement the data does not support
    and would drag any mean taken over the vector. ``nan`` says "not present", which is what a
    per-class table should print and what :func:`mean_iou` skips. This is also what
    ``torchmetrics.classification.MulticlassJaccardIndex(average=None)`` does.
    """
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=0).astype(np.float64) + matrix.sum(axis=1).astype(np.float64) - intersection
    return np.divide(intersection, union, out=np.full_like(intersection, np.nan), where=union > 0)


def mean_iou(matrix: np.ndarray) -> float:
    """Mean IoU over the classes PRESENT — the unweighted mean of :func:`iou_per_class`, nan-skipping.

    Skipping the absent classes rather than scoring them 0 is what makes this agree with
    ``MulticlassJaccardIndex(average="macro")``, whose default `zero_division` behaviour is the
    same. It matters most on exactly the split where a naive implementation looks fine: a small
    validation set that never shows one class would otherwise be penalised for the label scheme
    rather than for the model.

    All-absent (an empty matrix) scores ``0.0`` rather than returning ``nan``, so a caller can put
    the value straight into a ``{name: float}`` metric channel.
    """
    per_class = iou_per_class(matrix)
    return float(np.nanmean(per_class)) if not np.all(np.isnan(per_class)) else 0.0


# --------------------------------------------------------------------------- #
# The SQUARED-ERROR family — image restoration's accumulator
# --------------------------------------------------------------------------- #
def zero_squared_error() -> np.ndarray:
    """An empty accumulator: ``[sum of squared errors, sum of absolute errors, element count]``.

    A three-element array rather than three floats, so a caller accumulates in place exactly as it
    does with a confusion matrix — one object per split, mutated per batch, read once.

    Both error sums ride together because a restoration loop wants BOTH the L2-derived metric
    (PSNR) and the L1 one (MAE) from a single pass, and walking the arrays twice to get them would
    double the only expensive part.
    """
    return np.zeros(3, dtype=np.float64)


def accumulate_squared_error(totals: np.ndarray, predictions: Any, targets: Any) -> np.ndarray:
    """Add one batch's squared and absolute errors into ``totals``, in place.

    Args:
        totals: The accumulator from :func:`zero_squared_error`.
        predictions: The restored images — any array numpy can read, any shape.
        targets: The clean images, the SAME shape.

    The shapes must match exactly: a broadcast between ``[N, C, H, W]`` and ``[N, H, W, C]`` would
    silently produce an ``[N, C, H, C]`` error tensor and a number that looks plausible, which is
    precisely the failure a cross-framework comparison cannot survive.
    """
    predicted = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    if predicted.shape != truth.shape:
        raise ValueError(
            f"predictions and targets must have the same shape, got {predicted.shape} and {truth.shape}. "
            "A channels-first prediction against a channels-last target is the usual cause; it would "
            "otherwise BROADCAST into a plausible-looking wrong number."
        )
    difference = predicted - truth
    totals[0] += float(np.sum(difference * difference))
    totals[1] += float(np.sum(np.abs(difference)))
    totals[2] += float(difference.size)
    return totals


def mean_squared_error(totals: np.ndarray) -> float:
    """The split's MSE — the accumulated squared error over the accumulated element count."""
    return float(totals[0] / totals[2]) if totals[2] else 0.0


def mean_absolute_error(totals: np.ndarray) -> float:
    """The split's MAE, from the same single pass."""
    return float(totals[1] / totals[2]) if totals[2] else 0.0


def psnr(totals: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB, from the ACCUMULATED error.

    ``10 * log10(data_range^2 / mse)`` — the definition, over the whole split rather than per
    batch. Agrees with ``torchmetrics.image.PeakSignalNoiseRatio`` and ``keras.ops.psnr`` to six
    decimal places on the same arrays, which is the entire reason a cross-engine table can have one
    column called ``psnr``.

    ``data_range`` is the span of the target's values and is NOT optional in spirit: a wrong one
    shifts every score by a constant (measured on ``[0, 1]`` images — 20.46 dB at ``1.0`` against
    68.59 dB at ``255.0``).

    A perfect reconstruction gives ``mse == 0`` and an infinite PSNR; that is clamped to a large
    finite value, because the caller puts this straight into a ``{name: float}`` channel where an
    ``inf`` is a value a tracker records and nobody can read.
    """
    error = mean_squared_error(totals)
    if error <= 0.0:
        return float(MAX_PSNR_DB)
    return float(10.0 * np.log10((float(data_range) ** 2) / error))


#: What :func:`psnr` reports for a perfect reconstruction, instead of ``inf``. High enough that it
#: cannot be confused with a real score (a strong denoiser reaches the low 30s).
MAX_PSNR_DB: float = 100.0

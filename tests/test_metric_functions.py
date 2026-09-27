"""The confusion-matrix metric helpers — checked against the libraries they must agree with.

The point of this module is COMPARABILITY: a macro-F1 from a hand-written loop has to be the
same number a torchmetrics-driven backend reports for the same predictions, or a cross-framework
table is comparing two different quantities. So the tests here are mostly agreement tests
against ``sklearn`` and ``torchmetrics`` rather than restatements of the arithmetic — the latter
would pass just as happily if both the code and the test were wrong in the same way.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix, f1_score

from matrainer.metric_functions import (
    accumulate_confusion,
    iou_per_class,
    macro_f1,
    mean_iou,
    overall_accuracy,
    zero_confusion,
)


def _predictions(rows: int, classes: int, accuracy: float, seed: int) -> tuple:
    """``(logits, targets)`` for a model that is right about ``accuracy`` of the time."""
    rng = np.random.default_rng(seed)
    targets = rng.integers(0, classes, rows)
    predicted = np.where(rng.random(rows) < accuracy, targets, rng.integers(0, classes, rows))
    logits = np.eye(classes)[predicted] * 5.0 + rng.normal(0, 0.1, (rows, classes))
    return logits, targets


# ---------------------------------------------------------------------------
# zero_confusion
# ---------------------------------------------------------------------------
def test_the_accumulator_is_square_and_integral() -> None:
    matrix = zero_confusion(4)
    assert matrix.shape == (4, 4)
    assert matrix.dtype == np.int64
    assert not matrix.any()


def test_a_class_count_below_one_is_an_error() -> None:
    """Nothing downstream can do anything useful with an empty matrix, so it is refused here."""
    with pytest.raises(ValueError, match="at least 1"):
        zero_confusion(0)


# ---------------------------------------------------------------------------
# accumulate_confusion
# ---------------------------------------------------------------------------
def test_one_batch_matches_sklearns_confusion_matrix() -> None:
    """Rows are TRUTH and columns PREDICTION — the same orientation sklearn uses."""
    logits, targets = _predictions(200, 5, accuracy=0.7, seed=0)
    matrix = accumulate_confusion(zero_confusion(5), logits, targets)
    assert np.array_equal(matrix, confusion_matrix(targets, logits.argmax(-1), labels=range(5)))


def test_accumulating_in_batches_equals_one_pass() -> None:
    """The whole reason the accumulator exists: a loop sees batches, and must not lose by it."""
    logits, targets = _predictions(997, 6, accuracy=0.5, seed=1)
    batched = zero_confusion(6)
    for start in range(0, len(targets), 32):
        accumulate_confusion(batched, logits[start : start + 32], targets[start : start + 32])
    assert np.array_equal(batched, accumulate_confusion(zero_confusion(6), logits, targets))


def test_it_mutates_in_place_and_returns_the_same_array() -> None:
    """Callers keep one accumulator per split; a copy per batch would defeat the point."""
    matrix = zero_confusion(3)
    returned = accumulate_confusion(matrix, np.eye(3)[[0, 1, 2]], np.array([0, 1, 2]))
    assert returned is matrix
    assert matrix.sum() == 3


def test_probabilities_and_raw_logits_give_the_same_answer() -> None:
    """``argmax`` is invariant under softmax, so no backend has to declare which it produces."""
    logits, targets = _predictions(120, 4, accuracy=0.6, seed=2)
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    from_logits = accumulate_confusion(zero_confusion(4), logits, targets)
    from_probabilities = accumulate_confusion(zero_confusion(4), probabilities, targets)
    assert np.array_equal(from_logits, from_probabilities)


def test_a_row_count_mismatch_is_an_error() -> None:
    with pytest.raises(ValueError, match="scored 3 rows but targets hold 2"):
        accumulate_confusion(zero_confusion(3), np.eye(3), np.array([0, 1]))


@pytest.mark.parametrize("bad", [np.array([0, 3]), np.array([0, -1])])
def test_an_out_of_range_target_is_an_error_rather_than_a_silent_wrap(bad: np.ndarray) -> None:
    """The flat ``truth * C + prediction`` index would fold a stray id into another class.

    The usual cause is an accumulator sized from a stale ``num_classes``, which is worth an
    exception: the counts would still add up and the metric would still look plausible.
    """
    with pytest.raises(ValueError, match=r"must lie in \[0, 3\)"):
        accumulate_confusion(zero_confusion(3), np.eye(3)[[0, 1]], bad)


def test_an_empty_batch_changes_nothing() -> None:
    """A short final batch can be empty; the range guard must not read an empty ``max()``."""
    matrix = zero_confusion(3)
    accumulate_confusion(matrix, np.zeros((0, 3)), np.zeros((0,), dtype=np.int64))
    assert not matrix.any()


# ---------------------------------------------------------------------------
# macro_f1
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("accuracy", [0.95, 0.5, 0.1])
def test_it_agrees_with_sklearn(accuracy: float) -> None:
    logits, targets = _predictions(997, 10, accuracy=accuracy, seed=3)
    matrix = accumulate_confusion(zero_confusion(10), logits, targets)
    expected = f1_score(targets, logits.argmax(-1), average="macro", labels=range(10), zero_division=0)
    assert macro_f1(matrix) == pytest.approx(expected)


@pytest.mark.parametrize("accuracy", [0.95, 0.5, 0.1])
def test_it_agrees_with_torchmetrics(accuracy: float) -> None:
    """The number a torch backend reports for the same predictions — the comparability claim."""
    torch = pytest.importorskip("torch")
    from torchmetrics.classification import MulticlassF1Score

    logits, targets = _predictions(997, 10, accuracy=accuracy, seed=4)
    matrix = accumulate_confusion(zero_confusion(10), logits, targets)
    expected = MulticlassF1Score(num_classes=10, average="macro")(torch.tensor(logits), torch.tensor(targets))
    assert macro_f1(matrix) == pytest.approx(float(expected), abs=1e-6)


def test_a_class_with_no_support_and_no_predictions_contributes_zero() -> None:
    """``zero_division=0``, matching both libraries — two of four classes, perfect on those."""
    matrix = accumulate_confusion(zero_confusion(4), np.eye(4)[[0, 1, 0, 1]], np.array([0, 1, 0, 1]))
    assert macro_f1(matrix) == pytest.approx(0.5)


def test_a_perfect_matrix_scores_one() -> None:
    matrix = accumulate_confusion(zero_confusion(3), np.eye(3), np.array([0, 1, 2]))
    assert macro_f1(matrix) == pytest.approx(1.0)


def test_an_empty_matrix_scores_zero_rather_than_dividing_by_zero() -> None:
    """A split that yielded no rows still has to produce a number, not a ``nan`` or a crash."""
    assert macro_f1(zero_confusion(5)) == 0.0


def test_batch_averaging_is_what_this_module_exists_to_avoid() -> None:
    """The measurement quoted in the module docstring, pinned so the claim stays checkable.

    Averaging a per-batch macro-F1 is biased DOWNWARD — a class absent from a batch scores 0 for
    that batch and drags the mean — and the gap is large enough to reorder a comparison table.
    """
    logits, targets = _predictions(1024, 10, accuracy=0.9, seed=5)
    exact = macro_f1(accumulate_confusion(zero_confusion(10), logits, targets))
    per_batch = np.mean(
        [
            macro_f1(accumulate_confusion(zero_confusion(10), logits[i : i + 32], targets[i : i + 32]))
            for i in range(0, 1024, 32)
        ]
    )
    assert per_batch < exact - 0.02, f"exact={exact:.4f} batch-averaged={per_batch:.4f}"


# ---------------------------------------------------------------------------
# ignore_index
# ---------------------------------------------------------------------------
def test_an_ignored_target_is_counted_nowhere() -> None:
    """The whole requirement: a void position must affect no cell, so no metric derived from the
    matrix can be moved by how much of an input was ignored."""
    logits = np.eye(3)[[0, 1, 2, 0]] * 5.0
    targets = np.array([0, 1, 2, 255])

    with_void = accumulate_confusion(zero_confusion(3), logits, targets, ignore_index=255)
    without = accumulate_confusion(zero_confusion(3), logits[:3], targets[:3])
    assert np.array_equal(with_void, without)
    assert int(with_void.sum()) == 3


@pytest.mark.parametrize("sentinel", [-100, 255])
def test_a_sentinel_OUTSIDE_the_class_range_is_dropped_rather_than_rejected(sentinel: int) -> None:
    """The two conventions in the wild (torch's -100, SegFormer's 255) are both out of `[0, C)`,
    so the drop has to happen BEFORE the range check — checking first would reject every batch
    that has one."""
    logits = np.eye(3)[[0, 1]] * 5.0
    targets = np.array([0, sentinel])

    assert int(accumulate_confusion(zero_confusion(3), logits, targets, ignore_index=sentinel).sum()) == 1


def test_an_out_of_range_id_that_is_NOT_the_ignore_index_is_still_an_error() -> None:
    """The guard the drop must not weaken: a stray id from a stale `num_classes` would otherwise
    wrap into another class's counts and the totals would still add up."""
    logits = np.eye(3)[[0, 0]] * 5.0
    with pytest.raises(ValueError, match=r"must lie in \[0, 3\)"):
        accumulate_confusion(zero_confusion(3), logits, np.array([0, 7]), ignore_index=255)


def test_ignoring_everything_leaves_an_empty_matrix_rather_than_raising() -> None:
    """An all-void input is legal — a crop that happens to contain no annotated pixel."""
    logits = np.eye(3)[[0, 1]] * 5.0
    assert not accumulate_confusion(zero_confusion(3), logits, np.array([255, 255]), ignore_index=255).any()


def test_the_ignore_index_agrees_with_torchmetrics() -> None:
    """The claim that makes the number comparable: dropping void positions here gives what
    torchmetrics' own `ignore_index=` gives for the same predictions."""
    torchmetrics = pytest.importorskip("torchmetrics")
    import torch

    rng = np.random.default_rng(11)
    targets = rng.integers(0, 4, 512)
    targets[rng.random(512) < 0.3] = 255
    logits = np.eye(4)[rng.integers(0, 4, 512)] * 5.0 + rng.normal(0, 0.1, (512, 4))

    matrix = accumulate_confusion(zero_confusion(4), logits, targets, ignore_index=255)
    theirs = torchmetrics.classification.MulticlassJaccardIndex(num_classes=4, average="macro", ignore_index=255)
    assert mean_iou(matrix) == pytest.approx(float(theirs(torch.tensor(logits), torch.tensor(targets))), abs=1e-6)


# ---------------------------------------------------------------------------
# overall_accuracy / iou_per_class / mean_iou
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("accuracy", [0.3, 0.7, 0.95])
def test_overall_accuracy_agrees_with_torchmetrics_micro(accuracy: float) -> None:
    torchmetrics = pytest.importorskip("torchmetrics")
    import torch

    logits, targets = _predictions(1024, 5, accuracy=accuracy, seed=3)
    matrix = accumulate_confusion(zero_confusion(5), logits, targets)
    theirs = torchmetrics.classification.MulticlassAccuracy(num_classes=5, average="micro")
    assert overall_accuracy(matrix) == pytest.approx(
        float(theirs(torch.tensor(logits), torch.tensor(targets))), abs=1e-6
    )


@pytest.mark.parametrize("accuracy", [0.3, 0.7, 0.95])
def test_mean_iou_agrees_with_torchmetrics_macro_jaccard(accuracy: float) -> None:
    torchmetrics = pytest.importorskip("torchmetrics")
    import torch

    logits, targets = _predictions(1024, 5, accuracy=accuracy, seed=4)
    matrix = accumulate_confusion(zero_confusion(5), logits, targets)
    theirs = torchmetrics.classification.MulticlassJaccardIndex(num_classes=5, average="macro")
    assert mean_iou(matrix) == pytest.approx(float(theirs(torch.tensor(logits), torch.tensor(targets))), abs=1e-6)


def test_iou_per_class_agrees_with_torchmetrics_elementwise() -> None:
    """`average=None`, the per-class VECTOR a segmentation report renders worst-first."""
    torchmetrics = pytest.importorskip("torchmetrics")
    import torch

    logits, targets = _predictions(512, 4, accuracy=0.6, seed=6)
    matrix = accumulate_confusion(zero_confusion(4), logits, targets)
    theirs = torchmetrics.classification.MulticlassJaccardIndex(num_classes=4, average=None)
    assert np.allclose(iou_per_class(matrix), theirs(torch.tensor(logits), torch.tensor(targets)).numpy(), atol=1e-6)


def test_a_class_present_nowhere_is_nan_rather_than_zero() -> None:
    """`nan` says "not present"; `0` would be a judgement the data does not support, and would
    drag `mean_iou` down for the label scheme rather than for the model."""
    logits = np.eye(3)[[0, 1, 0, 1]] * 5.0
    matrix = accumulate_confusion(zero_confusion(3), logits, np.array([0, 1, 0, 1]))
    per_class = iou_per_class(matrix)

    assert np.isnan(per_class[2])
    assert mean_iou(matrix) == pytest.approx(1.0), "the absent class must be skipped, not scored 0"


def test_a_perfect_matrix_scores_one_on_both() -> None:
    matrix = accumulate_confusion(zero_confusion(3), np.eye(3) * 5.0, np.arange(3))
    assert overall_accuracy(matrix) == pytest.approx(1.0)
    assert mean_iou(matrix) == pytest.approx(1.0)


def test_an_empty_matrix_scores_zero_on_both_rather_than_nan() -> None:
    """A caller puts these straight into a `{name: float}` metric channel, where a `nan` is a
    value a tracker will happily record and nobody can read."""
    assert overall_accuracy(zero_confusion(4)) == 0.0
    assert mean_iou(zero_confusion(4)) == 0.0


def _accumulated(logits: np.ndarray, targets: np.ndarray, size: int) -> np.ndarray:
    """One matrix fed in ``size``-row batches — the shape every hand-written loop uses."""
    matrix = zero_confusion(10)
    for i in range(0, len(targets), size):
        accumulate_confusion(matrix, logits[i : i + size], targets[i : i + size])
    return matrix


def _batch_averaged_iou(logits: np.ndarray, targets: np.ndarray, size: int) -> float:
    """What a loop that scores each batch and averages would report."""
    return float(
        np.mean(
            [
                mean_iou(accumulate_confusion(zero_confusion(10), logits[i : i + size], targets[i : i + size]))
                for i in range(0, len(targets), size)
            ]
        )
    )


def test_an_ACCUMULATED_mean_iou_does_not_depend_on_the_batch_size() -> None:
    """The property that makes a cross-engine table readable, and the one a per-batch average
    cannot have: the same predictions must give one number however they were fed in.

    It is the check to reach for when reviewing a hand-written loop's metric — restating the
    arithmetic would pass just as happily if the code and the test were wrong the same way.
    """
    logits, targets = _predictions(1024, 10, accuracy=0.7, seed=7)
    values = {size: mean_iou(_accumulated(logits, targets, size)) for size in (4, 16, 128, 1024)}

    assert len({round(value, 12) for value in values.values()}) == 1, values


def test_batch_averaging_a_mean_iou_MOVES_WITH_the_batch_size() -> None:
    """The segmentation half of this module's reason to exist, and it is worse than macro-F1's.

    A class absent from a batch is SKIPPED by that batch's mean rather than scored 0, so batches
    with fewer classes present are over-weighted — and unlike macro-F1's, the error is not even
    consistently signed. Measured on 10 classes at accuracy 0.7 against an exact 0.5922:
    batch 4 reads 0.6164 (+0.024), batch 32 reads 0.5798 (-0.012). So it is not a constant offset
    a reader could learn to correct for; it is a different number per configuration.
    """
    logits, targets = _predictions(1024, 10, accuracy=0.7, seed=7)
    exact = mean_iou(_accumulated(logits, targets, 1024))
    small, large = _batch_averaged_iou(logits, targets, 4), _batch_averaged_iou(logits, targets, 32)

    assert small > exact > large, f"exact={exact:.4f} batch4={small:.4f} batch32={large:.4f}"
    assert small - large > 0.03, "the two batch sizes must disagree by more than rounding"

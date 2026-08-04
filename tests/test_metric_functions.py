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

from matrainer.metric_functions import accumulate_confusion, macro_f1, zero_confusion


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

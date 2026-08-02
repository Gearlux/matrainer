"""Tests for ``render_matrix_as_table`` — the markdown rendering of a square matrix.

Small surface, but it feeds two destinations with different requirements: a plain log (where
alignment is what makes it readable) and ``Tracker.log_text`` (where a markdown renderer decides
whether it sees a TABLE or one run-on paragraph). The blank line after the title is what the
second one needs, so it is asserted rather than assumed.
"""

from typing import Any, List

import pytest

from marainer.tracking import render_matrix_as_table


class _FakeTensor:
    """Stands in for a torch tensor: the renderer must go through ``detach().cpu().tolist()``."""

    def __init__(self, rows: List[List[int]]) -> None:
        self.rows = rows
        self.detached = False

    def detach(self) -> "_FakeTensor":
        self.detached = True
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def tolist(self) -> List[List[int]]:
        return self.rows


def test_rows_are_true_classes_and_columns_predicted() -> None:
    """The torchmetrics convention: ``C[i, j]`` counts ``true=i, pred=j``.

    Getting this backwards produces a plausible-looking table that says the opposite thing,
    so the orientation is pinned rather than eyeballed.
    """
    rendered = render_matrix_as_table("val/confusion", [[5, 1], [0, 4]], ["cat", "dog"])

    lines = rendered.splitlines()
    assert lines[0] == "val/confusion:"
    assert "pred:cat" in lines[2] and "pred:dog" in lines[2]
    assert lines[4].startswith("| true:cat") and lines[4].split("|")[2].strip() == "5"
    assert lines[5].startswith("| true:dog") and lines[5].split("|")[3].strip() == "4"


def test_a_blank_line_separates_the_title_from_the_table() -> None:
    """Markdown needs it: without the blank line a renderer sees one paragraph, not a table.

    This is what makes the same string usable in TensorBoard's Text tab and as an MLflow `.md`
    artifact, not only in a log.
    """
    rendered = render_matrix_as_table("val/confusion", [[1]], ["a"])

    assert rendered.splitlines()[1] == ""


def test_a_tensor_is_read_through_detach_cpu_tolist() -> None:
    """A live metric value is on a device and possibly in a graph; a bare `tolist` is not enough."""
    matrix = _FakeTensor([[2, 0], [1, 3]])

    rendered = render_matrix_as_table("cm", matrix, ["a", "b"])

    assert matrix.detached
    assert "| true:a" in rendered and "| true:b" in rendered


def test_a_plain_nested_list_needs_no_conversion() -> None:
    """The payload `confusion_matrices_payload` hands over is already a nested list of ints."""
    assert "| true:a" in render_matrix_as_table("cm", [[1, 0], [0, 1]], ["a", "b"])


@pytest.mark.parametrize("value", [[[1, 2], [3, 4]], _FakeTensor([[1, 2], [3, 4]])])
def test_every_count_reaches_the_table(value: Any) -> None:
    rendered = render_matrix_as_table("cm", value, ["a", "b"])

    for count in ("1", "2", "3", "4"):
        assert count in rendered

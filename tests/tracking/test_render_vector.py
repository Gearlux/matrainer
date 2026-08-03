"""Tests for the PER-CLASS VECTOR renderers — the shape ``average=None`` produces.

The vector twin of ``test_render.py``, and it exists because a per-class metric is not a
degenerate matrix: it has one axis of meaning, its values share a scale, and what a reader wants
from it is a RANKING (which class is the model worst at). Everything asserted below follows from
that — the worst-first sort, the mean reference, and the selection rule that keeps scalars and
matrices out.
"""

from typing import Any, List

import pytest

from marainer.tracking import per_class_vectors, render_matrix_as_table, render_vector_as_figure, render_vector_as_table

CLASSES = ["pet", "background", "boundary"]
IOU = [0.81, 0.12, 0.55]


class _FakeTensor:
    """Stands in for a torch tensor: the renderer must go through ``detach().cpu().tolist()``."""

    def __init__(self, values: Any) -> None:
        self.values = values
        self.detached = False

    def detach(self) -> "_FakeTensor":
        self.detached = True
        return self

    def cpu(self) -> "_FakeTensor":
        return self

    def tolist(self) -> Any:
        return self.values


# --------------------------------------------------------------------------- #
# The selection rule
# --------------------------------------------------------------------------- #
def test_it_selects_by_SHAPE_not_by_metric_name() -> None:
    """A runnable emits scalars, vectors and matrices in ONE mapping with no knowledge of which
    is which, so a name-based rule would need a naming convention nothing enforces."""
    found = per_class_vectors(
        {
            "val/miou": _FakeTensor(0.49),
            "val/iou_per_class": _FakeTensor(IOU),
            "val/confusion": _FakeTensor([[1, 0], [0, 1]]),
            "val/dice_per_class": [0.2, 0.4],
        },
        class_names=CLASSES,
    )
    assert [f["name"] for f in found] == ["val/iou_per_class", "val/dice_per_class"]


def test_a_scalar_is_not_a_length_one_vector() -> None:
    """The easy mistake: a scalar metric belongs in `log_dict`, not in a table of its own."""
    assert per_class_vectors({"val/miou": 0.49, "val/one": [0.3]}) == []


def test_a_matrix_falls_through_to_its_own_renderer() -> None:
    """One mapping is passed to BOTH selection rules, so each must take only its own shape."""
    metrics = {"val/confusion": [[5, 1], [0, 4]]}
    assert per_class_vectors(metrics) == []
    assert "pred:cat" in render_matrix_as_table("val/confusion", metrics["val/confusion"], ["cat", "dog"])


def test_a_bare_vector_is_accepted_and_named() -> None:
    found = per_class_vectors(_FakeTensor(IOU), class_names=CLASSES)
    assert [f["name"] for f in found] == ["per_class"]
    assert found[0]["values"] == IOU


def test_class_names_label_the_values_in_ORDER() -> None:
    """The labels are the metric's class axis, so pairing them wrong mislabels every row."""
    found = per_class_vectors({"iou": IOU}, class_names=CLASSES)[0]
    assert found["class_names"] == CLASSES


def test_a_length_mismatch_falls_back_to_indices_rather_than_dropping_the_value() -> None:
    """Usually a metric with its OWN axis (a per-threshold curve), not a bug worth hiding over."""
    found = per_class_vectors({"pr_curve": [0.1, 0.2, 0.3, 0.4]}, class_names=CLASSES)[0]
    assert found["class_names"] == ["0", "1", "2", "3"]


def test_no_class_names_means_indices() -> None:
    assert per_class_vectors({"iou": IOU})[0]["class_names"] == ["0", "1", "2"]


def test_a_non_numeric_sequence_is_ignored() -> None:
    assert per_class_vectors({"names": ["a", "b", "c"]}) == []


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def test_the_table_is_sorted_WORST_FIRST() -> None:
    """The whole reason it exists beside the scalar mean: the mean already says how the model is
    doing, so what this adds is *which class it is worst at* — the first row, not something to
    scan for."""
    lines = [ln for ln in render_vector_as_table("val/iou", IOU, CLASSES).splitlines() if ln.startswith("|")]
    ordered = [ln.split("|")[1].strip() for ln in lines[2:]]
    assert ordered == ["background", "boundary", "pet"]


def test_the_mean_is_restated_in_the_title() -> None:
    """A reader looking at per-class values wants to know at once whether one low class is
    dragging an otherwise fine average."""
    assert "(mean 0.4933)" in render_vector_as_table("val/iou", IOU, CLASSES)


def test_a_blank_line_separates_the_title_from_the_table() -> None:
    """`Tracker.log_text` feeds a markdown renderer, which needs it to see a TABLE rather than
    one run-on paragraph. In a plain log it costs a blank line."""
    assert render_vector_as_table("val/iou", IOU, CLASSES).split("\n")[1] == ""


def test_the_values_keep_the_precision_the_column_advertises() -> None:
    """tabulate PARSES a numeric-looking string and reformats it, so pre-formatting the value
    would let it be silently re-rounded — the column has to ask for the precision itself."""
    assert "0.1200" in render_vector_as_table("val/iou", IOU, CLASSES)


def test_the_column_header_drops_the_split_prefix() -> None:
    """The title already carries `val/`; repeating it in the column wastes the width."""
    header = render_vector_as_table("val/iou_per_class", IOU, CLASSES).splitlines()[2]
    assert "iou_per_class" in header and "val/" not in header


def test_a_tensor_renders_without_a_manual_conversion() -> None:
    tensor = _FakeTensor(IOU)
    assert "0.8100" in render_vector_as_table("val/iou", tensor, CLASSES)
    assert tensor.detached, "a live metric value is on a device and possibly in a graph"


def test_no_class_names_falls_back_to_indices() -> None:
    """The label column is then right-aligned by tabulate (it parses the indices as numbers),
    which is why this reads the cells rather than matching a literal `"| 0 "`."""
    rows = [ln for ln in render_vector_as_table("iou", IOU, []).splitlines() if ln.startswith("|")][2:]
    assert [ln.split("|")[1].strip() for ln in rows] == ["1", "2", "0"]


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
pytest.importorskip("matplotlib", reason="the figure form is the `plot` extra")


def test_the_figure_is_not_registered_with_pyplot() -> None:
    """Built through `Figure` directly: a pyplot figure lives in a process-global manager until
    someone closes it, so a value rendered once an epoch leaks one per epoch."""
    import matplotlib.pyplot as plt

    before = len(plt.get_fignums())
    render_vector_as_figure("val/iou", IOU, CLASSES)
    assert len(plt.get_fignums()) == before


def test_the_bars_are_sorted_worst_first_top_down() -> None:
    """Matching the table's first row, and the reason the y axis is inverted."""
    figure = render_vector_as_figure("val/iou", IOU, CLASSES)
    axes = figure.axes[0]
    labels = [t.get_text() for t in axes.get_yticklabels()]
    assert labels == ["background", "boundary", "pet"]
    bottom, top = axes.get_ylim()
    assert bottom > top, "inverted, so the worst class is at the TOP"


def test_every_value_becomes_a_bar() -> None:
    figure = render_vector_as_figure("val/iou", IOU, CLASSES)
    widths = sorted(round(patch.get_width(), 4) for patch in figure.axes[0].patches)
    assert widths == sorted(round(v, 4) for v in IOU)


def test_the_mean_is_drawn_as_a_reference_line() -> None:
    """ "Which classes are below average" is the question this figure is looked at to answer."""
    figure = render_vector_as_figure("val/iou", IOU, CLASSES)
    positions = [line.get_xdata()[0] for line in figure.axes[0].lines]
    assert any(abs(float(p) - sum(IOU) / len(IOU)) < 1e-6 for p in positions)


def test_annotations_are_dropped_when_there_are_too_many_classes() -> None:
    """Past ~40 they collide with the bars, and the table carries the numbers anyway."""
    many: List[float] = [i / 100 for i in range(50)]
    figure = render_vector_as_figure("val/iou", many, [str(i) for i in range(50)])
    texts = [t.get_text() for t in figure.axes[0].texts]
    assert all("mean" in t for t in texts), f"unexpected per-bar annotations: {texts[:3]}"


def test_the_figure_grows_with_the_class_count_within_bounds() -> None:
    """Two classes must not be a postage stamp, a hundred must not be a poster."""
    small = render_vector_as_figure("iou", [0.1, 0.2], ["a", "b"]).get_size_inches()[1]
    large = render_vector_as_figure("iou", [0.1] * 100, [str(i) for i in range(100)]).get_size_inches()[1]
    assert 2.5 <= small < large <= 20.0


def test_a_tensor_renders_without_a_manual_conversion_in_the_figure() -> None:
    figure = render_vector_as_figure("val/iou", _FakeTensor(IOU), CLASSES)
    assert len(figure.axes[0].patches) == len(IOU)

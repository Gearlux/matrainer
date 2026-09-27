"""The non-scalar reporting fan-outs (`report_matrices` / `report_vectors`).

These pin the WHEN/WHERE loop the training runnables delegate to: selection is by SHAPE
(recordstream's matrix rule, this package's vector rule), the table always reaches the log and
every tracker's text channel, and a figure is rendered once per payload only when asked for.
"""

from typing import Any, List, Optional, Tuple

import pytest

from matrainer.tracking.render import report_matrices, report_vectors


class _TrackerRecorder:
    def __init__(self) -> None:
        self.texts: List[Tuple[str, str, Optional[int]]] = []
        self.figures: List[Tuple[str, Any, Optional[int]]] = []

    def log_text(self, name: str, text: str, step: Optional[int] = None) -> None:
        self.texts.append((name, text, step))

    def log_figure(self, name: str, figure: Any, step: Optional[int] = None) -> None:
        self.figures.append((name, figure, step))


def test_vectors_reach_every_tracker_text_channel_with_the_step() -> None:
    trackers = [_TrackerRecorder(), _TrackerRecorder()]
    values = {"iou_per_class": [0.9, 0.1, 0.5], "mean_iou": 0.5}
    report_vectors(values, ["bg", "pet", "edge"], trackers, step=3)
    for tracker in trackers:
        assert len(tracker.texts) == 1
        name, table, step = tracker.texts[0]
        assert name == "iou_per_class" and step == 3
        assert "pet" in table
        assert tracker.figures == [], "plot=False must render no figure"


def test_a_scalar_only_mapping_reports_nothing() -> None:
    tracker = _TrackerRecorder()
    report_vectors({"loss": 0.5}, None, [tracker])
    report_matrices({"loss": 0.5}, None, [tracker])
    assert tracker.texts == [] and tracker.figures == []


def test_matrices_are_selected_by_shape_and_fan_out() -> None:
    tracker = _TrackerRecorder()
    values = {"confusion": [[5, 1], [0, 4]], "accuracy": 0.9}
    report_matrices(values, ["cat", "dog"], [tracker], step=1)
    assert [name for name, _, _ in tracker.texts] == ["confusion"]
    assert "pred:dog" in tracker.texts[0][1]


def test_plot_renders_one_figure_shared_by_every_tracker() -> None:
    pytest.importorskip("matplotlib")
    trackers = [_TrackerRecorder(), _TrackerRecorder()]
    report_vectors({"iou_per_class": [0.9, 0.1]}, None, trackers, plot=True)
    figures = [tracker.figures[0][1] for tracker in trackers]
    assert figures[0] is figures[1], "a figure is real work, rendered ONCE per payload"


def test_no_trackers_still_logs_the_table() -> None:
    """The log is the always-on destination; an untracked run still shows its table."""
    report_vectors({"iou_per_class": [0.9, 0.1]}, None, [])
    report_matrices({"confusion": [[1, 0], [0, 1]]}, None, [])

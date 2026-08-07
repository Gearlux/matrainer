"""Tests for the Tracker ABC and the TrackerGroup fan-out composite."""

from pathlib import Path
from typing import Any, List, Mapping, Optional

import pytest

from matrainer.tracking.base import Tracker, TrackerGroup, text_file_stem


class RecordingTracker(Tracker):
    """A minimal in-memory Tracker for exercising the ABC surface."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.metrics: List[dict] = []
        self.hparams: dict = {}
        self.artifacts: List[tuple] = []
        self.figures: List[str] = []
        self.texts: List[tuple] = []
        self.finalized: Optional[str] = None

    @property
    def experiment(self) -> str:
        return "handle"

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        self.hparams.update(params)

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        self.metrics.append({"step": step, **dict(metrics)})

    def log_artifact(self, path: Any, artifact_dir: str = "") -> None:
        self.artifacts.append((str(path), artifact_dir))

    def log_figure(self, name: str, figure: Any, step: Optional[int] = None) -> None:
        self.figures.append(name)

    def finalize(self, status: str = "success") -> None:
        self.finalized = status


class TextRecordingTracker(RecordingTracker):
    """Records ``log_text`` instead of serving it.

    Separate from :class:`RecordingTracker` on purpose: the ABC's ``log_text`` default WRITES
    a file, which is the behaviour the tests below exercise directly — a recording override on
    the shared fixture would hide it (and, in the fan-out test, write into ``./runs``).
    """

    def log_text(self, name: str, text: str, step: Optional[int] = None) -> None:
        self.texts.append((name, text, step))


def test_zero_arg_construction_draws_random_run_name() -> None:
    t = RecordingTracker()
    assert t.experiment_name == "matrainer"
    assert t.run_name  # a random slug
    assert t.name == "matrainer"
    assert t.version == t.run_name


def test_explicit_run_name_preserved() -> None:
    t = RecordingTracker(experiment_name="exp", run_name="my_run")
    assert t.run_name == "my_run"
    assert t.version == "my_run"


def test_default_extensions_are_noops_not_errors() -> None:
    # Tracker's default log_artifact/log_figure/log_dir must never raise.
    class Bare(Tracker):
        @property
        def experiment(self) -> str:
            return "x"

        def log_hyperparams(self, params: Mapping[str, Any]) -> None:
            pass

        def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
            pass

    b = Bare()
    b.log_artifact("/nonexistent")  # no raise
    b.log_figure("fig", object())  # no raise
    assert b.log_dir is None
    b.set_tags({"a": "b"})
    assert b.tags["a"] == "b"


# ---------------------------------------------------------------------------
# log_text — the channel for a run value that is neither a number nor an image
# ---------------------------------------------------------------------------
def test_text_file_stem_makes_a_metric_name_a_filename() -> None:
    """A metric name is `val/confusion`-shaped, and `/` is a path separator."""
    assert text_file_stem("val/confusion") == "val_confusion"
    assert text_file_stem("val/confusion", step=2) == "val_confusion_2"
    assert text_file_stem("val/confusion", step=0) == "val_confusion_0"  # step 0 is a step
    assert text_file_stem("/") == "text"  # nothing left to name it by


def test_log_text_defaults_to_a_file_in_the_backends_own_tree(tmp_path: Path) -> None:
    """Unlike `log_figure`, the default is real work: text is trivially serializable.

    A caller therefore never has to ask whether THIS backend can hold a table, which is the
    point of putting the default on the ABC rather than in each backend.
    """

    class WithDir(RecordingTracker):
        @property
        def log_dir(self) -> Optional[Path]:
            return tmp_path

    t = WithDir()
    t.log_text("val/confusion", "|a|b|", step=1)

    written = tmp_path / "text" / "val_confusion_1.md"
    assert written.read_text() == "|a|b|"
    # No `log_artifact`: the file is already where this backend's reader looks, and copying it
    # into an `artifacts/` subdirectory of the same tree would just duplicate it.
    assert t.artifacts == []


def test_a_remote_only_backend_uploads_the_text_file(tmp_path: Path) -> None:
    """No `log_dir` means no tree to write into, so the file goes through `log_artifact`."""
    t = RecordingTracker(save_dir=str(tmp_path))
    assert t.log_dir is None
    t.log_text("summary", "hello")

    assert (tmp_path / "text" / "summary.md").read_text() == "hello"
    assert t.artifacts == [(str(tmp_path / "text" / "summary.md"), "text")]


def test_each_step_is_its_own_file(tmp_path: Path) -> None:
    """A per-epoch value keeps its history even on a backend whose channel has no step axis."""

    class WithDir(RecordingTracker):
        @property
        def log_dir(self) -> Optional[Path]:
            return tmp_path

    t = WithDir()
    for step in range(3):
        t.log_text("val/confusion", f"epoch {step}", step=step)

    assert sorted(p.name for p in (tmp_path / "text").iterdir()) == [
        "val_confusion_0.md",
        "val_confusion_1.md",
        "val_confusion_2.md",
    ]


def test_tracker_group_fans_out_all_calls() -> None:
    a, b = TextRecordingTracker(), TextRecordingTracker()
    group = TrackerGroup([a, b])
    group.log_hyperparams({"lr": 0.1})
    group.log_metrics({"loss": 1.0}, step=3)
    group.log_figure("cm", object())
    group.log_text("val/confusion", "|a|b|", step=3)
    group.log_artifact("/tmp/a.txt", "config")
    group.set_tags({"stage": "smoke"})
    group.save()
    group.finalize("success")
    for t in (a, b):
        assert t.hparams == {"lr": 0.1}
        assert t.metrics == [{"step": 3, "loss": 1.0}]
        assert t.figures == ["cm"]
        assert t.texts == [("val/confusion", "|a|b|", 3)]
        assert t.artifacts == [("/tmp/a.txt", "config")]
        assert t.tags == {"stage": "smoke"}
        assert t.finalized == "success"


def test_tracker_group_swallows_member_failure() -> None:
    class Boom(RecordingTracker):
        def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
            raise RuntimeError("backend down")

    good = RecordingTracker()
    group = TrackerGroup([Boom(), good])
    group.log_metrics({"loss": 2.0})  # must not raise
    assert good.metrics == [{"step": None, "loss": 2.0}]


def test_tracker_group_log_dir_returns_first_available(tmp_path: Path) -> None:
    class WithDir(RecordingTracker):
        @property
        def log_dir(self) -> Optional[Path]:
            return tmp_path

    group = TrackerGroup([RecordingTracker(), WithDir()])
    assert group.log_dir == tmp_path


def test_empty_group_log_dir_is_none() -> None:
    assert TrackerGroup([]).log_dir is None
    assert TrackerGroup().experiment == []


def test_tracker_is_abstract() -> None:
    with pytest.raises(TypeError):
        Tracker()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# flow_trackers — the run-time half of the trackers-slot contract
# ---------------------------------------------------------------------------
class TestFlowTrackers:
    def test_none_becomes_an_empty_list(self) -> None:
        from matrainer.tracking.base import flow_trackers

        assert flow_trackers(None) == []

    def test_a_single_live_tracker_becomes_a_singleton_list(self, tmp_path: object) -> None:
        from matrainer.tracking.base import flow_trackers
        from matrainer.tracking.csv_backend import CsvTracker

        tracker = CsvTracker(save_dir=str(tmp_path))
        assert flow_trackers(tracker) == [tracker]

    def test_a_deferred_marker_is_built(self, tmp_path: object) -> None:
        from confluid import LazyClass

        from matrainer.tracking.base import flow_trackers
        from matrainer.tracking.csv_backend import CsvTracker

        live = flow_trackers([LazyClass(CsvTracker, save_dir=str(tmp_path))])
        assert len(live) == 1 and isinstance(live[0], CsvTracker)

    def test_a_live_list_passes_through_by_identity(self, tmp_path: object) -> None:
        """flow() on a live object is idempotent — the property the write-back idiom rests on."""
        from matrainer.tracking.base import flow_trackers
        from matrainer.tracking.csv_backend import CsvTracker

        trackers = [CsvTracker(save_dir=str(tmp_path))]
        assert flow_trackers(trackers)[0] is trackers[0]

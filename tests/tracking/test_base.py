"""Tests for the Tracker ABC and the TrackerGroup fan-out composite."""

from pathlib import Path
from typing import Any, List, Mapping, Optional

import pytest

from marainer.tracking.base import Tracker, TrackerGroup


class RecordingTracker(Tracker):
    """A minimal in-memory Tracker for exercising the ABC surface."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.metrics: List[dict] = []
        self.hparams: dict = {}
        self.artifacts: List[tuple] = []
        self.figures: List[str] = []
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


def test_zero_arg_construction_draws_random_run_name() -> None:
    t = RecordingTracker()
    assert t.experiment_name == "marainer"
    assert t.run_name  # a random slug
    assert t.name == "marainer"
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


def test_tracker_group_fans_out_all_calls() -> None:
    a, b = RecordingTracker(), RecordingTracker()
    group = TrackerGroup([a, b])
    group.log_hyperparams({"lr": 0.1})
    group.log_metrics({"loss": 1.0}, step=3)
    group.log_figure("cm", object())
    group.log_artifact("/tmp/a.txt", "config")
    group.set_tags({"stage": "smoke"})
    group.save()
    group.finalize("success")
    for t in (a, b):
        assert t.hparams == {"lr": 0.1}
        assert t.metrics == [{"step": 3, "loss": 1.0}]
        assert t.figures == ["cm"]
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

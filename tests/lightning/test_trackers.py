"""Tests for the TrackerLogger bridge (Tracker → pytorch_lightning logger)."""

from typing import Any, List, Mapping, Optional

from marainer.lightning.trackers import TrackerLogger
from marainer.tracking.base import Tracker


class _RecordingTracker(Tracker):
    def __init__(self) -> None:
        super().__init__()
        self.metrics: List[dict] = []
        self.hparams: dict = {}
        self.saved = False
        self.finalized: Optional[str] = None

    @property
    def experiment(self) -> str:
        return "handle"

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        self.hparams.update(params)

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        self.metrics.append({"step": step, **dict(metrics)})

    def save(self) -> None:
        self.saved = True

    def finalize(self, status: str = "success") -> None:
        self.finalized = status


def test_tracker_logger_delegates_to_tracker() -> None:
    t = _RecordingTracker()
    tl = TrackerLogger(t)
    tl.log_metrics({"loss": 0.5}, step=1)
    tl.log_hyperparams({"lr": 0.1})
    tl.save()
    tl.finalize("success")
    assert t.metrics == [{"step": 1, "loss": 0.5}]
    assert t.hparams == {"lr": 0.1}
    assert t.saved is True
    assert t.finalized == "success"


def test_tracker_logger_name_and_version_from_tracker() -> None:
    t = _RecordingTracker()
    tl = TrackerLogger(t)
    assert tl.name == t.name
    assert tl.version == t.version
    assert tl.experiment == "handle"


def test_tracker_logger_hyperparams_accepts_namespace() -> None:
    import argparse

    t = _RecordingTracker()
    TrackerLogger(t).log_hyperparams(argparse.Namespace(lr=0.01, epochs=3))
    assert t.hparams == {"lr": 0.01, "epochs": 3}

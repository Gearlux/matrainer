"""Tests for the TrackerLogger bridge (Tracker → pytorch_lightning logger)."""

from typing import Any, List, Mapping, Optional

from matrainer.lightning.trackers import TrackerLogger
from matrainer.metric_names import MetricNames
from matrainer.tracking.base import Tracker


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
    # `train/loss`, not `loss`: the adapter names what it forwards (see `MetricNames`), so a
    # Lightning run's curves can be read beside a Keras or fastai run's.
    assert t.metrics == [{"step": 1, "train/loss": 0.5}]
    assert t.hparams == {"lr": 0.1}
    assert t.saved is True
    assert t.finalized == "success"


def test_the_adapter_names_what_it_forwards() -> None:
    """A ``LightningModule``'s own keys, whichever of the two shapes it logs, land canonical."""
    t = _RecordingTracker()

    TrackerLogger(t).log_metrics({"val/loss": 0.4, "val_accuracy": 0.9, "epoch": 2.0}, step=7)

    # `val/loss` passes through untouched, `val_accuracy` is read as the val metric it is, and
    # `epoch` is an AXIS — prefixing it would invent a second series for the x-value.
    assert t.metrics == [{"step": 7, "val/loss": 0.4, "val/accuracy": 0.9, "epoch": 2.0}]


def test_a_wired_metric_names_records_in_another_engines_spelling() -> None:
    """The `metric_names:` knob reaching the Lightning path — the same object a config wires."""
    t = _RecordingTracker()

    TrackerLogger(t, metric_names=MetricNames(target="keras")).log_metrics({"val/loss": 0.4}, step=1)

    assert t.metrics == [{"step": 1, "val_loss": 0.4}]


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

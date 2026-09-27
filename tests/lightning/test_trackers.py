"""Tests for the TrackerLogger bridge (Tracker → pytorch_lightning logger)."""

from pathlib import Path
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


def test_the_adapter_reports_the_trackers_save_dir() -> None:
    """A projection of the tracker's own value, never a second source of truth."""
    t = _RecordingTracker()
    assert TrackerLogger(t).save_dir == t.save_dir == "./runs"

    t.save_dir = "./somewhere_else"
    assert TrackerLogger(t).save_dir == "./somewhere_else"


def test_checkpoints_land_under_the_trackers_save_dir_not_the_launch_directory(tmp_path: Any) -> None:
    """The reason `save_dir` exists — pinned against Lightning's REAL resolution.

    ``ModelCheckpoint.__resolve_ckpt_dir`` reads ``loggers[0].save_dir`` and falls back to
    ``Trainer.default_root_dir`` (the CWD) when it is ``None``. The base ``Logger.save_dir``
    IS ``None``, so before this property every checkpoint landed in
    ``<cwd>/<experiment_name>/<run_name>/checkpoints`` — i.e. a run-named directory written
    into the repository whenever a run was launched from a repo root.
    """
    import pytorch_lightning as L
    from pytorch_lightning.callbacks import ModelCheckpoint

    t = _RecordingTracker()
    t.experiment_name, t.run_name, t.save_dir = "Sun_Aug_9_2026_1", "lightning_convnet", str(tmp_path / "runs")

    callback = ModelCheckpoint()
    trainer = L.Trainer(logger=TrackerLogger(t), callbacks=[callback], default_root_dir=str(tmp_path / "cwd"))
    # The name-mangled private method IS the behaviour under test — Lightning exposes no public
    # equivalent, and restating its rule here instead would pass just as happily if both were wrong.
    resolve = getattr(callback, "_ModelCheckpoint__resolve_ckpt_dir")
    resolved = Path(resolve(trainer))

    assert resolved == tmp_path / "runs" / "Sun_Aug_9_2026_1" / "lightning_convnet" / "checkpoints"
    assert (tmp_path / "cwd") not in resolved.parents, "never the launch directory"

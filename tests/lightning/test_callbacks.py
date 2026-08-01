"""Tests for the Lightning callbacks: tracking, checkpoint-policy translation, system metrics."""

from typing import Any, List, Mapping, Optional
from unittest.mock import MagicMock

from pytorch_lightning.callbacks import ModelCheckpoint

from marainer.checkpoints import SaveBest, SaveEveryEpoch, SaveLast
from marainer.lightning.callbacks import (
    TrackingCallback,
    activate_for_lightning,
    build_checkpoint_callbacks,
    has_mlflow_logger,
)
from marainer.tracking.base import Tracker


class _RecordingTracker(Tracker):
    def __init__(self) -> None:
        super().__init__()
        self.hparams: dict = {}
        self.artifacts: List[tuple] = []

    @property
    def experiment(self) -> str:
        return "h"

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        self.hparams.update(params)

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        pass

    def log_artifact(self, path: Any, artifact_dir: str = "") -> None:
        self.artifacts.append((str(path), artifact_dir))


def test_tracking_callback_logs_context_once(monkeypatch: Any) -> None:
    from marainer.tracking import provenance

    monkeypatch.setattr(provenance, "get_git_commit_hash", lambda: "abc")
    monkeypatch.setattr(provenance, "config_script_path", lambda: None)
    monkeypatch.setattr(provenance, "included_config_paths", lambda: [])
    monkeypatch.setattr(provenance, "resolve_log_file", lambda: None)

    t = _RecordingTracker()
    cb = TrackingCallback([t], params={"lr": 0.1})
    cb.on_fit_start(MagicMock(), MagicMock())
    cb.on_test_start(MagicMock(), MagicMock())  # second call must not re-log
    cb.on_fit_end(MagicMock(), MagicMock())
    assert t.hparams["lr"] == 0.1
    assert t.hparams["git.commit"] == "abc"
    assert cb._logged is True


def _quiet_provenance(monkeypatch: Any, tmp_path: Any) -> None:
    """Neutralise the ambient run context so a test sees only what it wired."""
    from marainer.tracking import provenance

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(provenance, "get_git_commit_hash", lambda: None)
    monkeypatch.setattr(provenance, "config_script_path", lambda: None)
    monkeypatch.setattr(provenance, "included_config_paths", lambda: [])
    monkeypatch.setattr(provenance, "resolve_log_file", lambda: None)
    monkeypatch.setattr(provenance, "code_state", list)


def test_tracking_callback_derives_hyperparams_from_a_runnable(monkeypatch: Any, tmp_path: Any) -> None:
    """Hand it the object, not a params dict — one derivation, not one per consumer."""
    from confluid import configurable

    @configurable
    class CallbackToy:
        def __init__(self, lr: float = 0.03, epochs: int = 9) -> None:
            self.lr, self.epochs = lr, epochs

    _quiet_provenance(monkeypatch, tmp_path)
    t = _RecordingTracker()
    cb = TrackingCallback([t], runnable=CallbackToy())
    cb.on_fit_start(MagicMock(), MagicMock())

    leaves = {key.rsplit(".", 1)[-1]: value for key, value in t.hparams.items()}
    assert leaves["lr"] == 0.03 and leaves["epochs"] == 9
    assert cb.manifest is not None


def test_explicit_params_win_over_the_runnable(monkeypatch: Any, tmp_path: Any) -> None:
    from confluid import configurable

    @configurable
    class OverrideCallbackToy:
        def __init__(self, lr: float = 0.03) -> None:
            self.lr = lr

    _quiet_provenance(monkeypatch, tmp_path)
    t = _RecordingTracker()
    cb = TrackingCallback([t], params={"lr": 0.99}, runnable=OverrideCallbackToy())
    cb.on_fit_start(MagicMock(), MagicMock())
    assert t.hparams["lr"] == 0.99


def test_fit_end_finishes_the_manifest(monkeypatch: Any, tmp_path: Any) -> None:
    _quiet_provenance(monkeypatch, tmp_path)
    cb = TrackingCallback([_RecordingTracker()])
    cb.on_fit_start(MagicMock(), MagicMock())
    cb.on_fit_end(MagicMock(), MagicMock())
    assert cb.manifest is not None and cb.manifest.status == "success"


def test_a_fit_then_test_run_keeps_the_first_terminal_status(monkeypatch: Any, tmp_path: Any) -> None:
    """Both ends fire in a fit+test run; the second must not relabel the first."""
    _quiet_provenance(monkeypatch, tmp_path)
    cb = TrackingCallback([_RecordingTracker()])
    cb.on_fit_start(MagicMock(), MagicMock())
    cb.on_fit_end(MagicMock(), MagicMock())
    cb.on_test_end(MagicMock(), MagicMock())
    assert cb.manifest is not None and cb.manifest.status == "success"


def test_an_exception_is_not_recorded_as_a_success(monkeypatch: Any, tmp_path: Any) -> None:
    _quiet_provenance(monkeypatch, tmp_path)
    cb = TrackingCallback([_RecordingTracker()])
    cb.on_fit_start(MagicMock(), MagicMock())
    cb.on_exception(MagicMock(), MagicMock(), RuntimeError("boom"))
    assert cb.manifest is not None and cb.manifest.status == "failed"


def test_a_ctrl_c_is_recorded_as_interrupted(monkeypatch: Any, tmp_path: Any) -> None:
    _quiet_provenance(monkeypatch, tmp_path)
    cb = TrackingCallback([_RecordingTracker()])
    cb.on_fit_start(MagicMock(), MagicMock())
    cb.on_exception(MagicMock(), MagicMock(), KeyboardInterrupt())
    assert cb.manifest is not None and cb.manifest.status == "interrupted"


def test_build_checkpoint_none_returns_empty() -> None:
    assert build_checkpoint_callbacks(None) == []


def test_build_checkpoint_save_last() -> None:
    (cb,) = build_checkpoint_callbacks(SaveLast(dirpath="/ckpts"))
    assert isinstance(cb, ModelCheckpoint)
    assert cb.save_last is True
    assert cb.dirpath == "/ckpts"


def test_build_checkpoint_save_best_maps_monitor() -> None:
    (cb,) = build_checkpoint_callbacks(SaveBest(monitor="val/acc", mode="max", top_k=3))
    assert cb.monitor == "val/acc"
    assert cb.mode == "max"
    assert cb.save_top_k == 3


def test_build_checkpoint_every_epoch() -> None:
    (cb,) = build_checkpoint_callbacks(SaveEveryEpoch(every_n_epochs=2))
    assert cb.every_n_epochs == 2
    assert cb.save_top_k == -1


def test_build_checkpoint_accepts_list() -> None:
    cbs = build_checkpoint_callbacks([SaveLast(), SaveBest()])
    assert len(cbs) == 2


def test_has_mlflow_logger_by_class_name() -> None:
    class MLFlowLogger:
        pass

    class CSVLogger:
        pass

    assert has_mlflow_logger([CSVLogger(), MLFlowLogger()]) is True
    assert has_mlflow_logger([CSVLogger()]) is False


def test_activate_for_lightning_noop_without_mlflow_logger() -> None:
    trainer = MagicMock()
    trainer.loggers = []
    before = len(trainer.callbacks)
    activate_for_lightning(trainer)
    # No MLFlowLogger wired -> no callback appended (append not called).
    trainer.callbacks.append.assert_not_called() if hasattr(trainer.callbacks, "append") else None
    _ = before

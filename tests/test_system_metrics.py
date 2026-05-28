"""Tests for :mod:`marainer.system_metrics`."""

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger

from marainer.system_metrics import (
    MlflowSystemMetricsCallback,
    MPSMetricsCallback,
    activate_for_lightning,
    enable_mlflow_system_metrics,
    has_mlflow_logger,
)

# ---------------------------------------------------------------------------
# has_mlflow_logger
# ---------------------------------------------------------------------------


def test_has_mlflow_logger_true_when_wired() -> None:
    fake = MagicMock()
    type(fake).__name__ = "MLFlowLogger"
    csv = CSVLogger(save_dir=".")
    assert has_mlflow_logger([csv, fake]) is True


def test_has_mlflow_logger_false_for_csv_only(tmp_path: Any) -> None:
    csv = CSVLogger(save_dir=str(tmp_path))
    assert has_mlflow_logger([csv]) is False


def test_has_mlflow_logger_false_for_empty() -> None:
    assert has_mlflow_logger([]) is False


# ---------------------------------------------------------------------------
# enable_mlflow_system_metrics
# ---------------------------------------------------------------------------


def test_enable_mlflow_system_metrics_calls_into_mlflow() -> None:
    with patch("mlflow.system_metrics.enable_system_metrics_logging") as enable_mock:
        assert enable_mlflow_system_metrics() is True
        enable_mock.assert_called_once()


def test_enable_returns_false_when_mlflow_system_metrics_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the import so ``import mlflow.system_metrics`` raises."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        if name == "mlflow.system_metrics" or (name == "mlflow" and "system_metrics" in (fromlist or ())):
            raise ImportError("forced for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setitem(sys.modules, "mlflow.system_metrics", None)  # type: ignore[arg-type]
    monkeypatch.setattr("builtins.__import__", fake_import)
    assert enable_mlflow_system_metrics() is False


def test_enable_honours_sampling_interval_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", "30")
    with (
        patch("mlflow.system_metrics.enable_system_metrics_logging"),
        patch("mlflow.set_system_metrics_sampling_interval") as set_interval,
    ):
        enable_mlflow_system_metrics()
        set_interval.assert_called_once_with(30.0)


# ---------------------------------------------------------------------------
# MPSMetricsCallback
# ---------------------------------------------------------------------------


def test_mps_callback_default_sample_interval_is_10s(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default cadence is 10 s when no env override is set."""
    monkeypatch.delenv("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", raising=False)
    cb = MPSMetricsCallback()
    assert cb.sample_interval_s == pytest.approx(10.0)


def test_mps_callback_honours_env_var_sample_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operators can override the default workspace-wide via the env var."""
    monkeypatch.setenv("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", "30")
    cb = MPSMetricsCallback()
    assert cb.sample_interval_s == pytest.approx(30.0)


def test_mps_callback_explicit_kwarg_overrides_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ``sample_interval_s=...`` wins over the env var."""
    monkeypatch.setenv("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", "30")
    cb = MPSMetricsCallback(sample_interval_s=5.0)
    assert cb.sample_interval_s == pytest.approx(5.0)


def test_mps_callback_falls_back_to_default_on_bogus_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", "not-a-number")
    cb = MPSMetricsCallback()
    assert cb.sample_interval_s == pytest.approx(10.0)


# --- _sample_metrics (snapshot collection) ---------------------------------


def test_sample_metrics_empty_when_mps_unavailable() -> None:
    with patch("torch.backends.mps.is_available", return_value=False):
        assert MPSMetricsCallback._sample_metrics() == {}


def test_sample_metrics_only_torch_when_iokit_returns_none() -> None:
    """When the IOAccelerator service isn't available (non-Darwin, or pyobjc
    bundle load failed) the reader returns None — only the torch.mps triplet
    lands."""
    with (
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.mps.current_allocated_memory", return_value=2 * 1024**2),
        patch("torch.mps.driver_allocated_memory", return_value=4 * 1024**2),
        patch("torch.mps.recommended_max_memory", return_value=10 * 1024**2),
        patch("marainer.system_metrics._read_mps_performance_statistics", return_value=None),
    ):
        metrics = MPSMetricsCallback._sample_metrics()
    assert metrics == {
        "system/mps_allocated_mb": pytest.approx(2.0),
        "system/mps_driver_mb": pytest.approx(4.0),
        "system/mps_recommended_max_mb": pytest.approx(10.0),
    }


def test_sample_metrics_includes_utilization_from_iokit_read() -> None:
    """When the IOAccelerator read returns a PerformanceStatistics dict, the
    six curated keys land in MLflow alongside the torch.mps triplet."""
    fake_stats = {
        "Device Utilization %": 75,
        "Renderer Utilization %": 60,
        "Tiler Utilization %": 12,
        "In use system memory": 2 * 1024**2,
        "In use system memory (driver)": 1 * 1024**2,
        "Alloc system memory": 8 * 1024**2,
        "recoveryCount": 0,  # Outside the curated allow-list — ignored.
    }
    with (
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.mps.current_allocated_memory", return_value=1024**2),
        patch("torch.mps.driver_allocated_memory", return_value=2 * 1024**2),
        patch("torch.mps.recommended_max_memory", return_value=3 * 1024**2),
        patch("marainer.system_metrics._read_mps_performance_statistics", return_value=fake_stats),
    ):
        metrics = MPSMetricsCallback._sample_metrics()
    assert metrics == {
        "system/mps_allocated_mb": pytest.approx(1.0),
        "system/mps_driver_mb": pytest.approx(2.0),
        "system/mps_recommended_max_mb": pytest.approx(3.0),
        "system/mps_device_util_pct": pytest.approx(75.0),
        "system/mps_renderer_util_pct": pytest.approx(60.0),
        "system/mps_tiler_util_pct": pytest.approx(12.0),
        "system/mps_in_use_system_memory_mb": pytest.approx(2.0),
        "system/mps_in_use_system_memory_driver_mb": pytest.approx(1.0),
        "system/mps_alloc_system_memory_mb": pytest.approx(8.0),
    }


def test_sample_metrics_skips_missing_perf_stat_keys() -> None:
    """Apple has renamed PerformanceStatistics keys across macOS versions —
    a partial dict yields only the present keys, no KeyError."""
    fake_stats = {"Device Utilization %": 42}
    with (
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.mps.current_allocated_memory", return_value=1024**2),
        patch("torch.mps.driver_allocated_memory", return_value=2 * 1024**2),
        patch("torch.mps.recommended_max_memory", return_value=3 * 1024**2),
        patch("marainer.system_metrics._read_mps_performance_statistics", return_value=fake_stats),
    ):
        metrics = MPSMetricsCallback._sample_metrics()
    assert "system/mps_device_util_pct" in metrics
    assert "system/mps_renderer_util_pct" not in metrics
    assert "system/mps_tiler_util_pct" not in metrics


def test_sample_metrics_demotes_iokit_exception_to_debug() -> None:
    """An exception inside the IOKit read must not propagate — the torch.mps
    triplet still emits so the sampler keeps reporting."""
    with (
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.mps.current_allocated_memory", return_value=1024**2),
        patch("torch.mps.driver_allocated_memory", return_value=2 * 1024**2),
        patch("torch.mps.recommended_max_memory", return_value=3 * 1024**2),
        patch(
            "marainer.system_metrics._read_mps_performance_statistics",
            side_effect=RuntimeError("IOKit hiccup"),
        ),
    ):
        metrics = MPSMetricsCallback._sample_metrics()  # must not raise
    assert {"system/mps_allocated_mb", "system/mps_driver_mb", "system/mps_recommended_max_mb"} <= set(metrics)
    assert not any(k.startswith("system/mps_device_util") for k in metrics)


# --- lifecycle (start / stop) ----------------------------------------------


def test_mps_callback_start_skips_when_mps_unavailable() -> None:
    cb = MPSMetricsCallback(sample_interval_s=0.01)
    fake_trainer = MagicMock()
    fake_trainer.loggers = []  # no MLflow logger anyway
    with patch("torch.backends.mps.is_available", return_value=False):
        cb.on_fit_start(fake_trainer, MagicMock())
    assert cb._thread is None


def test_mps_callback_start_skips_without_mlflow_logger() -> None:
    cb = MPSMetricsCallback(sample_interval_s=0.01)
    fake_trainer = MagicMock()
    fake_trainer.loggers = []
    with patch("torch.backends.mps.is_available", return_value=True):
        cb.on_fit_start(fake_trainer, MagicMock())
    assert cb._thread is None


def test_mps_callback_starts_thread_and_pushes_to_mlflow_client() -> None:
    """End-to-end: on_fit_start spawns the daemon thread, which samples
    once and pushes every metric to ``MlflowClient.log_metric``. We use
    a tiny interval so the test runs fast, then on_fit_end joins."""
    import time as _time

    cb = MPSMetricsCallback(sample_interval_s=0.01)
    fake_logger = MagicMock()
    type(fake_logger).__name__ = "MLFlowLogger"
    fake_logger.run_id = "run-77"
    fake_logger._tracking_uri = "http://localhost:5005"
    fake_trainer = MagicMock()
    fake_trainer.loggers = [fake_logger]

    fake_client_instance = MagicMock()
    with (
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.mps.current_allocated_memory", return_value=1024**2),
        patch("torch.mps.driver_allocated_memory", return_value=2 * 1024**2),
        patch("torch.mps.recommended_max_memory", return_value=3 * 1024**2),
        patch(
            "marainer.system_metrics._read_mps_performance_statistics",
            return_value={"Device Utilization %": 50},
        ),
        patch("mlflow.tracking.MlflowClient", return_value=fake_client_instance),
    ):
        cb.on_fit_start(fake_trainer, MagicMock())
        # Give the daemon at least one sampling tick.
        _time.sleep(0.1)
        cb.on_fit_end(fake_trainer, MagicMock())
    # At least one push happened, every call carries the right run_id.
    assert fake_client_instance.log_metric.call_count >= 1
    for call in fake_client_instance.log_metric.call_args_list:
        assert call.kwargs["run_id"] == "run-77"
    pushed_keys = {call.kwargs["key"] for call in fake_client_instance.log_metric.call_args_list}
    assert {"system/mps_allocated_mb", "system/mps_device_util_pct"} <= pushed_keys
    # Thread joined cleanly.
    assert cb._thread is None


def test_mps_callback_stop_is_idempotent_when_never_started() -> None:
    cb = MPSMetricsCallback(sample_interval_s=0.01)
    cb.on_fit_end(MagicMock(), MagicMock())  # must not raise


def test_mps_callback_thread_is_not_daemon() -> None:
    """Non-daemon ensures Python's atexit waits for the sampler to exit before
    tearing down torch.mps / pyobjc / mlflow C state — daemon=True races them
    and segfaults on process shutdown."""
    import time as _time

    cb = MPSMetricsCallback(sample_interval_s=0.01)
    fake_logger = MagicMock()
    type(fake_logger).__name__ = "MLFlowLogger"
    fake_logger.run_id = "run-99"
    fake_logger._tracking_uri = "http://localhost:5005"
    fake_trainer = MagicMock()
    fake_trainer.loggers = [fake_logger]

    with (
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.mps.current_allocated_memory", return_value=1024**2),
        patch("torch.mps.driver_allocated_memory", return_value=2 * 1024**2),
        patch("torch.mps.recommended_max_memory", return_value=3 * 1024**2),
        patch("marainer.system_metrics._read_mps_performance_statistics", return_value=None),
        patch("mlflow.tracking.MlflowClient", return_value=MagicMock()),
    ):
        cb.on_fit_start(fake_trainer, MagicMock())
        thread = cb._thread
        assert thread is not None
        assert thread.daemon is False, "sampler thread must be non-daemon to survive interpreter teardown safely"
        _time.sleep(0.05)
        cb.on_fit_end(fake_trainer, MagicMock())
    assert cb._thread is None


def test_mps_callback_registers_atexit_hook_for_safe_shutdown() -> None:
    """An atexit hook fires `_stop` defensively if `on_fit_end` is bypassed
    (KeyboardInterrupt mid-fit, exception in user code, etc.). The hook uses
    a weakref so it can't keep the callback alive after Lightning drops it."""
    import atexit as _atexit

    registered_before = len(_atexit._exithandlers) if hasattr(_atexit, "_exithandlers") else None
    cb = MPSMetricsCallback(sample_interval_s=0.01)
    fake_logger = MagicMock()
    type(fake_logger).__name__ = "MLFlowLogger"
    fake_logger.run_id = "run-x"
    fake_logger._tracking_uri = None
    fake_trainer = MagicMock()
    fake_trainer.loggers = [fake_logger]

    with (
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.mps.current_allocated_memory", return_value=1024**2),
        patch("torch.mps.driver_allocated_memory", return_value=2 * 1024**2),
        patch("torch.mps.recommended_max_memory", return_value=3 * 1024**2),
        patch("marainer.system_metrics._read_mps_performance_statistics", return_value=None),
        patch("mlflow.tracking.MlflowClient", return_value=MagicMock()),
    ):
        cb.on_fit_start(fake_trainer, MagicMock())
        assert cb._atexit_registered is True
        # Second start (e.g. test phase after fit) doesn't double-register.
        cb.on_fit_end(fake_trainer, MagicMock())
        cb.on_fit_start(fake_trainer, MagicMock())
        cb.on_fit_end(fake_trainer, MagicMock())
        assert cb._atexit_registered is True

    # Touch the bookkeeping if the implementation exposes it; mainly we
    # want to assert the flag stays set after a stop/restart cycle.
    _ = registered_before  # silence unused-var when the attr is missing


# ---------------------------------------------------------------------------
# activate_for_lightning
# ---------------------------------------------------------------------------


def _make_lightning_trainer(mlflow_wired: bool) -> L.Trainer:
    """Lightning Trainer fixture with an optional MLflow-shaped logger."""
    if mlflow_wired:
        fake = MagicMock()
        type(fake).__name__ = "MLFlowLogger"
        loggers: list[Any] = [fake]
    else:
        loggers = []
    return L.Trainer(
        logger=loggers,
        max_epochs=1,
        accelerator="cpu",
        enable_checkpointing=False,
    )


def test_activate_for_lightning_noop_without_mlflow_logger(tmp_path: Any) -> None:
    trainer = L.Trainer(
        logger=[CSVLogger(save_dir=str(tmp_path))],
        max_epochs=1,
        accelerator="cpu",
        enable_checkpointing=False,
    )
    n_callbacks = len(trainer.callbacks)  # type: ignore[attr-defined]
    activate_for_lightning(trainer)
    assert len(trainer.callbacks) == n_callbacks  # type: ignore[attr-defined]


def test_activate_for_lightning_appends_both_callbacks_on_apple() -> None:
    """On Apple-Silicon with an MLflow logger, two callbacks land on the trainer:
    the Lightning-side SystemMetricsMonitor driver AND the MPS memory callback."""
    trainer = _make_lightning_trainer(mlflow_wired=True)
    n_callbacks = len(trainer.callbacks)  # type: ignore[attr-defined]
    with patch("torch.backends.mps.is_available", return_value=True):
        activate_for_lightning(trainer)
    assert len(trainer.callbacks) == n_callbacks + 2  # type: ignore[attr-defined]
    appended = trainer.callbacks[-2:]  # type: ignore[attr-defined]
    assert isinstance(appended[0], MlflowSystemMetricsCallback)
    assert isinstance(appended[1], MPSMetricsCallback)


def test_activate_for_lightning_skips_mps_callback_off_apple() -> None:
    trainer = _make_lightning_trainer(mlflow_wired=True)
    n_callbacks = len(trainer.callbacks)  # type: ignore[attr-defined]
    with patch("torch.backends.mps.is_available", return_value=False):
        activate_for_lightning(trainer)
    # Only the MLflow callback gets appended, not the MPS one.
    assert len(trainer.callbacks) == n_callbacks + 1  # type: ignore[attr-defined]
    assert isinstance(trainer.callbacks[-1], MlflowSystemMetricsCallback)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# MlflowSystemMetricsCallback
# ---------------------------------------------------------------------------


def test_mlflow_system_metrics_callback_starts_monitor_on_fit_start() -> None:
    """``on_fit_start`` reads the MLflow logger's ``run_id`` and starts a
    SystemMetricsMonitor against it. Lightning's ``MLFlowLogger`` bypasses
    ``mlflow.start_run`` so the global enable flag alone never attaches one
    — this callback is the workaround."""
    cb = MlflowSystemMetricsCallback()
    fake_logger = MagicMock()
    type(fake_logger).__name__ = "MLFlowLogger"
    fake_logger.run_id = "run-42"
    fake_logger._tracking_uri = "http://localhost:5005"

    fake_trainer = MagicMock()
    fake_trainer.loggers = [fake_logger]
    fake_module = MagicMock()

    monitor_instance = MagicMock()
    with patch(
        "mlflow.system_metrics.system_metrics_monitor.SystemMetricsMonitor",
        return_value=monitor_instance,
    ) as monitor_cls:
        cb.on_fit_start(fake_trainer, fake_module)
    monitor_cls.assert_called_once()
    kwargs = monitor_cls.call_args.kwargs
    assert kwargs["run_id"] == "run-42"
    assert kwargs["tracking_uri"] == "http://localhost:5005"
    monitor_instance.start.assert_called_once()


def test_mlflow_system_metrics_callback_finish_idempotent() -> None:
    """``on_fit_end`` is safe to call when the monitor never started
    (e.g. no MLflow logger was wired)."""
    cb = MlflowSystemMetricsCallback()
    cb.on_fit_end(MagicMock(), MagicMock())  # should not raise


def test_mlflow_system_metrics_callback_noop_when_no_mlflow_logger(tmp_path: Any) -> None:
    cb = MlflowSystemMetricsCallback()
    fake_trainer = MagicMock()
    fake_trainer.loggers = [CSVLogger(save_dir=str(tmp_path))]
    with patch("mlflow.system_metrics.system_metrics_monitor.SystemMetricsMonitor") as monitor_cls:
        cb.on_fit_start(fake_trainer, MagicMock())
    monitor_cls.assert_not_called()


def test_mlflow_system_metrics_callback_finishes_monitor_on_fit_end() -> None:
    cb = MlflowSystemMetricsCallback()
    fake_logger = MagicMock()
    type(fake_logger).__name__ = "MLFlowLogger"
    fake_logger.run_id = "run-42"
    fake_logger._tracking_uri = None
    fake_trainer = MagicMock()
    fake_trainer.loggers = [fake_logger]

    monitor_instance = MagicMock()
    with patch(
        "mlflow.system_metrics.system_metrics_monitor.SystemMetricsMonitor",
        return_value=monitor_instance,
    ):
        cb.on_fit_start(fake_trainer, MagicMock())
    cb.on_fit_end(fake_trainer, MagicMock())
    monitor_instance.finish.assert_called_once()

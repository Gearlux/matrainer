"""Tests for the framework-free system-metrics helpers."""

import subprocess
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from matrainer.tracking import system


def test_resolve_interval_explicit_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", "30")
    assert system._resolve_sample_interval(5.0) == 5.0


def test_resolve_interval_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", "30")
    assert system._resolve_sample_interval(None) == 30.0


def test_resolve_interval_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", raising=False)
    assert system._resolve_sample_interval(None) == system.DEFAULT_MPS_SAMPLE_INTERVAL_S


def test_resolve_interval_bogus_env_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", "not-a-number")
    assert system._resolve_sample_interval(None) == system.DEFAULT_MPS_SAMPLE_INTERVAL_S


def test_enable_returns_false_when_mlflow_system_metrics_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "mlflow.system_metrics":
            raise ImportError("no system metrics")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    assert system.enable_mlflow_system_metrics() is False


def test_start_monitor_none_run_id_returns_none() -> None:
    assert system.start_system_metrics_monitor(None) is None


def test_mps_sampler_start_noop_without_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(system, "_mps_available", lambda: False)
    sampler = system.MpsMetricsSampler(MagicMock(), "run-1")
    sampler.start()  # no-op
    assert sampler._thread is None
    sampler.stop()  # idempotent no-op


def test_mps_sampler_metrics_empty_without_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate torch not importable inside the static sampler.
    monkeypatch.setitem(sys.modules, "torch", None)
    assert system.MpsMetricsSampler._record_metrics() == {}


def test_mps_sampler_default_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", raising=False)
    sampler = system.MpsMetricsSampler(MagicMock(), "r")
    assert sampler.sample_interval_s == system.DEFAULT_MPS_SAMPLE_INTERVAL_S


def test_a_process_that_raises_with_a_running_sampler_still_exits(tmp_path: Any) -> None:
    """A run that fails never reaches ``stop()``. Python joins every non-daemon thread BEFORE it runs exit hooks,
    so a sampler stopped only by its exit hook kept the process alive for ever. The process must end, and the
    exit hook must have stopped the sampler before the interpreter is torn down."""
    script = tmp_path / "failing_run.py"
    script.write_text(
        "import atexit\n"
        "from unittest.mock import MagicMock\n"
        "from matrainer.tracking import system\n"
        "system._mps_available = lambda: True\n"
        "system.MpsMetricsSampler._record_metrics = staticmethod(lambda: {})\n"
        "sampler = system.MpsMetricsSampler(MagicMock(), 'run', sample_interval_s=60)\n"
        "atexit.register(lambda: print('sampler alive at exit:', sampler._thread is not None))\n"
        "sampler.start()\n"
        "assert sampler._thread.is_alive()\n"
        "raise RuntimeError('training failed')\n"
    )
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=30)
    assert "RuntimeError: training failed" in done.stderr
    assert "sampler alive at exit: False" in done.stdout

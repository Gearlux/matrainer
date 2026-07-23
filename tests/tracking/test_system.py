"""Tests for the framework-free system-metrics helpers."""

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from marainer.tracking import system


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
    assert system.MpsMetricsSampler._sample_metrics() == {}


def test_mps_sampler_default_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", raising=False)
    sampler = system.MpsMetricsSampler(MagicMock(), "r")
    assert sampler.sample_interval_s == system.DEFAULT_MPS_SAMPLE_INTERVAL_S

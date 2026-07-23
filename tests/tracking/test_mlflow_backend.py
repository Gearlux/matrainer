"""Tests for MlflowTracker with mlflow mocked via sys.modules injection."""

import sys
import types
from typing import Any, List

import pytest

from marainer.tracking.mlflow_backend import MlflowTracker


class _FakeRun:
    def __init__(self, run_id: str) -> None:
        self.info = types.SimpleNamespace(run_id=run_id)


class _FakeClient:
    """Records calls; stands in for mlflow.tracking.MlflowClient."""

    instances: List["_FakeClient"] = []

    def __init__(self, tracking_uri: Any = None) -> None:
        self.tracking_uri = tracking_uri
        self.params: dict = {}
        self.metrics: List[tuple] = []
        self.tags: dict = {}
        self.artifacts: List[tuple] = []
        self.terminated: Any = None
        _FakeClient.instances.append(self)

    def get_experiment_by_name(self, name: str) -> Any:
        return None

    def create_experiment(self, name: str) -> str:
        return "exp-1"

    def create_run(self, experiment_id: str, run_name: str, tags: Any = None) -> _FakeRun:
        return _FakeRun("run-42")

    def log_param(self, run_id: str, key: str, value: Any) -> None:
        self.params[key] = value

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0, **kw: Any) -> None:
        self.metrics.append((key, value, step))

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        self.tags[key] = value

    def log_artifact(self, run_id: str, path: str, artifact_dir: Any = None) -> None:
        self.artifacts.append((path, artifact_dir))

    def set_terminated(self, run_id: str, status: str) -> None:
        self.terminated = status


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> Any:
    _FakeClient.instances = []
    mlflow_mod = types.ModuleType("mlflow")
    tracking_mod = types.ModuleType("mlflow.tracking")
    tracking_mod.MlflowClient = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_mod)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking_mod)
    return _FakeClient


def test_import_error_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """When mlflow is unimportable, experiment access raises with the pip hint."""

    def fake_require(module: str, extra: str, feature: str = "") -> Any:
        raise ImportError(f"{feature} requires the '{module}' package. Install with: pip install marainer[{extra}]")

    monkeypatch.setattr("marainer.tracking.mlflow_backend.require", fake_require)
    t = MlflowTracker(log_system_metrics=False)
    with pytest.raises(ImportError, match=r"pip install marainer\[mlflow\]"):
        _ = t.experiment


def test_experiment_creates_run_lazily(fake_mlflow: Any) -> None:
    t = MlflowTracker(experiment_name="exp", run_name="r", log_system_metrics=False)
    assert not fake_mlflow.instances  # nothing built at construction
    _ = t.experiment
    assert t.run_id == "run-42"
    assert len(fake_mlflow.instances) == 1


def test_log_params_metrics_tags(fake_mlflow: Any) -> None:
    t = MlflowTracker(run_name="r", log_system_metrics=False)
    t.log_hyperparams({"lr": 0.1})
    t.log_metrics({"loss": 0.5}, step=2)
    t.set_tags({"git.commit": "abc"})
    client = fake_mlflow.instances[0]
    assert client.params == {"lr": 0.1}
    assert client.metrics == [("loss", 0.5, 2)]
    assert client.tags == {"git.commit": "abc"}


def test_log_artifact_forwards(fake_mlflow: Any, tmp_path: Any) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("x")
    t = MlflowTracker(run_name="r", log_system_metrics=False)
    t.log_artifact(f, "config")
    assert fake_mlflow.instances[0].artifacts == [(str(f), "config")]


def test_finalize_terminates_run(fake_mlflow: Any) -> None:
    t = MlflowTracker(run_name="r", log_system_metrics=False)
    _ = t.experiment
    t.finalize("success")
    assert fake_mlflow.instances[0].terminated == "FINISHED"
    t2 = MlflowTracker(run_name="r2", log_system_metrics=False)
    _ = t2.experiment
    t2.finalize("failed")
    assert fake_mlflow.instances[1].terminated == "FAILED"

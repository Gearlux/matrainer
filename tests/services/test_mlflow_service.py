"""The MLflow tracking server as a workspace service (`matrainer/services/mlflow_service.py`).

The server keeps its runs in a SQLite store and its artifacts in a folder, both inside the service's own data folder,
and answers `/health` once it is up.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from matrainer.services import mlflow_service
from matrainer.services.mlflow_service import DEFAULT_PORT, MLflow
from tests.services.context import Context, Failed


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mlflow_service, "_installed", lambda: True)


def test_start_serves_the_services_own_store_and_artifacts_on_mlflows_port(tmp_path: Path, installed: None) -> None:
    data = tmp_path / "mlflow"
    ctx = Context(data)
    MLflow().start(ctx)
    [run] = ctx.runs
    assert run["command"] == [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--host",
        "localhost",
        "--port",
        "5005",
        "--backend-store-uri",
        f"sqlite:///{data / 'mlflow.db'}",
        "--default-artifact-root",
        str(data / "artifacts"),
    ]
    assert (run["port"], run["ready"]) == (5005, "/health")
    assert (data / "artifacts").is_dir()  # made before the server writes into it
    assert ctx.lines == [f"MLflow: http://localhost:{DEFAULT_PORT}"]
    assert DEFAULT_PORT == 5005


def test_port_moves_the_server(tmp_path: Path, installed: None) -> None:
    ctx = Context(tmp_path / "mlflow", port=5006)
    MLflow().start(ctx)
    assert ctx.runs[0]["command"][7] == "5006"
    assert ctx.runs[0]["port"] == 5006


def test_without_mlflow_start_names_the_command_that_adds_it_and_runs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mlflow_service, "_installed", lambda: False)
    ctx = Context(tmp_path / "mlflow", chosen=["tensorboard"])
    with pytest.raises(Failed) as failed:
        MLflow().start(ctx)
    assert str(failed.value) == (
        "MLflow is not installed in this environment.\n"
        "  Add it to the tracker selection, so the choice is recorded and survives the next setup:\n"
        "      aisland workspace configure tracker set tensorboard mlflow\n"
        "  Or, for this machine only:\n"
        "      aisland workspace configure tracker set tensorboard mlflow --machine-only"
    )
    assert ctx.runs == [] and not (tmp_path / "mlflow").exists()


def test_info_names_the_url_the_store_and_the_artifacts(tmp_path: Path) -> None:
    data = tmp_path / "mlflow"
    ctx = Context(data, ports=[5006])
    MLflow().info(ctx)
    assert ctx.lines == [
        "  URL           : http://localhost:5006",
        f"  Backend store : sqlite:///{data / 'mlflow.db'}",
        f"  Artifacts dir : {data / 'artifacts'}",
    ]
    stopped = Context(data)  # no port known while the server is not running: the default is named
    MLflow().info(stopped)
    assert stopped.lines[0] == "  URL           : http://localhost:5005"


def test_it_is_a_service_with_no_options_of_its_own() -> None:
    assert MLflow.kind == "service"
    assert not getattr(MLflow, "options", {})


def test_the_module_imports_nothing_of_the_service_manager() -> None:
    code = "import sys; sys.modules['aisland'] = None\nfrom matrainer.services.mlflow_service import MLflow\n"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path(__file__).parent)
    assert result.returncode == 0, result.stderr


def test_it_is_declared_for_the_service_manager() -> None:
    from importlib.metadata import entry_points

    declared = {ep.name: ep.value for ep in entry_points(group="aisland.plugins")}
    assert declared["mlflow"] == "matrainer.services.mlflow_service:MLflow"

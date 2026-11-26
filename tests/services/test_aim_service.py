"""The Aim UI as a workspace service (`matrainer/services/aim_service.py`).

Aim needs no server: runs are written into a repository folder and `aim up` serves the UI from it. The one thing the
service must get right is that the repository it serves is the one the tracker writes to, so its path is published
to the workspace `.env` as `AIM_REPO`, which a run config reads as `$AIM_REPO`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from matrainer.services import aim_service
from matrainer.services.aim_service import DEFAULT_PORT, REPO_ENV_KEY, Aim
from tests.services.context import Context, Failed


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(aim_service, "_installed", lambda: True)


def test_start_serves_the_services_own_repository_on_aims_port(tmp_path: Path, installed: None) -> None:
    ctx = Context(tmp_path / "aim")
    Aim().start(ctx)
    [run] = ctx.runs
    assert run["command"][1:] == [
        "up",
        "--host",
        "localhost",
        "--port",
        "43800",
        "--repo",
        str(tmp_path / "aim"),
        # Aim asks before it initialises a repository it has not seen; a supervised process has no one to answer.
        "--yes",
    ]
    assert (run["port"], run["ready"]) == (43800, "/")
    assert (tmp_path / "aim").is_dir()  # made before Aim opens it
    assert DEFAULT_PORT == 43800


def test_the_repository_served_is_published_for_run_configs(tmp_path: Path, installed: None) -> None:
    ctx = Context(tmp_path / "aim", env={REPO_ENV_KEY: "/old/repo", "DATA_ROOT": "/data"})
    Aim().start(ctx)
    assert ctx.env == {REPO_ENV_KEY: str(tmp_path / "aim"), "DATA_ROOT": "/data"}


def test_repo_serves_another_repository_and_publishes_it(tmp_path: Path, installed: None) -> None:
    elsewhere = tmp_path / "other"
    ctx = Context(tmp_path / "aim", options={"repo": str(elsewhere)}, port=5000)
    Aim().start(ctx)
    assert str(elsewhere.resolve()) in ctx.runs[0]["command"]
    assert ctx.env[REPO_ENV_KEY] == str(elsewhere.resolve())
    assert ctx.runs[0]["port"] == 5000


def test_without_aim_start_names_the_command_that_adds_it_and_runs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aim_service, "_installed", lambda: False)
    ctx = Context(tmp_path / "aim", chosen=["tensorboard"])
    with pytest.raises(Failed) as failed:
        Aim().start(ctx)
    assert str(failed.value) == (
        "Aim is not installed in this environment.\n"
        "  Add it to the tracker selection, so the choice is recorded and survives the next setup:\n"
        "      aisland workspace configure tracker set tensorboard aim\n"
        "  Or, for this machine only:\n"
        "      aisland workspace configure tracker set tensorboard aim --machine-only"
    )
    assert ctx.runs == [] and ctx.env == {}


def test_the_repository_option_is_offered() -> None:
    assert Aim.kind == "service"
    assert set(Aim.options) == {"repo"}


def test_info_names_the_address_and_the_repository(tmp_path: Path) -> None:
    ctx = Context(tmp_path / "aim", ports=[5000])
    Aim().info(ctx)
    assert ctx.lines == [
        "  URL           : http://localhost:5000",
        f"  Repository    : {tmp_path / 'aim'}",
        f"  {REPO_ENV_KEY:<14}: published to the workspace .env for run configs",
    ]


def test_the_binary_is_the_venvs_first() -> None:
    """An unactivated shell must not serve another Aim, or none."""
    binary = aim_service._aim_binary()
    assert binary.startswith(sys.prefix) or Path(binary).name == "aim"


def test_the_module_imports_nothing_of_the_service_manager() -> None:
    code = (
        "import sys; sys.modules['aisland'] = None\nfrom matrainer.services.aim_service import Aim\nprint(Aim.kind)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=Path(__file__).parent)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "service"

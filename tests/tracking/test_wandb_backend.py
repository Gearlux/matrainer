"""Tests for WandbTracker with wandb mocked via sys.modules injection."""

import sys
import types
from typing import Any, List

import pytest

from marainer.tracking.wandb_backend import WandbTracker


class _FakeConfig:
    def __init__(self) -> None:
        self.data: dict = {}

    def update(self, d: dict, allow_val_change: bool = False) -> None:
        self.data.update(d)


class _FakeRun:
    def __init__(self) -> None:
        self.id = "wb-1"
        self.dir = "/tmp/wb"
        self.config = _FakeConfig()
        self.logged: List[tuple] = []
        self.saved: List[str] = []
        self.finished: Any = None

    def log(self, data: dict, step: Any = None) -> None:
        self.logged.append((data, step))

    def save(self, path: str, base_path: str = "") -> None:
        self.saved.append(path)

    def finish(self, exit_code: int = 0) -> None:
        self.finished = exit_code


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> Any:
    mod = types.ModuleType("wandb")
    runs: List[_FakeRun] = []

    def init(**kwargs: Any) -> _FakeRun:
        r = _FakeRun()
        runs.append(r)
        return r

    mod.init = init  # type: ignore[attr-defined]
    mod.runs = runs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", mod)
    return mod


def test_import_error_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_require(module: str, extra: str, feature: str = "") -> Any:
        raise ImportError(f"pip install marainer[{extra}]")

    monkeypatch.setattr("marainer.tracking.wandb_backend.require", fake_require)
    t = WandbTracker()
    with pytest.raises(ImportError, match=r"pip install marainer\[wandb\]"):
        _ = t.experiment


def test_run_created_lazily(fake_wandb: Any) -> None:
    t = WandbTracker(experiment_name="proj", run_name="r")
    assert not fake_wandb.runs
    _ = t.experiment
    assert len(fake_wandb.runs) == 1
    assert t.version == "wb-1"


def test_log_metrics_and_hparams(fake_wandb: Any) -> None:
    t = WandbTracker(run_name="r")
    t.log_hyperparams({"lr": 0.1})
    t.log_metrics({"loss": 0.4}, step=2)
    run = fake_wandb.runs[0]
    assert run.config.data == {"lr": 0.1}
    assert run.logged == [({"loss": 0.4}, 2)]


def test_finalize_finishes_run(fake_wandb: Any) -> None:
    t = WandbTracker(run_name="r")
    _ = t.experiment
    t.finalize("failed")
    assert fake_wandb.runs[0].finished == 1

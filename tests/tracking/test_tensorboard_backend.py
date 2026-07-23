"""Tests for TensorBoardTracker with the SummaryWriter mocked via sys.modules."""

import sys
import types
from pathlib import Path
from typing import Any, List

import pytest

from marainer.tracking.tensorboard_backend import TensorBoardTracker


class _FakeWriter:
    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self.scalars: List[tuple] = []
        self.figures: List[str] = []
        self.texts: List[tuple] = []
        self.closed = False

    def add_scalar(self, tag: str, value: float, global_step: Any = None) -> None:
        self.scalars.append((tag, value, global_step))

    def add_figure(self, tag: str, figure: Any, global_step: Any = None) -> None:
        self.figures.append(tag)

    def add_text(self, tag: str, text: str) -> None:
        self.texts.append((tag, text))

    def add_hparams(self, hparams: dict, metrics: dict) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_tb(monkeypatch: pytest.MonkeyPatch) -> Any:
    mod = types.ModuleType("torch.utils.tensorboard")
    mod.SummaryWriter = _FakeWriter  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", mod)
    return mod


def test_import_error_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_require(module: str, extra: str, feature: str = "") -> Any:
        raise ImportError(f"pip install marainer[{extra}]")

    monkeypatch.setattr("marainer.tracking.tensorboard_backend.require", fake_require)
    t = TensorBoardTracker()
    with pytest.raises(ImportError, match=r"pip install marainer\[tensorboard\]"):
        _ = t.experiment


def test_writer_created_lazily_under_run_dir(fake_tb: Any, tmp_path: Path) -> None:
    t = TensorBoardTracker(experiment_name="e", run_name="r", save_dir=str(tmp_path))
    writer = t.experiment
    assert Path(writer.log_dir) == tmp_path / "e" / "r"
    assert (tmp_path / "e" / "r").is_dir()


def test_log_metrics_forwards_scalars(fake_tb: Any, tmp_path: Path) -> None:
    t = TensorBoardTracker(experiment_name="e", run_name="r", save_dir=str(tmp_path))
    t.log_metrics({"loss": 0.3}, step=5)
    assert t.experiment.scalars == [("loss", 0.3, 5)]


def test_log_figure_and_finalize(fake_tb: Any, tmp_path: Path) -> None:
    t = TensorBoardTracker(experiment_name="e", run_name="r", save_dir=str(tmp_path))
    writer = t.experiment
    t.log_figure("cm", object(), step=1)
    assert writer.figures == ["cm"]
    t.finalize()
    assert writer.closed


def test_log_dir_reported_before_writer(tmp_path: Path) -> None:
    t = TensorBoardTracker(experiment_name="e", run_name="r", save_dir=str(tmp_path))
    assert t.log_dir == tmp_path / "e" / "r"

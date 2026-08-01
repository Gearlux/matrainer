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


# --------------------------------------------------------------------------- #
# manifest projection
# --------------------------------------------------------------------------- #


def test_log_manifest_renders_the_manifest_into_the_text_tab(fake_tb: Any, tmp_path: Path) -> None:
    """TensorBoard has no artifact store — `add_text` is the only slot the UI will show."""
    from marainer.tracking.manifest import RunManifest

    m = RunManifest(run_dir=tmp_path / "manifest-home", handle_interrupts=False)
    m.set("params", {"lr": 0.1})
    t = TensorBoardTracker(experiment_name="exp", run_name="r", save_dir=str(tmp_path / "tb"))
    t.log_manifest(m)

    texts = dict(t.experiment.texts)
    assert "manifest" in texts
    assert "```yaml" in texts["manifest"] and "lr: 0.1" in texts["manifest"]


def test_log_manifest_still_copies_the_files_and_logs_hparams(fake_tb: Any, tmp_path: Path) -> None:
    """Native slots are ADDITIONAL — the artifact copy must not be replaced by add_text."""
    from marainer.tracking.manifest import RunManifest

    m = RunManifest(run_dir=tmp_path / "manifest-home", handle_interrupts=False)
    m.set("params", {"lr": 0.1})
    run_dir = tmp_path / "tb" / "exp" / "r"
    t = TensorBoardTracker(experiment_name="exp", run_name="r", save_dir=str(tmp_path / "tb"))
    t.log_manifest(m)
    assert (run_dir / "artifacts" / "manifest.yaml").exists()
    assert (run_dir / "artifacts" / "manifest.json").exists()


def test_log_hyperparams_drops_what_tensorboard_rejects(fake_tb: Any, tmp_path: Path) -> None:
    """`add_hparams` accepts only int/float/str/bool — a list raises ValueError inside it."""
    recorded: List[dict] = []

    def capture(self: Any, hparams: dict, metrics: dict) -> None:
        recorded.append(hparams)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(_FakeWriter, "add_hparams", capture)
    try:
        t = TensorBoardTracker(experiment_name="exp", run_name="r", save_dir=str(tmp_path / "tb"))
        t.log_hyperparams({"lr": 0.1, "size": [8, 8], "name": "x"})
    finally:
        monkey.undo()
    assert recorded == [{"lr": 0.1, "name": "x"}]

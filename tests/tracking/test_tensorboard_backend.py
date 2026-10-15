"""Tests for TensorBoardTracker with the SummaryWriter mocked via sys.modules."""

import sys
import types
from pathlib import Path
from typing import Any, List

import pytest

from matrainer.tracking.tensorboard_backend import TensorBoardTracker


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

    def add_text(self, tag: str, text: str, global_step: Any = None) -> None:
        # `global_step` mirrors the real `SummaryWriter.add_text` signature — `log_text` passes
        # it (that is what drives the Text tab's step slider), `set_tags` does not.
        self.texts.append((tag, text, global_step))

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
        raise ImportError(f"pip install matrainer[{extra}]")

    monkeypatch.setattr("matrainer.tracking.tensorboard_backend.require", fake_require)
    t = TensorBoardTracker()
    with pytest.raises(ImportError, match=r"pip install matrainer\[tensorboard\]"):
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
    from matrainer.tracking.manifest import RunManifest

    m = RunManifest(run_dir=tmp_path / "manifest-home", handle_interrupts=False)
    m.set("params", {"lr": 0.1})
    t = TensorBoardTracker(experiment_name="exp", run_name="r", save_dir=str(tmp_path / "tb"))
    t.log_manifest(m)

    texts = {tag: text for tag, text, _step in t.experiment.texts}
    assert "manifest" in texts
    assert "```yaml" in texts["manifest"] and "lr: 0.1" in texts["manifest"]


def test_log_text_goes_to_the_text_tab_with_its_step(fake_tb: Any, tmp_path: Path) -> None:
    """The one native non-numeric slot TensorBoard has — and `step` is what makes it browsable.

    No base-class `.md` file beside it: the value is already durable in the event file, so the
    inherited default would only add a second copy nobody reads.
    """
    t = TensorBoardTracker(experiment_name="e", run_name="r", save_dir=str(tmp_path))
    t.log_text("val/confusion", "|a|b|", step=2)

    assert t.experiment.texts == [("val/confusion", "|a|b|", 2)]
    assert not (tmp_path / "e" / "r" / "text").exists()


def test_log_manifest_still_copies_the_files_and_logs_hparams(fake_tb: Any, tmp_path: Path) -> None:
    """Native slots are ADDITIONAL — the artifact copy must not be replaced by add_text."""
    from matrainer.tracking.manifest import RunManifest

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


# --- raise_errors: a refused call is a debug line, and raised as well when asked ----------------

SAID_NO = RuntimeError("tensorboard said no")


def _said_no(*args: Any, **kwargs: Any) -> Any:
    raise SAID_NO


def _manifest(tmp_path: Path) -> Any:
    from matrainer.tracking.manifest import RunManifest

    return RunManifest(run_dir=tmp_path / "manifest-home", handle_interrupts=False)


#: (writer method refused, the tracker call that writes through it)
_ONE_REQUEST = [
    ("add_hparams", lambda t, tmp: t.log_hyperparams({"lr": 0.1})),
    ("add_figure", lambda t, tmp: t.log_figure("val/cm", object())),
    ("add_text", lambda t, tmp: t.log_text("val/report", "|a|")),
    ("add_text", lambda t, tmp: t.log_manifest(_manifest(tmp))),
]


@pytest.mark.parametrize("raise_errors", [False, True])
@pytest.mark.parametrize("method, call", _ONE_REQUEST, ids=[f"{m}-{i}" for i, (m, _) in enumerate(_ONE_REQUEST)])
def test_a_refused_write_is_raised_only_when_asked(
    fake_tb: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, call: Any, raise_errors: bool
) -> None:
    tracker = TensorBoardTracker(save_dir=str(tmp_path / "tb"), raise_errors=raise_errors)
    monkeypatch.setattr(tracker.experiment, method, _said_no)
    if raise_errors:
        with pytest.raises(RuntimeError, match="tensorboard said no"):
            call(tracker, tmp_path)
    else:
        call(tracker, tmp_path)

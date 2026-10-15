"""Tests for the stdlib-only CsvTracker (the always-available default backend)."""

import csv
import json
from pathlib import Path

import pytest

from matrainer.tracking.csv_backend import CsvTracker


def _tracker(tmp_path: Path) -> CsvTracker:
    return CsvTracker(experiment_name="exp", run_name="run1", save_dir=str(tmp_path))


def test_run_dir_created_lazily(tmp_path: Path) -> None:
    t = _tracker(tmp_path)
    assert not (tmp_path / "exp").exists()  # nothing on construction
    _ = t.experiment
    assert (tmp_path / "exp" / "run1").is_dir()


def test_log_metrics_writes_csv_with_unioned_columns(tmp_path: Path) -> None:
    t = _tracker(tmp_path)
    t.log_metrics({"loss": 1.0}, step=0)
    t.log_metrics({"loss": 0.5, "acc": 0.9}, step=1)
    rows = list(csv.DictReader((tmp_path / "exp" / "run1" / "metrics.csv").open()))
    assert rows[0]["step"] == "0" and rows[0]["loss"] == "1.0"
    assert rows[1]["acc"] == "0.9"
    assert "acc" in rows[0]  # unioned header, blank for the first row


def test_log_hyperparams_merges(tmp_path: Path) -> None:
    t = _tracker(tmp_path)
    t.log_hyperparams({"lr": 0.1})
    t.log_hyperparams({"epochs": 3})
    data = json.loads((tmp_path / "exp" / "run1" / "hparams.json").read_text())
    assert data == {"lr": 0.1, "epochs": 3}


def test_log_artifact_copies_file(tmp_path: Path) -> None:
    src = tmp_path / "config.yaml"
    src.write_text("a: 1")
    t = _tracker(tmp_path)
    t.log_artifact(src, "config")
    assert (tmp_path / "exp" / "run1" / "artifacts" / "config" / "config.yaml").read_text() == "a: 1"


def test_log_artifact_missing_file_is_skipped(tmp_path: Path) -> None:
    t = _tracker(tmp_path)
    t.log_artifact(tmp_path / "nope.txt")  # no raise


def test_tags_written_on_first_access(tmp_path: Path) -> None:
    t = CsvTracker(experiment_name="e", run_name="r", save_dir=str(tmp_path), tags={"k": "v"})
    _ = t.experiment
    assert json.loads((tmp_path / "e" / "r" / "tags.json").read_text()) == {"k": "v"}


def test_log_figure_saves_png(tmp_path: Path) -> None:
    class FakeFig:
        def __init__(self) -> None:
            self.saved: str = ""

        def savefig(self, path: object) -> None:
            Path(str(path)).write_bytes(b"png")
            self.saved = str(path)

    t = _tracker(tmp_path)
    t.log_figure("test/cm", FakeFig())
    assert (tmp_path / "exp" / "run1" / "artifacts" / "test_cm.png").exists()


def test_log_dir_is_run_dir(tmp_path: Path) -> None:
    t = _tracker(tmp_path)
    assert t.log_dir == tmp_path / "exp" / "run1"


@pytest.mark.parametrize("raise_errors", [False, True])
def test_a_figure_that_cannot_be_saved_is_raised_only_when_asked(tmp_path: Path, raise_errors: bool) -> None:
    class RefusingFig:
        def savefig(self, path: object) -> None:
            raise RuntimeError("disk said no")

    t = CsvTracker(experiment_name="exp", run_name="run1", save_dir=str(tmp_path), raise_errors=raise_errors)
    if raise_errors:
        with pytest.raises(RuntimeError, match="disk said no"):
            t.log_figure("test/cm", RefusingFig())
    else:
        t.log_figure("test/cm", RefusingFig())

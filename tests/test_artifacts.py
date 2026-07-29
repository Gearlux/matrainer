"""Run artifacts beside a checkpoint — the convention and the logging, not the format.

The format is recordstream's (``LabelMap.save`` / ``.load``); three consumers had each
hand-rolled that JSON. What marainer owns is WHERE a run artifact lands and that it gets
logged — which is why these take a plain directory and a plain path.
"""

import json
from pathlib import Path
from typing import Any, List, Union, cast

import pytest
from recordstream import LabelMap

from marainer.artifacts import CLASS_NAMES_FILE, load_class_names, save_class_names, write_metrics_sidecar
from marainer.tracking.base import Tracker


class _RecordingTracker:
    """Records what was logged. Not a `Tracker` subclass — these helpers touch only
    `log_artifact`, and implementing three abstract methods to prove that would test
    the ABC rather than the artifact helpers."""

    def __init__(self) -> None:
        self.artifacts: List[Any] = []

    def log_artifact(self, path: Union[str, Path], artifact_dir: str = "") -> None:
        self.artifacts.append(Path(path))


def _map() -> LabelMap:
    return LabelMap.from_class_names(["bird", "cat", "dog"])


# --------------------------------------------------------------------------- #
# save / load
# --------------------------------------------------------------------------- #


def test_saving_writes_recordstreams_format_not_a_hand_rolled_one(tmp_path: Path) -> None:
    path = save_class_names(_map(), tmp_path)

    assert path == tmp_path / CLASS_NAMES_FILE
    assert json.loads(path.read_text()) == {"class_names": ["bird", "cat", "dog"], "num_classes": 3}


def test_a_bare_sequence_of_names_works_too(tmp_path: Path) -> None:
    save_class_names(["a", "b"], tmp_path)

    assert json.loads((tmp_path / CLASS_NAMES_FILE).read_text())["class_names"] == ["a", "b"]


def test_the_written_file_is_logged_as_an_artifact(tmp_path: Path) -> None:
    tracker = _RecordingTracker()

    path = save_class_names(_map(), tmp_path, [cast(Tracker, tracker)])

    assert tracker.artifacts == [path]


def test_nothing_to_write_is_not_an_error(tmp_path: Path) -> None:
    """A run on integer labels has no vocabulary; a framework may not know its own dir."""
    assert save_class_names(None, tmp_path) is None
    assert save_class_names([], tmp_path) is None
    assert save_class_names(_map(), None) is None
    assert not list(tmp_path.iterdir())


def test_load_finds_the_file_beside_the_checkpoint(tmp_path: Path) -> None:
    """The whole point: evaluate/predict reuse the training run's ordering, never refit."""
    save_class_names(_map(), tmp_path)

    loaded = load_class_names(tmp_path / "epoch=3.ckpt")

    assert loaded is not None
    assert loaded.class_names == ["bird", "cat", "dog"]
    assert loaded.mapping == _map().mapping


def test_load_degrades_rather_than_raising(tmp_path: Path) -> None:
    """A caller can still proceed on integer labels, so none of these are fatal."""
    assert load_class_names(None) is None
    assert load_class_names(tmp_path / "no_such.ckpt") is None

    (tmp_path / CLASS_NAMES_FILE).write_text("{ not json")
    assert load_class_names(tmp_path / "x.ckpt") is None


def test_save_and_load_round_trip_the_exact_ordering(tmp_path: Path) -> None:
    original = LabelMap.fit(["dog", "cat", "bird", "cat"])
    save_class_names(original, tmp_path)

    loaded = load_class_names(tmp_path / "last.ckpt")
    assert loaded is not None and loaded.mapping == original.mapping


# --------------------------------------------------------------------------- #
# metrics sidecar
# --------------------------------------------------------------------------- #


def test_the_sidecar_stamps_the_run_identity(tmp_path: Path) -> None:
    """A sidecar found on disk must identify the run it came from."""
    path = write_metrics_sidecar(tmp_path / "m.json", {"val/acc": 0.99}, "my_exp", "my_run")

    assert path is not None
    assert json.loads(path.read_text()) == {
        "experiment_name": "my_exp",
        "run_name": "my_run",
        "metrics": {"val/acc": 0.99},
    }


def test_the_sidecar_creates_missing_parents(tmp_path: Path) -> None:
    path = write_metrics_sidecar(tmp_path / "deep" / "er" / "m.json", {"a": 1.0})

    assert path is not None and path.is_file()


def test_no_path_writes_nothing(tmp_path: Path) -> None:
    """So a run with no sidecar configured needs no guard at the call site."""
    assert write_metrics_sidecar(None, {"a": 1.0}) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

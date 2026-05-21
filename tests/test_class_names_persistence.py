"""Tests for the Trainer's ``class_names.json`` persistence.

Covers :func:`marainer.trainer._resolve_label_names`,
:func:`marainer.trainer._resolve_checkpoint_dir`, and
:func:`marainer.trainer._write_class_names` — the helpers that capture the
training-time class names so the Evaluator can label the confusion matrix
without the user re-specifying anything.
"""

import json
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

from marainer.trainer import _resolve_checkpoint_dir, _resolve_label_names, _write_class_names


class _DatasetWithLabels:
    def __init__(self, names: List[str]) -> None:
        self.label_names = tuple(names)


class _DatasetWithoutLabels:
    pass


def test_resolve_label_names_from_dataset() -> None:
    ds = _DatasetWithLabels(["a", "b", "c"])
    assert _resolve_label_names(ds) == ["a", "b", "c"]


def test_resolve_label_names_returns_none_when_missing() -> None:
    assert _resolve_label_names(_DatasetWithoutLabels()) is None


def test_resolve_label_names_returns_none_when_empty() -> None:
    ds = _DatasetWithLabels([])
    assert _resolve_label_names(ds) is None


def test_resolve_label_names_returns_none_for_non_iterable() -> None:
    ds = _DatasetWithoutLabels()
    ds.label_names = 42  # type: ignore[attr-defined]
    assert _resolve_label_names(ds) is None


def test_resolve_checkpoint_dir_prefers_callback_dirpath(tmp_path: Path) -> None:
    """When a ModelCheckpoint callback is wired with ``dirpath``, use it."""
    lt = MagicMock()
    lt.checkpoint_callback.dirpath = str(tmp_path / "ckpts")
    lt.default_root_dir = str(tmp_path / "ignored")
    assert _resolve_checkpoint_dir(lt) == Path(str(tmp_path / "ckpts"))


def test_resolve_checkpoint_dir_falls_back_to_default_root_dir(tmp_path: Path) -> None:
    """No callback dirpath → fall back to default_root_dir."""
    lt = MagicMock()
    lt.checkpoint_callback = None
    lt.default_root_dir = str(tmp_path)
    assert _resolve_checkpoint_dir(lt) == tmp_path


def test_resolve_checkpoint_dir_returns_none_when_both_missing() -> None:
    lt = MagicMock()
    lt.checkpoint_callback = None
    lt.default_root_dir = None
    assert _resolve_checkpoint_dir(lt) is None


def test_write_class_names_writes_to_checkpoint_dir(tmp_path: Path) -> None:
    """Filesystem path: ``<ckpt_dir>/class_names.json`` with the public schema."""
    lt = MagicMock()
    lt.checkpoint_callback.dirpath = str(tmp_path / "ckpts")
    lt.loggers = []  # no MLflow

    _write_class_names(lt, ["alpha", "beta", "gamma"])

    out = tmp_path / "ckpts" / "class_names.json"
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload == {"class_names": ["alpha", "beta", "gamma"], "num_classes": 3}


def test_write_class_names_uses_default_root_dir_when_no_callback(
    tmp_path: Path,
) -> None:
    lt = MagicMock()
    lt.checkpoint_callback = None
    lt.default_root_dir = str(tmp_path)
    lt.loggers = []

    _write_class_names(lt, ["a", "b"])

    assert (tmp_path / "class_names.json").exists()


def test_write_class_names_logs_to_mlflow_logger(tmp_path: Path) -> None:
    """MLflow-style loggers (``experiment.log_dict(run_id, payload, path)``) get
    a ``class_names.json`` artifact for the run."""
    lt = MagicMock()
    lt.checkpoint_callback.dirpath = str(tmp_path)

    mlflow_logger = MagicMock()
    mlflow_logger.run_id = "run-abc"
    mlflow_logger.experiment.log_dict = MagicMock()

    other_logger = MagicMock()
    other_logger.experiment = MagicMock(spec=[])  # no log_dict
    other_logger.run_id = None

    lt.loggers = [mlflow_logger, other_logger]

    _write_class_names(lt, ["x", "y"])

    mlflow_logger.experiment.log_dict.assert_called_once_with(
        "run-abc", {"class_names": ["x", "y"], "num_classes": 2}, "class_names.json"
    )


def test_write_class_names_skips_mlflow_when_no_run_id(tmp_path: Path) -> None:
    """A logger that exposes ``log_dict`` but no ``run_id`` is silently skipped."""
    lt = MagicMock()
    lt.checkpoint_callback.dirpath = str(tmp_path)

    fake_logger = MagicMock()
    fake_logger.run_id = None
    fake_logger.experiment.log_dict = MagicMock()
    lt.loggers = [fake_logger]

    _write_class_names(lt, ["x"])

    fake_logger.experiment.log_dict.assert_not_called()


def test_write_class_names_handles_no_checkpoint_dir_gracefully() -> None:
    """No filesystem destination → no crash; MLflow path still runs."""
    lt = MagicMock()
    lt.checkpoint_callback = None
    lt.default_root_dir = None

    mlflow_logger = MagicMock()
    mlflow_logger.run_id = "rid"
    mlflow_logger.experiment.log_dict = MagicMock()
    lt.loggers = [mlflow_logger]

    _write_class_names(lt, ["only"])

    mlflow_logger.experiment.log_dict.assert_called_once_with(
        "rid", {"class_names": ["only"], "num_classes": 1}, "class_names.json"
    )


def test_write_class_names_tolerates_mlflow_log_dict_failure(tmp_path: Path) -> None:
    """A raising backend doesn't break the filesystem write."""
    lt = MagicMock()
    lt.checkpoint_callback.dirpath = str(tmp_path)

    fake_logger = MagicMock()
    fake_logger.run_id = "rid"
    fake_logger.experiment.log_dict.side_effect = RuntimeError("MLflow not running")
    lt.loggers = [fake_logger]

    _write_class_names(lt, ["a"])  # must not raise

    assert (tmp_path / "class_names.json").exists()


def test_write_class_names_is_idempotent(tmp_path: Path, caplog: Any) -> None:
    """Re-running overwrites cleanly with the same content."""
    lt = MagicMock()
    lt.checkpoint_callback.dirpath = str(tmp_path)
    lt.loggers = []

    _write_class_names(lt, ["one", "two"])
    _write_class_names(lt, ["one", "two"])

    payload = json.loads((tmp_path / "class_names.json").read_text())
    assert payload["class_names"] == ["one", "two"]

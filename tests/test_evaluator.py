"""Tests for the task-agnostic Marainer Evaluator.

Detection-flavored tests live in ``raidar/tests/test_evaluator_detection.py``;
here we exercise the classification-shaped path only.
"""

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")
import matplotlib.figure  # noqa: E402
import pytest  # noqa: E402
import pytorch_lightning as L  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from confluid import Class  # noqa: E402
from dataflux.core import Flux  # noqa: E402
from dataflux.sample import Sample  # noqa: E402
from torch import Tensor  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchmetrics import MetricCollection  # noqa: E402
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix  # noqa: E402

from marainer.evaluator import (  # noqa: E402
    LightningEvaluator,
    _coerce_metric_value,
    _ensure_class_names,
    _is_square_2d_tensor,
)
from marainer.loggers import _render_matrix_as_table, log_figure_to_loggers  # noqa: E402


class _RecordingSink:
    """Minimal sink that records everything handed to write()."""

    def __init__(self) -> None:
        self.writes: List[Any] = []
        self.closed = False

    def write(self, prediction: Any, metadata: Dict[str, Any]) -> None:
        self.writes.append((prediction, metadata))

    def close(self) -> None:
        self.closed = True


class _ClassificationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)  # type: ignore[no-any-return]


def test_evaluator_runs_on_classification_task(tmp_path: Any) -> None:
    model = _ClassificationModel()
    samples = [Sample(input=torch.randn(4), target=i % 3, metadata={"idx": i}) for i in range(8)]
    flux = Flux(source=samples)

    metrics = MetricCollection({"acc": MulticlassAccuracy(num_classes=3)})
    sink = _RecordingSink()

    evaluator = _make_evaluator(
        model=model,
        dataset=flux,
        loader=Class(DataLoader, batch_size=4),
        metrics=metrics,
        predictions_sink=sink,
        lightning=Class(
            L.Trainer,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            default_root_dir=str(tmp_path),
        ),
    )
    evaluator.run()

    assert sink.closed is True
    # Two batches of 4 → evaluator calls write at least once per batch.
    # _write_predictions treats a raw tensor output as a single prediction.
    assert len(sink.writes) >= 2


def test_evaluator_writes_metrics_sidecar_when_path_set(tmp_path: Any) -> None:
    """When ``metrics_sidecar_path`` is set, the final compute() dict is JSON-dumped."""
    model = _ClassificationModel()
    samples = [Sample(input=torch.randn(4), target=i % 3, metadata={"idx": i}) for i in range(8)]
    flux = Flux(source=samples)
    metrics = MetricCollection({"acc": MulticlassAccuracy(num_classes=3)})
    sidecar = tmp_path / "eval_metrics.json"

    evaluator = _make_evaluator(
        model=model,
        dataset=flux,
        loader=Class(DataLoader, batch_size=4),
        metrics=metrics,
        lightning=Class(
            L.Trainer,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            default_root_dir=str(tmp_path),
        ),
        experiment_name="test_exp",
        run_name="test_run",
        metrics_sidecar_path=str(sidecar),
    )
    evaluator.run()

    assert sidecar.exists(), "Evaluator must write the sidecar at the configured path."
    payload = json.loads(sidecar.read_text())
    assert payload["experiment_name"] == "test_exp"
    assert payload["run_name"] == "test_run"
    # Each surfaced scalar key must be a JSON-safe number.
    assert "metrics" in payload
    metric_values = payload["metrics"]
    # MulticlassAccuracy under MetricCollection surfaces the prefixed key "acc".
    assert "acc" in metric_values
    assert isinstance(metric_values["acc"], (int, float))


def test_evaluator_skips_sidecar_when_path_none(tmp_path: Any) -> None:
    """No sidecar path → no file written, even when metrics is set."""
    model = _ClassificationModel()
    samples = [Sample(input=torch.randn(4), target=i % 3, metadata={"idx": i}) for i in range(4)]
    flux = Flux(source=samples)
    metrics = MetricCollection({"acc": MulticlassAccuracy(num_classes=3)})

    evaluator = _make_evaluator(
        model=model,
        dataset=flux,
        loader=Class(DataLoader, batch_size=4),
        metrics=metrics,
        lightning=Class(
            L.Trainer,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            default_root_dir=str(tmp_path),
        ),
        # metrics_sidecar_path omitted → defaults to None
    )
    evaluator.run()

    # No JSON files should have been written under tmp_path by the sidecar path.
    json_files = list(tmp_path.rglob("*.json"))
    sidecar_files = [p for p in json_files if "metric" in p.name.lower()]
    assert sidecar_files == [], f"Unexpected sidecar artifacts: {sidecar_files}"


def test_evaluator_skips_sidecar_when_no_metrics(tmp_path: Any) -> None:
    """metrics=None → sidecar path is honored only when there's something to write."""
    model = _ClassificationModel()
    samples = [Sample(input=torch.randn(4), target=i % 3, metadata={"idx": i}) for i in range(4)]
    flux = Flux(source=samples)
    sidecar = tmp_path / "eval_metrics.json"

    evaluator = _make_evaluator(
        model=model,
        dataset=flux,
        loader=Class(DataLoader, batch_size=4),
        metrics=None,  # explicit no metrics
        lightning=Class(
            L.Trainer,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            default_root_dir=str(tmp_path),
        ),
        metrics_sidecar_path=str(sidecar),
    )
    evaluator.run()

    assert not sidecar.exists(), "Sidecar must NOT be written when metrics is None."


# --- _coerce_metric_value -------------------------------------------------


def test_coerce_metric_value_handles_scalar_tensor() -> None:
    assert _coerce_metric_value(torch.tensor(3.14)) == pytest.approx(3.14)
    assert _coerce_metric_value(torch.tensor(7, dtype=torch.long)) == 7


def test_coerce_metric_value_handles_vector_tensor() -> None:
    out = _coerce_metric_value(torch.tensor([1.0, 2.0, 3.0]))
    assert out == [pytest.approx(1.0), pytest.approx(2.0), pytest.approx(3.0)]


def test_coerce_metric_value_recurses_into_dict_and_list() -> None:
    out = _coerce_metric_value({"map": torch.tensor(0.5), "per_class": [torch.tensor(0.1), torch.tensor(0.2)]})
    assert out == {
        "map": pytest.approx(0.5),
        "per_class": [pytest.approx(0.1), pytest.approx(0.2)],
    }


def test_coerce_metric_value_passthrough_for_native_scalars() -> None:
    assert _coerce_metric_value(42) == 42
    assert _coerce_metric_value(3.14) == 3.14
    assert _coerce_metric_value("hello") == "hello"
    assert _coerce_metric_value(True) is True
    assert _coerce_metric_value(None) is None


def test_coerce_metric_value_falls_back_to_str_for_unknown() -> None:
    class _Custom:
        def __repr__(self) -> str:
            return "<custom>"

    assert _coerce_metric_value(_Custom()) == "<custom>"


# --- confusion-matrix render pass ---------------------------------------------


def test_is_square_2d_tensor_accepts_only_square_matrices() -> None:
    assert _is_square_2d_tensor(torch.zeros(3, 3))
    assert _is_square_2d_tensor(torch.ones(1, 1))
    assert not _is_square_2d_tensor(torch.zeros(2, 3))
    assert not _is_square_2d_tensor(torch.zeros(4))
    assert not _is_square_2d_tensor(torch.tensor(0.5))
    assert not _is_square_2d_tensor([[1, 2], [3, 4]])  # not a tensor
    assert not _is_square_2d_tensor(torch.zeros(0, 0))


def test_ensure_class_names_returns_input_when_length_matches() -> None:
    assert _ensure_class_names(["a", "b", "c"], 3) == ["a", "b", "c"]


def test_ensure_class_names_falls_back_when_none_or_mismatched() -> None:
    assert _ensure_class_names(None, 3) == ["0", "1", "2"]
    assert _ensure_class_names(["a", "b"], 3) == ["0", "1", "2"]


def test_render_matrix_as_table_contains_labels_and_values() -> None:
    cm = torch.tensor([[5, 1], [2, 7]])
    rendered = _render_matrix_as_table("test/confusion_matrix", cm, ["cat", "dog"])

    assert "test/confusion_matrix" in rendered
    assert "pred:cat" in rendered and "pred:dog" in rendered
    assert "true:cat" in rendered and "true:dog" in rendered
    for cell in ("5", "1", "2", "7"):
        assert cell in rendered


def test_log_figure_to_loggers_forwards_to_tensorboard_style() -> None:
    """A logger whose ``experiment.add_figure`` is callable receives the figure."""
    lt = MagicMock()
    tb_logger = MagicMock()
    tb_logger.experiment.add_figure = MagicMock()
    lt.loggers = [tb_logger]
    lt.global_step = 42

    fig = matplotlib.figure.Figure()
    log_figure_to_loggers(lt, "cm", fig)

    tb_logger.experiment.add_figure.assert_called_once_with("cm", fig, 42)


def test_log_figure_to_loggers_forwards_to_mlflow_style() -> None:
    """A logger with only ``log_figure`` + a ``run_id`` is dispatched via MLflow path."""
    lt = MagicMock()

    mlflow_logger = MagicMock()
    # Make sure add_figure path is NOT triggered — drop the attribute.
    mlflow_logger.experiment = MagicMock(spec=["log_figure"])
    mlflow_logger.experiment.log_figure = MagicMock()
    mlflow_logger.run_id = "run-xyz"
    lt.loggers = [mlflow_logger]
    lt.global_step = 0

    fig = matplotlib.figure.Figure()
    log_figure_to_loggers(lt, "test/cm", fig)

    mlflow_logger.experiment.log_figure.assert_called_once_with("run-xyz", fig, "test_cm.png")


def test_log_figure_to_loggers_handles_no_trainer() -> None:
    """``trainer=None`` (called outside a Lightning run) must not raise."""
    log_figure_to_loggers(None, "cm", matplotlib.figure.Figure())  # must not raise


def test_evaluator_logs_confusion_matrix_text_table(tmp_path: Any) -> None:
    """End-to-end: a confusion-matrix metric in the collection triggers a rendered
    table emitted via logflow's loguru sink."""
    from loguru import logger as _loguru_logger

    captured: List[str] = []
    sink_id = _loguru_logger.add(lambda m: captured.append(str(m)), level="INFO")
    try:
        model = _ClassificationModel()
        samples = [Sample(input=torch.randn(4), target=i % 3, metadata={"idx": i}) for i in range(12)]
        flux = Flux(source=samples)
        metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=3),
                "cm": MulticlassConfusionMatrix(num_classes=3),
            }
        )
        evaluator = _make_evaluator(
            model=model,
            dataset=flux,
            loader=Class(DataLoader, batch_size=4),
            metrics=metrics,
            class_names=["cat", "dog", "bird"],
            lightning=Class(
                L.Trainer,
                accelerator="cpu",
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                default_root_dir=str(tmp_path),
            ),
        )
        evaluator.run()
    finally:
        _loguru_logger.remove(sink_id)

    rendered = "".join(captured)
    assert "cm:" in rendered
    assert "pred:cat" in rendered and "true:bird" in rendered


def test_evaluator_resolves_class_names_from_dataset(tmp_path: Any) -> None:
    """Without an explicit ``class_names``, the Evaluator picks up
    ``dataset.label_names`` and uses it for axis labels."""

    class _Labeled(Flux):
        label_names = ("a", "b", "c")

    samples = [Sample(input=torch.randn(4), target=i % 3, metadata={"idx": i}) for i in range(6)]
    flux = _Labeled(source=samples)

    evaluator = _make_evaluator(
        model=_ClassificationModel(),
        dataset=flux,
        loader=Class(DataLoader, batch_size=3),
        metrics=MetricCollection({"cm": MulticlassConfusionMatrix(num_classes=3)}),
        lightning=Class(
            L.Trainer,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            default_root_dir=str(tmp_path),
        ),
    )
    evaluator.run()
    assert evaluator._resolved_class_names == ["a", "b", "c"]


def test_evaluator_resolves_class_names_from_json_file(tmp_path: Any) -> None:
    """An explicit ``class_names=<path>`` reads the public JSON format the
    Trainer writes."""
    names_path = tmp_path / "class_names.json"
    names_path.write_text(json.dumps({"class_names": ["x", "y", "z"]}))

    samples = [Sample(input=torch.randn(4), target=i % 3, metadata={"idx": i}) for i in range(6)]
    evaluator = _make_evaluator(
        model=_ClassificationModel(),
        dataset=Flux(source=samples),
        loader=Class(DataLoader, batch_size=3),
        metrics=MetricCollection({"cm": MulticlassConfusionMatrix(num_classes=3)}),
        class_names=str(names_path),
        lightning=Class(
            L.Trainer,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            default_root_dir=str(tmp_path),
        ),
    )
    evaluator.run()
    assert evaluator._resolved_class_names == ["x", "y", "z"]


def test_evaluator_resolves_class_names_next_to_checkpoint(tmp_path: Any) -> None:
    """No explicit param and no ``dataset.label_names`` → the Evaluator looks
    for ``class_names.json`` next to ``model.checkpoint_path``."""
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"")
    (tmp_path / "class_names.json").write_text(json.dumps({"class_names": ["p", "q", "r"]}))

    model = _ClassificationModel()
    # Monkey-patch a checkpoint_path attribute the way the real model classes
    # (e.g. raidar_ultralytics.yolo26.YOLO26Model) expose it for self-loading.
    setattr(model, "checkpoint_path", str(ckpt))

    samples = [Sample(input=torch.randn(4), target=i % 3, metadata={"idx": i}) for i in range(3)]
    evaluator = _make_evaluator(
        model=model,
        dataset=Flux(source=samples),
        loader=Class(DataLoader, batch_size=3),
        metrics=MetricCollection({"cm": MulticlassConfusionMatrix(num_classes=3)}),
        lightning=Class(
            L.Trainer,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            default_root_dir=str(tmp_path),
        ),
    )
    evaluator.run()
    assert evaluator._resolved_class_names == ["p", "q", "r"]


def test_evaluator_falls_back_to_integer_indices(tmp_path: Any) -> None:
    """Every resolution source missing → render uses ``["0", "1", ...]``."""
    samples = [Sample(input=torch.randn(4), target=i % 3, metadata={"idx": i}) for i in range(3)]
    evaluator = _make_evaluator(
        model=_ClassificationModel(),
        dataset=Flux(source=samples),
        loader=Class(DataLoader, batch_size=3),
        metrics=MetricCollection({"cm": MulticlassConfusionMatrix(num_classes=3)}),
        lightning=Class(
            L.Trainer,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            default_root_dir=str(tmp_path),
        ),
    )
    evaluator.run()
    assert evaluator._resolved_class_names is None  # no source → render falls back at table time


def test_evaluator_render_pass_skips_non_square_metrics(tmp_path: Any) -> None:
    """Scalar accuracy alone produces no rendered table."""
    from loguru import logger as _loguru_logger

    captured: List[str] = []
    sink_id = _loguru_logger.add(lambda m: captured.append(str(m)), level="INFO")
    try:
        samples = [Sample(input=torch.randn(4), target=i % 3, metadata={"idx": i}) for i in range(3)]
        evaluator = _make_evaluator(
            model=_ClassificationModel(),
            dataset=Flux(source=samples),
            loader=Class(DataLoader, batch_size=3),
            metrics=MetricCollection({"acc": MulticlassAccuracy(num_classes=3)}),
            lightning=Class(
                L.Trainer,
                accelerator="cpu",
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                default_root_dir=str(tmp_path),
            ),
        )
        evaluator.run()
    finally:
        _loguru_logger.remove(sink_id)
    assert "pred:" not in "".join(captured)


def test_evaluator_sidecar_unchanged_when_confusion_matrix_present(
    tmp_path: Any,
) -> None:
    """The JSON sidecar still receives the 2D-list payload for backwards
    compatibility with downstream tooling (``read_run_metrics``)."""
    sidecar = tmp_path / "eval_metrics.json"
    samples = [Sample(input=torch.randn(4), target=i % 3, metadata={"idx": i}) for i in range(6)]
    evaluator = _make_evaluator(
        model=_ClassificationModel(),
        dataset=Flux(source=samples),
        loader=Class(DataLoader, batch_size=3),
        metrics=MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=3),
                "cm": MulticlassConfusionMatrix(num_classes=3),
            }
        ),
        metrics_sidecar_path=str(sidecar),
        lightning=Class(
            L.Trainer,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            default_root_dir=str(tmp_path),
        ),
    )
    evaluator.run()

    payload = json.loads(sidecar.read_text())
    cm = payload["metrics"]["cm"]
    assert isinstance(cm, list) and len(cm) == 3 and all(len(row) == 3 for row in cm)


def _make_evaluator(**overrides: Any) -> LightningEvaluator:
    """Build a LightningEvaluator, applying overrides as post-construction attributes.

    Mirrors the minimal-ctor pattern: only 'model', 'dataset', 'experiment_name', 'run_name' are
    constructor params; deferred slots (``loader`` / ``lightning`` / ``metrics`` /
    ``predictions_sink`` / ``metrics_sidecar_path`` / ``class_names``) are set on
    the instance after construction.
    """
    _ctor_keys = (
        "model",
        "dataset",
        "experiment_name",
        "run_name",
    )
    ctor = {k: overrides.pop(k) for k in _ctor_keys if k in overrides}
    ev = LightningEvaluator(**ctor)
    for _k, _v in overrides.items():
        setattr(ev, _k, _v)
    return ev

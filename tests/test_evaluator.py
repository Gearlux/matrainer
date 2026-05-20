"""Tests for the task-agnostic Marainer Evaluator.

Detection-flavored tests live in ``raidar/tests/test_evaluator_detection.py``;
here we exercise the classification-shaped path only.
"""

import json
from typing import Any, Dict, List

import pytest
import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class
from dataflux.core import Flux
from dataflux.sample import Sample
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy

from marainer.evaluator import LightningEvaluator, _coerce_metric_value


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

    evaluator = LightningEvaluator(
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

    evaluator = LightningEvaluator(
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

    evaluator = LightningEvaluator(
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

    evaluator = LightningEvaluator(
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
    assert out == {"map": pytest.approx(0.5), "per_class": [pytest.approx(0.1), pytest.approx(0.2)]}


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

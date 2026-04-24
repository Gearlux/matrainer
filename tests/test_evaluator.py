"""Tests for the task-agnostic Marainer Evaluator.

Detection-flavored tests live in ``raidar/tests/test_evaluator_detection.py``;
here we exercise the classification-shaped path only.
"""

from typing import Any, Dict, List

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

from marainer.evaluator import Evaluator


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
        return self.fc(x)


def test_evaluator_runs_on_classification_task(tmp_path: Any) -> None:
    model = _ClassificationModel()
    samples = [Sample(input=torch.randn(4), target=i % 3, metadata={"idx": i}) for i in range(8)]
    flux = Flux(source=samples)

    metrics = MetricCollection({"acc": MulticlassAccuracy(num_classes=3)})
    sink = _RecordingSink()

    evaluator = Evaluator(
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

"""Tests for generic dataset-derived dimension injection.

Covers :mod:`marainer.dimensions` (the helpers) and its use in
:meth:`marainer.trainer.LightningTrainer.run` — a config may declare the model
as ``!lazy:`` and omit ``num_classes``; the dataset derives it and the trainer
injects it at setup, while marainer itself stays task-agnostic.
"""

from typing import Any, Dict, cast

import pytest
import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class, configurable
from confluid.fluid import Lazy  # the Fluid class produced by ``!lazy:`` (not the Lazy[T] annotation)
from dataflux.sample import Sample
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy

from marainer.dimensions import collect_derived_dimensions, flow_node_with_dims, needs_derivation
from marainer.trainer import LightningTrainer


@configurable
class _Model(nn.Module):
    def __init__(self, input_size: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.fc(x))


class _CountingDataset:
    """Map-style dataset exposing the ``derived_dimensions()`` hook (like sonair's)."""

    def __init__(self, n: int, num_classes: int, input_size: int = 4) -> None:
        self._samples = [Sample(input=torch.randn(input_size), target=i % num_classes, metadata={}) for i in range(n)]
        self._num_classes = num_classes

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, i: int) -> Sample:
        return self._samples[i]

    def derived_dimensions(self) -> Dict[str, Any]:
        return {"num_classes": self._num_classes}


class _PlainDataset:
    """Dataset WITHOUT the hook (e.g. a detection/segmentation dataset)."""

    def __init__(self, n: int, input_size: int = 4) -> None:
        self._samples = [Sample(input=torch.randn(input_size), target=0, metadata={}) for _ in range(n)]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, i: int) -> Sample:
        return self._samples[i]


# --------------------------------------------------------------------------- #
# needs_derivation
# --------------------------------------------------------------------------- #


def test_needs_derivation_true_when_any_node_is_lazy() -> None:
    assert needs_derivation([Lazy(_Model)])
    assert needs_derivation([Class(_Model), Lazy(_Model), None])


def test_needs_derivation_false_without_lazy() -> None:
    assert not needs_derivation([Class(_Model), _Model(), None])


# --------------------------------------------------------------------------- #
# collect_derived_dimensions
# --------------------------------------------------------------------------- #


def test_collect_takes_max_across_splits() -> None:
    train = _CountingDataset(4, num_classes=3)
    val = _CountingDataset(4, num_classes=5)
    assert collect_derived_dimensions([train, val, None]) == {"num_classes": 5}


def test_collect_ignores_datasets_without_hook() -> None:
    assert collect_derived_dimensions([_PlainDataset(4), None]) == {}


# --------------------------------------------------------------------------- #
# flow_node_with_dims
# --------------------------------------------------------------------------- #


def test_flow_injects_into_lazy_model() -> None:
    model = flow_node_with_dims(Lazy(_Model, input_size=4), {"num_classes": 7})
    assert isinstance(model, _Model)
    assert model.fc.out_features == 7


def test_flow_does_not_override_explicit_value() -> None:
    model = flow_node_with_dims(Class(_Model, num_classes=3), {"num_classes": 7})
    assert isinstance(model, _Model)
    assert model.fc.out_features == 3  # explicit wins over derived


def test_flow_live_node_is_idempotent() -> None:
    live = _Model(num_classes=2)
    assert flow_node_with_dims(live, {"num_classes": 9}) is live


def test_flow_lazy_without_dims_raises() -> None:
    with pytest.raises(RuntimeError, match="declared lazy"):
        flow_node_with_dims(Lazy(_Model), {})


def test_flow_injects_num_classes_into_nested_metric_collection() -> None:
    """The load-bearing broadcast: flowing a lazy MetricCollection with
    num_classes reaches the nested metrics (which accept it) and is filtered
    out of the collection's own constructor."""
    coll = flow_node_with_dims(
        Lazy(MetricCollection, metrics={"acc": Class(MulticlassAccuracy, average="micro")}),
        {"num_classes": 6},
    )
    assert isinstance(coll, MetricCollection)
    metrics = list(coll.values())
    assert metrics
    assert all(getattr(m, "num_classes", None) == 6 for m in metrics)


# --------------------------------------------------------------------------- #
# Integration: LightningTrainer.run() derives num_classes from the dataset
# --------------------------------------------------------------------------- #


def test_trainer_run_derives_num_classes_for_lazy_model(tmp_path: Any) -> None:
    n_classes = 5
    trainer = LightningTrainer(
        model=Lazy(_Model, input_size=4),  # num_classes omitted -> derived
        train_set=_CountingDataset(16, num_classes=n_classes),
    )
    # Minimal-ctor pattern: configure the deferred slots post-construction.
    trainer.optimizer = Class(torch.optim.SGD, lr=0.01)
    trainer.train_loader = Class(DataLoader, batch_size=8)
    trainer.lightning = Class(
        L.Trainer,
        max_epochs=1,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        limit_train_batches=1,
        default_root_dir=str(tmp_path),
    )
    trainer.log_system_metrics = False
    trainer.run()
    assert isinstance(trainer.model, _Model)
    assert trainer.model.fc.out_features == n_classes


def test_trainer_run_leaves_plain_dataset_untouched(tmp_path: Any) -> None:
    """A dataset without the hook + an explicit model => no derivation, no error."""
    trainer = LightningTrainer(
        model=Class(_Model, input_size=4, num_classes=2),
        train_set=_PlainDataset(16),
    )
    # Minimal-ctor pattern: configure the deferred slots post-construction.
    trainer.optimizer = Class(torch.optim.SGD, lr=0.01)
    trainer.train_loader = Class(DataLoader, batch_size=8)
    trainer.lightning = Class(
        L.Trainer,
        max_epochs=1,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        limit_train_batches=1,
        default_root_dir=str(tmp_path),
    )
    trainer.log_system_metrics = False
    trainer.run()
    assert trainer.model.fc.out_features == 2

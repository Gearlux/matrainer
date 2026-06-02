"""Tests for the flexible metrics system."""

import pytest
import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import configurable
from dataflux.core import Flux
from dataflux.sample import Sample
from torch.utils.data import DataLoader
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall

from marainer.metrics import MetricPreset
from marainer.trainer import LightningTrainer


@configurable
class TinyModel(nn.Module):
    def __init__(self, input_size: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # type: ignore[no-any-return]


def make_samples(n: int = 32, input_size: int = 4) -> Flux:
    return Flux(source=[Sample(input=torch.randn(input_size), target=i % 2, metadata={}) for i in range(n)])


# ---------------------------------------------------------------------------
# MetricPreset tests
# ---------------------------------------------------------------------------


class TestMetricPreset:
    def test_default_category(self) -> None:
        preset = MetricPreset()
        assert preset.category == "classification"

    def test_classification_multiclass_discovery(self) -> None:
        preset = MetricPreset(category="classification", task="multiclass", num_classes=3)
        metrics = preset.to_metrics()
        assert len(metrics) > 0
        for name, m in metrics.items():
            assert name.startswith("Multiclass")
            assert isinstance(m, Metric)

    def test_classification_binary_discovery(self) -> None:
        preset = MetricPreset(category="classification", task="binary")
        metrics = preset.to_metrics()
        assert len(metrics) > 0
        for name in metrics:
            assert name.startswith("Binary")

    def test_regression_discovery(self) -> None:
        preset = MetricPreset(category="regression")
        metrics = preset.to_metrics()
        assert len(metrics) > 0
        for m in metrics.values():
            assert isinstance(m, Metric)

    def test_segmentation_discovery(self) -> None:
        preset = MetricPreset(category="segmentation", num_classes=5)
        metrics = preset.to_metrics()
        assert len(metrics) > 0

    def test_exclude_filters_metrics(self) -> None:
        preset_all = MetricPreset(category="classification", task="multiclass", num_classes=3)
        preset_excluded = MetricPreset(
            category="classification",
            task="multiclass",
            num_classes=3,
            exclude=["MulticlassAccuracy"],
        )
        metrics_all = preset_all.to_metrics()
        metrics_excluded = preset_excluded.to_metrics()
        assert "MulticlassAccuracy" in metrics_all
        assert "MulticlassAccuracy" not in metrics_excluded
        assert len(metrics_excluded) == len(metrics_all) - 1

    def test_nonscalar_metrics_excluded_by_default(self) -> None:
        preset = MetricPreset(category="classification", task="multiclass", num_classes=3)
        metrics = preset.to_metrics()
        assert "MulticlassConfusionMatrix" not in metrics
        assert "MulticlassROC" not in metrics
        assert "MulticlassPrecisionRecallCurve" not in metrics
        assert "MulticlassStatScores" not in metrics

    def test_num_classes_propagated(self) -> None:
        preset = MetricPreset(category="classification", task="multiclass", num_classes=7)
        metrics = preset.to_metrics()
        acc = metrics.get("MulticlassAccuracy")
        assert acc is not None
        assert acc.num_classes == 7

    def test_average_propagated(self) -> None:
        preset = MetricPreset(
            category="classification",
            task="multiclass",
            num_classes=3,
            average="weighted",
        )
        metrics = preset.to_metrics()
        f1 = metrics.get("MulticlassF1Score")
        assert f1 is not None
        assert f1.average == "weighted"

    def test_available_metrics_returns_names(self) -> None:
        preset = MetricPreset(category="classification", task="multiclass")
        names = preset.available_metrics()
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)
        assert all(n.startswith("Multiclass") for n in names)
        assert "MulticlassConfusionMatrix" not in names  # excluded by default

    def test_invalid_category_raises(self) -> None:
        preset = MetricPreset(category="nonexistent_category_xyz")
        with pytest.raises(ModuleNotFoundError):
            preset.to_metrics()

    def test_empty_task_returns_all_tasks(self) -> None:
        preset = MetricPreset(category="classification", task="", num_classes=3)
        metrics = preset.to_metrics()
        has_binary = any(n.startswith("Binary") for n in metrics)
        has_multiclass = any(n.startswith("Multiclass") for n in metrics)
        assert has_binary
        assert has_multiclass


# ---------------------------------------------------------------------------
# Trainer integration tests
# ---------------------------------------------------------------------------


class TestTrainerMetricsIntegration:
    def _make_trainer(
        self,
        train_metrics: MetricCollection | None = None,
        val_metrics: MetricCollection | None = None,
    ) -> LightningTrainer:
        trainer = LightningTrainer(
            model=TinyModel(),
            train_set=make_samples(16),
        )
        if train_metrics is not None:
            trainer.train_metrics = train_metrics
        if val_metrics is not None:
            trainer.val_metrics = val_metrics
        return trainer

    def test_no_metrics_backward_compat(self) -> None:
        trainer = self._make_trainer()
        assert trainer.train_metrics is None
        assert trainer.val_metrics is None
        flux = Flux(source=make_samples(16))
        loader: DataLoader = DataLoader(flux, batch_size=8)
        loss = trainer.training_step(next(iter(loader)), 0)
        assert loss.dim() == 0

    def test_metrics_set_via_configurable(self) -> None:
        train_mc = MetricCollection(
            [MulticlassAccuracy(num_classes=2), MulticlassF1Score(num_classes=2)],
            prefix="train/",
        )
        val_mc = MetricCollection(
            [MulticlassAccuracy(num_classes=2), MulticlassF1Score(num_classes=2)],
            prefix="val/",
        )
        trainer = self._make_trainer(train_metrics=train_mc, val_metrics=val_mc)
        assert trainer.train_metrics is not None
        assert trainer.val_metrics is not None
        assert isinstance(trainer.train_metrics, MetricCollection)
        assert isinstance(trainer.val_metrics, MetricCollection)

    def test_train_metrics_prefixed(self) -> None:
        train_mc = MetricCollection([MulticlassAccuracy(num_classes=2)], prefix="train/")
        trainer = self._make_trainer(train_metrics=train_mc)
        assert trainer.train_metrics is not None
        assert any("train/" in str(k) for k in trainer.train_metrics)

    def test_val_metrics_prefixed(self) -> None:
        val_mc = MetricCollection([MulticlassAccuracy(num_classes=2)], prefix="val/")
        trainer = self._make_trainer(val_metrics=val_mc)
        assert trainer.val_metrics is not None
        assert any("val/" in str(k) for k in trainer.val_metrics)

    def test_training_step_updates_metrics(self) -> None:
        train_mc = MetricCollection([MulticlassAccuracy(num_classes=2)], prefix="train/")
        trainer = self._make_trainer(train_metrics=train_mc)
        flux = Flux(source=make_samples(16))
        loader: DataLoader = DataLoader(flux, batch_size=8)
        loss = trainer.training_step(next(iter(loader)), 0)
        assert loss.dim() == 0

    def test_validation_step_updates_metrics(self) -> None:
        val_mc = MetricCollection([MulticlassAccuracy(num_classes=2)], prefix="val/")
        trainer = self._make_trainer(val_metrics=val_mc)
        flux = Flux(source=make_samples(16))
        loader: DataLoader = DataLoader(flux, batch_size=8)
        loss = trainer.validation_step(next(iter(loader)), 0)
        assert loss.dim() == 0

    def test_full_epoch_with_metrics(self) -> None:
        train_mc = MetricCollection(
            [
                MulticlassAccuracy(num_classes=2),
                MulticlassPrecision(num_classes=2),
                MulticlassRecall(num_classes=2),
            ],
            prefix="train/",
        )
        trainer = self._make_trainer(train_metrics=train_mc)
        flux = Flux(source=make_samples(32))
        loader: DataLoader = DataLoader(flux, batch_size=16)
        lt = L.Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        lt.fit(model=trainer, train_dataloaders=loader)

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

from marainer.metrics import MetricPreset, build_metric_collection
from marainer.trainer import Trainer


@configurable
class TinyModel(nn.Module):
    def __init__(self, input_size: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # type: ignore[no-any-return]


def make_samples(n: int = 32, input_size: int = 4) -> list[Sample]:
    return [Sample(input=torch.randn(input_size), target=i % 2, metadata={}) for i in range(n)]


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
        assert acc.num_classes == 7  # type: ignore[attr-defined]

    def test_average_propagated(self) -> None:
        preset = MetricPreset(category="classification", task="multiclass", num_classes=3, average="weighted")
        metrics = preset.to_metrics()
        f1 = metrics.get("MulticlassF1Score")
        assert f1 is not None
        assert f1.average == "weighted"  # type: ignore[attr-defined]

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
# build_metric_collection tests
# ---------------------------------------------------------------------------


class TestBuildMetricCollection:
    def test_from_explicit_metrics(self) -> None:
        metrics = [MulticlassAccuracy(num_classes=3), MulticlassF1Score(num_classes=3)]
        collection = build_metric_collection(metrics)
        assert isinstance(collection, MetricCollection)
        assert "MulticlassAccuracy" in collection
        assert "MulticlassF1Score" in collection

    def test_from_preset(self) -> None:
        preset = MetricPreset(category="classification", task="multiclass", num_classes=3)
        collection = build_metric_collection([preset])
        assert isinstance(collection, MetricCollection)
        assert len(collection) > 0

    def test_mixed_preset_and_explicit(self) -> None:
        preset = MetricPreset(
            category="classification", task="multiclass", num_classes=3, exclude=["MulticlassF1Score"]
        )
        explicit_f1 = MulticlassF1Score(num_classes=3, average="weighted")
        collection = build_metric_collection([preset, explicit_f1])
        assert isinstance(collection, MetricCollection)
        assert "MulticlassF1Score" in collection

    def test_prefix_applied(self) -> None:
        metrics = [MulticlassAccuracy(num_classes=3)]
        collection = build_metric_collection(metrics, prefix="train/")
        result = collection(torch.randn(4, 3), torch.randint(0, 3, (4,)))
        assert any(k.startswith("train/") for k in result)

    def test_duplicate_metric_names_handled(self) -> None:
        m1 = MulticlassAccuracy(num_classes=3, average="micro")
        m2 = MulticlassAccuracy(num_classes=3, average="macro")
        collection = build_metric_collection([m1, m2])
        assert "MulticlassAccuracy" in collection
        assert "MulticlassAccuracy_2" in collection

    def test_invalid_item_raises(self) -> None:
        with pytest.raises(TypeError, match="Expected Metric or MetricPreset"):
            build_metric_collection(["not_a_metric"])  # type: ignore[list-item]

    def test_empty_list(self) -> None:
        collection = build_metric_collection([])
        assert isinstance(collection, MetricCollection)
        assert len(collection) == 0


# ---------------------------------------------------------------------------
# Trainer integration tests
# ---------------------------------------------------------------------------


class TestTrainerMetricsIntegration:
    def _make_trainer(self, metrics: list[Metric] | None = None) -> Trainer:
        return Trainer(
            model=TinyModel(),
            train_set=make_samples(16),
            metrics=metrics,
        )

    def test_no_metrics_backward_compat(self) -> None:
        trainer = self._make_trainer(metrics=None)
        assert trainer.train_metrics is None
        assert trainer.val_metrics is None
        flux = Flux(source=make_samples(16))
        loader: DataLoader = DataLoader(flux, batch_size=8)  # type: ignore[arg-type]
        loss = trainer.training_step(next(iter(loader)), 0)
        assert loss.dim() == 0

    def test_setup_metrics_creates_collections(self) -> None:
        metrics = [
            MulticlassAccuracy(num_classes=2),
            MulticlassF1Score(num_classes=2),
        ]
        trainer = self._make_trainer(metrics=metrics)
        trainer._setup_metrics()
        assert trainer.train_metrics is not None
        assert trainer.val_metrics is not None
        assert isinstance(trainer.train_metrics, MetricCollection)
        assert isinstance(trainer.val_metrics, MetricCollection)

    def test_train_metrics_prefixed(self) -> None:
        metrics = [MulticlassAccuracy(num_classes=2)]
        trainer = self._make_trainer(metrics=metrics)
        trainer._setup_metrics()
        assert any("train/" in k for k in trainer.train_metrics)  # type: ignore[union-attr]

    def test_val_metrics_prefixed(self) -> None:
        metrics = [MulticlassAccuracy(num_classes=2)]
        trainer = self._make_trainer(metrics=metrics)
        trainer._setup_metrics()
        assert any("val/" in k for k in trainer.val_metrics)  # type: ignore[union-attr]

    def test_training_step_updates_metrics(self) -> None:
        metrics = [MulticlassAccuracy(num_classes=2)]
        trainer = self._make_trainer(metrics=metrics)
        trainer._setup_metrics()
        flux = Flux(source=make_samples(16))
        loader: DataLoader = DataLoader(flux, batch_size=8)  # type: ignore[arg-type]
        loss = trainer.training_step(next(iter(loader)), 0)
        assert loss.dim() == 0

    def test_validation_step_updates_metrics(self) -> None:
        metrics = [MulticlassAccuracy(num_classes=2)]
        trainer = self._make_trainer(metrics=metrics)
        trainer._setup_metrics()
        flux = Flux(source=make_samples(16))
        loader: DataLoader = DataLoader(flux, batch_size=8)  # type: ignore[arg-type]
        loss = trainer.validation_step(next(iter(loader)), 0)
        assert loss.dim() == 0

    def test_full_epoch_with_metrics(self) -> None:
        metrics = [
            MulticlassAccuracy(num_classes=2),
            MulticlassPrecision(num_classes=2),
            MulticlassRecall(num_classes=2),
        ]
        trainer = self._make_trainer(metrics=metrics)
        trainer._setup_metrics()
        flux = Flux(source=make_samples(32))
        loader: DataLoader = DataLoader(flux, batch_size=16)  # type: ignore[arg-type]
        lt = L.Trainer(max_epochs=1, accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)
        lt.fit(model=trainer, train_dataloaders=loader)

    def test_preset_in_trainer(self) -> None:
        preset = MetricPreset(category="classification", task="multiclass", num_classes=2)
        trainer = self._make_trainer(metrics=[preset])  # type: ignore[list-item]
        trainer._setup_metrics()
        assert trainer.train_metrics is not None
        assert len(trainer.train_metrics) > 0

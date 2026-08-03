"""Tests for the flexible metrics system (MetricPreset + build_metric_collection)."""

import pytest
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from matrainer.torch.metrics import MetricPreset

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


# NOTE: the training-loop integration tests (step functions / full-epoch fit) moved to the
# consumer projects — matrainer no longer owns a Lightning training class; its metrics module is
# just the build_metric_collection + MetricPreset utilities tested here.


# ---------------------------------------------------------------------------
# build_metric_collection — normalizes any metrics spec into a MetricCollection.
# A single metric (incl. a deferred LazyClass from a StreamStudio metric node's
# `class` output) must work, not just a dict / ready-made collection.
# ---------------------------------------------------------------------------


class TestBuildMetricCollection:
    def test_single_lazy_template_wraps_and_flows_with_dims(self) -> None:
        # The reported crash: a single LazyClass (a metric node's `class` output) had no `.items()`.
        from confluid import LazyClass
        from torchmetrics.classification import MulticlassConfusionMatrix

        from matrainer.torch.metrics import build_metric_collection

        mc = build_metric_collection(LazyClass(MulticlassConfusionMatrix), num_classes=10)
        assert isinstance(mc, MetricCollection)
        assert list(mc.keys()) == ["MulticlassConfusionMatrix"]
        # The deferred class sized itself off the run-time num_classes (NOT the 0-valued widget).
        assert mc["MulticlassConfusionMatrix"].num_classes == 10

    def test_single_live_metric_wraps(self) -> None:
        from matrainer.torch.metrics import build_metric_collection

        mc = build_metric_collection(MulticlassAccuracy(num_classes=3), num_classes=3)
        assert list(mc.keys()) == ["MulticlassAccuracy"]

    def test_list_of_templates_wraps_each(self) -> None:
        from confluid import LazyClass
        from torchmetrics.classification import MulticlassConfusionMatrix

        from matrainer.torch.metrics import build_metric_collection

        mc = build_metric_collection(
            [LazyClass(MulticlassAccuracy), LazyClass(MulticlassConfusionMatrix)], num_classes=4
        )
        assert set(mc.keys()) == {"MulticlassAccuracy", "MulticlassConfusionMatrix"}

    def test_dict_of_templates_still_flows(self) -> None:
        from confluid import LazyClass

        from matrainer.torch.metrics import build_metric_collection

        mc = build_metric_collection(
            {"acc": LazyClass(MulticlassAccuracy), "f1": LazyClass(MulticlassF1Score, average="macro")},
            num_classes=2,
        )
        assert set(mc.keys()) == {"acc", "f1"}

    def test_ready_made_collection_passes_through_unchanged(self) -> None:
        from matrainer.torch.metrics import build_metric_collection

        ready = MetricCollection({"acc": MulticlassAccuracy(num_classes=2)})
        assert build_metric_collection(ready, num_classes=2) is ready

    def test_single_metric_result_clones_per_split(self) -> None:
        # The caller clones with a `test/` prefix; that must work on the single-metric result.
        from confluid import LazyClass
        from torchmetrics.classification import MulticlassConfusionMatrix

        from matrainer.torch.metrics import build_metric_collection

        mc = build_metric_collection(LazyClass(MulticlassConfusionMatrix), num_classes=5).clone(prefix="test/")
        assert list(mc.keys()) == ["test/MulticlassConfusionMatrix"]

"""Tests for marainer.presets — MetricPreset with !ref:obj.method() integration."""

import pytest
from confluid import get_registry, load
from confluid.loader import _register_constructors
from torchmetrics import Metric, MetricCollection

from marainer.presets import MetricPreset


@pytest.fixture(autouse=True)
def setup_registry() -> None:
    get_registry().clear()
    _register_constructors()
    # Re-import to re-register MetricPreset
    import marainer.presets  # noqa: F401


class TestMetricPresetBasic:
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

    def test_nonscalar_metrics_excluded_by_default(self) -> None:
        preset = MetricPreset(category="classification", task="multiclass", num_classes=3)
        metrics = preset.to_metrics()
        assert "MulticlassConfusionMatrix" not in metrics

    def test_num_classes_propagated(self) -> None:
        preset = MetricPreset(category="classification", task="multiclass", num_classes=7)
        metrics = preset.to_metrics()
        acc = metrics.get("MulticlassAccuracy")
        assert acc is not None
        assert acc.num_classes == 7

    def test_available_metrics_returns_names(self) -> None:
        preset = MetricPreset(category="classification", task="multiclass")
        names = preset.available_metrics()
        assert isinstance(names, list)
        assert all(n.startswith("Multiclass") for n in names)

    def test_invalid_category_raises(self) -> None:
        preset = MetricPreset(category="nonexistent_category_xyz")
        with pytest.raises(ModuleNotFoundError):
            preset.to_metrics()


class TestMetricPresetYAML:
    """Test MetricPreset integrated with !ref:obj.method() in YAML."""

    def test_preset_to_metrics_via_ref(self) -> None:
        """!ref:preset.to_metrics() produces a dict of Metric instances."""
        yaml_str = """
import: marainer.presets

preset: !class:marainer.presets.MetricPreset()
  category: classification
  task: multiclass
  num_classes: 3
result: !ref:preset.to_metrics()
"""
        data = load(yaml_str)
        assert isinstance(data["result"], dict)
        assert len(data["result"]) > 0
        for m in data["result"].values():
            assert isinstance(m, Metric)

    def test_preset_in_metric_collection(self) -> None:
        """MetricPreset.to_metrics() feeds into MetricCollection via !ref:."""
        yaml_str = """
import: marainer.presets

num_classes: 3

classification_metrics: !class:marainer.presets.MetricPreset()
  category: classification
  task: multiclass
  num_classes: !ref:num_classes

train_metrics: !class:torchmetrics.MetricCollection()
  metrics: !ref:classification_metrics.to_metrics()
  prefix: "train/"
"""
        data = load(yaml_str)
        assert isinstance(data["train_metrics"], MetricCollection)
        assert len(data["train_metrics"]) > 0
        # Verify prefix applied
        import torch

        dummy_preds = torch.randn(4, 3)
        dummy_target = torch.randint(0, 3, (4,))
        result = data["train_metrics"](dummy_preds, dummy_target)
        assert all(str(k).startswith("train/") for k in result)

    def test_train_val_independent_collections(self) -> None:
        """Train and val MetricCollections from same preset are independent."""
        yaml_str = """
import: marainer.presets

num_classes: 3

classification_metrics: !class:marainer.presets.MetricPreset()
  category: classification
  task: multiclass
  num_classes: !ref:num_classes

train_metrics: !class:torchmetrics.MetricCollection()
  metrics: !ref:classification_metrics.to_metrics()
  prefix: "train/"

val_metrics: !class:torchmetrics.MetricCollection()
  metrics: !ref:classification_metrics.to_metrics()
  prefix: "val/"
"""
        data = load(yaml_str)
        assert isinstance(data["train_metrics"], MetricCollection)
        assert isinstance(data["val_metrics"], MetricCollection)

        # Verify different prefixes
        import torch

        dummy_preds = torch.randn(4, 3)
        dummy_target = torch.randint(0, 3, (4,))
        train_result = data["train_metrics"](dummy_preds, dummy_target)
        val_result = data["val_metrics"](dummy_preds, dummy_target)
        assert all(str(k).startswith("train/") for k in train_result)
        assert all(str(k).startswith("val/") for k in val_result)

        # Verify independence — different MetricCollection objects
        assert data["train_metrics"] is not data["val_metrics"]

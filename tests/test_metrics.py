"""Tests for the flexible metrics system (MetricPreset + build_metric_collection)."""

from pathlib import Path
from typing import Any, cast

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
# A single metric (incl. a deferred PartialClass from a StreamStudio metric node's
# `class` output) must work, not just a dict / ready-made collection.
# ---------------------------------------------------------------------------


class TestBuildMetricCollection:
    def test_single_lazy_template_wraps_and_flows_with_dims(self) -> None:
        # The reported crash: a single PartialClass (a metric node's `class` output) had no `.items()`.
        from confluid import PartialClass
        from torchmetrics.classification import MulticlassConfusionMatrix

        from matrainer.torch.metrics import build_metric_collection

        mc = build_metric_collection(PartialClass(MulticlassConfusionMatrix), num_classes=10)
        assert isinstance(mc, MetricCollection)
        assert list(mc.keys()) == ["MulticlassConfusionMatrix"]
        # The deferred class sized itself off the run-time num_classes (NOT the 0-valued widget).
        assert mc["MulticlassConfusionMatrix"].num_classes == 10

    def test_single_live_metric_wraps(self) -> None:
        from matrainer.torch.metrics import build_metric_collection

        mc = build_metric_collection(MulticlassAccuracy(num_classes=3), num_classes=3)
        assert list(mc.keys()) == ["MulticlassAccuracy"]

    def test_list_of_templates_wraps_each(self) -> None:
        from confluid import PartialClass
        from torchmetrics.classification import MulticlassConfusionMatrix

        from matrainer.torch.metrics import build_metric_collection

        mc = build_metric_collection(
            [PartialClass(MulticlassAccuracy), PartialClass(MulticlassConfusionMatrix)], num_classes=4
        )
        assert set(mc.keys()) == {"MulticlassAccuracy", "MulticlassConfusionMatrix"}

    def test_dict_of_templates_still_flows(self) -> None:
        from confluid import PartialClass

        from matrainer.torch.metrics import build_metric_collection

        mc = build_metric_collection(
            {"acc": PartialClass(MulticlassAccuracy), "f1": PartialClass(MulticlassF1Score, average="macro")},
            num_classes=2,
        )
        assert set(mc.keys()) == {"acc", "f1"}

    def test_ready_made_collection_passes_through_unchanged(self) -> None:
        from matrainer.torch.metrics import build_metric_collection

        ready = MetricCollection({"acc": MulticlassAccuracy(num_classes=2)})
        assert build_metric_collection(ready, num_classes=2) is ready

    def test_single_metric_result_clones_per_split(self) -> None:
        # The caller clones with a `test/` prefix; that must work on the single-metric result.
        from confluid import PartialClass
        from torchmetrics.classification import MulticlassConfusionMatrix

        from matrainer.torch.metrics import build_metric_collection

        mc = build_metric_collection(PartialClass(MulticlassConfusionMatrix), num_classes=5).clone(prefix="test/")
        assert list(mc.keys()) == ["test/MulticlassConfusionMatrix"]


# ---------------------------------------------------------------------------
# The sizing dimension is applied PER TEMPLATE, not blanket
# ---------------------------------------------------------------------------


class TestSizingKwargs:
    """``flow_dims`` reaches only the templates that can take it and have not pinned it.

    The blanket form worked for classification by luck — every ``torchmetrics.classification``
    metric happens to take ``num_classes``. An image task's ``data_range`` does not: three of the
    seven full-reference metrics accept it and four do not, so the blanket form made a valid
    metric break the whole collection.
    """

    def test_a_dimension_the_metric_cannot_take_is_not_passed(self) -> None:
        """``SpectralAngleMapper`` has no ``data_range``; passing one raises inside confluid."""
        from confluid import PartialClass
        from torchmetrics.image import SpectralAngleMapper

        from matrainer.torch.metrics import build_metric_collection

        mc = build_metric_collection({"sam": PartialClass(SpectralAngleMapper)}, data_range=1.0)
        assert isinstance(mc["sam"], SpectralAngleMapper)

    def test_a_dimension_the_metric_requires_still_reaches_it(self) -> None:
        """``PeakSignalNoiseRatio(data_range)`` has NO default — an un-injected template cannot
        even be built, which is torchmetrics deliberately refusing to guess a scale."""
        from confluid import PartialClass
        from torchmetrics.image import PeakSignalNoiseRatio

        from matrainer.torch.metrics import build_metric_collection

        mc = build_metric_collection({"psnr": PartialClass(PeakSignalNoiseRatio)}, data_range=255.0)
        assert float(cast(Any, mc["psnr"]).data_range) == 255.0

    def test_a_template_that_pins_the_dimension_keeps_its_own_value(self) -> None:
        """The injection is a FLOOR: a runtime kwarg would otherwise OVERRIDE the marker's own,
        making an explicitly-written value unwritable."""
        from confluid import PartialClass
        from torchmetrics.image import PeakSignalNoiseRatio

        from matrainer.torch.metrics import build_metric_collection

        mc = build_metric_collection({"psnr": PartialClass(PeakSignalNoiseRatio, data_range=2.0)}, data_range=1.0)
        assert float(cast(Any, mc["psnr"]).data_range) == 2.0

    def test_a_mixed_collection_builds_whole(self) -> None:
        """The case that motivated this: metrics that take the dimension beside ones that do not."""
        from confluid import PartialClass
        from torchmetrics.image import PeakSignalNoiseRatio, SpectralAngleMapper, StructuralSimilarityIndexMeasure

        from matrainer.torch.metrics import build_metric_collection

        mc = build_metric_collection(
            {
                "psnr": PartialClass(PeakSignalNoiseRatio),
                "ssim": PartialClass(StructuralSimilarityIndexMeasure),
                "sam": PartialClass(SpectralAngleMapper),
            },
            data_range=1.0,
        )
        assert set(mc.keys()) == {"psnr", "ssim", "sam"}

    def test_the_classification_dimension_still_propagates(self) -> None:
        """The pre-existing behaviour every other task depends on is unchanged."""
        from confluid import PartialClass

        from matrainer.torch.metrics import build_metric_collection

        mc = build_metric_collection({"acc": PartialClass(MulticlassAccuracy)}, num_classes=7)
        assert mc["acc"].num_classes == 7


# ---------------------------------------------------------------------------
# write_collection_sidecar — the compute-once sidecar composition
# ---------------------------------------------------------------------------
class TestWriteCollectionSidecar:
    def _collection(self) -> object:
        import torch
        from torchmetrics import MetricCollection
        from torchmetrics.classification import MulticlassAccuracy

        collection = MetricCollection({"accuracy": MulticlassAccuracy(num_classes=2)}).clone(prefix="val/")
        collection.update(torch.tensor([[0.9, 0.1], [0.2, 0.8]]), torch.tensor([0, 1]))
        return collection

    def test_a_live_collection_is_computed_and_written(self, tmp_path: Path) -> None:
        import json

        from matrainer.torch.metrics import write_collection_sidecar

        path = tmp_path / "metrics.json"
        write_collection_sidecar(str(path), self._collection(), "exp", "run")
        payload = json.loads(path.read_text())
        assert payload["metrics"]["val/accuracy"] == 1.0
        assert payload["experiment_name"] == "exp" and payload["run_name"] == "run"

    def test_a_non_collection_slot_is_a_silent_no_op(self, tmp_path: Path) -> None:
        """A spec-holding or None slot means the run legitimately built no metrics."""
        from matrainer.torch.metrics import write_collection_sidecar

        path = tmp_path / "metrics.json"
        write_collection_sidecar(str(path), {"still": "a spec"}, "exp", "run")
        write_collection_sidecar(str(path), None, "exp", "run")
        assert not path.exists()

    def test_the_reportable_filter_runs_before_the_scalar_drop(self, tmp_path: Path) -> None:
        """A task's strip rule (detection MAP's `classes`) applies to the sidecar too."""
        import json

        from matrainer.torch.metrics import write_collection_sidecar

        path = tmp_path / "metrics.json"
        write_collection_sidecar(
            str(path),
            self._collection(),
            "exp",
            "run",
            reportable=lambda values: {k: v for k, v in values.items() if k != "val/accuracy"},
        )
        assert json.loads(path.read_text())["metrics"] == {}


def test_the_metrics_spec_alias_is_shared_from_here() -> None:
    """Declared ONCE (it existed verbatim in eight-plus consumer modules); both subpackage
    surfaces re-export it with the other metric helpers."""
    import matrainer.torch as torch_pkg
    from matrainer.torch import MetricsSpec
    from matrainer.torch.metrics import MetricsSpec as canonical

    assert "MetricsSpec" in torch_pkg.__all__
    assert canonical is MetricsSpec

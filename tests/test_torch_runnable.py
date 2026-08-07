"""The torch FAMILY trainer base (`matrainer.torch.runnable.TorchTrainerRunnable`).

The layer where the framework types are pinned: the typed slots, the sentinel resolution, the
`_metrics_built`-guarded builders, the sidecar write, and the shared outputs — everything the
lightning / bare-torch / fastai / huggingface machinery bases inherit rather than restate.
"""

from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402
from confluid import Fluid, LazyClass  # noqa: E402
from torchmetrics import MetricCollection  # noqa: E402
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix  # noqa: E402

from matrainer.torch.runnable import TorchTrainerRunnable  # noqa: E402


class _Stub(TorchTrainerRunnable):
    """The smallest torch-family concrete: entrypoints recorded, hooks minimal."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls: List[str] = []

    def initialize(self) -> None:
        self.calls.append("initialize")

    def fit(self) -> None:
        self.calls.append("fit")

    def evaluate(self) -> None:
        self.calls.append("evaluate")

    def test(self) -> None:
        self.calls.append("test")

    def predict(self) -> None:
        self.calls.append("predict")

    @property
    def computed_metrics(self) -> Dict[str, float]:
        return self._computed_eval_metrics()


class _WithDefaults(_Stub):
    def _default_loss(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def _default_metrics(self) -> Any:
        return {"accuracy": LazyClass(MulticlassAccuracy)}

    def _metric_flow_dims(self) -> Dict[str, "int | float"]:
        return {"num_classes": 3}


# ---------------------------------------------------------------------------
# typed slots + the autograd marker
# ---------------------------------------------------------------------------
def test_the_family_slots_store_verbatim() -> None:
    model = nn.Identity()
    runnable = _Stub(model=model, metrics={"a": 1}, val_metrics={"b": 2})
    assert runnable.model is model
    assert runnable.loss_fn is None and runnable._metrics_built is False
    assert runnable.metrics == {"a": 1} and runnable.val_metrics == {"b": 2}
    assert runnable.train_metrics is None and runnable.test_metrics is None


def test_needs_autograd_is_per_task() -> None:
    assert _Stub(task="fit").__needs_autograd__ is True
    assert _Stub(task="test").__needs_autograd__ is False
    assert _Stub(task="predict").__needs_autograd__ is False


def test_the_targets_default_names_the_two_ways_out() -> None:
    """Not abstract, deliberately — a detection task overrides the steps and never calls it —
    so the default must explain itself instead of being a bare NotImplementedError."""
    with pytest.raises(NotImplementedError, match="task base|overrides the step"):
        _Stub()._targets({"target": 1})


# ---------------------------------------------------------------------------
# sentinel resolution
# ---------------------------------------------------------------------------
def test_resolve_task_defaults_fills_the_sentinels_via_the_hooks() -> None:
    runnable = _WithDefaults()
    runnable._resolve_task_defaults()
    assert isinstance(runnable.loss_fn, nn.CrossEntropyLoss)
    assert runnable.metrics is not None and "accuracy" in runnable.metrics


def test_a_user_wired_value_is_never_replaced() -> None:
    """That is what the `None` sentinel buys — "not chosen" vs "chosen to be the default"."""
    wired_loss = nn.L1Loss()
    runnable = _WithDefaults(metrics={"mine": LazyClass(MulticlassAccuracy, num_classes=3)})
    runnable.loss_fn = wired_loss
    runnable._resolve_task_defaults()
    assert runnable.loss_fn is wired_loss and set(runnable.metrics) == {"mine"}


def test_the_hook_defaults_leave_the_sentinels_unresolved_and_resolution_idempotent() -> None:
    runnable = _Stub()
    runnable._resolve_task_defaults()
    assert runnable.loss_fn is None and runnable.metrics is None


# ---------------------------------------------------------------------------
# the metric builders
# ---------------------------------------------------------------------------
def test_build_fit_metrics_builds_all_three_prefixed_once() -> None:
    runnable = _WithDefaults(val_metrics={"confusion": LazyClass(MulticlassConfusionMatrix)})
    runnable._resolve_task_defaults()
    runnable._build_fit_metrics()
    assert isinstance(runnable.train_metrics, MetricCollection)
    assert set(runnable.val_metrics.keys()) == {"val/accuracy", "val/confusion"}, "split spec merges over shared"
    assert set(runnable.test_metrics.keys()) == {"test/accuracy"}
    # The `_metrics_built` guard: a second initialize must NOT re-prefix into `val/val/...`.
    runnable._build_fit_metrics()
    assert set(runnable.val_metrics.keys()) == {"val/accuracy", "val/confusion"}


def test_build_fit_metrics_can_skip_the_train_split() -> None:
    """A task with no train slot (detection: train mode returns only a loss dict)."""
    runnable = _WithDefaults()
    runnable._resolve_task_defaults()
    runnable._build_fit_metrics(include_train=False)
    assert runnable.train_metrics is None
    assert isinstance(runnable.val_metrics, MetricCollection)


def test_build_eval_metrics_builds_only_the_scored_split() -> None:
    runnable = _WithDefaults(task="evaluate")
    runnable._resolve_task_defaults()
    runnable._build_eval_metrics()
    assert isinstance(runnable.val_metrics, MetricCollection)
    assert runnable.test_metrics is None and runnable.train_metrics is None


def test_the_flow_dims_size_the_templates() -> None:
    runnable = _WithDefaults()
    runnable._resolve_task_defaults()
    runnable._build_fit_metrics()
    assert runnable.val_metrics["val/accuracy"].num_classes == 3


# ---------------------------------------------------------------------------
# reporting plumbing + outputs
# ---------------------------------------------------------------------------
def _scored_stub(task: str = "test") -> _Stub:
    runnable = _WithDefaults(task=task)
    runnable._resolve_task_defaults()
    runnable._build_eval_metrics()
    slot = runnable.val_metrics if task == "evaluate" else runnable.test_metrics
    slot.update(torch.eye(3), torch.tensor([0, 1, 2]))
    return runnable


def test_computed_eval_metrics_reads_the_scored_split_through_reportable() -> None:
    runnable = _scored_stub("test")
    assert runnable.computed_metrics == {"test/accuracy": 1.0}


def test_computed_metrics_raises_before_the_run_built_anything() -> None:
    with pytest.raises(RuntimeError, match="after run"):
        _ = _Stub(task="test").computed_metrics


def test_the_sidecar_write_goes_through_the_shared_helper(tmp_path: Any) -> None:
    import json

    runnable = _scored_stub("test")
    runnable.metrics_sidecar_path = str(tmp_path / "metrics.json")
    runnable._write_metrics_sidecar(runnable.test_metrics)
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload["metrics"] == {"test/accuracy": 1.0}
    assert payload["experiment_name"] == "matrainer" and payload["run_name"] == runnable.run_name


def test_the_sidecar_skips_a_slot_that_was_never_built(tmp_path: Any) -> None:
    runnable = _Stub()
    runnable.metrics_sidecar_path = str(tmp_path / "metrics.json")
    runnable._write_metrics_sidecar(runnable.test_metrics)  # still a spec (None) — silent no-op
    assert not (tmp_path / "metrics.json").exists()


def test_the_reportable_hook_filters_every_reporting_path(tmp_path: Any) -> None:
    class _Stripping(_WithDefaults):
        def _reportable(self, values: Any) -> Any:
            return {k: v for k, v in values.items() if not k.endswith("accuracy")}

    runnable = _Stripping(task="test")
    runnable._resolve_task_defaults()
    runnable._build_eval_metrics()
    runnable.test_metrics.update(torch.eye(3), torch.tensor([0, 1, 2]))
    assert runnable.computed_metrics == {}


def test_trained_model_raises_while_deferred_and_returns_the_live_module() -> None:
    live = nn.Identity()
    assert _Stub(model=live).trained_model is live
    deferred = _Stub(model=LazyClass(nn.Identity))
    assert isinstance(deferred.model, Fluid)
    with pytest.raises(RuntimeError, match="still a deferred Fluid"):
        _ = deferred.trained_model

"""The bare-torch machinery base (`matrainer.torch.loop.TorchLoopRunnable`), driven end to end.

A minimal regression task implements the hooks and the base does everything else — these pin
the loop machinery the task projects inherit: device bracketing around the task's own
`initialize`, the epoch driver, the per-split scoring, the sidecar, the history, and the
`_after_fit` hook's place in `fit()`. Task-specific behaviour is pinned per consumer.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, cast

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from confluid import PartialClass, configurable, flow  # noqa: E402
from recordstream import Stream, batch_tensor  # noqa: E402
from torchmetrics import MetricCollection  # noqa: E402
from torchmetrics.regression import MeanAbsoluteError  # noqa: E402

from matrainer.torch import TorchLoopRunnable  # noqa: E402
from matrainer.torch.metrics import MetricsSpec  # noqa: E402


class _MiniRestorer(TorchLoopRunnable):
    """The smallest real task: learn the identity map on 4-vectors, on an owned loop."""

    log_label = "mini"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("device", "cpu")  # deterministic + fast for the suite
        super().__init__(**kwargs)
        self.after_fit_calls: List[Any] = []

    def initialize(self) -> None:
        self.model = flow(self.model)
        self._resolve_task_defaults()
        if self.task == "fit":
            self._build_fit_metrics()
        else:
            self._build_eval_metrics()

    def _targets(self, batch: Any) -> torch.Tensor:
        return batch_tensor(batch, self.target_key, self.device)

    def _default_loss(self) -> nn.Module:
        return nn.MSELoss()

    def _default_metrics(self) -> MetricsSpec:
        return {"mae": PartialClass(MeanAbsoluteError)}

    def _prediction_output(self, preds: Any) -> Dict[str, Any]:
        return {"restored": preds}

    def _after_fit(self, trainer: Any, train_ds: Any, val_ds: Any, test_ds: Any) -> None:
        self.after_fit_calls.append((trainer, train_ds, val_ds, test_ds))


class _SinkRecorder:
    def __init__(self) -> None:
        self.writes: List[Any] = []

    def write(self, prediction: Any, metadata: Any) -> None:
        self.writes.append((prediction, metadata))

    def close(self) -> None:
        self.closed = True


def _records(n: int = 8) -> Stream:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        x = rng.normal(size=4).astype("float32")
        rows.append({"image": x, "target": x.copy(), "idx": i})
    return Stream(source=rows)


def _mini(**kwargs: Any) -> _MiniRestorer:
    kwargs.setdefault("model", nn.Linear(4, 4))
    kwargs.setdefault("batch_size", 4)
    kwargs.setdefault("max_epochs", 2)
    return _MiniRestorer(**kwargs)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------
def test_fit_runs_end_to_end_and_the_loss_descends() -> None:
    runnable = _mini(train_set=_records(), max_epochs=8)
    runnable.val_set = _records(4)
    runnable.run()
    history = runnable.history
    assert history["train/loss"][-1] < history["train/loss"][0], "the loop must actually train"
    assert "val/mae" in runnable.computed_metrics and "val/loss" in runnable.computed_metrics
    assert len(history["val/mae"]) == 8, "one entry per epoch — what a Trainer's logger would keep"


def test_the_device_brackets_wrap_the_task_initialize() -> None:
    """`run()` resolves the device BEFORE the task's initialize and moves model+loss AFTER."""
    runnable = _mini(train_set=_records(), max_epochs=1)
    runnable.run()
    assert runnable.device.type == "cpu"
    assert isinstance(runnable.loss_fn, nn.MSELoss)
    assert next(cast(nn.Module, runnable.model).parameters()).device.type == "cpu"


def test_after_fit_runs_before_the_post_fit_test() -> None:
    runnable = _mini(train_set=_records(), test_set=_records(4), max_epochs=1)
    runnable.run()
    assert len(runnable.after_fit_calls) == 1
    trainer, _, _, test_ds = runnable.after_fit_calls[0]
    assert trainer is runnable, "the runnable IS the driver on an owned loop"
    assert test_ds is not None
    assert "test/mae" in runnable.computed_metrics, "the post-fit test pass scored the held-out split"


def test_a_checkpoint_is_a_plain_state_dict(tmp_path: Path) -> None:
    runnable = _mini(train_set=_records(), max_epochs=1, checkpoint_dir=str(tmp_path))
    runnable.run()
    weights = torch.load(tmp_path / "model.pt", weights_only=True)
    assert "weight" in weights and "bias" in weights, "no framework format — a bare Linear state_dict"


# ---------------------------------------------------------------------------
# evaluate / test / predict
# ---------------------------------------------------------------------------
def test_a_test_run_scores_and_writes_the_sidecar(tmp_path: Path) -> None:
    runnable = _mini(test_set=_records(4), task="test")
    runnable.metrics_sidecar_path = str(tmp_path / "metrics.json")
    runnable.run()
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert "test/mae" in payload["metrics"] and "test/loss" in payload["metrics"]


def test_an_evaluate_run_scores_the_val_split() -> None:
    runnable = _mini(task="evaluate")
    runnable.val_set = _records(4)
    runnable.run()
    assert set(runnable.computed_metrics) == {"val/loss", "val/mae"}


def test_predict_writes_one_prediction_per_record_with_its_metadata() -> None:
    runnable = _mini(test_set=_records(4), task="predict")
    sink = _SinkRecorder()
    runnable.predictions_sink = sink
    runnable.run()
    assert len(sink.writes) == 4
    assert sink.writes[2][1] == {"idx": 2}, "metadata pairs per record, input/target excluded"
    assert getattr(sink, "closed", False)


# ---------------------------------------------------------------------------
# the batch-hook seam (what a loss-in-forward task overrides)
# ---------------------------------------------------------------------------
def test_an_eval_batch_returning_none_omits_the_loss_key() -> None:
    """Detection's contract: eval mode yields predictions, not losses — no val/loss key."""

    class _LossLess(_MiniRestorer):
        def _eval_batch(self, batch: Any, collection: Any) -> None:
            if isinstance(collection, MetricCollection):
                x = batch_tensor(batch, self.input_key, self.device)
                collection.update(cast(nn.Module, self.model)(x), self._targets(batch))
            return None

    runnable = _LossLess(model=nn.Linear(4, 4), test_set=_records(4), task="test", batch_size=4)
    runnable.run()
    assert "test/loss" not in runnable.computed_metrics
    assert "test/mae" in runnable.computed_metrics


def test_persistent_workers_is_overridable_and_still_derives_by_default() -> None:
    """The knob a HOST-scoped config sets: `persistent_workers: false` with workers kept.

    macOS terminates persistent workers slowly enough that a short run spends longer
    stopping than training, and no derivation from `num_workers` can see that."""
    derived = _mini(num_workers=2)
    forced = _mini(num_workers=2, persistent_workers=False)
    for split in ("train_loader", "val_loader", "test_loader"):
        assert getattr(derived, split).kwargs["persistent_workers"] is True, split
        marker = getattr(forced, split)
        assert marker.kwargs["persistent_workers"] is False, split
        assert marker.kwargs["num_workers"] == 2, split


def test_the_run_hands_mps_low_watermark_to_torch_before_the_task_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Before `initialize` builds the model: torch reads the value when the GPU is first used."""
    seen: List[Any] = []
    monkeypatch.setattr("matrainer.torch.runnable.apply_mps_low_watermark", seen.append)
    runnable = _mini(train_set=_records(), task="fit", max_epochs=1)
    runnable.mps_low_watermark = 0.5
    at_initialize: List[List[Any]] = []
    original = runnable.initialize

    def initialize() -> None:
        at_initialize.append(list(seen))
        original()

    monkeypatch.setattr(runnable, "initialize", initialize)
    runnable.run()
    assert seen == [0.5] and at_initialize == [[0.5]]


@configurable
class _FlatRestorer(_MiniRestorer):
    """A real trainer's shape: every setting named in its own constructor, no ``**kwargs``."""

    def __init__(self, *, model: Any = None, max_epochs: int = 1, batch_size: int = 4) -> None:
        super().__init__(model=model, max_epochs=max_epochs, batch_size=batch_size)


def test_mps_low_watermark_is_a_setting_a_config_names() -> None:
    """A body slot of the torch family base: no concrete constructor lists it, every config can set it."""
    from confluid import load, to_pydantic

    assert _mini().mps_low_watermark is None
    assert "mps_low_watermark" in to_pydantic(_FlatRestorer).model_fields
    runnable = load("runnable: !class:tests.test_torch_loop._FlatRestorer\n  mps_low_watermark: 0.5\n")["runnable"]
    assert isinstance(runnable, _FlatRestorer) and runnable.mps_low_watermark == 0.5

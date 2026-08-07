"""The Lightning machinery base (`matrainer.lightning.LightningRunnable`), driven end to end.

A minimal regression task implements the hooks and the base does everything else — these pin
the machinery the task projects inherit: dispatch through the inherited markers, the loader
slots, the metric building + epoch reporting, the sidecar, the outputs, and the `_after_fit`
hook's place in `fit()`. Task-specific behaviour is pinned in each consumer's own suite.
"""

from typing import Any, Dict, List

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")

import pytorch_lightning as L  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from confluid import LazyClass, flow  # noqa: E402
from recordstream import Stream, batch_tensor  # noqa: E402
from torchmetrics import MetricCollection  # noqa: E402
from torchmetrics.regression import MeanAbsoluteError  # noqa: E402

from matrainer.lightning import LightningRunnable  # noqa: E402
from matrainer.torch.metrics import MetricsSpec  # noqa: E402


class _MiniRestorer(LightningRunnable):
    """The smallest real task: learn the identity map on 4-vectors."""

    log_label = "mini"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.after_fit_calls: List[Any] = []
        # A fast, quiet Trainer for the suite; a consumer's config replaces this slot the
        # same way (whole-value), which is the override path the loaders test pins too.
        self.lightning = LazyClass(
            L.Trainer,
            max_epochs=1,
            accelerator="cpu",
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            log_every_n_steps=1,
        )
        self.log_system_metrics = False

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
        return {"mae": LazyClass(MeanAbsoluteError)}

    def _prediction_output(self, preds: Any) -> Dict[str, Any]:
        # The task's prediction CONTRACT (a real task returns restoration_output & friends);
        # a batched mapping is what `_write_predictions` slices per record.
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
    return _MiniRestorer(**kwargs)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------
def test_fit_runs_end_to_end_with_the_inherited_dispatch() -> None:
    runnable = _mini(train_set=_records(), task="fit")
    runnable.val_set = _records(4)
    runnable.run()
    # The per-split collections were built prefixed, and the fit half of the output reads
    # Lightning's own callback metrics.
    assert isinstance(runnable.train_metrics, MetricCollection)
    assert set(runnable.val_metrics.keys()) == {"val/mae"}
    computed = runnable.computed_metrics
    assert "train/loss" in computed and "val/mae" in computed


def test_after_fit_runs_between_training_and_the_post_fit_test() -> None:
    runnable = _mini(train_set=_records(), test_set=_records(4), task="fit")
    runnable.run()
    assert len(runnable.after_fit_calls) == 1
    trainer, train_ds, val_ds, test_ds = runnable.after_fit_calls[0]
    assert isinstance(trainer, L.Trainer)
    assert val_ds is None and test_ds is not None


def test_fit_without_a_train_set_fails_loudly_with_the_class_name() -> None:
    runnable = _mini(task="fit")
    runnable.initialize()
    with pytest.raises(ValueError, match=r"_MiniRestorer\.fit\(\) requires a 'train_set'"):
        runnable.fit()


# ---------------------------------------------------------------------------
# evaluate / test
# ---------------------------------------------------------------------------
def test_a_test_run_scores_and_writes_the_sidecar(tmp_path: Any) -> None:
    import json

    runnable = _mini(test_set=_records(4), task="test")
    runnable.metrics_sidecar_path = str(tmp_path / "metrics.json")
    runnable.run()
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert "test/mae" in payload["metrics"]
    assert runnable.computed_metrics.keys() == payload["metrics"].keys()


def test_an_evaluate_run_scores_the_val_split() -> None:
    runnable = _mini(task="evaluate")
    runnable.val_set = _records(4)
    runnable.run()
    assert set(runnable.computed_metrics) == {"val/mae"}


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------
def test_predict_writes_one_prediction_per_record_with_its_metadata() -> None:
    runnable = _mini(test_set=_records(4), task="predict")
    sink = _SinkRecorder()
    runnable.predictions_sink = sink  # type: ignore[assignment]
    runnable.run()
    assert len(sink.writes) == 4
    assert sink.writes[2][1] == {"idx": 2}, "metadata pairs per record, input/target excluded"
    assert getattr(sink, "closed", False), "the sink is closed at the end of the run"


# ---------------------------------------------------------------------------
# machinery details
# ---------------------------------------------------------------------------
def test_the_loader_slots_carry_the_batch_size_and_flow_with_the_dataset() -> None:
    runnable = _mini()
    loader = flow(runnable.val_loader, dataset=_records(8))
    assert loader.batch_size == 4
    assert len(list(loader)) == 2


def test_the_trainer_slot_is_replaceable_and_gains_only_the_loggers() -> None:
    """A wired `lightning:` block keeps every knob it set — the runtime kwarg adds loggers."""
    runnable = _mini()
    trainer = runnable._build_trainer()
    assert trainer.max_epochs == 1, "the stored slot's knobs survive"
    assert trainer.logger is None or trainer.loggers == [], "no trackers wired -> no loggers"

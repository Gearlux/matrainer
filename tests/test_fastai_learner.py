"""The fastai machinery base (`matrainer.fastai.learner.FastaiRunnable`), driven end to end.

A minimal regression task implements the hooks and the base does everything else — these pin
the machinery the task projects inherit: the Learner/DataLoaders assembly, the positional
`_to_tuple` built on the shared `_targets` hook, the Recorder read-back into canonical names,
the fastai-shaped sentinel resolution, the alias-first metric resolution, and `_after_fit`'s
place in `fit()`. Task-specific behaviour is pinned per consumer.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

pytest.importorskip("fastai")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from confluid import flow  # noqa: E402
from recordstream import Stream, batch_tensor  # noqa: E402

from matrainer.fastai import FastaiRunnable  # noqa: E402


class _MiniRestorer(FastaiRunnable):
    """The smallest real task: learn the identity map on 4-vectors, under a Learner."""

    log_label = "mini"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.after_fit_calls: List[Any] = []

    def initialize(self) -> None:
        self.train_set = self._prepare(self.train_set)
        self.val_set = self._prepare(self.val_set)
        self.test_set = self._prepare(self.test_set)
        self.model = flow(self.model)
        self._resolve_task_defaults()

    def _targets(self, batch: Any) -> torch.Tensor:
        return batch_tensor(batch, self.target_key, self.device)

    def _default_loss(self) -> nn.Module:
        return nn.MSELoss()

    def _default_fastai_metrics(self) -> List[str]:
        return ["mse"]

    def _metric_aliases(self) -> Dict[str, Any]:
        return {"the_answer": lambda: 42}

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
def test_fit_runs_end_to_end_and_records_canonical_names() -> None:
    runnable = _mini(train_set=_records(), max_epochs=2)
    runnable.val_set = _records(4)
    runnable.run()
    # The Recorder's `train_loss, train_mse, valid_loss, valid_mse` row, translated out of
    # fastai's vocabulary into the canonical `<split>/<name>` every other engine records.
    assert set(runnable.computed_metrics) == {"train/loss", "train/mse", "val/loss", "val/mse"}


def test_after_fit_runs_and_is_handed_the_learner() -> None:
    runnable = _mini(train_set=_records(), max_epochs=1)
    runnable.run()
    assert len(runnable.after_fit_calls) == 1
    trainer, train_ds, _, _ = runnable.after_fit_calls[0]
    from fastai.learner import Learner

    assert isinstance(trainer, Learner), "the driver on this base IS the Learner"
    assert train_ds is not None


def test_the_fit_ends_its_run_however_it_ends() -> None:
    """The learner's fit and everything after it run inside ``ending_the_run``: a raise ends the run as failed."""
    import inspect

    from matrainer.fastai.learner import FastaiRunnable

    body = inspect.getsource(FastaiRunnable.fit)
    assert body.index("log_run_context(") < body.index("with ending_the_run(trackers, manifest):")
    assert body.index("with ending_the_run(trackers, manifest):") < body.index("_build_learner(")
    assert "capture_log_file(" not in body


def test_the_epoch_history_reaches_the_trackers() -> None:
    class _Tracker:
        def __init__(self) -> None:
            self.rows: List[Any] = []

        def log_metrics(self, metrics: Dict[str, float], step: Any = None) -> None:
            self.rows.append((step, dict(metrics)))

    tracker = _Tracker()
    runnable = _mini(train_set=_records(), max_epochs=3)
    runnable.trackers = [tracker]  # type: ignore[list-item]
    from matrainer.metric_names import as_metric_names

    runnable.initialize()
    learner = runnable._build_learner(runnable._dataloaders(), "val")
    learner.fit(3, lr=1e-3)
    runnable._log_epoch_history(learner, [tracker], as_metric_names(None))  # type: ignore[list-item]
    assert [step for step, _ in tracker.rows] == [0, 1, 2], "one tracker row per recorded epoch"


# ---------------------------------------------------------------------------
# evaluate / test / predict
# ---------------------------------------------------------------------------
def test_a_test_run_scores_and_writes_the_sidecar(tmp_path: Path) -> None:
    runnable = _mini(test_set=_records(4), task="test")
    runnable.metrics_sidecar_path = str(tmp_path / "metrics.json")
    runnable.run()
    payload = json.loads((tmp_path / "metrics.json").read_text())
    # `validate()` fills the VALIDATION columns whichever split it walked — val/ names.
    assert "val/mse" in payload["metrics"] and "val/loss" in payload["metrics"]


def test_an_evaluate_run_scores_the_val_split() -> None:
    runnable = _mini(task="evaluate")
    runnable.val_set = _records(4)
    runnable.run()
    assert set(runnable.computed_metrics) == {"val/loss", "val/mse"}


def test_predict_writes_one_prediction_per_record_with_its_metadata() -> None:
    runnable = _mini(test_set=_records(4), task="predict")
    sink = _SinkRecorder()
    runnable.predictions_sink = sink
    runnable.run()
    assert len(sink.writes) == 4
    assert sink.writes[2][1] == {"idx": 2}, "metadata pairs per record, input/target excluded"
    assert getattr(sink, "closed", False)


# ---------------------------------------------------------------------------
# the fastai-shaped seams
# ---------------------------------------------------------------------------
def test_the_loss_property_is_the_family_loss_fn_slot() -> None:
    """ONE slot, two names: a config's `loss:` and the family hooks' `loss_fn` must agree."""
    chosen = nn.L1Loss()
    runnable = _mini(loss=chosen)
    assert runnable.loss is chosen and runnable.loss_fn is chosen
    runnable.loss = None
    assert runnable.loss_fn is None


def test_a_yaml_string_learning_rate_is_coerced() -> None:
    """YAML 1.1 parses `1e-4` (no decimal point) as a STRING; fastai's scheduler then dies in
    a numpy divide three frames from the config line."""
    runnable = _mini(learning_rate="1e-4")
    assert runnable.learning_rate == pytest.approx(1e-4)


def test_an_alias_is_resolved_before_the_upstream_lookup() -> None:
    runnable = _mini()
    assert runnable._resolve_metric("the_answer") == 42


def test_an_unknown_metric_name_names_the_aliases() -> None:
    runnable = _mini()
    with pytest.raises(ValueError, match="the_answer"):
        runnable._resolve_metric("no_such_metric")


def test_the_schedule_import_is_load_bearing_and_present() -> None:
    """`fit_one_cycle` is a `@patch` installed as an import SIDE EFFECT; without the import it
    falls through fastcore's delegation to the MODEL and raises an AttributeError. Imported
    ONCE, here — every consumer inherits it."""
    import matrainer.fastai.learner as mod

    assert "from fastai.callback import schedule" in Path(mod.__file__).read_text()


def test_every_data_seam_warms_its_source_in_the_parent() -> None:
    """`import fastai` sets the start method to `fork`, so a lazy source first built in a
    DataLoader worker SIGSEGVs through `_scproxy` on macOS — every seam that hands a dataset
    to a loader must `_prepare` (normalize + materialize) it in the PARENT first."""
    import inspect

    from matrainer.fastai.learner import FastaiRunnable as R

    for seam in (R._dataloaders, R._validate_split, R.predict):
        assert "self._prepare(" in inspect.getsource(seam), f"{seam.__name__} does not warm its source"


def test_the_fit_metric_list_is_the_union_and_the_eval_one_narrows() -> None:
    """A fitting Learner reports ONE list on both passes (union); an eval path builds its own
    Learner and narrows."""
    runnable = _mini(metrics=["mse"], val_metrics=["mae"], test_metrics=["msle"])
    runnable.task = "fit"
    assert list(runnable._learner_metrics("val")) == ["mse", "mae"]
    runnable.task = "test"
    assert list(runnable._learner_metrics("test")) == ["mse", "msle"]


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

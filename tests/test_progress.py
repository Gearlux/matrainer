"""Tests for :mod:`marainer.progress` — the comfy-free progress-reporting protocol.

Two halves:

* :class:`~marainer.progress.ProgressReporting` — the runnable-side mixin (a runnable records
  ``(value, total, desc)`` into an executor-injected sink).
* :class:`~marainer.progress.LightningProgressCallback` + :func:`~marainer.progress.attach_lightning_progress`
  — the Lightning bridge that maps per-batch hooks to that sink.

The Lightning trainer / state objects are faked (a real ``fit`` is far too heavy for a unit test);
the callback only reads a handful of public Trainer attributes, so a ``SimpleNamespace`` suffices.
"""

import math
from types import SimpleNamespace
from typing import Any, List, Tuple

from marainer.evaluator import LightningEvaluator
from marainer.processing import DatasetProcessor
from marainer.progress import (
    LightningProgressCallback,
    ProgressReporting,
    _finite,
    _stage_total,
    attach_lightning_progress,
)
from marainer.trainer import LightningTrainer

# --------------------------------------------------------------------------- ProgressReporting mixin


class _Reporter(ProgressReporting):
    """Trivial runnable exposing only the progress protocol."""


def test_report_progress_is_a_noop_without_a_callback() -> None:
    # No sink injected (a plain CLI run) ⇒ _report_progress does nothing and never raises.
    _Reporter()._report_progress(1, 10, "x")  # no exception, no effect


def test_report_progress_forwards_floats_to_the_callback() -> None:
    reports: List[Tuple[float, float, str]] = []
    r = _Reporter()
    r.set_progress_callback(lambda value, total, desc: reports.append((value, total, desc)))
    r._report_progress(3, 10, "Training")
    assert reports == [(3.0, 10.0, "Training")]
    # Values are coerced to float regardless of the int inputs.
    assert all(isinstance(v, float) and isinstance(t, float) for v, t, _ in reports)


def test_report_progress_skips_non_positive_total() -> None:
    reports: List[Tuple[float, float, str]] = []
    r = _Reporter()
    r.set_progress_callback(lambda value, total, desc: reports.append((value, total, desc)))
    r._report_progress(1, 0, "x")  # unknown/streaming length ⇒ skip (bar stays indeterminate)
    r._report_progress(1, -5, "x")
    assert reports == []


def test_set_progress_callback_can_be_cleared() -> None:
    reports: List[Tuple[float, float, str]] = []
    r = _Reporter()
    r.set_progress_callback(lambda value, total, desc: reports.append((value, total, desc)))
    r.set_progress_callback(None)  # clear it again
    r._report_progress(1, 10, "x")
    assert reports == []


def test_report_progress_swallows_a_raising_callback() -> None:
    # A broken sink must never abort the run — the exception is logged and dropped.
    def _boom(value: float, total: float, desc: str) -> None:
        raise RuntimeError("sink exploded")

    r = _Reporter()
    r.set_progress_callback(_boom)
    r._report_progress(1, 10, "x")  # does not raise


def test_runnables_inherit_progress_reporting() -> None:
    # The three long-running runnables all expose the protocol.
    assert issubclass(LightningTrainer, ProgressReporting)
    assert issubclass(LightningEvaluator, ProgressReporting)
    assert issubclass(DatasetProcessor, ProgressReporting)


# ----------------------------------------------------------------------------------- _finite helper


def test_finite_accepts_positive_finite_rejects_inf_zero_and_nonnumeric() -> None:
    assert _finite(10) == 10.0
    assert _finite(2.5) == 2.5
    assert _finite(0) is None
    assert _finite(-3) is None
    assert _finite(math.inf) is None
    assert _finite(math.nan) is None
    assert _finite(None) is None
    assert _finite("nope") is None


def test_stage_total_sums_a_per_dataloader_list() -> None:
    assert _stage_total([100]) == 100.0
    assert _stage_total([60, 40]) == 100.0
    assert _stage_total(50) == 50.0
    assert _stage_total([]) is None  # no batches ⇒ no total
    assert _stage_total([math.inf]) is None


# ------------------------------------------------------------------------ LightningProgressCallback


def _fake_trainer(**attrs: object) -> Any:
    """A stand-in for the bits of ``pytorch_lightning.Trainer`` the callback reads."""
    base = dict(global_step=0, estimated_stepping_batches=0, num_test_batches=[0], num_predict_batches=[0])
    base.update(attrs)
    return SimpleNamespace(**base)


def test_train_batch_end_reports_global_step_over_estimated_total() -> None:
    reports: List[Tuple[float, float, str]] = []
    cb = LightningProgressCallback(lambda value, total, desc: reports.append((value, total, desc)))
    trainer = _fake_trainer(global_step=45, estimated_stepping_batches=200)
    # Lightning passes (trainer, pl_module, outputs, batch, batch_idx); the callback only needs trainer.
    cb.on_train_batch_end(trainer, object(), None, object(), 12)
    assert reports == [(45.0, 200.0, "Training")]


def test_train_batch_end_skips_an_unbounded_total() -> None:
    # max_epochs=-1 / max_steps=-1 ⇒ estimated_stepping_batches is inf — no determinate bar.
    reports: List[Tuple[float, float, str]] = []
    cb = LightningProgressCallback(lambda value, total, desc: reports.append((value, total, desc)))
    cb.on_train_batch_end(
        _fake_trainer(global_step=5, estimated_stepping_batches=math.inf), object(), None, object(), 0
    )
    assert reports == []


def test_test_batch_end_reports_batch_index_over_stage_total() -> None:
    reports: List[Tuple[float, float, str]] = []
    cb = LightningProgressCallback(lambda value, total, desc: reports.append((value, total, desc)))
    trainer = _fake_trainer(num_test_batches=[10])
    cb.on_test_batch_end(trainer, object(), None, object(), 3)  # batch_idx 3 ⇒ 4 of 10 done
    assert reports == [(4.0, 10.0, "Testing")]


def test_predict_batch_end_reports_against_predict_total() -> None:
    reports: List[Tuple[float, float, str]] = []
    cb = LightningProgressCallback(lambda value, total, desc: reports.append((value, total, desc)))
    cb.on_predict_batch_end(_fake_trainer(num_predict_batches=[8]), object(), None, object(), 0)
    assert reports == [(1.0, 8.0, "Predicting")]


# --------------------------------------------------------------------- attach_lightning_progress glue


def test_attach_lightning_progress_appends_only_when_a_sink_is_set() -> None:
    runnable = _Reporter()
    lightning: Any = SimpleNamespace(callbacks=[])

    # No callback injected ⇒ nothing appended (a plain `marainer` run carries no extra callback).
    attach_lightning_progress(lightning, runnable)
    assert lightning.callbacks == []

    # With a sink injected ⇒ one LightningProgressCallback wired to the runnable's reporter.
    reports: List[Tuple[float, float, str]] = []
    runnable.set_progress_callback(lambda value, total, desc: reports.append((value, total, desc)))
    attach_lightning_progress(lightning, runnable)
    assert len(lightning.callbacks) == 1
    cb = lightning.callbacks[0]
    assert isinstance(cb, LightningProgressCallback)

    # The appended callback forwards to the runnable's injected sink.
    cb.on_train_batch_end(_fake_trainer(global_step=2, estimated_stepping_batches=4), object(), None, object(), 0)
    assert reports == [(2.0, 4.0, "Training")]

"""Comfy-free progress-reporting protocol for marainer runnables.

A long-running runnable (:class:`~marainer.trainer.LightningTrainer`,
:class:`~marainer.evaluator.LightningEvaluator`,
:class:`~marainer.processing.DatasetProcessor`) reports progress through a plain
Python callback so a GUI executor — FluxStudio, which drives ComfyUI's native
per-node progress bar — can surface it WITHOUT marainer importing comfy. This
mirrors the :class:`~marainer.runner.TorchRunner` split: marainer declares *that*
a runnable can report progress; the executor decides *how* to display it.

The protocol has two halves:

* :class:`ProgressReporting` — a stateless mixin a runnable inherits. The executor
  calls :meth:`~ProgressReporting.set_progress_callback` to inject a
  ``(value, total, desc) -> None`` sink; the runnable calls
  :meth:`~ProgressReporting._report_progress` from its loop (a no-op when no
  callback was injected, so a plain CLI / ``marainer`` run is unaffected).
* :class:`LightningProgressCallback` — a :class:`pytorch_lightning.Callback` that
  bridges Lightning's per-batch hooks to a :data:`ProgressCallback`, so Lightning
  runnables get whole-run progress for free. Training maps to ``global_step /
  estimated_stepping_batches`` (ONE bar over ALL epochs); ``test`` / ``predict``
  map to the per-stage batch count.
"""

import math
from typing import Any, Callable, Optional

import pytorch_lightning as L
from logflow import get_logger

logger = get_logger(__name__)

#: A progress sink: ``(value, total, description) -> None``. ``value`` / ``total`` are
#: floats in the same unit (optimizer steps, samples); ``description`` is a short stage label.
ProgressCallback = Callable[[float, float, str], None]


class ProgressReporting:
    """Mixin giving a runnable a comfy-free progress callback (mirrors :class:`~marainer.runner.TorchRunner`).

    Pure mixin — no ``__init__``, no required state — so it composes cleanly with
    ``pytorch_lightning.LightningModule`` (and any other base). The executor injects a sink via
    :meth:`set_progress_callback`; the runnable drains it from its loop via :meth:`_report_progress`.
    When no sink is injected (a plain ``marainer train`` / ``evaluate`` / ``process`` run, or any
    non-GUI caller) every call is a silent no-op, so the runnable's behaviour is unchanged.
    """

    #: Set by the executor via :meth:`set_progress_callback`; ``None`` ⇒ progress reporting is a no-op.
    _progress_callback: Optional[ProgressCallback] = None

    def set_progress_callback(self, callback: Optional[ProgressCallback]) -> None:
        """Inject (or clear, with ``None``) the progress sink the executor drains. Idempotent."""
        self._progress_callback = callback

    def _report_progress(self, value: float, total: Optional[float], desc: str = "") -> None:
        """Report ``value`` of ``total`` to the injected callback; no-op when unset or invalid.

        Never raises — a broken progress sink must never abort a training / eval / processing run
        (it is cosmetic). A ``None`` or non-positive ``total`` (an unbounded / unknown length, e.g. a
        streaming source) is skipped so the executor's bar stays in its indeterminate state rather
        than dividing by zero.
        """
        callback = self._progress_callback
        if callback is None or not total or total <= 0:
            return
        try:
            callback(float(value), float(total), desc)
        except Exception:  # noqa: BLE001 - progress reporting must never break the run
            logger.debug("progress callback raised; ignoring", exc_info=True)


def attach_lightning_progress(lightning_trainer: L.Trainer, runnable: ProgressReporting) -> None:
    """Append a :class:`LightningProgressCallback` to ``lightning_trainer`` when a sink is set.

    Called by the Lightning runnables AFTER ``flow(self.lightning)`` — the same post-flow point
    :func:`marainer.system_metrics.activate_for_lightning` appends its callbacks (Lightning iterates
    ``trainer.callbacks`` at hook-call time, so a post-construction append is honoured). A no-op when
    no progress callback was injected, so a plain ``marainer`` run never carries the extra callback.
    """
    if getattr(runnable, "_progress_callback", None) is not None:
        lightning_trainer.callbacks.append(  # type: ignore[attr-defined]
            LightningProgressCallback(runnable._report_progress)
        )


class LightningProgressCallback(L.Callback):
    """Bridge Lightning's per-batch hooks to a :data:`ProgressCallback`.

    **Training** reports ONE bar over the whole run — ``value = trainer.global_step`` (optimizer steps
    completed) out of ``total = trainer.estimated_stepping_batches`` (the planned step count across
    ALL epochs, honouring ``max_epochs`` / ``max_steps`` / ``accumulate_grad_batches`` /
    ``limit_train_batches``) — so the bar advances smoothly across epoch boundaries instead of
    resetting each epoch. Standalone **test** / **predict** report the per-stage batch count.
    Validation is intentionally NOT reported: during ``fit`` it interleaves with training (the
    global-step bar already covers the run), and the marainer runnables never call ``validate()``
    standalone. (Multi-dataloader test/predict reports against the first loader's index — our
    evaluator uses a single dataloader.)
    """

    def __init__(self, report: ProgressCallback) -> None:
        self._report = report

    def on_train_batch_end(self, trainer: L.Trainer, *args: Any, **kwargs: Any) -> None:
        total = _finite(trainer.estimated_stepping_batches)
        if total:
            self._report(float(trainer.global_step), total, "Training")

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._report_stage(trainer.num_test_batches, batch_idx, "Testing")

    def on_predict_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._report_stage(trainer.num_predict_batches, batch_idx, "Predicting")

    def _report_stage(self, num_batches: Any, batch_idx: int, desc: str) -> None:
        total = _stage_total(num_batches)
        if total:
            self._report(float(batch_idx + 1), total, desc)


def _finite(value: Any) -> Optional[float]:
    """Return ``value`` as a positive, finite float, else ``None`` (e.g. an unbounded ``inf`` total)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def _stage_total(num_batches: Any) -> Optional[float]:
    """Total batches for a test / predict stage (Lightning exposes a per-dataloader list, or an int)."""
    if isinstance(num_batches, (list, tuple)):
        return _finite(sum(n for n in num_batches if _finite(n)))
    return _finite(num_batches)

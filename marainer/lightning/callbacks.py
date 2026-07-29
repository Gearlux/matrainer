"""Lightning tracking utilities: a tracking callback, progress bridge, system metrics, checkpoints.

Optional Lightning adapters over the framework-free tracking core — a standard
``pytorch_lightning`` training class attaches whichever it wants (or none, using the
plain :mod:`marainer.tracking.provenance` functions directly):

* :class:`TrackingCallback` — the all-in-one convenience: on fit/test start it logs
  run provenance (git / config artifacts / params) to the given trackers; on end it
  captures the log file. One line in ``pl.Trainer(callbacks=[...])``.
* :class:`LightningProgressCallback` / :func:`attach_lightning_progress` — bridge
  Lightning's per-batch hooks to a :data:`~recordstream.runnable.ProgressCallback`.
* :class:`MlflowSystemMetricsCallback` / :class:`MPSMetricsCallback` /
  :func:`activate_for_lightning` — attach MLflow system-metric sampling
  (:mod:`marainer.tracking.system`) to a Lightning run whose ``MLFlowLogger``
  bypasses the fluent ``mlflow.start_run``.
* :func:`build_checkpoint_callbacks` — translate framework-neutral
  :mod:`marainer.checkpoints` policies into ``pytorch_lightning`` ``ModelCheckpoint``\\ s.
"""

from __future__ import annotations

import math
from typing import Any, List, Mapping, Optional, Sequence

import pytorch_lightning as L
import torch
from loggair import get_logger
from pytorch_lightning.callbacks import ModelCheckpoint
from recordstream.runnable import ProgressCallback, ProgressReporting

from marainer.checkpoints import CheckpointPolicy, SaveBest, SaveEveryEpoch, SaveLast
from marainer.tracking import provenance
from marainer.tracking.base import Tracker
from marainer.tracking.system import MpsMetricsSampler, start_system_metrics_monitor

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# All-in-one provenance tracking callback
# --------------------------------------------------------------------------- #
class TrackingCallback(L.Callback):
    """Log run provenance to marainer trackers around a standard Lightning run.

    Attach it to ``pl.Trainer(callbacks=[TrackingCallback([tracker])])`` and the
    training class stays a plain ``LightningModule``: on fit/test start this logs the
    git commit, config artifacts, and ``params`` to every tracker; on end it captures
    the loggair log file. Remove marainer → drop this one callback → the class still
    trains.

    Args:
        trackers: The marainer trackers to log provenance to.
        params: Optional hyperparameters to record (only JSON-scalar values forwarded).
        capture_log: Upload the loggair log file as an artifact on end (default ``True``).
    """

    def __init__(
        self,
        trackers: Sequence[Tracker],
        params: Optional[Mapping[str, Any]] = None,
        capture_log: bool = True,
    ) -> None:
        super().__init__()
        self.trackers = list(trackers)
        self.params = params
        self.capture_log = capture_log
        self._logged = False

    def _log_context(self) -> None:
        if not self._logged:
            provenance.log_run_context(self.trackers, self.params)
            self._logged = True

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._log_context()

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._log_context()

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.capture_log:
            provenance.capture_log_file(self.trackers)

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.capture_log:
            provenance.capture_log_file(self.trackers)


# --------------------------------------------------------------------------- #
# Progress bridge
# --------------------------------------------------------------------------- #
def attach_lightning_progress(lightning_trainer: L.Trainer, runnable: ProgressReporting) -> None:
    """Append a :class:`LightningProgressCallback` to ``lightning_trainer`` when a sink is set.

    Called AFTER ``flow(self.lightning)`` — the same post-flow point
    :func:`activate_for_lightning` appends its callbacks (Lightning iterates
    ``trainer.callbacks`` at hook-call time). A no-op when no progress callback was injected.
    """
    if getattr(runnable, "_progress_callback", None) is not None:
        lightning_trainer.callbacks.append(  # type: ignore[attr-defined]
            LightningProgressCallback(runnable._report_progress)
        )


class LightningProgressCallback(L.Callback):
    """Bridge Lightning's per-batch hooks to a :data:`~recordstream.runnable.ProgressCallback`.

    **Training** reports ONE bar over the whole run — ``global_step`` out of
    ``estimated_stepping_batches`` (across ALL epochs). Standalone **test** / **predict**
    report the per-stage batch count. Validation is intentionally NOT reported.
    """

    def __init__(self, report: ProgressCallback) -> None:
        self._report = report

    def on_train_batch_end(self, trainer: L.Trainer, *args: Any, **kwargs: Any) -> None:
        total = _finite(trainer.estimated_stepping_batches)
        if total:
            self._report(float(trainer.global_step), total, "Training")

    def on_test_batch_end(
        self, trainer: L.Trainer, pl_module: Any, outputs: Any, batch: Any, batch_idx: int, *a: Any, **k: Any
    ) -> None:
        self._report_stage(trainer.num_test_batches, batch_idx, "Testing")

    def on_predict_batch_end(
        self, trainer: L.Trainer, pl_module: Any, outputs: Any, batch: Any, batch_idx: int, *a: Any, **k: Any
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


# --------------------------------------------------------------------------- #
# System-metrics callbacks
# --------------------------------------------------------------------------- #
def has_mlflow_logger(loggers: List[Any]) -> bool:
    """True when at least one Lightning logger is an ``MLFlowLogger`` (detected by class name)."""
    return any(type(lg).__name__ == "MLFlowLogger" for lg in loggers)


def _find_mlflow_logger(loggers: List[Any]) -> Optional[Any]:
    for lg in loggers:
        if type(lg).__name__ == "MLFlowLogger":
            return lg
    return None


def _logger_run(lg: Any) -> tuple[Optional[str], Optional[str]]:
    """Materialise the MLflow run and return ``(run_id, tracking_uri)``."""
    try:
        _ = lg.experiment  # materialise lazily
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"MLflow run materialisation failed: {exc}")
        return None, None
    run_id = getattr(lg, "run_id", None)
    tracking_uri = getattr(lg, "_tracking_uri", None) or getattr(lg, "tracking_uri", None)
    return run_id, tracking_uri


class MlflowSystemMetricsCallback(L.Callback):
    """Attach MLflow's ``SystemMetricsMonitor`` (CPU/RAM/disk) to the Lightning ``MLFlowLogger``'s run.

    Lightning's ``MLFlowLogger`` uses ``MlflowClient.create_run`` directly, bypassing the fluent
    ``mlflow.start_run`` the global enable hooks into — so the monitor is started explicitly here.
    """

    def __init__(self) -> None:
        super().__init__()
        self._monitor: Optional[Any] = None

    def _start(self, trainer: L.Trainer) -> None:
        if self._monitor is not None:
            return
        lg = _find_mlflow_logger(trainer.loggers)
        if lg is None:
            return
        run_id, tracking_uri = _logger_run(lg)
        if run_id:
            self._monitor = start_system_metrics_monitor(run_id, tracking_uri)

    def _finish(self) -> None:
        if self._monitor is None:
            return
        try:
            self._monitor.finish()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"system-metrics monitor finish failed: {exc}")
        self._monitor = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._start(trainer)

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._start(trainer)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._finish()

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._finish()


class MPSMetricsCallback(L.Callback):
    """Drive a :class:`~marainer.tracking.system.MpsMetricsSampler` for a Lightning ``MLFlowLogger`` run.

    A thin Lightning adapter: on ``fit`` / ``test`` start it materialises the MLflow run, builds an
    ``MlflowClient``, and starts the framework-free sampler; on the matching end hook it stops it.

    Args:
        sample_interval_s: Seconds between records (``None`` = env var then 10s).
    """

    def __init__(self, sample_interval_s: Optional[float] = None) -> None:
        super().__init__()
        self.sample_interval_s = sample_interval_s
        self._sampler: Optional[MpsMetricsSampler] = None

    def _start(self, trainer: L.Trainer) -> None:
        if self._sampler is not None:
            return
        if not torch.backends.mps.is_available():
            return
        lg = _find_mlflow_logger(trainer.loggers)
        if lg is None:
            return
        run_id, tracking_uri = _logger_run(lg)
        if not run_id:
            return
        try:
            from mlflow.tracking import MlflowClient
        except ImportError:
            logger.warning("mlflow.tracking.MlflowClient unavailable — MPS metrics suppressed.")
            return
        self._sampler = MpsMetricsSampler(MlflowClient(tracking_uri=tracking_uri), run_id, self.sample_interval_s)
        self._sampler.start()

    def _stop(self) -> None:
        if self._sampler is not None:
            self._sampler.stop()
            self._sampler = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._start(trainer)

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._start(trainer)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._stop()

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._stop()


def activate_for_lightning(lightning_trainer: L.Trainer) -> None:
    """Append the system-metrics callbacks to a Lightning Trainer when an ``MLFlowLogger`` is wired.

    A no-op without an ``MLFlowLogger`` (saves the monitor overhead on CSV-only / no-logger runs).
    Appends :class:`MlflowSystemMetricsCallback` (CPU/RAM/disk) and, on Apple Silicon,
    :class:`MPSMetricsCallback`.
    """
    if not has_mlflow_logger(lightning_trainer.loggers):
        logger.debug("No MLFlowLogger wired — skipping system-metrics activation.")
        return
    lightning_trainer.callbacks.append(MlflowSystemMetricsCallback())  # type: ignore[attr-defined]
    if torch.backends.mps.is_available():
        lightning_trainer.callbacks.append(MPSMetricsCallback())  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Checkpoint-policy translation
# --------------------------------------------------------------------------- #
def checkpoint_dir(trainer: Any) -> Optional[str]:
    """Where this Lightning trainer writes checkpoints, if it can say.

    The Lightning-specific half of :func:`marainer.artifacts.save_class_names`: that function
    takes a plain directory, and THIS is what turns a ``pl.Trainer`` into one. Prefers the
    checkpoint callback's ``dirpath`` (where checkpoints actually land) and falls back to the
    trainer's ``default_root_dir``, so a run with checkpointing disabled still has somewhere
    to put a run artifact.

    Returns ``None`` when neither is set — the caller writes nothing rather than guessing.

    Example::

        save_class_names(self.label_map, checkpoint_dir(trainer), self._trackers)
    """
    callback = getattr(trainer, "checkpoint_callback", None)
    dirpath = getattr(callback, "dirpath", None) if callback is not None else None
    return str(dirpath) if dirpath else (getattr(trainer, "default_root_dir", None) or None)


def build_checkpoint_callbacks(policies: Any) -> List[ModelCheckpoint]:
    """Translate framework-neutral :mod:`marainer.checkpoints` policies into ``ModelCheckpoint``\\ s.

    Accepts a single policy, a list, or ``None`` (⇒ no checkpointing). Each policy maps to one
    ``pytorch_lightning.callbacks.ModelCheckpoint``:

    * :class:`~marainer.checkpoints.SaveLast` → ``save_last=True, save_top_k=0``
    * :class:`~marainer.checkpoints.SaveEveryEpoch` → ``every_n_epochs=N, save_top_k=-1``
    * :class:`~marainer.checkpoints.SaveBest` → ``monitor, mode, save_top_k=top_k``
    """
    if policies is None:
        return []
    if isinstance(policies, CheckpointPolicy):
        policies = [policies]
    callbacks: List[ModelCheckpoint] = []
    for policy in policies:
        common = {"dirpath": policy.dirpath}
        if policy.filename is not None:
            common["filename"] = policy.filename
        if isinstance(policy, SaveBest):
            callbacks.append(
                ModelCheckpoint(monitor=policy.monitor, mode=policy.mode, save_top_k=policy.top_k, **common)
            )
        elif isinstance(policy, SaveEveryEpoch):
            callbacks.append(ModelCheckpoint(every_n_epochs=policy.every_n_epochs, save_top_k=-1, **common))
        elif isinstance(policy, SaveLast):
            callbacks.append(ModelCheckpoint(save_last=True, save_top_k=0, **common))
        else:  # a bare CheckpointPolicy — default to save-last
            callbacks.append(ModelCheckpoint(save_last=True, save_top_k=0, **common))
    return callbacks


__all__ = [
    "TrackingCallback",
    "LightningProgressCallback",
    "MPSMetricsCallback",
    "MlflowSystemMetricsCallback",
    "activate_for_lightning",
    "attach_lightning_progress",
    "build_checkpoint_callbacks",
    "has_mlflow_logger",
]

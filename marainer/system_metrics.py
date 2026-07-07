"""MLflow system-metrics activation for marainer trainers.

Three pieces:

1. **Explicit Lightning-side monitor attachment**
   (:class:`MlflowSystemMetricsCallback`) — pytorch-lightning's
   ``MLFlowLogger`` uses ``MlflowClient.create_run()`` directly which
   bypasses the fluent ``mlflow.start_run()`` that
   :func:`mlflow.system_metrics.enable_system_metrics_logging` hooks into.
   The global enable flag therefore has no observable effect on a
   Lightning run — CPU / RAM / disk samples are silently dropped. The
   callback works around this by instantiating
   :class:`mlflow.system_metrics.system_metrics_monitor.SystemMetricsMonitor`
   against the Lightning logger's ``run_id`` directly when ``fit`` /
   ``test`` starts.

2. **Process-wide enable for `mlflow.start_run`-based clients**
   (:func:`enable_mlflow_system_metrics`) — ultralytics' MLflow callback
   does use ``mlflow.start_run()`` so the global flag is sufficient there.
   :class:`raidar_ultralytics.UltralyticsTrainer` keeps calling this
   directly.

3. **Apple-Silicon MPS metrics** (:class:`MPSMetricsCallback`) — runs a
   background **non-daemon** thread between ``on_fit_start`` and
   ``on_fit_end`` (and the matching ``test_*`` hooks) that samples MPS
   GPU stats every ``sample_interval_s`` seconds and pushes them straight
   to MLflow via :class:`mlflow.tracking.MlflowClient.log_metric`. The
   cadence is decoupled from training-step pacing so it matches MLflow's
   built-in CPU / RAM / disk monitor — both default to 10 s and both
   honour ``MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL``. Two data sources
   feed the sampler: ``torch.mps.*_allocated_memory`` (per-process) and
   the inlined IOKit ``IOAccelerator.PerformanceStatistics`` reader
   (:func:`_read_mps_performance_statistics`) — system-level utilisation
   + memory, no sudo.

The :func:`activate_for_lightning` helper composes (1) and (3). Trainers
call it once in ``run()`` before they log hyperparams.
"""

import atexit
import os
import threading
import time
import weakref
from typing import Any, Dict, List, Optional

import pytorch_lightning as L
import torch
from loggair import get_logger

logger = get_logger(__name__)

# IOAccelerator PerformanceStatistics reader (Apple-Silicon GPU utilisation,
# no sudo). Inlined here rather than depending on the ``apple-gpu`` PyPI
# package, which ships two cleanup bugs that corrupt refcounts and segfault
# during ``gc.collect`` / interpreter teardown:
#
#   1. ``CFRelease(err)`` where ``err`` is a Python int (a ``kern_return_t``
#      error code), not a ``CFTypeRef``.
#   2. ``CFRelease(io_service_t)`` — ``io_service_t`` is a ``mach_port_t``
#      handle and must be released with ``IOObjectRelease``, not
#      ``CFRelease``.
#
# Our version below releases each handle with the correct API and skips the
# bogus int release.
_IOKIT_FUNCS: Dict[str, Any] = {}
try:
    import CoreFoundation as _CoreFoundation
    import Foundation as _Foundation
    import objc as _objc

    _iokit_bundle = _Foundation.NSBundle.bundleWithIdentifier_("com.apple.framework.IOKit")
    _objc.loadBundleFunctions(
        _iokit_bundle,
        _IOKIT_FUNCS,
        [
            ("IOServiceGetMatchingService", b"II@"),
            ("IOServiceMatching", b"@*"),
            ("IORegistryEntryCreateCFProperties", b"IIo^@@I"),
            ("IOObjectRelease", b"II"),
        ],
    )
    _MPS_PERF_STATS_AVAILABLE: bool = True
except Exception:  # ImportError on non-Darwin, runtime error if IOKit shape changes
    _MPS_PERF_STATS_AVAILABLE = False


def _read_mps_performance_statistics() -> Optional[Dict[str, Any]]:
    """Return the ``IOAccelerator.PerformanceStatistics`` dict via IOKit.

    Returns ``None`` when pyobjc/IOKit isn't usable on this host or when
    the service isn't matched. Both handles are released with the
    correct API (``IOObjectRelease`` for the ``io_service_t``,
    ``CFRelease`` for the CFDictionary), and the integer kern_return is
    never released — this is the fix for the upstream ``apple_gpu``
    segfault.
    """
    if not _MPS_PERF_STATS_AVAILABLE:
        return None
    accelerator = _IOKIT_FUNCS["IOServiceGetMatchingService"](0, _IOKIT_FUNCS["IOServiceMatching"](b"IOAccelerator"))
    if accelerator == 0:
        return None
    try:
        err, props = _IOKIT_FUNCS["IORegistryEntryCreateCFProperties"](accelerator, None, None, 0)
        if err != 0 or props is None:
            return None
        try:
            stats = props.get("PerformanceStatistics")
            return dict(stats) if stats is not None else None
        finally:
            _CoreFoundation.CFRelease(props)
    finally:
        _IOKIT_FUNCS["IOObjectRelease"](accelerator)


# Default sampling cadence for the MPS background sampler, aligned with
# MLflow's built-in ``SystemMetricsMonitor`` default. Operators can override
# globally via ``MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL`` (seconds, float) or
# explicitly per-callback via the ``sample_interval_s`` ctor kwarg.
DEFAULT_MPS_SAMPLE_INTERVAL_S: float = 10.0


def _resolve_sample_interval(explicit: Optional[float]) -> float:
    """Pick the sampling cadence: explicit > env var > 10s default."""
    if explicit is not None:
        return float(explicit)
    env_val = os.environ.get("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL")
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            logger.debug(f"MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL={env_val!r} not a float; using default.")
    return DEFAULT_MPS_SAMPLE_INTERVAL_S


# Subset of keys probed live from ``IOAccelerator.PerformanceStatistics`` on
# Apple Silicon. Each entry maps an MLflow metric name to its source key plus
# a unit conversion (no-op for percentages, bytes→MB for memory). Apple has
# historically renamed these keys across macOS releases (see exelban/stats
# issue #2130), so the sampler skips any key that isn't present rather than
# fail the cycle.
_APPLE_GPU_METRICS: List[tuple[str, str, float]] = [
    ("system/mps_device_util_pct", "Device Utilization %", 1.0),
    ("system/mps_renderer_util_pct", "Renderer Utilization %", 1.0),
    ("system/mps_tiler_util_pct", "Tiler Utilization %", 1.0),
    ("system/mps_in_use_system_memory_mb", "In use system memory", 1.0 / (1024**2)),
    ("system/mps_in_use_system_memory_driver_mb", "In use system memory (driver)", 1.0 / (1024**2)),
    ("system/mps_alloc_system_memory_mb", "Alloc system memory", 1.0 / (1024**2)),
]


def has_mlflow_logger(loggers: List[Any]) -> bool:
    """True when at least one Lightning logger is an ``MLFlowLogger``.

    Detection by class name (rather than ``isinstance``) avoids an
    unconditional ``pytorch_lightning.loggers.MLFlowLogger`` import — the
    logger is only needed at runtime when the user actually wired one.
    """
    for lg in loggers:
        if type(lg).__name__ == "MLFlowLogger":
            return True
    return False


def _find_mlflow_logger(loggers: List[Any]) -> Optional[Any]:
    for lg in loggers:
        if type(lg).__name__ == "MLFlowLogger":
            return lg
    return None


def enable_mlflow_system_metrics() -> bool:
    """Globally enable ``mlflow.system_metrics`` for subsequent runs.

    Honours an optional ``MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL`` env
    var (seconds, float). Returns ``True`` on success, ``False`` when
    ``mlflow.system_metrics`` is missing (e.g. mlflow < 2.8 or psutil not
    installed).

    **For Lightning trainers this is NOT sufficient** — pytorch-lightning
    creates its MLflow run via ``MlflowClient.create_run`` which bypasses
    the fluent ``mlflow.start_run`` that this flag hooks into. The
    Lightning path uses :class:`MlflowSystemMetricsCallback` instead.
    This entry point is kept for ``mlflow.start_run``-based clients
    (notably ultralytics' built-in MLflow callback).
    """
    try:
        import mlflow
        import mlflow.system_metrics
    except ImportError:
        logger.warning("mlflow.system_metrics unavailable — install psutil + mlflow>=2.8 to enable.")
        return False
    mlflow.system_metrics.enable_system_metrics_logging()
    interval = os.environ.get("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL")
    if interval is not None:
        try:
            mlflow.set_system_metrics_sampling_interval(float(interval))
        except Exception as exc:  # pragma: no cover — defensive, depends on mlflow internals
            logger.debug(f"sampling-interval override ignored: {exc}")
    return True


class MlflowSystemMetricsCallback(L.Callback):
    """Attach a :class:`SystemMetricsMonitor` to the Lightning ``MLFlowLogger``'s run.

    Workaround for ``pytorch_lightning.loggers.MLFlowLogger`` — it calls
    ``MlflowClient.create_run()`` directly, bypassing the fluent
    ``mlflow.start_run`` entry that the global
    ``enable_system_metrics_logging`` flag hooks into. Without this
    callback the CPU / RAM / disk traces simply never reach the run.

    Lifecycle: starts the monitor on ``fit`` / ``test`` start (after the
    logger has materialised the run via the first ``log_hyperparams``
    call) and finishes it on the matching end hook.
    """

    def __init__(self) -> None:
        super().__init__()
        self._monitor: Optional[Any] = None

    def _start(self, trainer: L.Trainer) -> None:
        if self._monitor is not None:
            return  # already running for this fit/test cycle
        lg = _find_mlflow_logger(trainer.loggers)
        if lg is None:
            return
        # Force the run to exist before we read run_id (Lightning materialises
        # lazily on first ``.experiment`` access; ``_marainer_log_hyperparams``
        # has already done so by the time fit_start fires, but accessing the
        # property again is harmless and idempotent).
        try:
            _ = lg.experiment
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug(f"MLflow run materialisation failed: {exc}")
            return
        run_id = getattr(lg, "run_id", None)
        if not run_id:
            return
        try:
            from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
        except ImportError:
            logger.warning("mlflow.system_metrics.SystemMetricsMonitor unavailable — install psutil + mlflow>=2.8.")
            return
        sampling = float(os.environ.get("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", "10"))
        tracking_uri = getattr(lg, "_tracking_uri", None) or getattr(lg, "tracking_uri", None)
        monitor = SystemMetricsMonitor(
            run_id=run_id,
            sampling_interval=sampling,
            tracking_uri=tracking_uri,
        )
        monitor.start()
        self._monitor = monitor
        logger.info(f"MLflow system-metrics monitor started for run {run_id} (interval={sampling}s).")

    def _finish(self) -> None:
        if self._monitor is None:
            return
        try:
            self._monitor.finish()
        except Exception as exc:  # pragma: no cover — defensive
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
    """Background MPS sampler — writes nine ``system/mps_*`` metrics
    straight to MLflow at a fixed time interval.

    Three per-process memory scalars come from ``torch.mps``:

    * ``system/mps_allocated_mb`` — current allocation of the calling process
    * ``system/mps_driver_mb`` — driver-side allocation (includes caches)
    * ``system/mps_recommended_max_mb`` — Apple's soft cap

    Six system-level scalars come from the ``apple_gpu`` package's
    ``IOAccelerator.PerformanceStatistics`` read (no sudo):

    * ``system/mps_device_util_pct`` — overall GPU busy %
    * ``system/mps_renderer_util_pct`` — render-stage busy %
    * ``system/mps_tiler_util_pct`` — tile-binning-stage busy %
    * ``system/mps_in_use_system_memory_mb``
    * ``system/mps_in_use_system_memory_driver_mb``
    * ``system/mps_alloc_system_memory_mb``

    Sampling runs in a **non-daemon** thread started on ``on_fit_start`` /
    ``on_test_start`` and joined on the matching end hook. Metrics are
    pushed via :class:`mlflow.tracking.MlflowClient.log_metric` so the
    cadence is decoupled from Lightning's step pacing and matches
    MLflow's built-in CPU / RAM / disk monitor. The callback is a no-op
    when MPS is unavailable, when no ``MLFlowLogger`` is wired, or when
    ``mlflow`` cannot be imported.

    The thread is non-daemon and an ``atexit`` hook also signals stop —
    both safeguards prevent the sampler from accessing ``torch.mps`` /
    ``apple_gpu`` / ``MlflowClient`` during interpreter teardown, which
    would segfault on macOS as the Metal device + IOKit handles + MLflow
    C state get torn down out from under it.

    Args:
        sample_interval_s: Seconds between samples. ``None`` (default)
            consults ``MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL`` env var
            then falls back to :data:`DEFAULT_MPS_SAMPLE_INTERVAL_S`
            (10 s). Explicit values override both.
    """

    def __init__(self, sample_interval_s: Optional[float] = None) -> None:
        super().__init__()
        self.sample_interval_s: float = _resolve_sample_interval(sample_interval_s)
        self._thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._mlflow_client: Optional[Any] = None
        self._run_id: Optional[str] = None
        self._step: int = 0

    # ------------------------------------------------------------------
    # Sampling primitives

    @staticmethod
    def _sample_metrics() -> Dict[str, float]:
        """Return one snapshot of MPS metrics — empty when MPS isn't available."""
        metrics: Dict[str, float] = {}
        if not torch.backends.mps.is_available():
            return metrics
        try:
            metrics["system/mps_allocated_mb"] = torch.mps.current_allocated_memory() / (1024**2)
            metrics["system/mps_driver_mb"] = torch.mps.driver_allocated_memory() / (1024**2)
            metrics["system/mps_recommended_max_mb"] = torch.mps.recommended_max_memory() / (1024**2)
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug(f"MPS metrics read failed: {exc}")
            return metrics
        try:
            stats = _read_mps_performance_statistics()
        except Exception as exc:
            logger.debug(f"IOAccelerator PerformanceStatistics read failed: {exc}")
            return metrics
        if stats is None:
            return metrics
        for metric_name, source_key, scale in _APPLE_GPU_METRICS:
            value = stats.get(source_key)
            if value is None:
                continue
            metrics[metric_name] = float(value) * scale
        return metrics

    def _sample_loop(self) -> None:
        """Sampler thread body: sample, push, sleep, repeat until stopped."""
        while not self._stop_event.is_set():
            metrics = self._sample_metrics()
            if metrics and self._mlflow_client is not None and self._run_id is not None:
                timestamp_ms = int(time.time() * 1000)
                step = self._step
                for key, value in metrics.items():
                    # Re-check the stop flag between pushes so the thread can
                    # exit fast when shutdown is requested mid-batch — keeps
                    # us from calling into MLflow C state during teardown.
                    if self._stop_event.is_set():
                        return
                    try:
                        self._mlflow_client.log_metric(
                            run_id=self._run_id,
                            key=key,
                            value=value,
                            timestamp=timestamp_ms,
                            step=step,
                        )
                    except Exception as exc:  # pragma: no cover — defensive
                        logger.debug(f"MPS metric '{key}' push failed: {exc}")
                self._step += 1
            self._stop_event.wait(self.sample_interval_s)

    # ------------------------------------------------------------------
    # Lifecycle

    def _start(self, trainer: L.Trainer) -> None:
        if self._thread is not None and self._thread.is_alive():
            return  # idempotent — already sampling
        if not torch.backends.mps.is_available():
            return
        lg = _find_mlflow_logger(trainer.loggers)
        if lg is None:
            return
        try:
            _ = lg.experiment  # materialise the run lazily
        except Exception as exc:  # pragma: no cover — defensive
            logger.debug(f"MLflow run materialisation failed: {exc}")
            return
        run_id = getattr(lg, "run_id", None)
        if not run_id:
            return
        try:
            from mlflow.tracking import MlflowClient
        except ImportError:
            logger.warning("mlflow.tracking.MlflowClient unavailable — MPS metrics suppressed.")
            return
        tracking_uri = getattr(lg, "_tracking_uri", None) or getattr(lg, "tracking_uri", None)
        self._mlflow_client = MlflowClient(tracking_uri=tracking_uri)
        self._run_id = run_id
        self._stop_event.clear()
        # daemon=False: ensures Python's atexit waits for the thread to exit
        # cleanly. The atexit hook below signals stop in case ``on_fit_end``
        # is bypassed (crash, KeyboardInterrupt mid-fit, etc.). Together they
        # guarantee the sampler never touches torch.mps / apple_gpu /
        # MlflowClient during interpreter teardown.
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="MPSMetricsSampler",
            daemon=False,
        )
        self._register_atexit_stop()
        self._thread.start()
        logger.info(f"MPS metrics sampler started for run {run_id} (interval={self.sample_interval_s}s).")

    def _register_atexit_stop(self) -> None:
        """Wire up a one-shot atexit hook that signals stop.

        Uses a ``weakref`` so the hook doesn't pin the callback alive
        (Python's ``atexit`` table holds strong refs by default). The
        hook re-runs ``_stop`` defensively; ``_stop`` is idempotent.
        """
        if getattr(self, "_atexit_registered", False):
            return
        cb_ref = weakref.ref(self)

        def _atexit_stop() -> None:
            cb = cb_ref()
            if cb is not None:
                cb._stop()

        atexit.register(_atexit_stop)
        self._atexit_registered = True

    def _stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        # Bound the join generously — the in-loop stop check + the
        # ``_stop_event.wait`` between cycles mean the thread should exit
        # within a single sampling interval. A timeout here would leave a
        # live thread accessing C state during teardown (segfault risk), so
        # we tolerate a longer wait and only log if it goes wrong.
        join_timeout = max(2.0, 2 * self.sample_interval_s)
        self._thread.join(timeout=join_timeout)
        if self._thread.is_alive():  # pragma: no cover — defensive
            logger.warning(
                f"MPS metrics sampler did not exit within {join_timeout}s — "
                f"keeping thread reference to block interpreter exit until it does."
            )
            # Don't drop the reference: the thread is non-daemon so the
            # interpreter will wait for it on its own atexit pass.
            return
        self._thread = None
        self._mlflow_client = None
        self._run_id = None
        self._step = 0

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._start(trainer)

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._start(trainer)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._stop()

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._stop()


def activate_for_lightning(lightning_trainer: L.Trainer) -> None:
    """Activate system metrics for a configured pytorch-lightning Trainer.

    No-op when no ``MLFlowLogger`` is wired — saves the monitor overhead
    on CSV-only / no-logger runs. The caller is responsible for the
    ``log_system_metrics`` opt-out; this function unconditionally enables
    when invoked.

    Two callbacks get appended on success:

    * :class:`MlflowSystemMetricsCallback` — drives MLflow's
      ``SystemMetricsMonitor`` against the Lightning logger's
      ``run_id`` (the global ``enable_system_metrics_logging`` flag is
      not enough because Lightning bypasses ``mlflow.start_run``).
    * :class:`MPSMetricsCallback` — only on Apple-Silicon hosts.
    """
    if not has_mlflow_logger(lightning_trainer.loggers):
        logger.debug("No MLFlowLogger wired — skipping system-metrics activation.")
        return
    lightning_trainer.callbacks.append(MlflowSystemMetricsCallback())  # type: ignore[attr-defined]
    if torch.backends.mps.is_available():
        lightning_trainer.callbacks.append(MPSMetricsCallback())  # type: ignore[attr-defined]


__all__ = [
    "MPSMetricsCallback",
    "MlflowSystemMetricsCallback",
    "activate_for_lightning",
    "enable_mlflow_system_metrics",
    "has_mlflow_logger",
]

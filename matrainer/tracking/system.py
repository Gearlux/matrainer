"""Framework-free system-metrics sampling for the tracking core.

This is the platform-neutral half of the old ``matrainer.system_metrics`` — the
pieces that push CPU / RAM / disk / Apple-Silicon-GPU scalars to MLflow **without
any Lightning coupling** (the ``pytorch_lightning.Callback`` adapters that wrap
these live in ``matrainer.lightning.callbacks``):

* :func:`enable_mlflow_system_metrics` — the process-wide MLflow enable for
  ``mlflow.start_run``-based clients (Ultralytics' MLflow callback, plain loops).
* :func:`start_system_metrics_monitor` — start MLflow's own
  ``SystemMetricsMonitor`` (CPU/RAM/disk) against a known ``run_id``.
* :class:`MpsMetricsSampler` — a plain ``start()`` / ``stop()`` background sampler
  that pushes nine ``system/mps_*`` scalars straight to MLflow via
  ``MlflowClient.log_metric`` (Apple Silicon; ``torch`` and ``mlflow`` are both
  lazily imported, so this module loads in the light core install).

The inlined ``IOAccelerator.PerformanceStatistics`` reader replaces the buggy
``apple-gpu`` PyPI package (whose cleanup path corrupts refcounts / segfaults);
``pyobjc`` is guarded so the module imports off-Darwin.
"""

from __future__ import annotations

import atexit
import os
import threading
import time
import weakref
from typing import Any, Dict, List, Optional

from loggair import get_logger

logger = get_logger(__name__)

# IOAccelerator PerformanceStatistics reader (Apple-Silicon GPU utilisation, no sudo).
# Inlined rather than depending on ``apple-gpu`` (two cleanup bugs corrupt refcounts and
# segfault during ``gc.collect`` / interpreter teardown): (1) ``CFRelease(err)`` where
# ``err`` is a Python int; (2) ``CFRelease(io_service_t)`` — a mach_port handle needing
# ``IOObjectRelease``. This version releases each handle with the correct API.
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
    """Return the ``IOAccelerator.PerformanceStatistics`` dict via IOKit, or ``None``.

    Both handles are released with the correct API (``IOObjectRelease`` for the
    ``io_service_t``, ``CFRelease`` for the CFDictionary), and the integer
    kern_return is never released — the fix for the upstream ``apple_gpu`` segfault.
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


DEFAULT_MPS_SAMPLE_INTERVAL_S: float = 10.0


def _resolve_sample_interval(explicit: Optional[float]) -> float:
    """Pick the sampling cadence: explicit > ``MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL`` > 10s."""
    if explicit is not None:
        return float(explicit)
    env_val = os.environ.get("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL")
    if env_val:
        try:
            return float(env_val)
        except ValueError:
            logger.debug(f"MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL={env_val!r} not a float; using default.")
    return DEFAULT_MPS_SAMPLE_INTERVAL_S


# MLflow metric name -> (PerformanceStatistics key, unit conversion). Apple renames these
# keys across macOS releases, so the sampler skips any absent key rather than fail the cycle.
_APPLE_GPU_METRICS: List[tuple[str, str, float]] = [
    ("system/mps_device_util_pct", "Device Utilization %", 1.0),
    ("system/mps_renderer_util_pct", "Renderer Utilization %", 1.0),
    ("system/mps_tiler_util_pct", "Tiler Utilization %", 1.0),
    ("system/mps_in_use_system_memory_mb", "In use system memory", 1.0 / (1024**2)),
    ("system/mps_in_use_system_memory_driver_mb", "In use system memory (driver)", 1.0 / (1024**2)),
    ("system/mps_alloc_system_memory_mb", "Alloc system memory", 1.0 / (1024**2)),
]


def enable_mlflow_system_metrics() -> bool:
    """Globally enable ``mlflow.system_metrics`` for subsequent ``mlflow.start_run`` runs.

    Returns ``True`` on success, ``False`` when ``mlflow.system_metrics`` is missing.
    NOTE: this is NOT sufficient for Lightning's ``MLFlowLogger`` (it bypasses the
    fluent ``start_run``) — that path uses the callback in
    ``matrainer.lightning.callbacks`` / an :class:`MlflowTracker`. Kept for
    ``start_run``-based clients (notably Ultralytics' MLflow callback).
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
        except Exception as exc:  # pragma: no cover - depends on mlflow internals
            logger.debug(f"sampling-interval override ignored: {exc}")
    return True


def start_system_metrics_monitor(run_id: Optional[str], tracking_uri: Optional[str] = None) -> Optional[Any]:
    """Start MLflow's ``SystemMetricsMonitor`` (CPU/RAM/disk) for ``run_id``; return it or ``None``.

    The caller owns the returned monitor and must call ``.finish()`` on it. Returns
    ``None`` when ``run_id`` is falsy or ``mlflow.system_metrics`` is unavailable.
    """
    if not run_id:
        return None
    try:
        from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
    except ImportError:
        logger.warning("mlflow.system_metrics.SystemMetricsMonitor unavailable — install psutil + mlflow>=2.8.")
        return None
    sampling = _resolve_sample_interval(None)
    monitor = SystemMetricsMonitor(run_id=run_id, sampling_interval=sampling, tracking_uri=tracking_uri)
    monitor.start()
    logger.info(f"MLflow system-metrics monitor started for run {run_id} (interval={sampling}s).")
    return monitor


def _mps_available() -> bool:
    """True when ``torch`` is importable and reports an available MPS backend."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.backends.mps.is_available())


class MpsMetricsSampler:
    """Background Apple-Silicon MPS metrics sampler pushing straight to MLflow.

    A plain ``start()`` / ``stop()`` object (no Lightning coupling) that records nine
    ``system/mps_*`` scalars every ``sample_interval_s`` and pushes them via
    :meth:`mlflow.tracking.MlflowClient.log_metric`. Three per-process memory scalars
    come from ``torch.mps``; six system-level scalars come from the inlined IOKit
    ``PerformanceStatistics`` read. A no-op when MPS/torch is unavailable.

    The thread is a **daemon** and an ``atexit`` hook stops and joins it, so the sampler
    never touches ``torch.mps`` / IOKit / MLflow C state during interpreter teardown (which
    segfaults on macOS). It must be a daemon for that hook to run at all: Python joins every
    non-daemon thread BEFORE its exit hooks, so a run that failed without calling
    :meth:`stop` never ended.

    Args:
        mlflow_client: A live ``mlflow.tracking.MlflowClient`` to push to.
        run_id: The MLflow run id metrics are logged against.
        sample_interval_s: Seconds between records (``None`` = env var then 10s).
    """

    def __init__(self, mlflow_client: Any, run_id: Optional[str], sample_interval_s: Optional[float] = None) -> None:
        self.sample_interval_s: float = _resolve_sample_interval(sample_interval_s)
        self._client = mlflow_client
        self._run_id = run_id
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._step = 0
        self._atexit_registered = False

    @staticmethod
    def _record_metrics() -> Dict[str, float]:
        """One MPS snapshot — empty when MPS/torch isn't available."""
        metrics: Dict[str, float] = {}
        try:
            import torch
        except ImportError:
            return metrics
        if not torch.backends.mps.is_available():
            return metrics
        try:
            metrics["system/mps_allocated_mb"] = torch.mps.current_allocated_memory() / (1024**2)
            metrics["system/mps_driver_mb"] = torch.mps.driver_allocated_memory() / (1024**2)
            metrics["system/mps_recommended_max_mb"] = torch.mps.recommended_max_memory() / (1024**2)
        except Exception as exc:  # pragma: no cover - defensive
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

    def _record_loop(self) -> None:
        while not self._stop_event.is_set():
            metrics = self._record_metrics()
            if metrics and self._client is not None and self._run_id is not None:
                timestamp_ms = int(time.time() * 1000)
                step = self._step
                for key, value in metrics.items():
                    if self._stop_event.is_set():
                        return
                    try:
                        self._client.log_metric(
                            run_id=self._run_id, key=key, value=value, timestamp=timestamp_ms, step=step
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.debug(f"MPS metric '{key}' push failed: {exc}")
                self._step += 1
            self._stop_event.wait(self.sample_interval_s)

    def start(self) -> None:
        """Start sampling (idempotent; a no-op when MPS is unavailable)."""
        if self._thread is not None and self._thread.is_alive():
            return
        if not _mps_available() or self._run_id is None or self._client is None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._record_loop, name="MPSMetricsSampler", daemon=True)
        self._register_atexit_stop()
        self._thread.start()
        logger.info(f"MPS metrics sampler started for run {self._run_id} (interval={self.sample_interval_s}s).")

    def _register_atexit_stop(self) -> None:
        if self._atexit_registered:
            return
        ref = weakref.ref(self)

        def _atexit_stop() -> None:
            obj = ref()
            if obj is not None:
                obj.stop()

        atexit.register(_atexit_stop)
        self._atexit_registered = True

    def stop(self) -> None:
        """Signal the sampler thread and join it (idempotent)."""
        if self._thread is None:
            return
        self._stop_event.set()
        join_timeout = max(2.0, 2 * self.sample_interval_s)
        self._thread.join(timeout=join_timeout)
        if self._thread.is_alive():  # pragma: no cover - defensive
            logger.warning(f"MPS metrics sampler did not exit within {join_timeout}s — leaving it to atexit.")
            return
        self._thread = None
        self._step = 0


__all__ = [
    "DEFAULT_MPS_SAMPLE_INTERVAL_S",
    "MpsMetricsSampler",
    "enable_mlflow_system_metrics",
    "start_system_metrics_monitor",
]

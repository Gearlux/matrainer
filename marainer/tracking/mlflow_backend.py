"""``MlflowTracker`` — MLflow experiment-tracking backend (``pip install marainer[mlflow]``).

Drives MLflow through :class:`mlflow.tracking.MlflowClient` so the run lifecycle
is explicit and independent of the fluent ``mlflow.start_run`` global state (the
same reason ``pytorch_lightning.loggers.MLFlowLogger`` uses the client directly).
``mlflow`` is imported lazily on first :attr:`experiment` access — importing this
module needs no MLflow installed.

When ``log_system_metrics`` is set, the run also starts MLflow's own
``SystemMetricsMonitor`` (CPU / RAM / disk) plus, on Apple Silicon, the
framework-free :class:`marainer.tracking.system.MpsMetricsSampler` — so a
non-Lightning training platform gets system metrics for free by using this
tracker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Union

from confluid import configurable
from loggair import get_logger

from marainer._imports import require
from marainer.tracking.base import Tracker

logger = get_logger(__name__)


@configurable
class MlflowTracker(Tracker):
    """Track an experiment run in MLflow.

    Zero-arg constructible; the MLflow experiment + run are created on first
    :attr:`experiment` access (which imports ``mlflow`` lazily).

    Args:
        experiment_name: MLflow experiment name (created if absent).
        run_name: MLflow run name; ``None`` draws a random slug.
        save_dir: Unused for a remote tracking server; kept for interface parity.
        tags: Tags stamped on the run at creation.
        tracking_uri: MLflow tracking URI; ``None`` uses the ambient
            ``MLFLOW_TRACKING_URI`` / local ``mlruns``.
        log_system_metrics: Start CPU/RAM/disk (+ Apple MPS) sampling for the run.
    """

    def __init__(
        self,
        experiment_name: str = "marainer",
        run_name: Optional[str] = None,
        save_dir: str = "./runs",
        tags: Optional[Mapping[str, str]] = None,
        tracking_uri: Optional[str] = None,
        log_system_metrics: bool = True,
    ) -> None:
        super().__init__(experiment_name, run_name, save_dir, tags)
        self.tracking_uri = tracking_uri
        self.log_system_metrics = log_system_metrics
        self._client: Optional[Any] = None
        self._run_id: Optional[str] = None
        self._system_monitor: Optional[Any] = None
        self._mps_sampler: Optional[Any] = None

    @property
    def run_id(self) -> Optional[str]:
        """The MLflow run id (available after first :attr:`experiment` access)."""
        return self._run_id

    @property
    def experiment(self) -> Any:
        """The :class:`mlflow.tracking.MlflowClient`, creating the experiment + run lazily."""
        if self._client is None:
            mlflow = require("mlflow", "mlflow", feature="MlflowTracker")
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=self.tracking_uri)
            exp = client.get_experiment_by_name(self.experiment_name)
            exp_id = exp.experiment_id if exp is not None else client.create_experiment(self.experiment_name)
            run = client.create_run(experiment_id=exp_id, run_name=self.run_name, tags=self.tags or None)
            self._client = client
            self._run_id = run.info.run_id
            logger.info(f"MLflow run started: {self.experiment_name}/{self.run_name} ({self._run_id})")
            if self.log_system_metrics:
                self._start_system_metrics(mlflow)
        return self._client

    def _start_system_metrics(self, mlflow: Any) -> None:
        from marainer.tracking.system import MpsMetricsSampler, start_system_metrics_monitor

        self._system_monitor = start_system_metrics_monitor(self._run_id, self.tracking_uri)
        self._mps_sampler = MpsMetricsSampler(self._client, self._run_id)
        self._mps_sampler.start()

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        client = self.experiment
        for key, value in params.items():
            try:
                client.log_param(self._run_id, str(key), value)
            except Exception as exc:  # pragma: no cover - mlflow rejects re-logged params
                logger.debug(f"MLflow log_param failed for {key}: {exc}")

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        client = self.experiment
        for key, value in metrics.items():
            try:
                client.log_metric(self._run_id, str(key), float(value), step=step or 0)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"MLflow log_metric failed for {key}: {exc}")

    def set_tags(self, tags: Mapping[str, str]) -> None:
        super().set_tags(tags)
        client = self.experiment
        for key, value in tags.items():
            try:
                client.set_tag(self._run_id, str(key), str(value))
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"MLflow set_tag failed for {key}: {exc}")

    def log_artifact(self, path: Union[str, Path], artifact_dir: str = "") -> None:
        src = Path(path)
        if not src.exists():
            logger.debug(f"MlflowTracker.log_artifact: {src} does not exist; skipping.")
            return
        client = self.experiment
        try:
            client.log_artifact(self._run_id, str(src), artifact_dir or None)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"MLflow log_artifact failed for {src}: {exc}")

    def log_figure(self, name: str, figure: Any, step: Optional[int] = None) -> None:
        client = self.experiment
        safe = name.replace("/", "_")
        try:
            client.log_figure(self._run_id, figure, f"{safe}.png")
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"MLflow log_figure failed for {name}: {exc}")

    def finalize(self, status: str = "success") -> None:
        if self._mps_sampler is not None:
            self._mps_sampler.stop()
            self._mps_sampler = None
        if self._system_monitor is not None:
            try:
                self._system_monitor.finish()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"system-metrics monitor finish failed: {exc}")
            self._system_monitor = None
        if self._client is not None and self._run_id is not None:
            mlflow_status = "FINISHED" if status == "success" else "FAILED"
            try:
                self._client.set_terminated(self._run_id, status=mlflow_status)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"MLflow set_terminated failed: {exc}")


__all__ = ["MlflowTracker"]

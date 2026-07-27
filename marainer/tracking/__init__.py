"""Marainer experiment-tracking core.

Platform-neutral tracking of parameters, metrics, git provenance, config
artifacts, log files, figures, and system metrics — for any logger library and any
training platform.

* :class:`Tracker` / :class:`TrackerGroup` — the backend ABC (shaped like
  ``pytorch_lightning.loggers.Logger``) + a fan-out composite.
* :class:`CsvTracker` — the always-available, zero-dependency default.
* :class:`MlflowTracker` / :class:`TensorBoardTracker` / :class:`WandbTracker` —
  backends behind the ``[mlflow]`` / ``[tensorboard]`` / ``[wandb]`` extras, each
  importing its library lazily (importing this package pulls in none of them).
* :mod:`~marainer.tracking.provenance` — git hash / config-artifact / log-file
  capture + the :func:`~marainer.tracking.provenance.log_run_context` orchestrator.
* :mod:`~marainer.tracking.system` — framework-free system-metric sampling.
* :func:`~marainer.tracking.render.render_matrix_as_table` — confusion-matrix table.

Entry-pointed under ``confluid.configurables`` (``marainer-tracking``) so the
backends are discoverable by navigaitor / StreamStudio.
"""

from marainer.tracking.base import Tracker, TrackerGroup
from marainer.tracking.csv_backend import CsvTracker
from marainer.tracking.mlflow_backend import MlflowTracker
from marainer.tracking.provenance import (
    capture_log_file,
    get_git_commit_hash,
    included_config_paths,
    log_run_context,
    resolve_log_file,
)
from marainer.tracking.render import render_matrix_as_table
from marainer.tracking.system import MpsMetricsSampler, enable_mlflow_system_metrics, start_system_metrics_monitor
from marainer.tracking.tensorboard_backend import TensorBoardTracker
from marainer.tracking.wandb_backend import WandbTracker

__all__ = [
    "Tracker",
    "TrackerGroup",
    "CsvTracker",
    "MlflowTracker",
    "TensorBoardTracker",
    "WandbTracker",
    "MpsMetricsSampler",
    "enable_mlflow_system_metrics",
    "start_system_metrics_monitor",
    "capture_log_file",
    "get_git_commit_hash",
    "included_config_paths",
    "log_run_context",
    "resolve_log_file",
    "render_matrix_as_table",
]

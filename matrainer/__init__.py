"""Matrainer — experiment tracking for any training platform.

Matrainer's core is a light experiment-tracking library: log and track parameters,
git provenance, log files, config artifacts, figures, metrics, and system metrics
to various logger libraries (MLflow / TensorBoard / CSV / W&B) from ANY training
platform. The heavy per-framework training code lives in opt-in subpackages —
:mod:`matrainer.lightning`, :mod:`matrainer.keras` — behind extras, and the
logger libraries are imported lazily by their backend classes only.

A plain ``import matrainer`` stays light (no torch / pytorch_lightning): it exposes
the tracking surface and the framework-neutral checkpoint policies.
"""

from importlib.metadata import PackageNotFoundError, version
from typing import Any, List

from matrainer.checkpoints import CheckpointPolicy, SaveBest, SaveEveryEpoch, SaveLast
from matrainer.metric_names import MetricNames
from matrainer.run_names import generate_run_name
from matrainer.tracking import ResilientTracker, RunManifest, Tracker, TrackerGroup, discover_backends, log_run_context

try:
    # Single source of truth: the installed distribution's metadata, i.e. pyproject.toml's
    # `version` (the distribution is `matrainer`). Never type the number in here as well.
    __version__ = version("matrainer")
except PackageNotFoundError:  # pragma: no cover - uninstalled source checkout
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "Tracker",
    "TrackerGroup",
    "ResilientTracker",
    "RunManifest",
    "log_run_context",
    "generate_run_name",
    "MetricNames",
    "CheckpointPolicy",
    "SaveLast",
    "SaveEveryEpoch",
    "SaveBest",
]

# The tracker backends are re-exported by DISCOVERY, not by name: listing them here would
# be a second place a new backend has to be registered (see matrainer.tracking).
__all__ += sorted(discover_backends())


def __getattr__(name: str) -> Any:
    """Re-export a discovered backend from the package root (PEP 562)."""
    backends = discover_backends()
    if name in backends:
        return backends[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(__all__)

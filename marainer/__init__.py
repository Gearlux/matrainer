"""Marainer — experiment tracking for any training platform.

Marainer's core is a light experiment-tracking library: log and track parameters,
git provenance, log files, config artifacts, figures, metrics, and system metrics
to various logger libraries (MLflow / TensorBoard / CSV / W&B) from ANY training
platform. The heavy per-framework training code lives in opt-in subpackages —
:mod:`marainer.lightning`, :mod:`marainer.keras` — behind extras, and the
logger libraries are imported lazily by their backend classes only.

A plain ``import marainer`` stays light (no torch / pytorch_lightning): it exposes
the tracking surface and the framework-neutral checkpoint policies.
"""

__version__ = "0.2.0"

from typing import Any, List

from marainer.checkpoints import CheckpointPolicy, SaveBest, SaveEveryEpoch, SaveLast
from marainer.run_names import generate_run_name
from marainer.tracking import RunManifest, Tracker, TrackerGroup, discover_backends, log_run_context

__all__ = [
    "__version__",
    "Tracker",
    "TrackerGroup",
    "RunManifest",
    "log_run_context",
    "generate_run_name",
    "CheckpointPolicy",
    "SaveLast",
    "SaveEveryEpoch",
    "SaveBest",
]

# The tracker backends are re-exported by DISCOVERY, not by name: listing them here would
# be a second place a new backend has to be registered (see marainer.tracking).
__all__ += sorted(discover_backends())


def __getattr__(name: str) -> Any:
    """Re-export a discovered backend from the package root (PEP 562)."""
    backends = discover_backends()
    if name in backends:
        return backends[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(__all__)

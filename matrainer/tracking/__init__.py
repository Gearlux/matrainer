"""Matrainer experiment-tracking core.

Platform-neutral tracking of parameters, metrics, git provenance, config
artifacts, log files, figures, and system metrics — for any logger library and any
training platform.

* :class:`Tracker` / :class:`TrackerGroup` — the backend ABC (shaped like
  ``pytorch_lightning.loggers.Logger``) + a fan-out composite.
* :class:`ResilientTracker` — a composite around ONE tracker that time-limits each call and
  queues what the network loses, replaying it in order (``resilient.py``).
* :class:`RunManifest` — the crash-safe union of everything static about a run
  (``manifest.yaml`` + ``manifest.json``), of which each backend's native slots are
  a projection.
* :mod:`~matrainer.tracking.provenance` — git / config / environment capture plus
  :func:`~matrainer.tracking.provenance.log_run_context`, the one call at run start.
* :mod:`~matrainer.tracking.system` — framework-free system-metric sampling.
* :func:`~matrainer.tracking.render.per_class_vectors` — the per-class VECTORS in a metrics
  result, selected by shape, each rendered by
  :func:`~matrainer.tracking.render.render_vector_as_table` (a worst-first markdown table) or
  :func:`~matrainer.tracking.render.render_vector_as_figure` (a sorted bar chart).
* :func:`~matrainer.tracking.render.render_matrix_as_table` /
  :func:`~matrainer.tracking.render.render_matrix_as_figure` — a confusion matrix as a
  markdown table (for ``log_text``) or a heatmap (for ``log_figure``).

**Backends are discovered, not listed.** Every ``*_backend.py`` module in this package
is scanned for :class:`Tracker` subclasses on first use (PEP 562 module ``__getattr__``),
so adding one is a single new file — no import line, no ``__all__`` entry, no registry.
``CsvTracker`` (stdlib only) is always available; ``MlflowTracker`` / ``TensorBoardTracker``
/ ``WandbTracker`` / ``AimTracker`` sit behind their extras and import their library
lazily, so importing this package pulls in none of them.

Entry-pointed under ``confluid.configurables`` (``matrainer-tracking``) so the backends are
discoverable by config-driven tooling — a backend shipped by a DIFFERENT distribution is
discovered there without any change here.
"""

import importlib
import pkgutil
from typing import Any, Dict, List, Optional, Type

from matrainer.tracking.base import Tracker, TrackerGroup
from matrainer.tracking.manifest import RunManifest, RunStatus
from matrainer.tracking.provenance import (
    capture_log_file,
    ending_the_run,
    get_git_commit_hash,
    included_config_paths,
    log_run_context,
    resolve_log_file,
)
from matrainer.tracking.render import (
    per_class_vectors,
    render_matrix_as_figure,
    render_matrix_as_table,
    render_vector_as_figure,
    render_vector_as_table,
    report_matrices,
    report_vectors,
)
from matrainer.tracking.resilient import ResilientTracker
from matrainer.tracking.system import MpsMetricsSampler, enable_mlflow_system_metrics, start_system_metrics_monitor

#: Suffix marking a module in this package as holding a backend.
BACKEND_SUFFIX = "_backend"

# Literal, so static readers (pyflakes, IDEs) credit the re-exports above. The discovered
# backends are appended below rather than listed here — that list is the thing a new
# backend must not have to edit.
__all__ = [
    "Tracker",
    "TrackerGroup",
    "ResilientTracker",
    "RunManifest",
    "RunStatus",
    "MpsMetricsSampler",
    "enable_mlflow_system_metrics",
    "start_system_metrics_monitor",
    "capture_log_file",
    "discover_backends",
    "ending_the_run",
    "get_git_commit_hash",
    "included_config_paths",
    "log_run_context",
    "resolve_log_file",
    "per_class_vectors",
    "render_matrix_as_figure",
    "render_matrix_as_table",
    "render_vector_as_figure",
    "render_vector_as_table",
    "report_matrices",
    "report_vectors",
]

#: Snapshot of the names that exist independent of backend discovery.
_CORE = list(__all__)

_backends: Optional[Dict[str, Type[Tracker]]] = None


def discover_backends() -> Dict[str, Type[Tracker]]:
    """Map ``class name -> Tracker subclass`` for every backend module in this package.

    Scanning beats a hand-written import list because the list is a second place to
    forget: a new backend used to need a new import line AND an ``__all__`` entry, and an
    out-of-tree backend could never appear in either. Importing the modules is cheap and
    safe — each imports its own library lazily inside its methods, which
    ``tests/test_light_import.py`` pins — so this pulls in no mlflow / wandb / aim / torch.

    Cached after the first call; the result is a copy, so a caller cannot poison it.
    """
    global _backends
    if _backends is None:
        found: Dict[str, Type[Tracker]] = {}
        for info in pkgutil.iter_modules(__path__):
            if not info.name.endswith(BACKEND_SUFFIX):
                continue
            module = importlib.import_module(f"{__name__}.{info.name}")
            for attr, value in vars(module).items():
                if isinstance(value, type) and issubclass(value, Tracker) and value is not Tracker:
                    found[attr] = value
        _backends = found
    return dict(_backends)


def __getattr__(name: str) -> Any:
    """Resolve a backend class on first access (PEP 562)."""
    backends = discover_backends()
    if name in backends:
        return backends[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(_CORE + list(discover_backends()))


# Appended from the same scan `__getattr__` resolves through, so the two can never
# disagree about which backends exist.
__all__ += sorted(discover_backends())

"""Metric construction + coercion for torch runnables — **Lightning-free**.

* :func:`build_metric_collection` — normalize any metrics spec (mapping / list /
  single metric / ``MetricCollection``) into ONE ``MetricCollection`` the caller
  clones per split.
* :func:`coerce_metric_value` / :func:`scalar_metrics` — JSON-safe coercion of a
  torchmetrics ``compute()`` result (used by the ``@output`` properties + the
  metrics sidecar).
* :class:`MetricPreset` — category-based torchmetrics discovery factory.

This module lived under ``matrainer.lightning`` until 2026-08-01, which was a statement about
its first CONSUMER rather than about its dependencies: nothing in it imports
``pytorch_lightning``, only confluid, torch and torchmetrics. The move was forced by a consumer
that trains on plain torch with no ``Trainer`` at all — importing the helpers from the old home
pulled ``pytorch_lightning`` in as a side effect of the package ``__init__``'s ``require()``,
which made a deliberately Lightning-free backend depend on Lightning. ``matrainer.lightning``
re-exports every name so its public surface is unchanged.
"""

import importlib
import inspect
from collections.abc import Mapping
from typing import Any, Callable, Dict, Optional, Union

from confluid import PartialClass, configurable, flow
from confluid.fluid import Fluid
from loggair import get_logger
from torch import Tensor
from torchmetrics import Metric, MetricCollection

from matrainer.artifacts import write_metrics_sidecar

logger = get_logger(__name__)

#: What a runnable's metrics slot admits: a ``{name: template}`` mapping, a ready-made
#: collection, or ``None`` for "not chosen" (the sentinel a task resolves in ``initialize()``).
#: Named ONCE here because four slots per runnable share it across every torch-family backend
#: and `to_pydantic` reads the annotation for form-specs / widget generation — it was declared
#: verbatim in eight-plus consumer modules before the 2026-08-06 consolidation.
MetricsSpec = Optional[Union[Dict[str, Union[Metric, PartialClass[Metric]]], MetricCollection]]


def _declares(target: Any, key: str) -> bool:
    """Does ``target``'s constructor DECLARE ``key`` by name (ignoring ``**kwargs``)?

    Since confluid 0.3.0 this IS :func:`confluid.declares_key` (added for exactly this case —
    ``accepts_key`` answers yes for any ``**kwargs`` target, which is EVERY torchmetrics metric,
    and the metric then REJECTS undeclared names with *"Unexpected keyword arguments"*). What
    stays OURS is the fallback policy below.

    An unresolvable target (a dotted string a marker carries, a callable with no signature) is
    treated as accepting the key, which preserves the pre-2026-08-05 behaviour for anything this
    cannot see rather than silently dropping a dimension the metric needed.
    """
    from confluid import declares_key
    from confluid.broadcast import _settability_target

    if _settability_target(target) is None:  # unresolvable — permissive, OUR policy (confluid answers False)
        return True
    return declares_key(target, key)


def _sizing_kwargs(template: Any, flow_dims: Mapping[str, Any]) -> Dict[str, Any]:
    """The subset of ``flow_dims`` this template can TAKE and has not PINNED for itself.

    Two filters, and each closes a way the blanket form fails:

    **Declares.** A task's sizing dimension is not universal across its own metric family. A
    classification run's ``num_classes`` happens to reach every ``torchmetrics.classification``
    metric, so the blanket form worked by luck; an image task's ``data_range`` reaches three of
    the seven full-reference metrics (measured against torchmetrics 1.9: ``PeakSignalNoiseRatio``
    — where it is REQUIRED — plus the two SSIMs; ``SpectralAngleMapper``,
    ``UniversalImageQualityIndex``, ``ErrorRelativeGlobalDimensionlessSynthesis`` and
    ``RootMeanSquaredErrorUsingSlidingWindow`` take no such argument). Passing it to one that does
    not is not a no-op — the metric raises — so wiring a perfectly valid metric beside the others
    would break the whole collection.

    **Pinned.** A runtime kwarg normally OVERRIDES a marker's own, which would make a template's
    explicit value unwritable: ``!lazy:PeakSignalNoiseRatio {data_range: 2.0}`` would silently be
    scored on the run's scale instead of the one it asked for. Skipping a key the template already
    carries makes the injection a FLOOR — the task supplies what the config did not.
    """
    if not flow_dims:
        return {}
    target = template.target if isinstance(template, Fluid) else template
    pinned = template.kwargs if isinstance(template, Fluid) else {}
    return {key: value for key, value in flow_dims.items() if key not in pinned and _declares(target, key)}


def build_metric_collection(metrics: Any, **flow_dims: Any) -> MetricCollection:
    """Normalize any metrics spec into a ``MetricCollection`` the caller clones per split.

    Accepts every shape a task's ``metrics`` slot can hold (so a GUI can wire a single
    metric node into it):

    * **MetricCollection** — returned as-is.
    * **Mapping** ``{name: template}`` — each value flowed with ``flow_dims``, keyed by name.
    * **a single metric OR a list/tuple** — each a live ``Metric``, a ``PartialClass``/``Fluid``
      template, or a metric class; flowed with ``flow_dims`` and wrapped (auto-named by class).
      This lets a single metric node (including its deferred ``class`` output) wire into the slot
      and size itself off the model head at run time.

    ``flow_dims`` is the task's SIZING dimension — ``num_classes`` for a classification or
    per-pixel task, ``data_range`` for an image-to-image one — and it is applied PER TEMPLATE
    rather than blanket: see :func:`_sizing_kwargs` for the two filters and what each one prevents.
    """
    if isinstance(metrics, MetricCollection):
        return metrics
    if isinstance(metrics, Mapping):
        return MetricCollection({k: flow(v, **_sizing_kwargs(v, flow_dims)) for k, v in metrics.items()})
    seq = metrics if isinstance(metrics, (list, tuple)) else [metrics]
    return MetricCollection([flow(v, **_sizing_kwargs(v, flow_dims)) for v in seq])


def coerce_metric_value(value: Any) -> Any:
    """Convert a torchmetrics ``compute()`` value into a JSON-safe Python scalar / nested list.

    Tensors become floats (or ints for integer scalars); nested dicts / lists / tuples recurse.
    Anything already JSON-safe passes through.
    """
    if isinstance(value, Tensor):
        if value.numel() == 1:
            try:
                return value.item()
            except RuntimeError:
                return str(value)
        return [coerce_metric_value(v) for v in value.detach().cpu().tolist()]
    if isinstance(value, dict):
        return {k: coerce_metric_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [coerce_metric_value(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def scalar_metrics(raw: Any) -> Dict[str, float]:
    """Coerce a torchmetrics ``compute()`` result to a flat ``{name: float}`` of SCALARS only.

    Non-scalar metrics (e.g. an ``N×N`` confusion matrix) are dropped: this feeds channels that
    hold one number per name (``Tracker.log_metrics``, a metrics sidecar), and a caller that wants
    a matrix reported renders it separately — see
    :func:`~matrainer.tracking.render.render_matrix_as_table`. ``bool`` is excluded (not a metric
    value).
    """
    payload = coerce_metric_value(raw if isinstance(raw, dict) else {"value": raw})
    out: Dict[str, float] = {}
    if isinstance(payload, dict):
        for key, val in payload.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                out[str(key)] = float(val)
    return out


def write_collection_sidecar(
    path: Any,
    collection: Any,
    experiment_name: str,
    run_name: str,
    *,
    reportable: Optional[Callable[[Any], Any]] = None,
) -> None:
    """Compute a split's ``MetricCollection`` once and hand its SCALARS to the metrics sidecar.

    The composition every torch-family runnable writes at the end of an eval run: guard that a
    live collection was actually built (a spec-holding or ``None`` slot is a silent no-op —
    the run may legitimately have no metrics), compute it, drop non-scalars via
    :func:`scalar_metrics`, and delegate the write to
    :func:`matrainer.artifacts.write_metrics_sidecar` (which owns where run artifacts land).
    A compute failure is a warning, never an abort — the sidecar is a record, not the run.

    Args:
        path: The sidecar path (``None`` lets ``write_metrics_sidecar`` pick its default).
        collection: The runnable's post-run metrics slot — anything that is not a live
            ``Metric`` / ``MetricCollection`` is skipped.
        experiment_name: Stamped into the sidecar as the run identity.
        run_name: Stamped into the sidecar as the run identity.
        reportable: Optional filter applied to the computed mapping BEFORE the scalar drop —
            a task that must strip a pseudo-metric from every reporting path (a detection
            MAP's ``classes`` key) passes its rule here.
    """
    if not isinstance(collection, (Metric, MetricCollection)):
        return
    try:
        values = collection.compute()
        if reportable is not None:
            values = reportable(values)
        payload = scalar_metrics(values)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Could not compute final metrics for the sidecar: {exc}")
        return
    write_metrics_sidecar(path, payload, experiment_name, run_name)


# Metrics that return non-scalar values and cannot be logged with self.log_dict(), by category.
_DEFAULT_EXCLUDE: dict[str, set[str]] = {
    "classification": {
        "BinaryConfusionMatrix",
        "BinaryROC",
        "BinaryPrecisionRecallCurve",
        "BinaryStatScores",
        "BinaryFairness",
        "BinaryGroupStatRates",
        "MulticlassConfusionMatrix",
        "MulticlassROC",
        "MulticlassPrecisionRecallCurve",
        "MulticlassStatScores",
        "MultilabelConfusionMatrix",
        "MultilabelROC",
        "MultilabelPrecisionRecallCurve",
        "MultilabelStatScores",
        "ConfusionMatrix",
        "ROC",
        "PrecisionRecallCurve",
        "StatScores",
    },
}

_TASK_PREFIX: dict[str, str] = {"binary": "Binary", "multiclass": "Multiclass", "multilabel": "Multilabel"}


@configurable
class MetricPreset:
    """Factory that discovers and instantiates all torchmetrics for a given category.

    When materialized from YAML, produces a set of metric instances via ``to_metrics()``.

    Args:
        category: torchmetrics sub-module name (e.g. ``"classification"``, ``"regression"``).
        task: Task variant filter (``"binary"`` / ``"multiclass"`` / ``"multilabel"``); filters
            metrics by class-name prefix.
        exclude: Additional metric class names to skip (merged with built-in non-scalar exclusions).
        num_classes: Passed to metrics that accept it.
        num_labels: Passed to metrics that accept it (multilabel tasks).
        average: Averaging strategy (``"macro"`` / ``"micro"`` / ``"weighted"``).

    Example YAML::

        metrics: !class:matrainer.lightning.MetricPreset()
          category: classification
          task: multiclass
          num_classes: 10
    """

    def __init__(
        self,
        category: str = "classification",
        task: str = "",
        exclude: list[str] | None = None,
        num_classes: int | None = None,
        num_labels: int | None = None,
        average: str = "macro",
    ) -> None:
        self.category = category
        self.task = task
        self.exclude = exclude
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.average = average

    def to_metrics(self) -> dict[str, Metric]:
        """Discover and instantiate all compatible metrics for this preset.

        Returns a dict mapping metric class names to instances; metrics whose required
        parameters cannot be satisfied are skipped with a warning.
        """
        module = importlib.import_module(f"torchmetrics.{self.category}")
        all_names: list[str] = getattr(module, "__all__", [])

        prefix = _TASK_PREFIX.get(self.task, self.task.capitalize() if self.task else "")
        if prefix:
            all_names = [n for n in all_names if n.startswith(prefix)]

        effective_exclude = set(self.exclude or [])
        effective_exclude.update(_DEFAULT_EXCLUDE.get(self.category, set()))
        all_names = [n for n in all_names if n not in effective_exclude]

        shared_kwargs: dict[str, Any] = {}
        if self.num_classes is not None:
            shared_kwargs["num_classes"] = self.num_classes
        if self.num_labels is not None:
            shared_kwargs["num_labels"] = self.num_labels
        shared_kwargs["average"] = self.average

        metrics: dict[str, Metric] = {}
        for name in sorted(all_names):
            cls = getattr(module, name, None)
            if cls is None or not (inspect.isclass(cls) and issubclass(cls, Metric)):
                continue
            # Only pass kwargs the metric's __init__ explicitly names — torchmetrics
            # rejects unexpected arguments even when **kwargs is present.
            try:
                sig = inspect.signature(cls.__init__)
            except (ValueError, TypeError):
                continue
            explicit_params = {
                p.name
                for p in sig.parameters.values()
                if p.name not in ("self", "cls") and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            }
            filtered_kwargs = {k: v for k, v in shared_kwargs.items() if k in explicit_params}
            try:
                metrics[name] = cls(**filtered_kwargs)
            except (TypeError, ValueError) as e:
                logger.warning(f"Skipping metric {name}: {e}")

        logger.info(f"MetricPreset({self.category}, task={self.task!r}) loaded {len(metrics)} metrics")
        return metrics

    def available_metrics(self) -> list[str]:
        """List metric class names that would be instantiated by this preset."""
        module = importlib.import_module(f"torchmetrics.{self.category}")
        all_names: list[str] = getattr(module, "__all__", [])
        prefix = _TASK_PREFIX.get(self.task, self.task.capitalize() if self.task else "")
        if prefix:
            all_names = [n for n in all_names if n.startswith(prefix)]
        effective_exclude = set(self.exclude or [])
        effective_exclude.update(_DEFAULT_EXCLUDE.get(self.category, set()))
        return sorted(n for n in all_names if n not in effective_exclude)


__all__ = [
    "MetricPreset",
    "MetricsSpec",
    "build_metric_collection",
    "coerce_metric_value",
    "scalar_metrics",
    "write_collection_sidecar",
]

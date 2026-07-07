"""Preset-based metric discovery for Marainer trainers.

Provides MetricPreset for category-based metric discovery and instantiation.
"""

import importlib
import inspect
from typing import Any

from confluid import configurable
from loggair import get_logger
from torchmetrics import Metric

logger = get_logger(__name__)

# Metrics that return non-scalar values and cannot be logged with self.log_dict().
# Organized by torchmetrics category.
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
        # Generic wrappers that require a 'task' argument
        "ConfusionMatrix",
        "ROC",
        "PrecisionRecallCurve",
        "StatScores",
    },
}

# Mapping from task shorthand to class name prefix
_TASK_PREFIX: dict[str, str] = {
    "binary": "Binary",
    "multiclass": "Multiclass",
    "multilabel": "Multilabel",
}


@configurable
class MetricPreset:
    """Factory that discovers and instantiates all torchmetrics for a given category.

    When materialized from YAML, produces a set of metric instances via
    ``to_metrics()`` for use in MetricCollection.

    Args:
        category: torchmetrics sub-module name (e.g. ``"classification"``,
            ``"regression"``, ``"segmentation"``).
        task: Task variant filter. For classification, one of ``"binary"``,
            ``"multiclass"``, or ``"multilabel"``. Filters metrics by class
            name prefix.
        exclude: Additional metric class names to skip (merged with
            built-in non-scalar exclusions).
        num_classes: Passed to metrics that accept it.
        num_labels: Passed to metrics that accept it (multilabel tasks).
        average: Averaging strategy (``"macro"``, ``"micro"``, ``"weighted"``).

    Example YAML::

        classification_metrics: !class:marainer.presets.MetricPreset()
          category: classification
          task_type: multiclass
          num_classes: !ref:num_classes

        train_metrics: !class:torchmetrics.MetricCollection()
          metrics: !ref:classification_metrics.to_metrics()
          prefix: "train/"
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

        Returns:
            Dictionary mapping metric class names to instantiated Metric objects.
            Metrics whose required parameters cannot be satisfied are skipped
            with a warning.
        """
        task = self.task
        module = importlib.import_module(f"torchmetrics.{self.category}")
        all_names: list[str] = getattr(module, "__all__", [])

        # Filter by task prefix
        prefix = _TASK_PREFIX.get(task, task.capitalize() if task else "")
        if prefix:
            all_names = [n for n in all_names if n.startswith(prefix)]

        # Apply exclusions
        effective_exclude = set(self.exclude or [])
        effective_exclude.update(_DEFAULT_EXCLUDE.get(self.category, set()))
        all_names = [n for n in all_names if n not in effective_exclude]

        # Build shared kwargs (only non-None values)
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

            # Only pass kwargs that the metric's __init__ explicitly names.
            # Avoid passing through **kwargs — torchmetrics validates internally
            # and rejects unexpected arguments even when **kwargs is present.
            try:
                sig = inspect.signature(cls.__init__)
            except (ValueError, TypeError):
                continue

            explicit_params = {
                p.name
                for p in sig.parameters.values()
                if p.name not in ("self", "cls") and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
            }

            filtered_kwargs: dict[str, Any] = {}
            for k, v in shared_kwargs.items():
                if k in explicit_params:
                    filtered_kwargs[k] = v

            try:
                metrics[name] = cls(**filtered_kwargs)
            except (TypeError, ValueError) as e:
                logger.warning(f"Skipping metric {name}: {e}")

        logger.info(f"MetricPreset({self.category}, task={task!r}) loaded {len(metrics)} metrics")
        return metrics

    def available_metrics(self) -> list[str]:
        """List metric class names that would be instantiated by this preset.

        Useful for discovery without actually creating metric instances.
        """
        task = self.task
        module = importlib.import_module(f"torchmetrics.{self.category}")
        all_names: list[str] = getattr(module, "__all__", [])

        prefix = _TASK_PREFIX.get(task, task.capitalize() if task else "")
        if prefix:
            all_names = [n for n in all_names if n.startswith(prefix)]

        effective_exclude = set(self.exclude or [])
        effective_exclude.update(_DEFAULT_EXCLUDE.get(self.category, set()))
        return sorted(n for n in all_names if n not in effective_exclude)

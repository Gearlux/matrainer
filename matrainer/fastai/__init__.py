"""The fastai machinery base (``pip install matrainer[fastai]``).

:class:`~matrainer.fastai.learner.FastaiRunnable` fills the trainer API in with a fastai
``Learner`` — the third machinery base beside the Lightning one and the owned torch loop. The
extra also carries two measured upstream bounds (``fastcore<2`` / ``fastprogress<1.1``); see
``pyproject.toml`` for the breakages they pin out.

Importing this subpackage requires ``fastai`` (raised with a clear
``pip install matrainer[fastai]`` message when absent).
"""

from matrainer._imports import require

require("fastai", "fastai", feature="matrainer.fastai")

from matrainer.fastai.learner import (  # noqa: E402
    FastaiLoss,
    FastaiMetric,
    FastaiOptimizer,
    FastaiRunnable,
    as_opt_func,
    log_recorded_epochs,
    metric_columns,
    named_values,
    recorded_metrics,
)

__all__ = [
    "FastaiLoss",
    "FastaiMetric",
    "FastaiOptimizer",
    "FastaiRunnable",
    "as_opt_func",
    "log_recorded_epochs",
    "metric_columns",
    "named_values",
    "recorded_metrics",
]

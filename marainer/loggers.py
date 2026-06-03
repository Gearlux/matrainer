"""Logger-side helpers for Marainer runnables.

This module holds the small utilities that *interact with* the wired
PyTorch-Lightning loggers — it deliberately defines no logger subclasses.
Configs wire the stock ``pytorch_lightning.loggers.*`` classes directly.

* :func:`log_figure_to_loggers` — forward a matplotlib ``Figure`` to every
  wired Lightning logger, bridging the TensorBoard-style
  (``experiment.add_figure``) and MLflow-style (``experiment.log_figure``)
  APIs. Used by :class:`marainer.evaluator.LightningEvaluator` to push
  confusion-matrix / diagnostic plots.
* :func:`_render_matrix_as_table` — render a square 2D tensor (e.g. a
  torchmetrics confusion matrix) as a github-flavored markdown table.
"""

from typing import Any, List, Optional

import pytorch_lightning as L
from logflow import get_logger
from torch import Tensor

logger = get_logger(__name__)


def _render_matrix_as_table(name: str, matrix: Tensor, class_names: List[str]) -> str:
    """Render a square 2D tensor as a github-flavored markdown table.

    Rows are true labels, columns are predicted labels (the torchmetrics
    confusion-matrix convention: ``C[i, j]`` = count of ``true=i, pred=j``).
    """
    from tabulate import tabulate

    arr = matrix.detach().cpu().tolist()
    headers = [""] + [f"pred:{c}" for c in class_names]
    rows = [[f"true:{class_names[i]}", *arr[i]] for i in range(len(arr))]
    rendered: str = tabulate(rows, headers=headers, tablefmt="github")
    return f"{name}:\n" + rendered


def log_figure_to_loggers(lightning_trainer: Optional[L.Trainer], name: str, figure: Any) -> None:
    """Forward a matplotlib Figure to every wired Lightning logger.

    TensorBoard-style loggers expose ``experiment.add_figure(tag, fig, step)``;
    MLflow-style loggers expose ``experiment.log_figure(run_id, fig, path)``.
    Loggers without either are silently skipped.
    """
    if lightning_trainer is None:
        return
    step = getattr(lightning_trainer, "global_step", 0)
    for lg in getattr(lightning_trainer, "loggers", []) or []:
        exp = getattr(lg, "experiment", None)
        if exp is None:
            continue
        add_figure = getattr(exp, "add_figure", None)
        if callable(add_figure):
            try:
                add_figure(name, figure, step)
                continue
            except Exception as exc:  # pragma: no cover - logger backends are external
                logger.debug(f"add_figure failed on {type(lg).__name__}: {exc}")
        log_figure = getattr(exp, "log_figure", None)
        run_id = getattr(lg, "run_id", None)
        if callable(log_figure) and run_id:
            try:
                log_figure(run_id, figure, f"{name.replace('/', '_')}.png")
            except Exception as exc:  # pragma: no cover - logger backends are external
                logger.debug(f"log_figure failed on {type(lg).__name__}: {exc}")


__all__ = [
    "log_figure_to_loggers",
]

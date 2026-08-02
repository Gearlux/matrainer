"""Human-readable rendering helpers for tracked artifacts.

A run value that is neither a number nor an image — the canonical case is a confusion matrix —
reaches a tracker through one of the two functions here:

* :func:`render_matrix_as_table` → a markdown table for :meth:`Tracker.log_text`. Cheap, works
  everywhere, greppable, and readable in a plain log.
* :func:`render_matrix_as_figure` → a matplotlib heatmap for :meth:`Tracker.log_figure`.
  Glanceable, and the only form in which a large matrix is readable at all — but it needs
  matplotlib (``pip install marainer[plot]``), which is why it is the opt-in of the two rather
  than the default.

Both are framework-free: the matrix only needs ``.detach().cpu().tolist()`` (a torch tensor),
``.tolist()`` (a numpy array), or to already be a nested list — so this module imports with the
light core and neither ``tabulate`` nor ``matplotlib`` is imported until it is used.
"""

from __future__ import annotations

from typing import Any, List, Sequence

from marainer._imports import require

__all__ = ["render_matrix_as_figure", "render_matrix_as_table"]


def _as_rows(matrix: Any) -> List[List[Any]]:
    """The matrix as a nested list, whatever array type it arrived as.

    A live torchmetrics value is on a device and possibly in a graph, so ``detach().cpu()``
    comes first; a numpy array only needs ``tolist``; a payload built by a caller is already
    a nested list.
    """
    if callable(getattr(matrix, "detach", None)):
        return list(matrix.detach().cpu().tolist())
    to_list = getattr(matrix, "tolist", None)
    if callable(to_list):
        return list(to_list())
    return list(matrix)


def render_matrix_as_table(name: str, matrix: Any, class_names: Sequence[str]) -> str:
    """Render a square 2-D matrix as a GitHub-flavored markdown table.

    Rows are true labels, columns are predicted labels (the torchmetrics
    confusion-matrix convention: ``C[i, j]`` = count of ``true=i, pred=j``).

    Args:
        name: A title printed above the table.
        matrix: A square 2-D array/tensor (anything with ``.detach().cpu().tolist()``)
            or an already-materialized nested list.
        class_names: Row/column labels (length = matrix dimension).
    """
    from tabulate import tabulate

    arr = _as_rows(matrix)
    headers = [""] + [f"pred:{c}" for c in class_names]
    rows = [[f"true:{class_names[i]}", *arr[i]] for i in range(len(arr))]
    rendered: str = tabulate(rows, headers=headers, tablefmt="github")
    # A BLANK line after the title, not just a newline: the result is handed to
    # `Tracker.log_text` as well as to a log, and every markdown renderer (TensorBoard's
    # Text tab, MLflow's `.md` artifact preview) needs one to see a table rather than one
    # run-on paragraph. In a plain log it costs a blank line.
    return f"{name}:\n\n{rendered}"


def render_matrix_as_figure(
    name: str,
    matrix: Any,
    class_names: Sequence[str],
    cmap: str = "Blues",
) -> Any:
    """Render a square 2-D matrix as a heatmap ``matplotlib.figure.Figure``.

    The counterpart of :func:`render_matrix_as_table` for :meth:`Tracker.log_figure`. Same
    orientation — rows are true labels, columns predicted — with each cell annotated, so the
    figure carries the numbers as well as the shading.

    Built through ``matplotlib.figure.Figure`` DIRECTLY rather than ``pyplot.subplots``, and
    that is the load-bearing detail: a pyplot figure is registered in a process-global manager
    and stays alive until someone calls ``pyplot.close``, so a value rendered once an epoch
    leaks a figure per epoch in a caller that has no reason to know about pyplot's state. A bare
    ``Figure`` is owned by whoever holds the reference and is collected with it.

    Cell text is dropped past 25 classes: the annotations become unreadable well before that and
    the render cost grows with N². The heatmap itself stays informative at any size, which is the
    reason this form exists beside the table.

    Args:
        name: The figure title.
        matrix: A square 2-D array/tensor, or an already-materialized nested list.
        class_names: Row/column labels (length = matrix dimension).
        cmap: Any matplotlib colormap name.
    """
    require("matplotlib", "plot", feature="render_matrix_as_figure")
    from matplotlib.figure import Figure

    rows = _as_rows(matrix)
    n = len(rows)
    # Scale with the class count so the labels stay legible, within bounds that keep a 2-class
    # figure from being a postage stamp and a 100-class one from being a poster.
    side = max(4.0, min(16.0, 0.5 * n + 2.0))
    figure = Figure(figsize=(side, side * 0.85), layout="constrained")
    axes = figure.subplots()

    image = axes.imshow(rows, cmap=cmap, interpolation="nearest")
    figure.colorbar(image, ax=axes, shrink=0.8)
    axes.set_title(name)
    axes.set_xlabel("predicted")
    axes.set_ylabel("true")
    axes.set_xticks(range(n), labels=[str(c) for c in class_names[:n]], rotation=45, ha="right")
    axes.set_yticks(range(n), labels=[str(c) for c in class_names[:n]])

    if n <= 25:
        # White on the dark cells, black on the light ones — a fixed colour is unreadable at one
        # end of any sequential colormap.
        threshold = max((max(row) for row in rows), default=0) / 2.0
        for i, row in enumerate(rows):
            for j, count in enumerate(row):
                axes.text(
                    j,
                    i,
                    format(count, "d" if isinstance(count, int) else ".2f"),
                    ha="center",
                    va="center",
                    color="white" if count > threshold else "black",
                    fontsize="small",
                )
    return figure

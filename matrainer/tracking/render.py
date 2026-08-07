"""Human-readable rendering helpers for tracked artifacts.

A run value that is neither a number nor an image reaches a tracker through one of the functions
here. There are TWO SHAPES, each with a table form and a figure form:

* a **square N×N matrix** — a confusion matrix — via :func:`render_matrix_as_table` /
  :func:`render_matrix_as_figure` (a heatmap);
* a **length-C vector** — a per-class metric, the value ``average=None`` produces — via
  :func:`render_vector_as_table` / :func:`render_vector_as_figure` (a bar chart).

The table form goes to :meth:`Tracker.log_text`: cheap, works everywhere, greppable, and readable
in a plain log. The figure form goes to :meth:`Tracker.log_figure`: glanceable, and for a large
matrix the only form that is readable at all — but it needs matplotlib
(``pip install matrainer[plot]``), which is why it is the opt-in of the two rather than the default.

**The two shapes need different renderings, which is why there are four functions and not two.**
A per-class IoU is not a degenerate matrix: it has one axis of meaning, its values share a scale
(they are all fractions), and what a reader wants from it is a RANKING — which class is the model
worst at. A heatmap of a 1×C strip answers that badly; a sorted bar chart answers it at a glance.
The classification task wants the first shape, a per-pixel task the second, and neither rendering
substitutes for the other.

:func:`per_class_vectors` is the SELECTION rule that finds the second shape in a metrics mapping,
by SHAPE rather than by metric name — the vector twin of recordstream's
``confusion_matrices_payload``. It lives here rather than upstream beside that one because unlike
a confusion matrix, a per-class vector feeds no GUI viewer and needs no payload: its only
consumers are the two renderers below it, and a selection rule is best read beside what selects
for it.

Everything here is framework-free: a value only needs ``.detach().cpu().tolist()`` (a torch
tensor), ``.tolist()`` (a numpy array), or to already be a list — so this module imports with the
light core and neither ``tabulate`` nor ``matplotlib`` is imported until it is used.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from matrainer._imports import require

__all__ = [
    "per_class_vectors",
    "render_matrix_as_figure",
    "render_matrix_as_table",
    "render_vector_as_figure",
    "render_vector_as_table",
    "report_matrices",
    "report_vectors",
]


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


# --------------------------------------------------------------------------- #
# Per-class vectors — the shape a `average=None` metric produces
# --------------------------------------------------------------------------- #
def _as_floats(vector: Any) -> Optional[List[float]]:
    """``vector`` as a flat list of floats, or ``None`` when it is not a 1-D numeric vector.

    Deliberately permissive about the CONTAINER (torch tensor, numpy array, list) and strict
    about the SHAPE, because the shape is the whole selection rule — a scalar and a matrix must
    both fall through, and a scalar in particular is easy to mistake for a length-1 vector.
    """
    if callable(getattr(vector, "detach", None)):
        vector = vector.detach().cpu()
    to_list = getattr(vector, "tolist", None)
    values = to_list() if callable(to_list) else vector
    if not isinstance(values, (list, tuple)) or not values:
        return None  # a scalar `tolist()`s to a float, and an empty value renders to nothing
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError):
        return None  # nested (a matrix), or non-numeric


def per_class_vectors(metrics: Any, class_names: Optional[Sequence[Any]] = None) -> List[dict]:
    """Every PER-CLASS VECTOR in a metrics result, as ``{"name", "values", "class_names"}``.

    The vector twin of ``recordstream.ops.image.confusion_matrices_payload``, and selected the
    same way — **by SHAPE, not by metric name**. A runnable emits its full metric results
    (``name -> value``: scalars, ``[C]`` vectors and ``N×N`` matrices together) with no knowledge
    of which is which, so a name-based rule would need every backend to agree on a naming
    convention that nothing enforces.

    A value qualifies when it is a flat numeric sequence of length >= 2. Scalars fall through (a
    scalar metric belongs in ``log_dict``, not in a table) and so do matrices (their renderer is
    :func:`render_matrix_as_table`) — which is what lets a caller pass ONE mapping to both
    selection rules and have each take its own.

    ``class_names`` labels every vector; a vector whose length disagrees with it is labelled by
    INDEX instead of dropped, because a length mismatch usually means a metric with its own axis
    (a per-threshold curve) rather than a bug worth hiding the value over.

    Args:
        metrics: ``name -> value`` metric results, or a single vector (named ``"per_class"``).
        class_names: Optional length-``C`` labels; defaults to ``"0"``, ``"1"``, ….

    Returns:
        One dict per match, in the order the mapping gave them; ``[]`` when there are none.
    """
    items = list(metrics.items()) if isinstance(metrics, dict) else [("per_class", metrics)]
    out: List[dict] = []
    for name, value in items:
        values = _as_floats(value)
        if values is None or len(values) < 2:
            continue
        labels = [str(c) for c in class_names] if class_names else []
        if len(labels) != len(values):
            labels = [str(i) for i in range(len(values))]
        out.append({"name": str(name), "values": values, "class_names": labels})
    return out


def render_vector_as_table(name: str, values: Any, class_names: Sequence[str]) -> str:
    """Render a per-class vector as a GitHub-flavored markdown table, WORST CLASS FIRST.

    Sorted ascending rather than left in class order, and that is the whole reason the table
    exists beside the scalar mean: the mean already says how the model is doing, so what this
    adds is *which class it is worst at* — which is the first line of the table rather than
    something to scan for. Class order is still recoverable (the labels are right there), and for
    a long class list scanning an unsorted column is exactly the work the sort removes.

    Args:
        name: A title printed above the table.
        values: A 1-D array/tensor/list of per-class values.
        class_names: Length-``C`` labels.
    """
    from tabulate import tabulate

    numbers = _as_floats(values) or []
    labels = [str(c) for c in class_names] or [str(i) for i in range(len(numbers))]
    rows = sorted(zip(labels, numbers), key=lambda pair: pair[1])
    # `floatfmt` rather than pre-formatted strings: tabulate PARSES a numeric-looking string and
    # reformats it, so `f"{v:.4f}"` would be silently re-rounded and the column would not have the
    # precision it appears to ask for.
    rendered: str = tabulate(
        [[label, value] for label, value in rows],
        headers=["class", name.rsplit("/", 1)[-1]],
        tablefmt="github",
        floatfmt=".4f",
    )
    mean = sum(numbers) / len(numbers) if numbers else float("nan")
    # A blank line after the title for the same reason the matrix table has one: this is handed
    # to `Tracker.log_text`, and every markdown renderer needs it to see a table rather than one
    # run-on paragraph. The mean is restated because a reader looking at per-class values wants
    # to know immediately whether a low class is dragging an otherwise fine average.
    return f"{name} (mean {mean:.4f}):\n\n{rendered}"


def render_vector_as_figure(
    name: str,
    values: Any,
    class_names: Sequence[str],
    color: str = "#4C72B0",
) -> Any:
    """Render a per-class vector as a horizontal bar chart ``matplotlib.figure.Figure``.

    The counterpart of :func:`render_vector_as_table` for :meth:`Tracker.log_figure`, and a BAR
    CHART rather than a 1×C heatmap strip: a per-class metric has one axis of meaning and a shared
    scale, so bar LENGTH is directly comparable where a colour is something the reader has to
    decode against a colorbar. Horizontal so the class labels read left-to-right at any count.

    Sorted worst-first like the table, and drawn with the mean as a dashed reference line, because
    "which classes are below average" is the question this figure is looked at to answer.

    Built through ``matplotlib.figure.Figure`` DIRECTLY rather than ``pyplot.subplots``, and that
    is load-bearing for the same reason as in :func:`render_matrix_as_figure`: a pyplot figure is
    registered in a process-global manager and leaks one per epoch in a caller with no reason to
    know about pyplot's state.

    Args:
        name: The figure title.
        values: A 1-D array/tensor/list of per-class values.
        class_names: Length-``C`` labels.
        color: Any matplotlib color for the bars.
    """
    require("matplotlib", "plot", feature="render_vector_as_figure")
    from matplotlib.figure import Figure

    numbers = _as_floats(values) or []
    labels = [str(c) for c in class_names] or [str(i) for i in range(len(numbers))]
    rows = sorted(zip(labels, numbers), key=lambda pair: pair[1])
    n = len(rows)

    # Height scales with the class count so the labels never overlap; width is fixed because the
    # value axis is a 0..1-ish fraction in every use this exists for.
    height = max(2.5, min(20.0, 0.32 * n + 1.4))
    figure = Figure(figsize=(7.0, height), layout="constrained")
    axes = figure.subplots()

    positions = range(n)
    axes.barh(list(positions), [value for _, value in rows], color=color)
    axes.set_yticks(list(positions), labels=[label for label, _ in rows])
    axes.invert_yaxis()  # worst at the TOP, matching the table's first row
    axes.set_xlabel(name.rsplit("/", 1)[-1])
    axes.set_title(name)

    if numbers:
        mean = sum(numbers) / len(numbers)
        axes.axvline(mean, linestyle="--", linewidth=1.0, color="0.35")
        axes.text(mean, -0.7, f" mean {mean:.3f}", va="bottom", fontsize="small", color="0.25")

    # Annotate only when there is room; past ~40 classes the numbers collide with the bars and
    # the table is the form that carries them anyway.
    if n <= 40:
        span = max(numbers, default=1.0) or 1.0
        for index, (_, value) in enumerate(rows):
            axes.text(value + 0.01 * span, index, f"{value:.3f}", va="center", fontsize="x-small")
    return figure


def report_matrices(
    values: Any,
    class_names: Optional[Sequence[str]],
    trackers: Sequence[Any],
    *,
    step: Optional[int] = None,
    plot: bool = False,
) -> None:
    """Send every confusion MATRIX in a computed metrics mapping to the log + each tracker.

    The reporting loop every classification-shaped runnable wrote at epoch end: the selection
    rule is recordstream's ``confusion_matrices_payload`` (square 2-D, by SHAPE not by metric
    name), the renderers are this module's, and this function is the WHEN/WHERE — the loggair
    log always, plus every wired tracker's text channel, plus its figure channel when ``plot``
    asks for one. A figure is rendered ONCE per payload, not once per tracker — real work every
    tracker can share.

    Args:
        values: A metrics ``compute()`` mapping (scalars and matrices mixed; scalars ignored).
        class_names: Row/column labels in class-id order, or ``None`` for index labels.
        trackers: LIVE trackers (flow first — see ``flow_trackers``); empty is fine.
        step: The step the text/figure entries are filed under (typically the epoch).
        plot: ALSO render each matrix as a heatmap for ``log_figure`` (needs matplotlib —
            ``matrainer[plot]`` — which is why it is opt-in; the table costs nothing).
    """
    from loggair import get_logger

    # Function-body import: the selection rule is recordstream's, and importing it here keeps
    # `import matrainer.tracking` from paying for recordstream's image stack (PIL) up front.
    from recordstream.ops.image import confusion_matrices_payload

    logger = get_logger(__name__)
    for payload in confusion_matrices_payload(values, list(class_names) if class_names else None):
        table = render_matrix_as_table(payload["name"], payload["counts"], payload["class_names"])
        logger.info(table)
        figure = (
            render_matrix_as_figure(payload["name"], payload["counts"], payload["class_names"])
            if plot and trackers
            else None
        )
        for tracker in trackers:
            tracker.log_text(payload["name"], table, step=step)
            if figure is not None:
                tracker.log_figure(payload["name"], figure, step=step)


def report_vectors(
    values: Any,
    class_names: Optional[Sequence[str]],
    trackers: Sequence[Any],
    *,
    step: Optional[int] = None,
    plot: bool = False,
) -> None:
    """Send every per-class VECTOR in a computed metrics mapping to the log + each tracker.

    The vector twin of :func:`report_matrices`, for the shape ``average=None`` produces (a
    per-class IoU, a per-class AP): selection by :func:`per_class_vectors` (by SHAPE), the
    worst-first table to the log and every tracker's text channel, the sorted bar chart to the
    figure channel when ``plot`` asks. Same once-per-payload figure rule.
    """
    from loggair import get_logger

    logger = get_logger(__name__)
    for payload in per_class_vectors(values, list(class_names) if class_names else None):
        table = render_vector_as_table(payload["name"], payload["values"], payload["class_names"])
        logger.info(table)
        figure = (
            render_vector_as_figure(payload["name"], payload["values"], payload["class_names"])
            if plot and trackers
            else None
        )
        for tracker in trackers:
            tracker.log_text(payload["name"], table, step=step)
            if figure is not None:
                tracker.log_figure(payload["name"], figure, step=step)

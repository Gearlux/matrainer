"""Human-readable rendering helpers for tracked artifacts.

Currently :func:`render_matrix_as_table` — render a square 2-D matrix (e.g. a
torchmetrics confusion matrix) as a GitHub-flavored markdown table. Kept
framework-free (the matrix only needs ``.detach().cpu().tolist()`` OR to already
be a nested list) so this module imports with the light core; ``tabulate`` is
imported lazily.
"""

from __future__ import annotations

from typing import Any, List

__all__ = ["render_matrix_as_table"]


def render_matrix_as_table(name: str, matrix: Any, class_names: List[str]) -> str:
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

    to_list = getattr(matrix, "tolist", None)
    if callable(getattr(matrix, "detach", None)):
        arr = matrix.detach().cpu().tolist()
    elif callable(to_list):
        arr = to_list()
    else:
        arr = matrix
    headers = [""] + [f"pred:{c}" for c in class_names]
    rows = [[f"true:{class_names[i]}", *arr[i]] for i in range(len(arr))]
    rendered: str = tabulate(rows, headers=headers, tablefmt="github")
    # A BLANK line after the title, not just a newline: the result is handed to
    # `Tracker.log_text` as well as to a log, and every markdown renderer (TensorBoard's
    # Text tab, MLflow's `.md` artifact preview) needs one to see a table rather than one
    # run-on paragraph. In a plain log it costs a blank line.
    return f"{name}:\n\n{rendered}"

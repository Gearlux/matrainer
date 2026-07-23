"""Lazy optional-dependency import helper.

Marainer's core install is deliberately light — the tracking backends and the
per-framework training subpackages pull heavy libraries (``mlflow``, ``wandb``,
``torch``, ``pytorch_lightning``, …) that live behind optional extras. A backend
class imports its library **lazily inside its methods** (never at module top
level), through :func:`require`, so:

* ``import marainer`` and ``import marainer.tracking`` stay import-safe with only
  the light core installed (a ``CsvTracker`` works with no extra at all), and
* a missing library produces ONE actionable error naming the exact
  ``pip install marainer[<extra>]`` incantation, instead of a bare
  ``ModuleNotFoundError`` the operator has to decode.

This mirrors the workspace lazy-backend convention (see ``distilaide`` /
``purifai``): validate/dispatch first, import the heavy library last.
"""

from __future__ import annotations

import importlib
from types import ModuleType


def require(module: str, extra: str, *, feature: str | None = None) -> ModuleType:
    """Import ``module``, or raise a clear ``pip install marainer[extra]`` error.

    Args:
        module: The importable module name (e.g. ``"mlflow"``, ``"wandb"``).
        extra: The marainer optional-dependency extra that provides it
            (e.g. ``"mlflow"``) — named in the error message.
        feature: Optional human name of the thing that needs it (e.g.
            ``"MlflowTracker"``); defaults to ``module``.

    Returns:
        The imported module.

    Raises:
        ImportError: When ``module`` cannot be imported, with a message telling
            the operator exactly which extra to install.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        what = feature or module
        raise ImportError(
            f"{what} requires the '{module}' package, which is not installed. "
            f"Install it with: pip install marainer[{extra}]"
        ) from exc


__all__ = ["require"]

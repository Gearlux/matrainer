"""Lazy optional-dependency import helper.

Matrainer's core install is deliberately light — the tracking backends and the
per-framework training subpackages pull heavy libraries (``mlflow``, ``wandb``,
``torch``, ``pytorch_lightning``, …) that live behind optional extras. A backend
class imports its library **lazily inside its methods** (never at module top
level), through :func:`require`, so:

* ``import matrainer`` and ``import matrainer.tracking`` stay import-safe with only
  the light core installed (a ``CsvTracker`` works with no extra at all), and
* a missing library produces ONE actionable error naming the exact
  ``pip install matrainer[<extra>]`` incantation, instead of a bare
  ``ModuleNotFoundError`` the operator has to decode.

This mirrors the workspace lazy-backend convention (see ``distilaide`` /
``purifai``): validate/dispatch first, import the heavy library last.
"""

from __future__ import annotations

import importlib
from types import ModuleType


def require(module: str, extra: str, *, feature: str | None = None, via: str | None = None) -> ModuleType:
    """Import ``module``, or raise a clear ``pip install matrainer[extra]`` error.

    Args:
        module: The importable module name (e.g. ``"mlflow"``, ``"wandb"``) — this is the
            name the ERROR message reports as missing, so it must be the thing an
            operator would install.
        extra: The matrainer optional-dependency extra that provides it
            (e.g. ``"mlflow"``) — named in the error message.
        feature: Optional human name of the thing that needs it (e.g.
            ``"MlflowTracker"``); defaults to ``module``.
        via: Import THIS module instead, when the library must be reached through a shim
            that has to run first. The message still names ``module``, because that is what
            the operator is missing. The one case today is keras, which must be imported
            through ``recordstream.keras`` (it sets ``KERAS_BACKEND`` before keras loads);
            naming ``recordstream.keras`` as missing would send the operator after a package
            that is already installed.

    Returns:
        The imported module — ``via``'s module when that is given, so a shim's own handle
        (``recordstream.keras.keras``) is read off the result.

    Raises:
        ImportError: When the module cannot be imported, with a message telling
            the operator exactly which extra to install.
    """
    try:
        return importlib.import_module(via or module)
    except ImportError as exc:
        what = feature or module
        raise ImportError(
            f"{what} requires the '{module}' package, which is not installed. "
            f"Install it with: pip install matrainer[{extra}]"
        ) from exc


__all__ = ["require"]

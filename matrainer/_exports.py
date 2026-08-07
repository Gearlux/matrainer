"""The lazy backend-name machinery a task project's ``__init__.py`` runs on (PEP 562).

Every training project's package root keeps its core backend-free: each backend-owned name is
mapped to ``(defining module, the extra that installs it)`` and resolved on FIRST attribute
access, so ``import <project>`` pulls no framework and touching a missing backend raises an
``ImportError`` naming the exact extra rather than a traceback from three libraries down.

The `_BACKEND_NAMES` table is project DATA and stays in each project's ``__init__.py``; what
was duplicated four-plus times is the resolver machinery, which is this factory::

    _BACKEND_NAMES = {"LightningClassifier": ("sonair.lightning_classifier", "lightning"), ...}
    __all__ = [*_BACKEND_NAMES]
    __getattr__, __dir__ = lazy_exports(__name__, _BACKEND_NAMES)

It lives in matrainer (light core, already every training project's dependency) rather than
confluid because it is trainer-project ergonomics, not config-system function — revisitable if
a non-training package ever needs it.
"""

from importlib import import_module
from typing import Any, Callable, List, Mapping, Tuple

__all__ = ["lazy_exports"]


def lazy_exports(
    package: str, names: Mapping[str, Tuple[str, str]]
) -> Tuple[Callable[[str], Any], Callable[[], List[str]]]:
    """Build a package's PEP 562 ``(__getattr__, __dir__)`` pair from its lazy-name table.

    Args:
        package: The consuming package's ``__name__`` — used in the error messages and to
            advertise the right ``pip install '<package>[<extra>]'`` fix.
        names: ``{exported name: (defining module, extra)}``. An EMPTY extra (``""``) marks a
            name whose module needs no extra — its ImportError passes through untranslated,
            because advertising an extra that does not exist would be a fix that fails.

    Returns:
        The ``(__getattr__, __dir__)`` pair to assign at the consuming package's top level.
    """
    table = dict(names)

    def __getattr__(name: str) -> Any:
        """Resolve a backend-owned name on first use, or explain which extra is missing."""
        entry = table.get(name)
        if entry is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        module_path, extra = entry
        try:
            return getattr(import_module(module_path), name)
        except ImportError as exc:
            if not extra:
                raise  # a core name — the failure is real, not a missing extra
            # The failure is a MISSING BACKEND, not a broken package — say so, and say what to
            # run. Chained (`from exc`) so the underlying "No module named ..." stays visible.
            raise ImportError(
                f"{package}.{name} needs the {extra!r} backend, which is not installed.\n"
                f"  pip install '{package}[{extra}]'\n"
                f"(underlying import error: {exc})"
            ) from exc

    def __dir__() -> List[str]:
        """Advertise the lazy names to ``dir()`` / tab-completion."""
        module = import_module(package)
        return sorted({*vars(module), *table})

    return __getattr__, __dir__

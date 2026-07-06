"""Plugin discovery for marainer's CLI.

External packages contribute new commands (and supporting Confluid-`@configurable`
classes) to ``marainer`` by exposing a ``register(app: LiquifyApp) -> None``
function via the ``marainer.commands`` entry-point group. When the marainer CLI
boots, it iterates every entry point registered under that group and calls each
one with the live :class:`liquifai.LiquifyApp`, giving plugins a chance to
decorate their commands with ``@app.script_command`` exactly as marainer's own
built-in commands do.

A plugin's ``pyproject.toml`` block looks like::

    [project.entry-points."marainer.commands"]
    my-plugin = "my_pkg.cli_plugin:register"

A failing plugin (import error, raised exception during registration) is logged
and skipped — one broken plugin must never take down the whole CLI.
"""

from importlib.metadata import entry_points
from typing import TYPE_CHECKING

from loggair import get_logger

if TYPE_CHECKING:
    from liquifai import LiquifyApp

logger = get_logger(__name__)

ENTRY_POINT_GROUP = "marainer.commands"


def load_plugins(app: "LiquifyApp") -> None:
    """Discover and register every ``marainer.commands`` entry point on ``app``.

    Each entry point must resolve to a callable with the signature
    ``register(app: LiquifyApp) -> None``. Errors during loading or
    registration are logged at WARNING level and the plugin is skipped — the
    CLI continues with whatever plugins did register successfully.
    """
    eps = entry_points(group=ENTRY_POINT_GROUP)
    for ep in eps:
        try:
            register = ep.load()
        except Exception as exc:
            logger.warning(f"Failed to load marainer plugin {ep.name!r} ({ep.value}): {exc}")
            continue
        try:
            register(app)
        except Exception as exc:
            logger.warning(f"Plugin {ep.name!r} ({ep.value}) raised during register(app): {exc}")
            continue
        logger.debug(f"Registered marainer plugin: {ep.name} ({ep.value})")


__all__ = ["ENTRY_POINT_GROUP", "load_plugins"]

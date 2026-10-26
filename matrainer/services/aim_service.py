"""The Aim UI as a workspace service, served from a repository folder.

Aim needs no server: runs are written into a repository folder and ``aim up`` serves the comparison UI from it. The
one thing this service must get right is that the repository it serves is the one the tracker writes to, so its path
is published to the workspace ``.env`` as ``AIM_REPO``, which a run config reads as ``$AIM_REPO``.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any

from matrainer.services._messages import not_installed

DEFAULT_PORT = 43800

#: The workspace ``.env`` key holding the repository's path, so a tracker and this UI agree.
REPO_ENV_KEY = "AIM_REPO"


def _installed() -> bool:
    """Whether Aim can be imported, without importing it."""
    return importlib.util.find_spec("aim") is not None


def _aim_binary() -> str:
    """The venv's ``aim``, else the one on PATH: an unactivated shell must not serve another Aim, or none."""
    candidate = Path(sys.prefix) / "bin" / "aim"
    if candidate.exists():
        return str(candidate)
    return shutil.which("aim") or "aim"


class Aim:
    """``aim up`` over the service's own repository folder, or the one ``--repo`` names."""

    kind = "service"
    options = {"repo": "The Aim repository to serve (default: the service's own folder)."}

    # The context is the service manager's; matrainer names no type of it, so that it never depends on the manager.
    def start(self, ctx: Any) -> None:
        if not _installed():
            ctx.fail(not_installed(ctx, "Aim", "aim"))
        port = ctx.port or DEFAULT_PORT
        repo = ctx.options.get("repo")
        target = Path(repo).expanduser().resolve() if repo else ctx.data_dir
        target.mkdir(parents=True, exist_ok=True)
        ctx.write_env({REPO_ENV_KEY: str(target)})
        ctx.echo(f"Aim: http://localhost:{port}")
        ctx.echo(f"  Repository: {target}")
        ctx.run(
            # `--yes`: Aim asks before it initialises a repository it has not seen, and a supervised process has no
            # one to answer — it would hang instead of starting.
            [_aim_binary(), "up", "--host", "localhost", "--port", str(port), "--repo", str(target), "--yes"],
            port=port,
            # Aim has no health endpoint: its root page answering is the sign it is up.
            ready="/",
        )

    def info(self, ctx: Any) -> None:
        ports = ctx.ports()
        ctx.echo(f"  URL           : http://localhost:{ports[0] if ports else DEFAULT_PORT}")
        ctx.echo(f"  Repository    : {ctx.data_dir}")
        ctx.echo(f"  {REPO_ENV_KEY:<14}: published to the workspace .env for run configs")

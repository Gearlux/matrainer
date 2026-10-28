"""The MLflow tracking server as a workspace service.

``mlflow server`` keeps its runs in a SQLite store and its artifacts in a folder, both inside the service's own data
folder, and answers ``/health`` once it is up — the probe the service manager waits on. It is run with this venv's
interpreter (``python -m mlflow``), so an unactivated shell never serves another MLflow.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from matrainer.services._messages import not_installed

DEFAULT_PORT = 5005


def _installed() -> bool:
    """Whether MLflow can be imported, without importing it."""
    return importlib.util.find_spec("mlflow") is not None


def _store(data_dir: Path) -> str:
    return f"sqlite:///{data_dir / 'mlflow.db'}"


class MLflow:
    """``mlflow server`` over the service's own store and artifact folder."""

    kind = "service"

    # The context is the service manager's; matrainer names no type of it, so that it never depends on the manager.
    def start(self, ctx: Any) -> None:
        if not _installed():
            ctx.fail(not_installed(ctx, "MLflow", "mlflow"))
        port = ctx.port or DEFAULT_PORT
        artifacts = ctx.data_dir / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        ctx.echo(f"MLflow: http://localhost:{port}")
        ctx.run(
            [
                sys.executable,
                "-m",
                "mlflow",
                "server",
                "--host",
                "localhost",
                "--port",
                str(port),
                "--backend-store-uri",
                _store(ctx.data_dir),
                "--default-artifact-root",
                str(artifacts),
            ],
            port=port,
            ready="/health",
        )

    def info(self, ctx: Any) -> None:
        ports = ctx.ports()
        ctx.echo(f"  URL           : http://localhost:{ports[0] if ports else DEFAULT_PORT}")
        ctx.echo(f"  Backend store : {_store(ctx.data_dir)}")
        ctx.echo(f"  Artifacts dir : {ctx.data_dir / 'artifacts'}")

"""A stand-in for the context the workspace's service manager hands a service class, recording every call.

The classes in ``matrainer/services/`` import nothing of the manager, so they are tested against this.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


class Failed(Exception):
    """What ``ctx.fail`` raises here (the manager turns it into one ``Error:`` line)."""


class Context:
    def __init__(
        self,
        data_dir: Path,
        port: Optional[int] = None,
        options: Optional[Dict[str, Optional[str]]] = None,
        env: Optional[Dict[str, str]] = None,
        ports: Optional[List[int]] = None,
        chosen: Optional[List[str]] = None,
        workspace_name: str = "acme",
    ) -> None:
        self.name = data_dir.name
        self.data_dir = data_dir
        self.port = port
        self.options: Dict[str, Optional[str]] = options or {}
        self.background = True
        self.workspace_name = workspace_name
        self.env: Dict[str, str] = dict(env or {})
        self._ports = ports or []
        self._chosen = chosen or []
        self.runs: List[Dict[str, Any]] = []
        self.lines: List[str] = []

    def run(
        self,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        port: Optional[int] = None,
        ready: Optional[str] = None,
    ) -> None:
        self.runs.append({"command": command, "env": env, "port": port, "ready": ready})

    def write_env(self, values: Dict[str, str]) -> None:
        self.env.update(values)

    def read_env(self) -> Dict[str, str]:
        return dict(self.env)

    def ports(self) -> List[int]:
        return self._ports

    def choose_command(self, selection: str, name: str) -> str:
        names = self._chosen + ([] if name in self._chosen else [name])
        return " ".join(["aisland workspace configure", selection, "set", *names])

    def echo(self, text: str) -> None:
        self.lines.append(text)

    def fail(self, message: str) -> None:
        raise Failed(message)

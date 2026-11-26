"""Every spelling of the framework reaches the config as its ``framework=<name>`` scope, through a real app.

``tests/test_cli.py`` pins how argv is read and rewritten; this runs the whole way: ``trainer_main`` → liquifai's
``run()`` → the config's ``!scope:framework=<name>`` block → the runnable the train verb builds and runs.
``--scope=framework=X`` is the spelling liquifai reads only since its global flags took ``--flag=VALUE``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, List

import confluid
import pytest
from confluid import configurable
from liquifai.context import set_context

from matrainer.cli import Framework, make_trainer_app, trainer_main

#: The engine of every probe run, in order.
RAN: List[str] = []


@configurable
class EngineProbe:
    """A runnable that records the engine the config built it with.

    Args:
        engine: The engine the config names.
    """

    def __init__(self, engine: str = "none") -> None:
        self.engine = engine

    def run(self) -> None:
        RAN.append(self.engine)


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    set_context(None)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("XDG_CONFIG_DIRS", str(tmp_path / "xdg_sys"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    RAN.clear()
    saved = confluid.get_app_name()
    yield
    confluid.set_app_name(saved)
    set_context(None)  # a run leaves its context active: later tests must not read it


@pytest.fixture
def config(tmp_path: Path) -> Path:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        f"runnable: !class:{__name__}.EngineProbe()\n"
        "  engine: none\n"
        "torch_engine: !scope:framework=torch\n"
        "  runnable:\n"
        "    engine: torch\n"
    )
    return path


def _engine(config: Path, monkeypatch: pytest.MonkeyPatch, *argv: str) -> str:
    monkeypatch.setattr(sys, "argv", ["probe", "train", str(config), *argv])
    trainer_main(make_trainer_app(name="probe"), {"torch": Framework("the reference engine")})
    assert len(RAN) == 1
    return RAN[0]


@pytest.mark.parametrize(
    "argv",
    [
        ["--framework", "torch"],
        ["--framework=torch"],
        ["--scope", "framework=torch"],
        ["--scope=framework=torch"],
        ["-s", "framework=torch"],
        ["-s=framework=torch"],
    ],
)
def test_every_spelling_activates_the_frameworks_scope(
    config: Path, monkeypatch: pytest.MonkeyPatch, argv: List[str]
) -> None:
    assert _engine(config, monkeypatch, *argv) == "torch"


def test_without_a_framework_the_config_builds_as_written(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert _engine(config, monkeypatch) == "none"


def test_a_config_without_a_framework_block_ignores_the_framework(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A keyed scope of a dimension the config never declares stays inert: the framework is passed to every config."""
    plain = tmp_path / "plain.yaml"
    plain.write_text(f"runnable: !class:{__name__}.EngineProbe()\n  engine: none\n")
    assert _engine(plain, monkeypatch, "--scope=framework=torch") == "none"

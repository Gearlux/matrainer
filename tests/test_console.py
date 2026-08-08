"""The `matrainer` console script — the generic runner + the sweep dispatch.

The verb machinery itself (the verb→task map, the taskless guard, the `--framework` rewrite,
the keras engine pin) is `matrainer.cli`'s and is pinned in `test_cli.py`; what belongs here
is the console's OWN surface: the `--registrations` strip (a flag liquifai would read as a
config override), the pin's ORDER against that import, the `sweep` dispatch, and — end to
end — a real `matrainer train` subprocess over a tiny runnable config, proving the flat
config's keys broadcast and the verb forces the task.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from matrainer import console

# ---------------------------------------------------------------------------
# The argv pre-steps the console owns
# ---------------------------------------------------------------------------


def test_registrations_are_stripped_before_liquifai_sees_them() -> None:
    """liquifai would read `--registrations` as a config override for a `registrations` key."""
    argv = ["train", "cfg.yaml", "--registrations", "a.mod,b.mod", "--framework", "torch"]
    modules = console._pop_registrations(argv)
    assert modules == ["a.mod", "b.mod"]
    assert "--registrations" not in argv

    argv = ["train", "cfg.yaml", "--registrations=c.mod"]
    assert console._pop_registrations(argv) == ["c.mod"]
    assert argv == ["train", "cfg.yaml"]


def test_the_pin_runs_before_the_registration_import() -> None:
    """The ORDER is the whole fix (measured in the per-project CLIs): registering can import a
    module that binds the keras engine, so the pin must land first. Pinned by reading main()."""
    import inspect

    source = inspect.getsource(console.main)
    assert source.index("pin_keras_backend") < source.index("_import_registrations")


def test_the_app_has_the_three_verbs_and_no_frameworks_command() -> None:
    """A project-agnostic runner has no FRAMEWORKS table — `--registrations` is its substitute."""
    assert set(console.VERB_TASKS) <= set(console.app._commands)
    assert "frameworks" not in console.app._commands


def test_sweep_dispatches_before_the_liquifai_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "argv", ["matrainer", "sweep", "next-experiment", "csv", "P", "--save-dir", str(tmp_path)])
    assert console.main() == 0


# ---------------------------------------------------------------------------
# End to end: `matrainer train` over a tiny runnable config, in a subprocess
# ---------------------------------------------------------------------------
_PROBE_MODULE = '''
"""A tiny runnable that records what the generic runner handed it."""
import json


class Probe:
    def __init__(self, max_epochs: int = 0, out: str = "", task: str = "fit") -> None:
        self.max_epochs = max_epochs
        self.out = out
        self.task = task

    def run(self) -> None:
        with open(self.out, "w") as handle:
            json.dump({"task": self.task, "max_epochs": self.max_epochs}, handle)
'''


@pytest.mark.parametrize(("verb", "expected_task"), [("train", "fit"), ("evaluate", "test")])
def test_matrainer_train_runs_any_runnable_config(tmp_path: Path, verb: str, expected_task: str) -> None:
    """The generic runner end to end: the flat config's top-level keys BROADCAST into the
    constructor (`materialize_runnable`, never a bare flow) and the verb forces the task."""
    (tmp_path / "probe_runnable.py").write_text(_PROBE_MODULE)
    result_path = tmp_path / "result.json"
    config = tmp_path / "cfg.yaml"
    config.write_text(f"runnable: !class:probe_runnable.Probe\nmax_epochs: 7\nout: {result_path}\ntask: predict\n")

    proc = subprocess.run(
        [sys.executable, "-m", "matrainer.console", verb, str(config)],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(tmp_path)},
    )
    assert proc.returncode == 0, proc.stderr
    recorded: Any = json.loads(result_path.read_text())
    assert recorded == {
        "task": expected_task,
        "max_epochs": 7,
    }, "max_epochs must reach the constructor (broadcast) and the verb must beat the YAML task"

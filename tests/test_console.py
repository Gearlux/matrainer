"""The `matrainer` console script — the generic runner + the sweep dispatch.

What this pins: the verb→task map (the verb wins over the YAML's `task:`), the taskless guard
(a runnable with no `task` attribute runs as-is for `train` ONLY), the `--framework` rewrite,
the keras engine pin's ORDER (before any registration import), the `--registrations` strip,
the `sweep` dispatch, and — end to end — a real `matrainer train` subprocess over a tiny
runnable config, proving the flat config's keys broadcast and the verb forces the task.
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
# The dispatch + the taskless guard
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.task = "fit"
        self.ran = False

    def run(self) -> None:
        self.ran = True


class _Taskless:
    def __init__(self) -> None:
        self.ran = False

    def run(self) -> None:
        self.ran = True


def test_the_verb_wins_over_the_yaml_task() -> None:
    runnable = _Recorder()
    runnable.task = "predict"  # what the YAML said
    console._dispatch(runnable, "test")
    assert runnable.task == "test" and runnable.ran


def test_a_taskless_runnable_trains_as_is_and_refuses_the_other_verbs() -> None:
    """The silent `hasattr` guard would let an evaluate verb TRAIN — refuse by name instead."""
    runnable = _Taskless()
    console._dispatch(runnable, "fit")
    assert runnable.ran

    with pytest.raises(ValueError, match="_Taskless"):
        console._dispatch(_Taskless(), "test")


def test_no_runnable_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="runnable"):
        console._dispatch(None, "fit")


# ---------------------------------------------------------------------------
# The argv pre-steps
# ---------------------------------------------------------------------------
def test_the_framework_flag_is_rewritten_never_appended() -> None:
    argv = ["train", "cfg.yaml", "--framework", "keras", "--epochs-like", "3"]
    console._normalize_framework_flag(argv)
    assert argv == ["train", "cfg.yaml", "--scope", "framework=keras", "--epochs-like", "3"]

    argv = ["train", "cfg.yaml", "--framework=torch"]
    console._normalize_framework_flag(argv)
    assert argv == ["train", "cfg.yaml", "--scope", "framework=torch"]


def test_registrations_are_stripped_before_liquifai_sees_them() -> None:
    """liquifai would read `--registrations` as a config override for a `registrations` key."""
    argv = ["train", "cfg.yaml", "--registrations", "a.mod,b.mod", "--framework", "torch"]
    modules = console._pop_registrations(argv)
    assert modules == ["a.mod", "b.mod"]
    assert "--registrations" not in argv


def test_the_tensorflow_framework_pins_the_keras_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KERAS_BACKEND", raising=False)
    console._pin_keras_backend("tensorflow")
    import os

    assert os.environ.get("KERAS_BACKEND") == "tensorflow"
    monkeypatch.setenv("KERAS_BACKEND", "torch")
    console._pin_keras_backend("tensorflow")
    assert os.environ["KERAS_BACKEND"] == "torch", "setdefault — an explicit choice always wins"


def test_the_pin_runs_before_the_registration_import() -> None:
    """The ORDER is the whole fix (measured in the per-project CLIs): registering can import a
    module that binds the keras engine, so the pin must land first. Pinned by reading main()."""
    import inspect

    source = inspect.getsource(console.main)
    assert source.index("_pin_keras_backend") < source.index("_import_registrations")


def test_the_requested_framework_is_read_in_every_spelling() -> None:
    read = console._requested_framework
    assert read(["train", "c.yaml", "--framework", "jax"]) == "jax"
    assert read(["train", "c.yaml", "--framework=mlx"]) == "mlx"
    assert read(["train", "c.yaml", "--scope", "framework=keras"]) == "keras"
    assert read(["train", "c.yaml"]) is None


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

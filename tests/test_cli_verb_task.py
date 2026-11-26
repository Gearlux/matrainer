"""The task a verb forces is in the configuration the command line resolves — so ``--dump-config`` shows it.

Each verb adds ``runnable.task: <its task>`` to the loaded configuration (liquifai's ``config_keys=``) when the
runnable's class takes a ``task``, so ``evaluate cfg.yaml --dump-config`` shows ``task: test`` where it showed the
config's own ``fit``. A runnable without one gets nothing added — a train run of it says nothing new — and the
other verbs still refuse it by name.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterator, List

import confluid
import pytest
import yaml
from confluid import configurable
from liquifai.context import set_context

from matrainer.cli import make_trainer_app

RAN: List[Any] = []


@configurable
class VerbTaskRunnable:
    def __init__(self, task: str = "fit", epochs: int = 1) -> None:
        self.task = task
        self.epochs = epochs

    def run(self) -> None:
        RAN.append(self)


@configurable
class VerbTasklessRunnable:
    def __init__(self, epochs: int = 1) -> None:
        self.epochs = epochs

    def run(self) -> None:
        RAN.append(self)


WITH_TASK = f"runnable: !class:{VerbTaskRunnable.__module__}.VerbTaskRunnable\n  task: fit\n"
TASKLESS = f"runnable: !class:{VerbTasklessRunnable.__module__}.VerbTasklessRunnable\n"


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    set_context(None)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("XDG_CONFIG_DIRS", str(tmp_path / "xdg_sys"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    RAN.clear()
    saved = confluid.get_app_name()
    yield tmp_path
    confluid.set_app_name(saved)
    set_context(None)  # a run leaves its context active: later tests must not read it


def _run(monkeypatch: pytest.MonkeyPatch, *args: str) -> int:
    app = make_trainer_app(name="probe")
    monkeypatch.setattr(sys, "argv", ["probe", *args])
    try:
        app.run()
    except SystemExit as stopped:
        return int(stopped.code or 0)
    return 0


@pytest.mark.parametrize(("verb", "task"), [("train", "fit"), ("evaluate", "test"), ("predict", "predict")])
def test_the_dumped_configuration_shows_the_task_the_verb_runs(
    verb: str, task: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "cfg.yaml").write_text(WITH_TASK)
    assert _run(monkeypatch, verb, "cfg.yaml", "--dump-config") == 0
    assert RAN == []
    assert yaml.safe_load(capsys.readouterr().out)["runnable"]["task"] == task


def test_the_verbs_task_wins_over_a_typed_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "cfg.yaml").write_text(WITH_TASK)
    assert _run(monkeypatch, "evaluate", "cfg.yaml", "--runnable.task", "predict") == 0
    assert RAN[0].task == "test"


def test_a_taskless_runnable_trains_with_nothing_added(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "cfg.yaml").write_text(TASKLESS)
    assert _run(monkeypatch, "train", "cfg.yaml", "--dump-config") == 0
    captured = capsys.readouterr()
    assert "task" not in yaml.safe_load(captured.out)["runnable"]
    assert "has no attribute 'task'" not in captured.err

    assert _run(monkeypatch, "train", "cfg.yaml") == 0
    assert isinstance(RAN[0], VerbTasklessRunnable)
    assert "has no attribute 'task'" not in capsys.readouterr().err


def test_a_taskless_runnable_is_still_refused_by_the_other_verbs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "cfg.yaml").write_text(TASKLESS)
    with pytest.raises(ValueError, match="VerbTasklessRunnable has no 'task' attribute"):
        _run(monkeypatch, "evaluate", "cfg.yaml")
    assert RAN == []

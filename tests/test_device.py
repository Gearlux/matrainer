"""Tests for :func:`matrainer.torch.resolve_device` — the ``"auto"`` accelerator choice.

The point of the helper is that an EXPLICIT choice is honoured verbatim (so a
CPU-vs-GPU benchmark cannot silently fall back) while ``"auto"`` walks one fixed
preference order. Both halves are pinned here with the availability probes patched, so
the test says the same thing on a CUDA box, an Apple laptop and a CPU-only runner.
"""

import os
import re
from typing import Any

import pytest
import torch

from matrainer.torch import Device, resolve_device


@pytest.fixture
def _no_accelerators(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither CUDA nor MPS present — the CPU-only runner's view."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


@pytest.mark.parametrize("choice", ["cpu", "cuda", "mps"])
def test_an_explicit_choice_is_taken_at_its_word(
    choice: Device, monkeypatch: pytest.MonkeyPatch, _no_accelerators: Any
) -> None:
    """No availability probe runs — an operator who wrote `cuda` gets torch's own error, later."""
    assert resolve_device(choice).type == choice


def test_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda"


def test_auto_falls_back_to_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto").type == "mps"


def test_auto_falls_back_to_cpu(_no_accelerators: Any) -> None:
    assert resolve_device("auto").type == "cpu"


def test_auto_is_the_default(_no_accelerators: Any) -> None:
    assert resolve_device() == resolve_device("auto")


def test_the_device_literal_enumerates_its_options() -> None:
    """A closed Literal is what lets a form spec / MCP schema list the choices."""
    from typing import get_args

    assert set(get_args(Device)) == {"auto", "cpu", "cuda", "mps"}


# --------------------------------------------------------------------------- the GPU's cache on a Mac

from matrainer.torch import device as device_module  # noqa: E402
from matrainer.torch.device import (  # noqa: E402
    HIGH_WATERMARK_VARIABLE,
    LOW_WATERMARK_VARIABLE,
    TORCH_DEFAULT_HIGH_WATERMARK,
    apply_mps_low_watermark,
)


class _Log:
    def __init__(self) -> None:
        self.lines: dict = {"debug": [], "info": [], "warning": []}

    def debug(self, message: str) -> None:
        self.lines["debug"].append(message)

    def info(self, message: str) -> None:
        self.lines["info"].append(message)

    def warning(self, message: str) -> None:
        self.lines["warning"].append(message)


@pytest.fixture
def log(monkeypatch: pytest.MonkeyPatch) -> _Log:
    recorder = _Log()
    monkeypatch.setattr(device_module, "logger", recorder)
    monkeypatch.delenv(LOW_WATERMARK_VARIABLE, raising=False)
    monkeypatch.delenv(HIGH_WATERMARK_VARIABLE, raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    return recorder


def test_torchs_default_high_watermark_is_the_one_measured() -> None:
    """1.69 accepted, 1.71 refused (`invalid low watermark ratio 1.71`) — measured on an M3 Pro, torch 2.13."""
    assert TORCH_DEFAULT_HIGH_WATERMARK == 1.7


def test_unset_sets_nothing(log: _Log) -> None:
    apply_mps_low_watermark(None)
    assert LOW_WATERMARK_VARIABLE not in os.environ and log.lines == {"debug": [], "info": [], "warning": []}


def test_a_value_is_handed_to_torch_through_its_variable(log: _Log) -> None:
    apply_mps_low_watermark(0.5)
    assert os.environ[LOW_WATERMARK_VARIABLE] == "0.5"
    assert log.lines["warning"] == []


def test_the_shells_value_wins_and_the_log_says_so(log: _Log, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOW_WATERMARK_VARIABLE, "0.4")
    apply_mps_low_watermark(0.5)
    assert os.environ[LOW_WATERMARK_VARIABLE] == "0.4"
    assert log.lines["info"] == [
        f"{LOW_WATERMARK_VARIABLE}=0.4 from the environment is used; mps_low_watermark=0.5 is not"
    ]


def test_CON_the_same_value_in_the_shell_says_nothing(log: _Log, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LOW_WATERMARK_VARIABLE, "0.5")
    apply_mps_low_watermark(0.5)
    assert log.lines["info"] == []


@pytest.mark.parametrize(
    "value, high, message",
    [
        (1.8, None, "mps_low_watermark 1.8 must be between 0 and the high watermark 1.7 (torch's default)"),
        (0.7, "0.6", f"mps_low_watermark 0.7 must be between 0 and the high watermark 0.6 ({HIGH_WATERMARK_VARIABLE})"),
        (-0.1, None, "mps_low_watermark -0.1 must be between 0 and the high watermark 1.7"),
    ],
)
def test_a_value_torch_would_refuse_is_refused_by_name(
    log: _Log, monkeypatch: pytest.MonkeyPatch, value: float, high: Any, message: str
) -> None:
    """torch's own refusal names neither the setting nor the bound: `RuntimeError: invalid low watermark ratio 1.4`."""
    if high is not None:
        monkeypatch.setenv(HIGH_WATERMARK_VARIABLE, high)
    with pytest.raises(ValueError, match=re.escape(message)):
        apply_mps_low_watermark(value)
    assert LOW_WATERMARK_VARIABLE not in os.environ


def test_a_gpu_already_in_use_is_warned_about(log: _Log, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 4096)
    apply_mps_low_watermark(0.5)
    assert len(log.lines["warning"]) == 1 and "already used the GPU" in log.lines["warning"][0]


def test_CON_a_gpu_not_yet_used_takes_it_quietly(log: _Log, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 0)
    apply_mps_low_watermark(0.5)
    assert log.lines["warning"] == [] and os.environ[LOW_WATERMARK_VARIABLE] == "0.5"


def test_CON_without_an_apple_gpu_nothing_is_probed(log: _Log, monkeypatch: pytest.MonkeyPatch) -> None:
    def probe() -> int:
        raise AssertionError("probed a GPU that is not there")

    monkeypatch.setattr(torch.mps, "current_allocated_memory", probe)
    apply_mps_low_watermark(0.5)
    assert os.environ[LOW_WATERMARK_VARIABLE] == "0.5"

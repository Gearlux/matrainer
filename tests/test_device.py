"""Tests for :func:`matrainer.torch.resolve_device` — the ``"auto"`` accelerator choice.

The point of the helper is that an EXPLICIT choice is honoured verbatim (so a
CPU-vs-GPU benchmark cannot silently fall back) while ``"auto"`` walks one fixed
preference order. Both halves are pinned here with the availability probes patched, so
the test says the same thing on a CUDA box, an Apple laptop and a CPU-only runner.
"""

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

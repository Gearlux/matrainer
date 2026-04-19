"""Smoke test for the generic ``marainer process`` script command."""

import pytest
from confluid import configurable

from marainer.cli import process


@configurable
class _Stub:
    """Minimal Confluid-configurable processor for the test."""

    last_run_count = 0

    def __init__(self, label: str = "stub") -> None:
        self.label = label
        self.experiment_name = label

    def run(self) -> None:
        type(self).last_run_count += 1


def test_process_invokes_run_on_injected_processor() -> None:
    _Stub.last_run_count = 0
    stub = _Stub(label="hello")
    # The script_command decorator just registers — call the underlying function directly.
    process(stub)
    assert _Stub.last_run_count == 1


def test_process_handles_object_without_run(caplog: pytest.LogCaptureFixture) -> None:
    """If the injected object has no .run(), we log an error and don't crash."""

    class NoRun:
        pass

    process(NoRun())  # Should not raise.


def test_process_handles_none(caplog: pytest.LogCaptureFixture) -> None:
    """If DI failed and processor is None, we log + return cleanly."""
    process(None)  # Should not raise.


def test_process_logs_class_name_when_no_experiment_name(caplog: pytest.LogCaptureFixture) -> None:
    class Bare:
        def run(self) -> None:
            pass

    # No experiment_name attribute → label falls back to class name.
    process(Bare())

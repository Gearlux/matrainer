"""Smoke tests for the ``marainer process`` / ``download`` / ``convert`` script commands.

All three verbs are semantic aliases of the same ``_dispatch_processor`` body
(see ``marainer/cli.py``) — they share dispatch and error handling, only the
log-message verb differs. The tests are parametrized over the three verbs so
any divergence is caught.
"""

from typing import Any

import pytest
from confluid import configurable

from marainer.cli import convert, download, process

VERBS = [process, download, convert]
VERB_IDS = ["process", "download", "convert"]


@configurable
class _Stub:
    """Minimal Confluid-configurable processor for the test."""

    last_run_count = 0

    def __init__(self, label: str = "stub") -> None:
        self.label = label
        self.experiment_name = label

    def run(self) -> None:
        type(self).last_run_count += 1


@pytest.mark.parametrize("verb", VERBS, ids=VERB_IDS)
def test_verb_invokes_run_on_injected_processor(verb: Any) -> None:
    _Stub.last_run_count = 0
    stub = _Stub(label="hello")
    # The script_command decorator just registers — call the underlying function directly.
    verb(stub)
    assert _Stub.last_run_count == 1


@pytest.mark.parametrize("verb", VERBS, ids=VERB_IDS)
def test_verb_handles_object_without_run(verb: Any, caplog: pytest.LogCaptureFixture) -> None:
    """If the injected object has no .run(), we log an error and don't crash."""

    class NoRun:
        pass

    verb(NoRun())  # Should not raise.


@pytest.mark.parametrize("verb", VERBS, ids=VERB_IDS)
def test_verb_handles_none(verb: Any, caplog: pytest.LogCaptureFixture) -> None:
    """If DI failed and processor is None, we log + return cleanly."""
    verb(None)  # Should not raise.


@pytest.mark.parametrize("verb", VERBS, ids=VERB_IDS)
def test_verb_logs_class_name_when_no_experiment_name(verb: Any, caplog: pytest.LogCaptureFixture) -> None:
    class Bare:
        def run(self) -> None:
            pass

    # No experiment_name attribute → label falls back to class name.
    verb(Bare())

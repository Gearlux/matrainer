"""Suite-wide fixtures.

The one thing here is the portable-root isolation below: without it, whether the suite passes
depends on the shell it was launched from.
"""

import pytest

from marainer.tracking.provenance import PORTABLE_ROOT_VARS


@pytest.fixture(autouse=True)
def no_ambient_portable_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset ``PROJECT_ROOT`` / ``DATA_ROOT`` / ``MODEL_ROOT`` for every test.

    Path contraction is opt-in by configuration, so a test that does not configure a root must
    not inherit one from the operator's shell — and the workspace's own activation script
    (``source project.bashrc``) exports ``PROJECT_ROOT``, which every CI run and every developer
    shell therefore has set. ``sys.executable`` lives under it, so the pasteable ``command.line``
    correctly came back as ``"$PROJECT_ROOT/.venv/bin/python3" -m ...`` while the tests that
    compare against a raw ``sys.executable`` expected the absolute path: green in a bare shell,
    six failures in an activated one, with nothing wrong in the code under test.

    A test that WANTS contraction sets its roots itself (the ``roots`` fixture in
    ``tests/tracking/test_provenance.py``); an explicitly requested fixture is set up after an
    autouse one at the same scope, so it wins.
    """
    for name in PORTABLE_ROOT_VARS:
        monkeypatch.delenv(name, raising=False)

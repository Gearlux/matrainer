"""Tests for marainer's entry-point-based plugin discovery."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from liquifai import LiquifyApp
from marainer import plugins


def _build_fake_entry_point(name: str, register_fn: Any) -> MagicMock:
    """Return a MagicMock that quacks like importlib.metadata.EntryPoint."""
    ep = MagicMock()
    ep.name = name
    ep.value = f"fake.{name}:register"
    ep.load.return_value = register_fn
    return ep


def test_load_plugins_invokes_register_with_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = LiquifyApp(name="marainer")

    seen: list[LiquifyApp] = []

    def register(passed_app: LiquifyApp) -> None:
        seen.append(passed_app)

        @passed_app.script_command(flow_mode="auto")
        def my_plugin_command(stuff: Any) -> None:
            return None

    ep = _build_fake_entry_point("my-plugin", register)
    monkeypatch.setattr(
        plugins,
        "entry_points",
        lambda group: [ep] if group == plugins.ENTRY_POINT_GROUP else [],
    )

    plugins.load_plugins(app)

    assert seen == [app]
    assert "my-plugin-command" in app._commands


def test_load_plugins_skips_plugin_that_raises_on_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken `ep.load()` must not abort the whole CLI bootstrap."""
    app = LiquifyApp(name="marainer")

    bad_ep = MagicMock()
    bad_ep.name = "broken"
    bad_ep.value = "broken_pkg:register"
    bad_ep.load.side_effect = ImportError("boom")

    def good_register(passed_app: LiquifyApp) -> None:
        @passed_app.script_command(flow_mode="auto")
        def survives(stuff: Any) -> None:
            return None

    good_ep = _build_fake_entry_point("good", good_register)

    monkeypatch.setattr(plugins, "entry_points", lambda group: [bad_ep, good_ep])

    plugins.load_plugins(app)

    assert "survives" in app._commands


def test_load_plugins_skips_plugin_that_raises_during_register(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception inside `register(app)` must not abort other plugins."""
    app = LiquifyApp(name="marainer")

    def bad_register(passed_app: LiquifyApp) -> None:
        raise RuntimeError("plugin bug")

    def good_register(passed_app: LiquifyApp) -> None:
        @passed_app.script_command(flow_mode="auto")
        def survives(stuff: Any) -> None:
            return None

    bad_ep = _build_fake_entry_point("bad", bad_register)
    good_ep = _build_fake_entry_point("good", good_register)

    monkeypatch.setattr(plugins, "entry_points", lambda group: [bad_ep, good_ep])

    plugins.load_plugins(app)

    assert "survives" in app._commands


def test_load_plugins_no_entry_points_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    app = LiquifyApp(name="marainer")
    monkeypatch.setattr(plugins, "entry_points", lambda group: [])
    before = dict(app._commands)
    plugins.load_plugins(app)
    assert app._commands == before

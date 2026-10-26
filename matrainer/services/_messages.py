"""What a tracking service says when its backend is not installed — it never installs it."""

from typing import Any


def not_installed(ctx: Any, what: str, tracker: str) -> str:
    """The message naming the command that adds ``tracker`` to the tracker selection, keeping the trackers chosen.

    A service never installs its backend: an install behind the selection's back leaves the venv holding something
    no selection asked for, gone again after the next setup. ``ctx`` is the service manager's context (any type)."""
    command = ctx.choose_command("tracker", tracker)
    return (
        f"{what} is not installed in this environment.\n"
        "  Add it to the tracker selection, so the choice is recorded and survives the next setup:\n"
        f"      {command}\n"
        "  Or, for this machine only:\n"
        f"      {command} --machine-only"
    )

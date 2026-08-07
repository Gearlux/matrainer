"""The ``matrainer`` console script — generic train verbs + the sweep group.

A user-authorized exception (2026-08-07) to the old "matrainer ships no binary" rule, in two
halves:

* ``matrainer train|evaluate|predict <config.yaml> [--framework F] [--tracker T] [--KEY VAL …]``
  — the GENERIC runner: any project's ``runnable:``-rooted confluid config runs through the
  same materialize-and-dispatch path the per-project CLIs use
  (:func:`recordstream.cli.materialize_runnable` — never a bare ``flow()``, which would drop
  every top-level key silently), with the verb forcing the ``task``. It works on any config
  whose class references are module-qualified — which every shipped example and sweep overlay
  is — and takes ``--registrations mod1,mod2`` for one that names bare registered classes.
* ``matrainer sweep backends|trackers|all <sweep.yaml>`` (+ the read-back commands ``report`` /
  ``next-experiment`` / ``json``) — dispatched to :mod:`matrainer.sweep` before the liquifai
  app is even built.

The per-project CLIs (their FRAMEWORKS tables, registrations, task-specific help) remain the
ergonomic surface; this runner is what a project-agnostic caller — the sweep, a CI job, an
operator with a config in hand — uses without knowing which project the config belongs to.

Two pre-steps generalized from the project CLIs, both of which must run BEFORE the app:

* the KERAS ENGINE PIN — ``--framework tensorflow`` ``setdefault``s ``KERAS_BACKEND`` so the
  first keras import binds the right engine (Keras 3 reads the variable once, at import);
* the ``--framework`` REWRITE to ``--scope framework=…`` — liquifai binds the ergonomic flag
  only for dimensions the config DECLARES, so against a config with no ``!scope:framework=…``
  block a bare ``--framework X`` would be read as a config override.
"""

import importlib
import os
import sys
from typing import Any, Callable, List, Optional

from liquifai import LiquifyApp
from loggair import get_logger

logger = get_logger(__name__)

__all__ = ["VERB_TASKS", "app", "main"]

app = LiquifyApp(name="matrainer", description="Generic runner + sweep for runnable configs")

#: Verb -> the ``task`` it forces. The verb wins over any ``task:`` in the YAML — the same map
#: every per-project CLI carries.
VERB_TASKS = {"train": "fit", "evaluate": "test", "predict": "predict"}

_VERB_DOCS = {
    "train": "Train the config's runnable (any project's)",
    "evaluate": "Score the config's runnable over its test split",
    "predict": "Stream the config's runnable's predictions to its sink",
}


def _dispatch(runnable: Any, task: str) -> None:
    """Build the bound runnable, force ``task``, and run it.

    ``materialize_runnable`` rather than a bare ``flow()`` — the flat config's top-level keys
    only broadcast into the constructor when the node is built against its document; a bare
    flow drops them SILENTLY (``train_set`` becomes None, ``max_epochs`` falls back to the
    ctor default without a trace).

    A runnable with NO ``task`` attribute runs as-is for ``train`` only; ``evaluate`` /
    ``predict`` raise naming the class — a silent ``hasattr`` guard would let an evaluate verb
    TRAIN (the guard one task project's upstream-loop backend needed).
    """
    if runnable is None:
        raise ValueError("No runnable was bound — provide one under 'runnable:' in your YAML.")

    from recordstream.cli import materialize_runnable

    runnable = materialize_runnable(runnable)
    if hasattr(runnable, "task"):
        runnable.task = task  # the verb wins over the YAML's task
    elif task != "fit":
        raise ValueError(
            f"{type(runnable).__name__} has no 'task' attribute, so only 'matrainer train' can "
            "run it — this verb would silently do something else."
        )
    runnable.run()


def _verb_command(verb: str) -> Callable[[Any], None]:
    """One verb — a factory, never a default-argument closure (liquifai reads the SIGNATURE,
    so a smuggled loop variable would surface as a ``--_verb`` flag). The one parameter must be
    named ``runnable``: that is the config key liquifai's DI binds it to."""

    def command(runnable: Any) -> None:
        _dispatch(runnable, VERB_TASKS[verb])

    command.__name__ = verb
    command.__doc__ = f"{_VERB_DOCS[verb]} (task={VERB_TASKS[verb]}). Pick a backend with --framework."
    return command


for _verb in VERB_TASKS:
    app.script_command(name=_verb, flow_mode="manual")(_verb_command(_verb))


def _pin_keras_backend(framework: Optional[str]) -> None:
    """``--framework tensorflow`` pins ``KERAS_BACKEND`` before ANYTHING imports keras.

    Keras 3 reads the variable once, at import, and the workspace default resolves to the
    first engine INSTALLED (torch, on most machines) — a torch-backed ``keras.Model`` cannot
    be trained with ``tf.GradientTape``. ``setdefault``, so an explicit ``KERAS_BACKEND=…``
    still wins and is reported as a mismatch by the backend itself rather than silently
    overridden. No other framework touches the variable — running on whichever engine is
    bound is the point of the plain ``keras`` backend.
    """
    if framework == "tensorflow":
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")


def _import_registrations(modules: List[str]) -> None:
    """``--registrations mod1,mod2`` — import registering modules so a config may use BARE
    registered names. Not needed by any shipped config (all module-qualified); best-effort,
    because absent registrations on a partial install are a state, not an error."""
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError as exc:
            logger.debug(f"registrations unavailable ({module}: {exc})")


def _pop_registrations(argv: List[str]) -> List[str]:
    """Strip ``--registrations a,b`` out of ``argv`` (liquifai would read it as a config
    override) and return the module list."""
    modules: List[str] = []
    out: List[str] = []
    i, n = 0, len(argv)
    while i < n:
        arg = argv[i]
        if arg.startswith("--registrations="):
            modules += arg.split("=", 1)[1].split(",")
        elif arg == "--registrations" and i + 1 < n:
            modules += argv[i + 1].split(",")
            i += 1
        else:
            out.append(arg)
        i += 1
    argv[:] = out
    return [m for m in modules if m]


def _normalize_framework_flag(argv: List[str]) -> None:
    """Rewrite ``--framework <name>`` into the ``--scope framework=<name>`` liquifai reads.

    Never appended: liquifai binds the ergonomic flag only for dimensions the config DECLARES,
    and rewriting makes the flag mean one thing regardless — an undeclared dimension stays an
    inert no-op, which is what lets a config grow into a backend it does not have yet.
    """
    out: List[str] = []
    i, n = 0, len(argv)
    while i < n:
        arg = argv[i]
        if arg.startswith("--framework="):
            out += ["--scope", f"framework={arg.split('=', 1)[1]}"]
        elif arg == "--framework" and i + 1 < n:
            out += ["--scope", f"framework={argv[i + 1]}"]
            i += 1
        else:
            out.append(arg)
        i += 1
    argv[:] = out


def _requested_framework(argv: List[str]) -> Optional[str]:
    """The backend named on the command line, in any spelling, or ``None`` — read from argv
    because the pin must land BEFORE the config is loaded."""
    for i, arg in enumerate(argv):
        if arg.startswith("--framework="):
            return arg.split("=", 1)[1]
        if arg.startswith(("--scope=", "-s=")) and "framework=" in arg:
            return arg.split("framework=", 1)[1]
        if arg in ("--framework", "--scope", "-s") and i + 1 < len(argv):
            following = argv[i + 1]
            if arg == "--framework":
                return following
            if following.startswith("framework="):
                return following.split("=", 1)[1]
    return None


def main() -> Optional[int]:
    argv = sys.argv[1:]
    if argv and argv[0] == "sweep":
        # Dispatched BEFORE the liquifai app: the sweep group is plain argv, and building the
        # app for it would only add a layer the report commands' pinned surface never had.
        from matrainer.sweep import main as sweep_main

        return sweep_main(argv[1:])

    registrations = _pop_registrations(sys.argv)
    framework = _requested_framework(sys.argv)
    _pin_keras_backend(framework)  # BEFORE any registration import binds an engine
    _import_registrations(registrations)
    _normalize_framework_flag(sys.argv)
    app.run()
    return None


if __name__ == "__main__":
    raise SystemExit(main())

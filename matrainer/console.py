"""The ``matrainer`` console script — generic train verbs + the sweep group.

A user-authorized exception (2026-08-07) to the old "matrainer ships no binary" rule, in two
halves:

* ``matrainer train|evaluate|predict <config.yaml> [--framework F] [--tracker T] [--KEY VAL …]``
  — the GENERIC runner: any project's ``runnable:``-rooted confluid config runs through the
  same materialize-and-dispatch path the per-project CLIs use. It works on any config whose
  class references are module-qualified — which every shipped example and sweep overlay is —
  and takes ``--registrations mod1,mod2`` for one that names bare registered classes.
* ``matrainer sweep backends|trackers|all <sweep.yaml>`` (+ the read-back commands ``report`` /
  ``next-experiment`` / ``json``) — dispatched to :mod:`matrainer.sweep` before the liquifai
  app is even built.

The per-project CLIs (their FRAMEWORKS tables, registrations, task-specific help) remain the
ergonomic surface; this runner is what a project-agnostic caller — the sweep, a CI job, an
operator with a config in hand — uses without knowing which project the config belongs to.

The whole verb machinery is :mod:`matrainer.cli` — the same factory the per-project CLIs are
built from, invoked WITHOUT a FRAMEWORKS table (a project-agnostic runner has none; what it
has instead is ``--registrations``, an explicit module list stripped from argv before liquifai
would read it as a config override). The rationale for each pre-step — the keras engine pin's
ORDER, the ``--framework`` rewrite, manual flow through ``materialize_runnable`` — lives on
the factory's own docstrings.
"""

import importlib
import sys
from typing import List, Optional

from loggair import get_logger

from matrainer.cli import VERB_TASKS, make_trainer_app, normalize_framework_flag, pin_keras_backend, requested_framework

logger = get_logger(__name__)

__all__ = ["VERB_TASKS", "app", "main"]

app = make_trainer_app(
    name="matrainer",
    description="Generic runner + sweep for runnable configs",
    verb_docs={
        "train": "Train the config's runnable (any project's)",
        "evaluate": "Score the config's runnable over its test split",
        "predict": "Stream the config's runnable's predictions to its sink",
    },
)


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
    override for a ``registrations`` key) and return the module list."""
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


def main() -> Optional[int]:
    argv = sys.argv[1:]
    if argv and argv[0] == "sweep":
        # Dispatched BEFORE the liquifai app: the sweep group is plain argv, and building the
        # app for it would only add a layer the report commands' pinned surface never had.
        from matrainer.sweep import main as sweep_main

        return sweep_main(argv[1:])

    registrations = _pop_registrations(sys.argv)
    framework = requested_framework(sys.argv)
    pin_keras_backend(framework)  # BEFORE any registration import binds an engine
    _import_registrations(registrations)
    normalize_framework_flag(sys.argv)
    app.run()
    return None


if __name__ == "__main__":
    raise SystemExit(main())

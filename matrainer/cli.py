"""The trainer-CLI FACTORY — the machinery of a task project's command line.

Every training project in a workspace grows the same CLI: three verbs (``train`` /
``evaluate`` / ``predict``) that force a ``task`` onto a ``runnable:``-rooted confluid config,
a ``--framework`` flag naming which backend's scope block activates, a ``frameworks`` listing
command, and two pre-steps that must run BEFORE the app loads the config. Four projects
carried that as ~280 near-identical lines each (measured 2026-08-07 — the modules differed in
their FRAMEWORKS table, their verb help texts, and nothing else that was not a docstring); the
machinery moved here and what stays in a project is its DATA::

    from matrainer.cli import Framework, make_trainer_app, trainer_main

    FRAMEWORKS = {"lightning": Framework("…", "myproject.torch_task"), ...}
    app = make_trainer_app(name="myproject", description="…", frameworks=FRAMEWORKS,
                           verb_docs={"train": "Train a …", ...})

    def main() -> None:
        trainer_main(app, FRAMEWORKS)

The behaviours the four CLIs established, each for its measured reason:

* **The verbs are ``flow_mode="manual"`` and build through
  ``recordstream.cli.materialize_runnable``** — never a bare ``flow()``: the handlers annotate
  ``runnable: Any`` (the runnable is polymorphic), so liquifai's auto deep-flow would drop
  every top-level config key SILENTLY (``train_set`` becomes None, ``max_epochs`` falls back
  to the ctor default without a trace).
* **The verb wins over the YAML's ``task:``**, so one config serves all three verbs — and a
  runnable with NO ``task`` attribute runs as-is for ``train`` ONLY; the other verbs raise
  naming the class (a silent ``hasattr`` guard would let an evaluate verb TRAIN, the failure
  one project's upstream-loop backend actually had).
* **``--framework`` is REWRITTEN to ``--scope framework=…``, never appended** — liquifai binds
  the ergonomic flag only for dimensions the config DECLARES, so against a config with no
  ``!scope:framework=…`` block a bare ``--framework X`` would be read as a config override.
* **The keras engine pin lands BEFORE anything imports keras** — Keras 3 reads
  ``KERAS_BACKEND`` once, at import, and registering a backend's classes can be that import.
* **Registrations are imported BEFORE ``app.run()``** — the eager instance form
  (``!class:Foo()`` with parens) is built during config load, earlier than any handler runs.
  With no ``--framework`` nothing is imported (there is no safe guess — importing the
  default's set would drag torch into a keras-only install), and a missing backend is a
  debug note, not an error: absent registrations on a partial install are the correct state.

The generic ``matrainer`` console script (:mod:`matrainer.console`) is built from the same
pieces — it is this factory minus the FRAMEWORKS table, which is exactly the difference
between a project-agnostic runner and a project CLI.
"""

import importlib
import os
import sys
from typing import Any, Callable, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union

from liquifai import LiquifyApp
from loggair import get_logger

logger = get_logger(__name__)

__all__ = [
    "DEFAULT_VERB_DOCS",
    "Framework",
    "VERB_TASKS",
    "default_dispatch",
    "make_trainer_app",
    "normalize_framework_flag",
    "pin_keras_backend",
    "register_frameworks",
    "registration_modules",
    "requested_framework",
    "trainer_main",
]

#: Verb -> the ``task`` it forces. The verb wins over any ``task:`` in the YAML.
VERB_TASKS = {"train": "fit", "evaluate": "test", "predict": "predict"}

#: The generic help texts — a project names its task with ``verb_docs=``.
DEFAULT_VERB_DOCS = {
    "train": "Train the config's runnable",
    "evaluate": "Score the config's runnable over its test split",
    "predict": "Stream the config's runnable's predictions to its sink",
}


class Framework(NamedTuple):
    """One training backend row: what to say about it, what to import for it.

    ``registrations`` is the module (or modules) whose IMPORT registers this engine's classes
    under a plain NAME — ``None`` means the backend has nothing to register, which is a real
    state rather than an oversight (a model that computes its own loss leaves no loss a config
    could wire; a loss that is a plain function leaves no class for a tag to resolve). A tuple
    makes a multi-module coupling readable IN the table rather than hidden as a side-effect
    import inside one of them.
    """

    description: str
    registrations: Union[None, str, Tuple[str, ...]] = None


def registration_modules(spec: Framework) -> Tuple[str, ...]:
    """The row's registration modules as a tuple — the one place the union is normalized."""
    if spec.registrations is None:
        return ()
    if isinstance(spec.registrations, str):
        return (spec.registrations,)
    return tuple(spec.registrations)


def default_dispatch(runnable: Any, task: str) -> None:
    """Build the bound runnable, force ``task``, and run it.

    ``materialize_runnable`` rather than a bare ``flow()`` — the flat config's top-level keys
    only broadcast into the constructor when the node is built against its document; a bare
    flow drops them SILENTLY.

    A runnable with NO ``task`` attribute runs as-is for ``fit`` only; the other tasks raise
    naming the class — a silent ``hasattr`` guard would let an evaluate verb TRAIN.
    """
    if runnable is None:
        raise ValueError("No runnable was bound — provide one under 'runnable:' in your YAML.")

    from recordstream.cli import materialize_runnable

    runnable = materialize_runnable(runnable)
    if hasattr(runnable, "task"):
        runnable.task = task  # the verb wins over the YAML's task
    elif task != "fit":
        raise ValueError(
            f"{type(runnable).__name__} has no 'task' attribute, so only the train verb can "
            "run it — this verb would silently do something else."
        )
    else:
        logger.info(f"{type(runnable).__name__} has no 'task' knob — running it as-is.")
    runnable.run()


def _verb_command(verb: str, doc: str, dispatch: Callable[[Any, str], None]) -> Callable[[Any], None]:
    """One verb — a factory, never a default-argument closure (liquifai reads the handler's
    SIGNATURE to build the command, so a smuggled loop variable would surface as a ``--_verb``
    flag in help and TAB completion). The one parameter must be named ``runnable``: that is
    the config key liquifai's DI binds it to."""

    def command(runnable: Any) -> None:
        dispatch(runnable, VERB_TASKS[verb])

    command.__name__ = verb
    command.__doc__ = f"{doc} (task={VERB_TASKS[verb]}). Pick a backend with --framework."
    return command


def make_trainer_app(
    name: str,
    description: str = "",
    frameworks: Optional[Mapping[str, Framework]] = None,
    verb_docs: Optional[Mapping[str, str]] = None,
    dispatch: Callable[[Any, str], None] = default_dispatch,
) -> LiquifyApp:
    """A liquifai app with the three trainer verbs, plus a ``frameworks`` listing command
    when a table is given (``None`` builds the project-agnostic shape the generic runner
    uses). ``dispatch`` is the extension point for a project whose dispatch needs more than
    :func:`default_dispatch` — it receives ``(bound runnable, task)``."""
    app = LiquifyApp(name=name, description=description)
    docs = {**DEFAULT_VERB_DOCS, **(verb_docs or {})}
    for verb in VERB_TASKS:
        app.script_command(name=verb, flow_mode="manual")(_verb_command(verb, docs[verb], dispatch))

    if frameworks is not None:
        table = dict(frameworks)

        @app.command(name="frameworks")
        def _frameworks() -> None:
            """List the training backends `--framework` accepts."""
            width = max(len(framework) for framework in table)
            print("Training backends — pass one as `--framework <name>`:\n")
            for framework, spec in table.items():
                print(f"  {framework:<{width}}  {spec.description}")
            print(
                "\nThe name must also match a `!scope:framework=<name>` block in the config;"
                "\nconfluid lists the values a config declares when it does not."
            )

    return app


def pin_keras_backend(framework: Optional[str]) -> None:
    """``--framework tensorflow`` pins ``KERAS_BACKEND`` before ANYTHING imports keras.

    Keras 3 reads the variable once, at import, and the workspace default resolves to the
    first engine INSTALLED (torch, on most machines) — a torch-backed ``keras.Model`` cannot
    be trained with ``tf.GradientTape``. ``setdefault``, so an explicit ``KERAS_BACKEND=…``
    still wins and is reported as a mismatch by the backend itself rather than silently
    overridden. No other framework touches the variable — running on whichever engine is
    bound is the point of a plain ``keras`` backend.
    """
    if framework == "tensorflow":
        os.environ.setdefault("KERAS_BACKEND", "tensorflow")


def register_frameworks(framework: Optional[str], frameworks: Optional[Mapping[str, Framework]]) -> None:
    """Import the named backend's registration modules so a config may use BARE names.

    A ``!class:`` target that is a registered name resolves only once the registering module
    has been imported; a dotted path resolves through importlib either way. Discovery services
    bootstrap every ``confluid.configurables`` entry point — a CLI does not, and it must not
    import ALL of them (that would drag every installed framework into every run).

    Best-effort: a missing backend is a normal state on a partial install, and its absent
    registrations are the correct outcome, not an error.
    """
    if framework is None or not frameworks:
        return
    spec = frameworks.get(framework)
    if spec is None:
        return
    for module in registration_modules(spec):
        try:
            importlib.import_module(module)
        except ImportError as exc:
            logger.debug(f"{framework}: registrations unavailable ({module}: {exc})")


def normalize_framework_flag(argv: List[str]) -> None:
    """Rewrite ``--framework <name>`` into the ``--scope framework=<name>`` liquifai reads.

    Never appended: liquifai binds the ergonomic flag only for dimensions the config DECLARES,
    and rewriting makes the flag mean one thing regardless — an undeclared dimension stays an
    inert no-op, which is what lets a config grow into a backend it does not have yet.
    Mutates ``argv`` in place because liquifai parses scopes from the command line; the
    explicit ``--scope framework=…`` spelling is left exactly as typed.
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


def requested_framework(argv: Sequence[str]) -> Optional[str]:
    """The backend named on the command line, in any spelling, or ``None`` — read from argv
    because both pre-steps need it BEFORE the config is loaded."""
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


def trainer_main(
    app: LiquifyApp,
    frameworks: Optional[Mapping[str, Framework]] = None,
    *,
    argv: Optional[List[str]] = None,
) -> None:
    """The established pre-step order, then ``app.run()``.

    The order is load-bearing and each arrow was a real bug when reversed:
    :func:`requested_framework` (from argv — the config is not loaded yet) →
    :func:`pin_keras_backend` (BEFORE any registration import binds an engine) →
    :func:`register_frameworks` (BEFORE ``app.run()`` — the eager ``!class:Foo()`` form is
    built during config load) → :func:`normalize_framework_flag` → ``app.run()``.

    ``argv`` defaults to ``sys.argv``, which is what liquifai's ``run()`` reads — pass a list
    only in tests that stub ``app.run``.
    """
    target = sys.argv if argv is None else argv
    framework = requested_framework(target)
    pin_keras_backend(framework)
    register_frameworks(framework, frameworks)
    normalize_framework_flag(target)
    app.run()

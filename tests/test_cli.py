"""The trainer-CLI factory (`matrainer.cli`) — the machinery every project CLI is built from.

What this pins: the verb→task map, `default_dispatch` (the verb wins over the YAML's `task:`,
the taskless guard, the materialize seam), the `Framework` row's registration normalization,
`register_frameworks`' best-effort import, the `--framework` rewrite, the keras engine pin,
the argv reader, `make_trainer_app`'s command surface (three verbs; the `frameworks` listing
only when a table is given; a custom `dispatch` reaching the handlers), and `trainer_main`'s
pre-step ORDER — each arrow of which was a real bug when reversed.
"""

from __future__ import annotations

import os
from typing import Any, List

import pytest

from matrainer import cli
from matrainer.cli import (
    DEFAULT_VERB_DOCS,
    VERB_TASKS,
    Framework,
    default_dispatch,
    make_trainer_app,
    normalize_framework_flag,
    pin_keras_backend,
    register_frameworks,
    registration_modules,
    requested_framework,
    trainer_main,
)

# --------------------------------------------------------------------------- #
# The dispatch
# --------------------------------------------------------------------------- #


class _Recorder:
    def __init__(self) -> None:
        self.task = "fit"
        self.ran = False

    def run(self) -> None:
        self.ran = True


class _Taskless:
    def __init__(self) -> None:
        self.ran = False

    def run(self) -> None:
        self.ran = True


def test_the_verb_task_map() -> None:
    assert VERB_TASKS == {"train": "fit", "evaluate": "test", "predict": "predict"}


def test_the_verb_wins_over_the_yaml_task() -> None:
    runnable = _Recorder()
    runnable.task = "predict"  # what the YAML said
    default_dispatch(runnable, "test")
    assert runnable.task == "test" and runnable.ran


def test_a_taskless_runnable_trains_as_is_and_refuses_the_other_verbs() -> None:
    """The silent `hasattr` guard would let an evaluate verb TRAIN — refuse by name instead."""
    runnable = _Taskless()
    default_dispatch(runnable, "fit")
    assert runnable.ran

    with pytest.raises(ValueError, match="_Taskless"):
        default_dispatch(_Taskless(), "test")


def test_no_runnable_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="runnable"):
        default_dispatch(None, "fit")


def test_dispatch_builds_through_the_shared_materialize_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """The seam itself: building goes through `recordstream.cli.materialize_runnable`, never a
    bare `flow()` — a bare flow drops every top-level config key silently."""
    import recordstream.cli

    seen: List[Any] = []

    def spy(runnable: Any) -> Any:
        seen.append(runnable)
        return runnable

    monkeypatch.setattr(recordstream.cli, "materialize_runnable", spy)
    runnable = _Recorder()
    default_dispatch(runnable, "fit")
    assert seen == [runnable]


# --------------------------------------------------------------------------- #
# The Framework row + registrations
# --------------------------------------------------------------------------- #
def test_registration_modules_normalizes_every_spelling() -> None:
    """`None` (nothing to register — a real state), one module, or a tuple that makes a
    multi-module coupling readable IN the table."""
    assert registration_modules(Framework("x")) == ()
    assert registration_modules(Framework("x", "a.mod")) == ("a.mod",)
    assert registration_modules(Framework("x", ("a.mod", "b.mod"))) == ("a.mod", "b.mod")


def test_register_frameworks_imports_the_named_rows_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    imported: List[str] = []
    monkeypatch.setattr(cli.importlib, "import_module", lambda module: imported.append(module))
    register_frameworks("torch", {"torch": Framework("", ("a.mod", "b.mod"))})
    assert imported == ["a.mod", "b.mod"]


def test_register_frameworks_is_a_noop_without_a_framework_or_a_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """No `--framework` imports NOTHING — there is no safe guess (importing the default's set
    would drag torch into a keras-only install)."""

    def explode(module: str) -> None:
        raise AssertionError(f"imported {module}")

    monkeypatch.setattr(cli.importlib, "import_module", explode)
    register_frameworks(None, {"torch": Framework("", "a.mod")})
    register_frameworks("torch", None)
    register_frameworks("unknown", {"torch": Framework("", "a.mod")})
    register_frameworks("bare", {"bare": Framework("")})


def test_a_missing_backend_is_best_effort_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent registrations on a partial install are the correct state, not a failure."""

    def missing(module: str) -> None:
        raise ImportError(f"No module named {module!r}")

    monkeypatch.setattr(cli.importlib, "import_module", missing)
    register_frameworks("torch", {"torch": Framework("", "not.there")})  # must not raise


# --------------------------------------------------------------------------- #
# The argv pre-steps
# --------------------------------------------------------------------------- #
def test_the_framework_flag_is_rewritten_never_appended() -> None:
    argv = ["train", "cfg.yaml", "--framework", "keras", "--epochs-like", "3"]
    normalize_framework_flag(argv)
    assert argv == ["train", "cfg.yaml", "--scope", "framework=keras", "--epochs-like", "3"]

    argv = ["train", "cfg.yaml", "--framework=torch"]
    normalize_framework_flag(argv)
    assert argv == ["train", "cfg.yaml", "--scope", "framework=torch"]


def test_an_explicit_scope_spelling_is_left_exactly_as_typed() -> None:
    argv = ["train", "cfg.yaml", "--scope", "framework=keras"]
    normalize_framework_flag(argv)
    assert argv == ["train", "cfg.yaml", "--scope", "framework=keras"]


def test_the_requested_framework_is_read_in_every_spelling() -> None:
    assert requested_framework(["train", "c.yaml", "--framework", "jax"]) == "jax"
    assert requested_framework(["train", "c.yaml", "--framework=mlx"]) == "mlx"
    assert requested_framework(["train", "c.yaml", "--scope", "framework=keras"]) == "keras"
    assert requested_framework(["train", "c.yaml", "--scope=framework=torch"]) == "torch"
    assert requested_framework(["train", "c.yaml"]) is None


def test_the_tensorflow_framework_pins_the_keras_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    # `delenv(raising=False)` records NOTHING when the variable is already absent, so the
    # direct `os.environ[...] = ...` inside `pin_keras_backend` would have no undo entry and
    # leak `KERAS_BACKEND=tensorflow` into every later test — where `import keras` then tries
    # to load a TensorFlow this workspace may not have. Setting it first gives monkeypatch a
    # record to restore from.
    monkeypatch.setenv("KERAS_BACKEND", "torch")
    monkeypatch.delenv("KERAS_BACKEND")
    pin_keras_backend("tensorflow")
    assert os.environ.get("KERAS_BACKEND") == "tensorflow"
    monkeypatch.setenv("KERAS_BACKEND", "torch")
    pin_keras_backend("tensorflow")
    assert os.environ["KERAS_BACKEND"] == "torch", "setdefault — an explicit choice always wins"


@pytest.mark.parametrize("framework", ["keras", "torch", "lightning", None])
def test_no_other_framework_touches_the_variable(monkeypatch: pytest.MonkeyPatch, framework: Any) -> None:
    """Running on whichever engine is bound is the point of a plain `keras` backend."""
    monkeypatch.setenv("KERAS_BACKEND", "torch")  # see above: gives monkeypatch a record to undo
    monkeypatch.delenv("KERAS_BACKEND")
    pin_keras_backend(framework)
    assert "KERAS_BACKEND" not in os.environ


# --------------------------------------------------------------------------- #
# The app factory
# --------------------------------------------------------------------------- #
def test_the_three_verbs_are_registered() -> None:
    app = make_trainer_app(name="probe")
    for verb in VERB_TASKS:
        assert verb in app._commands


def test_the_frameworks_command_exists_only_with_a_table() -> None:
    """`None` builds the project-agnostic shape the generic runner uses."""
    assert "frameworks" not in make_trainer_app(name="probe")._commands
    assert "frameworks" in make_trainer_app(name="probe", frameworks={"torch": Framework("")})._commands


def test_the_frameworks_command_prints_the_table(capsys: pytest.CaptureFixture[str]) -> None:
    app = make_trainer_app(name="probe", frameworks={"torch": Framework("the control experiment")})
    app._commands["frameworks"]()
    out = capsys.readouterr().out
    assert "torch" in out and "the control experiment" in out


def test_verb_docs_override_the_generic_texts() -> None:
    app = make_trainer_app(name="probe", verb_docs={"train": "Train a widget"})
    train = app._commands["train"]
    assert train.__doc__ is not None and "Train a widget" in train.__doc__
    evaluate = app._commands["evaluate"]
    assert evaluate.__doc__ is not None and DEFAULT_VERB_DOCS["evaluate"] in evaluate.__doc__


def test_a_custom_dispatch_reaches_every_verb() -> None:
    """The extension point for a project whose dispatch needs more than the default."""
    seen: List[Any] = []
    app = make_trainer_app(name="probe", dispatch=lambda runnable, task: seen.append((runnable, task)))
    app._commands["evaluate"](runnable="the-node")
    assert seen == [("the-node", "test")]


def test_the_handler_signature_is_exactly_runnable() -> None:
    """The parameter NAME is what liquifai's DI binds to the config's `runnable:` key, and a
    smuggled loop variable would surface as a `--_verb` flag."""
    import inspect

    app = make_trainer_app(name="probe")
    for verb in VERB_TASKS:
        assert list(inspect.signature(app._commands[verb]).parameters) == ["runnable"]


# --------------------------------------------------------------------------- #
# trainer_main — the pre-step order
# --------------------------------------------------------------------------- #
def test_the_pre_step_order_is_the_established_one() -> None:
    """Each arrow was a real bug when reversed: the pin BEFORE the registration import (the
    import is what binds the wrong engine), registration BEFORE app.run (the eager
    `!class:Foo()` form is built during config load). Read from the SOURCE — calling it would
    run the app."""
    import inspect

    # The BODY only — the docstring narrates the same order and would shadow the code's.
    source = inspect.getsource(trainer_main).split('"""')[-1]
    order = [
        source.index("requested_framework("),
        source.index("pin_keras_backend("),
        source.index("register_frameworks("),
        source.index("normalize_framework_flag("),
        source.index("app.run()"),
    ]
    assert order == sorted(order)


def test_trainer_main_rewrites_argv_registers_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    imported: List[str] = []
    monkeypatch.setattr(cli.importlib, "import_module", lambda module: imported.append(module))

    class _App:
        def __init__(self) -> None:
            self.ran = False

        def run(self) -> None:
            self.ran = True

    app = _App()
    argv = ["probe", "train", "cfg.yaml", "--framework", "torch"]
    trainer_main(app, {"torch": Framework("", "a.mod")}, argv=argv)  # type: ignore[arg-type]
    assert app.ran
    assert imported == ["a.mod"]
    assert argv == ["probe", "train", "cfg.yaml", "--scope", "framework=torch"]

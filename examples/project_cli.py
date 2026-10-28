"""A training project's command line, built from a frameworks table and two calls.

Asserts what `docs/cli.md` claims:

* `make_trainer_app` gives the three verbs — `train` / `evaluate` / `predict` — with no project
  code beyond the table;
* the `frameworks` listing appears only when a project passes a table, so the generic runner
  (which has none) does not advertise one;
* `--framework X` is REWRITTEN to the `--scope framework=X` liquifai reads, never appended;
* `requested_framework` reads the choice out of argv in any spelling, before any config loads —
  which is what lets `--framework tensorflow` pin `KERAS_BACKEND` before keras is imported;
* `lazy_exports` keeps a package root framework-free: touching a name whose backend is missing
  raises an `ImportError` naming the exact extra to install.

Nothing is executed: building the app and reading argv needs no config, no dataset and no
framework. A real project ships `FRAMEWORKS` plus a `main()` that calls `trainer_main(app, …)`.

Run:  python examples/project_cli.py
"""

from matrainer._exports import lazy_exports
from matrainer.cli import Framework, make_trainer_app, normalize_framework_flag, requested_framework


def a_projects_whole_cli() -> None:
    """Everything a project writes: a table of its backends."""
    frameworks = {
        "lightning": Framework("the reference backend", "myproject.torch_task"),
        "keras": Framework("Keras 3 on any engine", "myproject.keras_task"),
        "huggingface": Framework("a Trainer that computes its own loss"),  # nothing to register
    }
    app = make_trainer_app(
        name="myproject",
        description="Train a toy classifier.",
        frameworks=frameworks,
        verb_docs={"train": "Train a classifier on a config."},
    )

    for verb in ("train", "evaluate", "predict"):
        assert verb in app._commands, verb
    # A project with backends gets the listing; the generic runner, which passes none, does not.
    assert "frameworks" in app._commands
    assert "frameworks" not in make_trainer_app(name="generic")._commands
    print(f"myproject: {', '.join(sorted(app._commands))}")


def the_framework_flag_is_rewritten() -> None:
    """One flag means one thing, whether or not the config declares the dimension."""
    argv = ["train", "config.yaml", "--framework", "keras"]
    normalize_framework_flag(argv)  # in place: liquifai parses scopes off the command line
    assert argv == ["train", "config.yaml", "--scope", "framework=keras"], argv
    assert "--framework" not in argv, "rewritten, never appended"

    # Read before anything loads, so a pre-step can act on it — in every spelling.
    assert requested_framework(["train", "c.yaml", "--framework", "keras"]) == "keras"
    assert requested_framework(["train", "c.yaml", "--framework=keras"]) == "keras"
    assert requested_framework(["train", "c.yaml", "--scope=framework=keras"]) == "keras"
    assert requested_framework(["train", "c.yaml"]) is None
    print("--framework keras  ->  --scope framework=keras")


def a_package_root_that_imports_nothing() -> None:
    """`{name: (module, extra)}` — the import happens on first touch, not at import time."""
    names = {"LightningThing": ("myproject.lightning_thing", "lightning")}
    # A root passes its OWN `__name__` — `dir_()` imports it to list the real attributes beside
    # the lazy ones, so a package that does not exist cannot stand in here. For this script that
    # name is `__main__`; in a project it is the package, and the advertised fix names it.
    getattr_, dir_ = lazy_exports(__name__, names)

    assert "LightningThing" in dir_()
    try:
        getattr_("LightningThing")  # the backend module is absent, as a missing extra leaves it
    except ImportError as exc:
        assert "lightning" in str(exc), exc  # names the extra...
        assert "pip install" in str(exc), exc  # ...as the command that installs it
        print(
            "a name whose backend is missing raises an ImportError naming the 'lightning' extra "
            "and the pip command, not a bare ModuleNotFoundError"
        )
    else:  # pragma: no cover - only if a module of that name really exists
        raise AssertionError("expected an ImportError naming the extra")


def main() -> None:
    a_projects_whole_cli()
    the_framework_flag_is_rewritten()
    a_package_root_that_imports_nothing()


if __name__ == "__main__":
    main()

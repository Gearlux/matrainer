# Building a project CLI

Your project's `train|evaluate|predict` binary, built from a frameworks table and two calls.

Runnable twin: [`examples/project_cli.py`](../examples/project_cli.py) — a project's app built from a frameworks table, and the flag rewrite.

## Building a project CLI (`matrainer.cli`, no extra needed)

A training project's command line is three verbs (`train` / `evaluate` / `predict`) over a
`runnable:`-rooted config, a `--framework` flag, and a `frameworks` listing — the machinery is
identical across projects, so it is a FACTORY here and a project ships only its DATA:

```python
from matrainer.cli import Framework, make_trainer_app, trainer_main

FRAMEWORKS = {
    "lightning": Framework("the reference backend", "myproject.torch_task"),
    "keras": Framework("Keras 3 on any engine", "myproject.keras_task"),
    "huggingface": Framework("a Trainer that computes its own loss"),   # nothing to register
}
app = make_trainer_app(name="myproject", description="…", frameworks=FRAMEWORKS,
                       verb_docs={"train": "Train a …"})

def main() -> None:                     # [project.scripts] myproject = "myproject.cli:main"
    trainer_main(app, FRAMEWORKS)
```

What the machinery does (each behaviour documented on its function): the verbs build through
`recordstream`'s `materialize_runnable` (never a bare flow — that drops every top-level config
key silently) and the verb wins over the YAML's `task:`; a runnable with no `task` attribute
trains as-is and REFUSES the other verbs; `--framework X` is rewritten to `--scope
framework=X`; `--framework tensorflow` pins `KERAS_BACKEND` before anything imports keras; and
the named backend's registration modules are imported before the config loads. `dispatch=` on
`make_trainer_app` is the extension point when the default dispatch is not enough.

Every verb takes the app framework's global flags, `--dump-config` among them: `myproject train config.yaml
--framework keras --dump-config` prints the configuration that run resolves to — every setting of every class, the
defaults included — and trains nothing; `--save-config PATH` writes it and then trains
([global flags](https://github.com/Gearlux/liquifai/blob/main/docs/global-flags.md)). The dumped `task:` is the one
the verb runs (`evaluate` → `test`): each verb adds `runnable.task` to the configuration when the runnable's class
takes a `task`, and adds nothing to one that does not.

A package root keeps its core framework-free with the same library — the PEP 562 resolver is
`matrainer._exports.lazy_exports`; the name table stays the project's:

```python
_BACKEND_NAMES = {"LightningThing": ("myproject.lightning_thing", "lightning")}
__all__ = [*_BACKEND_NAMES]
__getattr__, __dir__ = lazy_exports(__name__, _BACKEND_NAMES)
```

Touching a name whose backend is missing raises an `ImportError` naming the exact extra; an
empty extra (`""`) marks a core name whose failure passes through untranslated.

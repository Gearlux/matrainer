# Marainer Architecture Decisions

Short decision records for the mechanisms whose *shape* isn't obvious from the code alone.
Each record: **Context → Decision → Consequences → Example → What you may change**. User-facing
"how do I use it" lives in the [README](../README.md); this file is the *why*.

Per the workspace "Docs Never Name Dependent Projects" mandate, these records describe generic
"consuming projects" / "a training platform", never a specific downstream package by name.

---

## 1. Marainer tracks; it does not train (2026-07)

**Context.** Marainer used to be a training *orchestrator*: a `LightningTrainer` / `LightningEvaluator`
pair with `training_step` / `validation_step` / `test_step`, a `run()` loop, a five-verb CLI, and
`pytorch_lightning` / `torch` hard-imported even in modules that only logged a git hash. Two problems
compounded. (a) Every consumer inherited a heavy base whose training loop it usually overrode anyway,
so the "shared" loop was a delegation maze, not a simplification. (b) A project that only wanted the
*tracking* (log params, git hash, config artifacts, system metrics) still paid the full Lightning
import cost and couldn't be removed without gutting the training code.

**Decision.** Marainer is an **experiment-tracking library only** — no training loop, no
trainer/evaluator class, no CLI. A class that trains reads like a standard example for its framework
and lives in the *consuming* project; it reaches for Marainer solely to track the run. The runnable
protocol (`TorchRunner`, `ProgressReporting`, the `entrypoint` method-annotation mechanism), the
workflow combinators, and `DatasetProcessor` moved to `recordstream`; the generic runner is
`recordstream run <config>` binding a top-level `runnable:` key.

**Consequences.**
- The guiding invariant is testable: **removing Marainer costs a consumer only a couple of line
  changes** — it keeps training (with loggair logging), losing only the rich tracking. If a change
  makes Marainer load-bearing for *training*, it's in the wrong project.
- `import marainer` imports no torch/lightning/mlflow (see record 2). The light core is
  `confluid`/`liquifai`/`loggair`/`coolname`/`numpy`.
- There is exactly one thing to reintroduce-and-reject in review: a `training_step`, a `run()` loop,
  or a `marainer.cli`. None belong here.

**Example.** A consumer's standard `LightningModule` opts into tracking without inheriting anything
from Marainer:

```python
import pytorch_lightning as pl
from marainer import CsvTracker, MlflowTracker
from marainer.lightning import TrackingCallback

trainer = pl.Trainer(
    max_epochs=10,
    callbacks=[TrackingCallback([CsvTracker(), MlflowTracker(experiment_name="exp")])],
)
trainer.fit(MyStandardLightningModule(), datamodule=dm)   # delete the callback -> still trains
```

**What you may change.** New tracking helpers, backends, provenance sources, and framework hookup
styles are all in scope. What you may *not* add back: a training/evaluation loop, a step function, a
`run()` dispatcher, a `DatasetProcessor`/workflow combinator (those live in `recordstream`), or a CLI.

---

## 2. The `Tracker` ABC + one backend per file, lazily imported (2026-07)

**Context.** Tracking must reach several logger libraries (MLflow, TensorBoard, CSV, Weights &
Biases) that a given install rarely all has. Importing all of them at package import — the old shape
— made a light `import marainer` drag in mlflow/wandb and fail on a machine that has neither, even
for a consumer that only wanted the CSV backend. Consumers also want ONE surface they can hand to any
framework's logger slot, not four bespoke logging call sites.

**Decision.** A single `Tracker` ABC (`tracking/base.py`) mirrors the
`pytorch_lightning.loggers.Logger` shape (`name`/`version`/`experiment`/`log_hyperparams`/
`log_metrics`/`save`/`finalize`) plus the extensions `set_tags`/`log_artifact`/`log_figure`/
`log_dir`. `TrackerGroup` fans out to a list behind the same surface. Each backend lives
**one-per-file** and imports its library **lazily inside its methods** via
`marainer._imports.require(module, extra)`; every backend is **zero-arg constructible**, building its
native handle on first `.experiment` access. The Tracker is the whole abstraction — no wrapper layer
sits over it (an earlier `as_tracker` / `_LoggerTrackerView` machinery was removed as redundant).

**Consequences.**
- `import marainer` / `import marainer.tracking` imports **none** of torch/mlflow/tensorboard/wandb;
  a `tests/test_light_import.py` pins that contract. A missing library surfaces as one actionable
  error (`pip install marainer[mlflow]`), never a bare `ModuleNotFoundError` at import time.
- `CsvTracker` is stdlib-only, so it is the always-available default and the backend every test can
  exercise without mocking.
- Zero-arg construction is what lets Trackers be `@configurable` and wire from YAML like every other
  workspace primitive (the workspace lazy-init rule).
- A framework hookup (a `pl.Callback`, a `TrackerLogger` bridge, a Keras callback) is written ONCE
  against `Tracker`, so all four backends work through any of the three hookup styles for free.

**Example.** Adding a backend is a new file that stays import-safe:

```python
# marainer/tracking/newthing_backend.py
from marainer._imports import require
from marainer.tracking.base import Tracker

class NewThingTracker(Tracker):
    @property
    def experiment(self):
        if self._exp is None:
            newthing = require("newthing", "newthing")   # lazy: import + extra-name for the error
            self._exp = newthing.Run(self.experiment_name)
        return self._exp
```

**What you may change.** Add or extend backends and Tracker methods freely. What you may not do:
import a backend's library at module top level, add a required constructor arg, or reintroduce a
view/adapter layer over `Tracker`.

---

## 3. The run manifest is the union; a native slot is a projection (2026-08-01)

> **Status: implemented.** The accumulator, the capture sources, `Tracker.log_manifest`, and
> native projections for the two backends that have native slots. `log_run_context` is the single
> orchestration entry: it fills a manifest and projects it. Backends added later must satisfy
> rule 2 below and must not break rule 3.

**Context.** Rebuilding a run needs a fixed set of *static* facts — the configuration that went in,
the configuration that came out after post-construction, the command line, the hyperparameters, the
state of every repository whose code executed, the models and datasets with their versions, and
where the checkpoints landed. Today those are captured piecemeal and unevenly: `tracking/provenance.py`
uploads the transitively-included YAML tree as a `config/` artifact and stamps one `git.commit` tag,
and that is the whole of it. The command line is never read. The *resolved* configuration is never
dumped, even though the configuration library can emit it. Datasets and models are never named. And
the single git hash describes only the current working directory, while a run's behaviour is
determined by every editable package on `sys.path` — a dirty configuration library that changed how
a value was injected is invisible in the record.

Two forces then pull in opposite directions. Backends have *different* native slots — one has a
first-class dataset-input API and standard source-control tags, another has an artifact-lineage
graph, a third has a hyperparameter plugin and nothing else, and a local file backend has none. If
each backend records only what it natively supports, the same run tracked two ways produces two
incomparable records, and moving a project between backends silently loses fields. If instead
everything is flattened to the lowest common denominator, the rich backends stop being useful and
their native UIs stay empty.

**Decision.** One `RunManifest` object accumulates every static fact about a run and is the
**union**. Each backend's native slot is a **projection** of it, never an alternative:

1. every field lands in the manifest unconditionally;
2. a backend with a native home for a field *also* writes it there;
3. no field is ever *only* native.

The manifest is serialized as **`manifest.yaml` and `manifest.json` together**, from a single atomic
write (temp file + `os.replace`) performed after **every** mutation. YAML is the readable artifact —
`yaml.safe_dump(..., sort_keys=False)`, plain and tag-free, with a fixed section order so two runs
diff cleanly instead of scrambling alphabetically. JSON is the machine artifact for tooling that
should not need a YAML parser. A terminal `status` (`running` → `success` / `failed` / `interrupted`)
is stamped from `atexit` plus `SIGINT`/`SIGTERM`, the same shutdown pattern the MPS sampler already
uses in `tracking/system.py`.

**Consequences.**
- **Crash-safety is a write policy, not a shutdown hook.** Because every mutation re-emits both
  files, an interrupted run leaves a complete record of everything gathered up to that point. The
  signal handlers only stamp the final status — if they never run, the manifest is still there and
  still valid, merely marked `running`. A design that buffered until run end would lose exactly the
  runs whose provenance matters most.
- **Runs stay comparable across backends.** The same manifest exists whether the run tracked to a
  hosted server or to a local directory, so a project can move between backends without a gap in its
  history.
- **The manifest owns its own format and no other.** The configuration library's dump emits
  `!class:` / `!ref:` / `!clone:` tags that only its own resolver can load; embedding that inside the
  manifest would make the manifest unreadable by a generic YAML tool, which defeats the point of
  choosing YAML. So the resolved configuration is written to its own `config/final.yaml` artifact and
  the manifest references it by path plus digest. This is the same rule `marainer.artifacts` already
  follows for the label vocabulary — persist it, log it, but never re-hand-roll someone else's
  format.
- **`pyyaml` becomes a declared core dependency.** It is currently reachable only transitively
  through the configuration library. A module that imports it directly must declare it.
- **Code provenance becomes a list.** `get_git_commit_hash()` stays as the single-repo helper it is;
  the manifest's `code:` section carries one record per repository, discovered from installed
  distributions' editable-install metadata, each with commit, dirty flag, branch and remote.
- **One new `Tracker` method**, `log_manifest(manifest)`, defaults to serialize-and-`log_artifact`.
  Every existing backend therefore works the day the manifest lands, and native projection is an
  opt-in per-backend override. No wrapper layer appears over `Tracker` — record 2 still holds.
- Hyperparameter derivation moves *into* this package. Walking a configured object graph to produce
  `{dotted-path: value}` is provenance work; leaving it to each consumer is why two consumers of the
  same tracking surface can give two different answers to "what were the hyperparameters?".
- **A lossy native slot may decline to answer.** Some backends have exactly one field for a source
  commit while the manifest holds a *list*. Filling that field with an arbitrary pick makes a coin
  flip look like a fact — a run was once stamped with an unrelated workspace package's commit
  purely because it sorted first. So the union→slot narrowing is explicit and ordered (the
  repository containing the working directory, then the one containing the entry script, then a
  lone record), and yields **nothing** when none applies. An empty native field sends the reader to
  the manifest, which is correct; a wrong one does not look wrong.
- **One orchestration entry, not two overlapping ones.** `log_run_context` fills the manifest and
  projects it, returning it so the caller can go on recording; `capture_log_file` finishes it. A
  consumer that had to choose between "call the provenance helper" and "drive a manifest" would
  produce two different partial answers to the same question — which is the failure this record
  exists to prevent, one level up.

**Example.**

```python
manifest = log_run_context(trackers, runnable=runnable)   # capture everything + project it
manifest.record_dataset(kind="hf", name="org/set", split="train", count=12_000)
manifest.record_output("epoch=3.ckpt", monitor="val/acc", value=0.91)
# ... every call re-emitted manifest.yaml AND manifest.json atomically ...
capture_log_file(trackers, manifest=manifest, status="success")   # Ctrl-C -> "interrupted"
```

and a backend that has native slots widens the projection without narrowing the union:

```python
class MlflowTracker(Tracker):
    def log_manifest(self, manifest):
        super().log_manifest(manifest)        # artifact + hyperparams: the union, always
        self._project_source_tags(manifest)   # mlflow.source.git.* — ADDITIONAL, never instead
        self._project_datasets(manifest)      # log_inputs: datasets become first-class
```

Three consequences of the rules above are visible in that snippet and worth stating outright.
`capture_code` returns a *list* because a run's behaviour is set by every editable package on
`sys.path`, not by whichever repository the shell was sitting in. `capture_environment` redacts
anything whose name looks like a credential, because the manifest becomes an uploaded artifact and
a captured API key would be a secret published to every backend. And `capture_config` writes the
resolved dump to a *separate* file: it carries the configuration library's `!class:` / `!ref` tags,
which would make the manifest unreadable by a generic YAML tool.

**What you may change.** New sections, new capture sources, new native projections, new backends
that project more of it. What you may *not* do: make a field native-only, embed another library's
serialization format inside the manifest, or defer the write to run end.

---

## 4. Backends are separately installable: tolerate, declare, select (2026-08-01)

> **Status: all three layers ship.** Tolerance was already there; discovery + `[tool.aisland]
> trackers` + the workspace `aisland tracker` command landed 2026-08-01. `AimTracker` and
> `ClearMLTracker` were the first backends added under the new rules — each one new file plus one
> extra, with no edit to any existing file, which is the claim this record makes.

**Context.** Backends are a *menu*, not a set. A machine that tracks to a local server has no reason
to download the client of a hosted service it has no account for, and a continuous-integration runner
may want every backend installed so every one stays tested. Record 2 solved half of this: lazy
in-method imports mean the code *tolerates* a missing backend, and a light `import marainer.tracking`
pulls in none of them. But nothing decides what a given machine *installs*, so in practice the
answer is "the `[all]` extra, or nothing" — the same gap the framework extras had before the
workspace gained a framework selection.

A second, subtler limit: `marainer/tracking/__init__.py` imports each backend module by name and
lists each class in `__all__`. That is fine for four, but it means every new backend edits a shared
file, and a backend shipped by a *different distribution* can never appear there at all.

**Decision.** Three layers, deliberately separate.

1. **Tolerate** (ships today, unchanged) — every backend imports its library lazily inside its
   methods through `require(module, extra)`, so a missing library is one actionable
   `pip install marainer[<extra>]` error and never an import-time failure.
2. **Declare** — a project declares which of its extras are tracker backends in its own
   `pyproject.toml`, in a table *sibling* to the framework declaration it already carries:
   `[tool.aisland] trackers = [...]`. Sibling rather than an addition to `frameworks`, because the
   axes are independent — the framework is a property of what the machine *trains with*, the backend
   a property of where results *go*. Folding them together would install a tracking client because
   someone selected a training engine.
3. **Select** — the workspace commits its selection to a root file, with a per-machine override in
   local git config, mirroring the framework axis exactly (committed layer shared and read by
   generated CI; machine layer never reaching a tracked file).

Backend *discovery* is by entry point (`confluid.configurables`), so a backend shipped by another
distribution is discoverable with no change to this package; and `marainer.tracking` exposes its own
backends through a module-level `__getattr__` (PEP 562) rather than a fixed import list, so adding
one does not touch an existing file.

**Consequences.**
- Adding a backend is one new file plus one extra, with no edit to `tracking/__init__.py` and no
  entry in a registry table. An out-of-tree backend needs nothing from this package at all.
- The declaration is the *menu*, the selection is what *lands* — declaring a backend does not
  install it. A machine can drop a backend it never uses while CI still installs the committed set
  and keeps every backend tested.
- The extra names the **library**, never a distribution that merely bundles it — the same rule the
  Keras helper's extra already encodes.
- A backend needing a local server also needs a supervised service in the workspace environment
  manager. A service definition must then *assert* that the backend's extra is selected rather than
  installing it ad hoc behind the operator's back; an ad-hoc install contradicts the selection layer
  it is supposed to honour.
- Selection criterion for which backends this package implements in-tree: **free and self-hostable**.
  A backend whose server is a paid or hosted-only product can still be supported — the abstraction is
  the same — but it is not what a fresh checkout is steered toward. The user-facing survey of the
  alternatives lives in the [README](../README.md#choosing-a-backend); this record carries only the
  rule that produced it.

**Example.** A project declares its backend extras beside its framework extras:

```toml
[tool.aisland]
frameworks = ["torch", "lightning", "keras"]   # what this machine trains with
trackers   = ["mlflow", "tensorboard", "aim"]  # where this machine's results go
```

and a new backend is a single self-contained file, discovered without editing anything:

```python
# marainer/tracking/aim_backend.py
from marainer._imports import require
from marainer.tracking.base import Tracker

class AimTracker(Tracker):
    @property
    def experiment(self):
        if self._run is None:
            aim = require("aim", "aim", feature="AimTracker")   # lazy: import + the extra to install
            self._run = aim.Run(experiment=self.experiment_name)
        return self._run
```

**What you may change.** Add backends and extras freely; add native projections (record 3). What you
may *not* do: import a backend's library at module top level, reintroduce a fixed import list or a
hand-maintained backend registry, or fold the backend names into the framework selection.

---

## 5. A helper's home names its dependencies, not its first consumer (2026-08-01)

**Context.** `build_metric_collection` / `coerce_metric_value` / `scalar_metrics` /
`MetricPreset` lived in `marainer/lightning/metrics.py` from the day they were
written, because a Lightning runnable was the only thing that called them. None of
them imports `pytorch_lightning` — they use confluid, torch and torchmetrics and
nothing else.

That stayed harmless until a consumer arrived that trains on plain torch with no
`Trainer` at all. `from marainer.lightning.metrics import build_metric_collection`
imports the parent package first, and `marainer/lightning/__init__.py` opens with
`require("pytorch_lightning", "lightning")` — so a deliberately Lightning-free
backend could not reach a Lightning-free helper. Measured: after
`import marainer.lightning.metrics`, `pytorch_lightning` is in `sys.modules`.

**Decision.** The module moved to `marainer/torch/metrics.py`, and `torchmetrics`
moved from the `[lightning]` extra to `[torch]` with it. `marainer.lightning`
re-exports all four names, so its public surface is unchanged.

This is the same correction as `marainer.tensorflow` → `marainer.keras` (record 2's
rule, "an optional helper's extra names the API it imports, never a distribution
that happens to bundle it") applied one level up: a subpackage's name is a claim
about what importing it costs.

**Consequences.**

- `pip install marainer[torch]` now gives a consumer the metric helpers. It also
  gives them `torchmetrics`, which is a real addition to that extra — accepted,
  because a torch consumer computing no metrics is not the case worth optimising.
- Every existing import keeps working. `raidar/detector.py` does
  `from marainer.lightning import scalar_metrics` and was not touched.
- `marainer.torch` grows for the first time since its 2026-07-29 emptying. The rule
  that emptied it still holds and this does not bend it: those helpers left because
  they were about the DATA (recordstream's business); these arrive because they are
  about torchmetrics, which is torch's.

**Example.**

```python
from marainer.torch import build_metric_collection, scalar_metrics   # no Lightning
from marainer.lightning import scalar_metrics                        # same object, still works
```

**What you may change.** Move a helper whenever its home over-states its
dependencies — check with `python -c "import <module>; import sys; print('pytorch_lightning' in sys.modules)"`
rather than by reading the imports, since a package `__init__` can require what the
module itself does not. Keep the re-export when you do: a consumer's import path is
not yours to break for tidiness.

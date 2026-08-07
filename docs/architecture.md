# Matrainer Architecture Decisions

Short decision records for the mechanisms whose *shape* isn't obvious from the code alone.
Each record: **Context → Decision → Consequences → Example → What you may change**. User-facing
"how do I use it" lives in the [README](../README.md); this file is the *why*.

Per the workspace "Docs Never Name Dependent Projects" mandate, these records describe generic
"consuming projects" / "a training platform", never a specific downstream package by name.

---

## 1. Matrainer tracks; it does not train (2026-07) — **superseded in part by record 9 (2026-08-06)**

**Context.** Matrainer used to be a training *orchestrator*: a `LightningTrainer` / `LightningEvaluator`
pair with `training_step` / `validation_step` / `test_step`, a `run()` loop, a five-verb CLI, and
`pytorch_lightning` / `torch` hard-imported even in modules that only logged a git hash. Two problems
compounded. (a) Every consumer inherited a heavy base whose training loop it usually overrode anyway,
so the "shared" loop was a delegation maze, not a simplification. (b) A project that only wanted the
*tracking* (log params, git hash, config artifacts, system metrics) still paid the full Lightning
import cost and couldn't be removed without gutting the training code.

**Decision.** Matrainer is an **experiment-tracking library only** — no training loop, no
trainer/evaluator class, no CLI. A class that trains reads like a standard example for its framework
and lives in the *consuming* project; it reaches for Matrainer solely to track the run. The runnable
protocol (`TorchRunner`, `ProgressReporting`, the `entrypoint` method-annotation mechanism), the
workflow combinators, and `DatasetProcessor` moved to `recordstream`; the generic runner is
`recordstream run <config>` binding a top-level `runnable:` key.

**Consequences.**
- The guiding invariant is testable: **removing Matrainer costs a consumer only a couple of line
  changes** — it keeps training (with loggair logging), losing only the rich tracking. If a change
  makes Matrainer load-bearing for *training*, it's in the wrong project.
- `import matrainer` imports no torch/lightning/mlflow (see record 2). The light core is
  `confluid`/`liquifai`/`loggair`/`coolname`/`numpy`.
- There is exactly one thing to reintroduce-and-reject in review: a `training_step`, a `run()` loop,
  or a `matrainer.cli`. None belong here.

**Example.** A consumer's standard `LightningModule` opts into tracking without inheriting anything
from Matrainer:

```python
import pytorch_lightning as pl
from matrainer import CsvTracker, MlflowTracker
from matrainer.lightning import TrackingCallback

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
— made a light `import matrainer` drag in mlflow/wandb and fail on a machine that has neither, even
for a consumer that only wanted the CSV backend. Consumers also want ONE surface they can hand to any
framework's logger slot, not four bespoke logging call sites.

**Decision.** A single `Tracker` ABC (`tracking/base.py`) mirrors the
`pytorch_lightning.loggers.Logger` shape (`name`/`version`/`experiment`/`log_hyperparams`/
`log_metrics`/`save`/`finalize`) plus the extensions `set_tags`/`log_artifact`/`log_figure`/
`log_dir`. `TrackerGroup` fans out to a list behind the same surface. Each backend lives
**one-per-file** and imports its library **lazily inside its methods** via
`matrainer._imports.require(module, extra)`; every backend is **zero-arg constructible**, building its
native handle on first `.experiment` access. The Tracker is the whole abstraction — no wrapper layer
sits over it (an earlier `as_tracker` / `_LoggerTrackerView` machinery was removed as redundant).

**Consequences.**
- `import matrainer` / `import matrainer.tracking` imports **none** of torch/mlflow/tensorboard/wandb;
  a `tests/test_light_import.py` pins that contract. A missing library surfaces as one actionable
  error (`pip install matrainer[mlflow]`), never a bare `ModuleNotFoundError` at import time.
- `CsvTracker` is stdlib-only, so it is the always-available default and the backend every test can
  exercise without mocking.
- Zero-arg construction is what lets Trackers be `@configurable` and wire from YAML like every other
  workspace primitive (the workspace lazy-init rule).
- A framework hookup (a `pl.Callback`, a `TrackerLogger` bridge, a Keras callback) is written ONCE
  against `Tracker`, so all four backends work through any of the three hookup styles for free.

**Example.** Adding a backend is a new file that stays import-safe:

```python
# matrainer/tracking/newthing_backend.py
from matrainer._imports import require
from matrainer.tracking.base import Tracker

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
  the manifest references it by path plus digest. This is the same rule `matrainer.artifacts` already
  follows for the label vocabulary — persist it, log it, but never re-hand-roll someone else's
  format. The installed-package freeze (`environment/requirements.txt`) is the second application of
  it: that is pip's format, so it too is a referenced artifact rather than an inlined list.
- **A referenced artifact travels with the manifest** (added 2026-08-02). Because those files hold
  formats the manifest deliberately does not own, it carries only a path and a digest for each — so
  a backend that receives the manifest and *not* the files is holding references to data that
  exists on exactly one machine, which is rule 3 broken by omission rather than by design.
  `RunManifest.artifacts()` lists them and the default `log_manifest` uploads them, skipping any
  already inside the backend's own `log_dir`. This was a live gap: `config/final.yaml` had been
  written since the manifest landed and never uploaded, invisible on a file-backed backend (where
  it happens to sit in the right tree) and missing on every remote-only one.
- **Machine-specific roots are contracted on the way out** (added 2026-08-02). A manifest full of
  absolute paths answers "what produced this run?" only on the machine that wrote it — the code,
  the data and the models are all somewhere else on the next one. So where `PROJECT_ROOT` /
  `DATA_ROOT` / `MODEL_ROOT` are set, paths under them are recorded as `$NAME/...`.

  Three choices make that safe rather than lossy. It happens **once, over the whole document**
  (`to_dict` *and* `section`, so what is written and what a native projection reads agree) rather
  than at each capture site — otherwise a section added later is silently non-portable, and the
  two readers disagree. **`os.path.expandvars` is the exact inverse**, so the contraction is a
  reversible encoding rather than a lossy rewrite; the values used are recorded verbatim under
  `env.roots`, which is the one subtree left alone, because contracting a definition into a
  reference to itself would destroy the only record of what the root was. And the match is on a
  path **boundary** with the most specific root winning, so `/store` cannot swallow the start of
  `/storefront` and a `MODEL_ROOT` nested inside `DATA_ROOT` still resolves to the nearer of the
  two.

  The consequence worth stating: contraction reaches the pasteable `command.line`, which is a
  *shell* string, so quoting had to change with it. A contracted word is double-quoted rather than
  single-quoted, because `'$PROJECT_ROOT'` is a literal dollar sign — the line would have looked
  correct and pointed at a directory that does not exist.
- **The command line is recorded twice, deliberately** (added 2026-08-02). `command.argv` is the
  raw list — what `primary_code()` parses and what a tool consumes — and `command.line` is a shell
  snippet a person can paste to run it again, `cd` included so a config referenced by a relative
  path resolves. One cannot be derived from the other after the fact: the workspace's runner is
  launched with `-m`, whose `argv[0]` is the module's *file* path and does not re-execute. Where
  there is no honest answer — an interactive session, or `python -c`, whose code never reaches
  `sys.argv` — `line` is omitted rather than guessed.
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
manifest.record_output("epoch=3.ckpt", monitor="val/acc", value=0.91)
# ... every call re-emitted manifest.yaml AND manifest.json atomically ...
capture_log_file(trackers, manifest=manifest, status="success")   # Ctrl-C -> "interrupted"
```

The datasets and models are *not* in that snippet, and their absence is the point (2026-08-02).
Asking each consumer to call `record_dataset` / `record_model` is one chance to forget per
consumer, and the ten runnables in this workspace all reach `log_run_context` already — so it
walks the configured object graph itself and records every source that can name the data it reads
and every wrapper that can name the weights it loads. The same argument put `Tracker.finalize`
inside `capture_log_file`, and for the same reason: a fact every consumer owes the record should
be gathered where every consumer already passes.

The walk both use is deliberately *not* the one hyperparameter derivation uses. That one flattens
the graph into its leaf scalars, because a hyperparameter is a value; this one needs the objects
themselves, because what identifies a dataset or a model is a source or a wrapper. And both
classify from the URI — the scheme is the kind, the path is the name — so this package learns
nothing about any particular dataset registry or model zoo, and a new source or wrapper type needs
no change here.

**The two identity protocols have deliberately different shapes, and the asymmetry is not drift.**
A dataset source exposes two plain strings (`dataset_uri` / `dataset_url`) because it reads *one*
dataset; a concatenation is the exception and has its own plural helper. A model wrapper exposes a
*list* (`model_refs`, in `matrainer.uri`) because it routinely **composes** artifacts into one
network: a pretrained backbone with a fine-tuned checkpoint loaded over it used two, and a run
record that named only one of them would be wrong about how the weights were obtained. Each
reference therefore carries the `role` it played — `pretrained` for weights from outside the run,
`checkpoint` for weights a previous run wrote — and a pipeline composing several models is the
same shape one level up. The shapes differ because the things differ.

Two consequences of that protocol are load-bearing and easy to undo:

- **Reporting an empty list is an answer, not a non-answer.** A model built from an architecture
  with random weights loaded nothing external; that is recorded by recording nothing, because the
  architecture is configuration and is already in `params`. The walk therefore stops at any object
  that *implements* the protocol, empty result included — falling through would follow the wrapper
  chain into `.model`, which on real wrappers is a property that builds the network.
- **Asking never builds.** These properties read stored configuration only, so a run's provenance
  can be recorded from a wrapper that has never been solidified. That is why the walk materializes
  a *deferred marker* rather than calling `flow()` on whatever it is handed: `flow()` on a live
  object runs the post-construction `solidify()` hook, which for a model wrapper downloads its
  weights — and raises outright when the runnable has not injected `num_classes` yet.

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

One native projection was evaluated and **deliberately not built** (2026-08-03), and it is worth
recording so "new native projections" does not read as an open invitation to it: a `clearml.Dataset`
for the manifest's datasets. ClearML addresses only `s3://` / `gs://` / `azure://` / `file://` /
`http(s)://`, and it does not *reject* anything else — it resolves an unknown scheme as a relative
path against the working directory, so a Hub reference would be registered as a local file that
does not exist. The projection would therefore be an empty versioned entity holding wrong paths,
strictly less informative than the comparable hyperparameter section it would sit beside. That is
also why the model half gates on an addressable scheme rather than passing every URI through.
See matrainer's `AGENTS.md` for the full reasoning.

---

## 4. Backends are separately installable: tolerate, declare, select (2026-08-01)

> **Status: all three layers ship.** Tolerance was already there; discovery + `[tool.aisland]
> trackers` + the workspace `aisland tracker` command landed 2026-08-01. `AimTracker` and
> `ClearMLTracker` were the first backends added under the new rules — each one new file plus one
> extra, with no edit to any existing file, which is the claim this record makes.

**Context.** Backends are a *menu*, not a set. A machine that tracks to a local server has no reason
to download the client of a hosted service it has no account for, and a continuous-integration runner
may want every backend installed so every one stays tested. Record 2 solved half of this: lazy
in-method imports mean the code *tolerates* a missing backend, and a light `import matrainer.tracking`
pulls in none of them. But nothing decides what a given machine *installs*, so in practice the
answer is "the `[all]` extra, or nothing" — the same gap the framework extras had before the
workspace gained a framework selection.

A second, subtler limit: `matrainer/tracking/__init__.py` imports each backend module by name and
lists each class in `__all__`. That is fine for four, but it means every new backend edits a shared
file, and a backend shipped by a *different distribution* can never appear there at all.

**Decision.** Three layers, deliberately separate.

1. **Tolerate** (ships today, unchanged) — every backend imports its library lazily inside its
   methods through `require(module, extra)`, so a missing library is one actionable
   `pip install matrainer[<extra>]` error and never an import-time failure.
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
distribution is discoverable with no change to this package; and `matrainer.tracking` exposes its own
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
# matrainer/tracking/aim_backend.py
from matrainer._imports import require
from matrainer.tracking.base import Tracker

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
`MetricPreset` lived in `matrainer/lightning/metrics.py` from the day they were
written, because a Lightning runnable was the only thing that called them. None of
them imports `pytorch_lightning` — they use confluid, torch and torchmetrics and
nothing else.

That stayed harmless until a consumer arrived that trains on plain torch with no
`Trainer` at all. `from matrainer.lightning.metrics import build_metric_collection`
imports the parent package first, and `matrainer/lightning/__init__.py` opens with
`require("pytorch_lightning", "lightning")` — so a deliberately Lightning-free
backend could not reach a Lightning-free helper. Measured: after
`import matrainer.lightning.metrics`, `pytorch_lightning` is in `sys.modules`.

**Decision.** The module moved to `matrainer/torch/metrics.py`, and `torchmetrics`
moved from the `[lightning]` extra to `[torch]` with it. `matrainer.lightning`
re-exports all four names, so its public surface is unchanged.

This is the same correction as `matrainer.tensorflow` → `matrainer.keras` (record 2's
rule, "an optional helper's extra names the API it imports, never a distribution
that happens to bundle it") applied one level up: a subpackage's name is a claim
about what importing it costs.

**Consequences.**

- `pip install matrainer[torch]` now gives a consumer the metric helpers. It also
  gives them `torchmetrics`, which is a real addition to that extra — accepted,
  because a torch consumer computing no metrics is not the case worth optimising.
- Every existing import keeps working. `raidar/detector.py` does
  `from matrainer.lightning import scalar_metrics` and was not touched.
- `matrainer.torch` grows for the first time since its 2026-07-29 emptying. The rule
  that emptied it still holds and this does not bend it: those helpers left because
  they were about the DATA (recordstream's business); these arrive because they are
  about torchmetrics, which is torch's.

**Example.**

```python
from matrainer.torch import build_metric_collection, scalar_metrics   # no Lightning
from matrainer.lightning import scalar_metrics                        # same object, still works
```

**What you may change.** Move a helper whenever its home over-states its
dependencies — check with `python -c "import <module>; import sys; print('pytorch_lightning' in sys.modules)"`
rather than by reading the imports, since a package `__init__` can require what the
module itself does not. Keep the re-export when you do: a consumer's import path is
not yours to break for tidiness.

## 6. Metric names are a PAIR of vocabularies, not a canonicaliser (2026-08-04)

**Context.** A `Tracker` records the key it is handed, verbatim — every backend does
(`mlflow_backend.log_metrics` calls `client.log_metric(str(key), …)`, the CSV backend makes it a
column, ClearML's `rpartition("/")` splits it into a graph title but renames nothing). So the
names in a tracker are whatever the training engine called them, and eight engines call the same
three quantities eight ways. Measured from one sweep that trains ONE config on every backend, the
header line of each run's `metrics.csv`:

```
lightning     train/loss_epoch  train/loss_step  train/accuracy_top1  val/loss  val/accuracy_top1
torch         train/loss        train/accuracy_top1                   val/loss  val/accuracy_top1
fastai        train_loss        valid_loss       accuracy             top_k_accuracy
keras         loss              accuracy         val_loss             val_accuracy
tensorflow    loss              sparse_categorical_accuracy  val_loss  val_sparse_categorical_accuracy
huggingface   train_loss        eval_loss        eval_accuracy        train_runtime  …
jax / mlx     loss              val_loss         val_accuracy
```

Nothing here is a bug in any backend — each is spelling its own engine's convention — but the
comparison the sweep exists to make is impossible: one quantity occupies four columns and every
row is mostly blank.

**Decision.** `matrainer/metric_names.py` holds a `MetricVocabulary` per framework (its
separator, how it writes each split, what an unprefixed key means, the base names it has its own
word for) and translates by PARSING with the source's vocabulary and RENDERING with the target's.
`matrainer`'s own vocabulary — `<split>/<name>`, `split ∈ {train, val, test}` — is one entry in
that table, not a privileged destination.

A one-way canonicaliser was the obvious smaller design and was rejected for two reasons, both
concrete. **It cannot parse.** "Which split is a bare `accuracy` from?" has no universal answer:
Keras' unprefixed keys are training metrics, fastai's are VALIDATION metrics (its `Recorder`
computes them on the validation pass and prefixes none of them), so a canonicaliser without a
per-engine table silently files fastai's val accuracy in the train column — a wrong number that
looks right. **And a consumer legitimately wants the other direction**: a team whose dashboards,
alerts and saved queries are written against `val_loss` can set `target: keras` and get every
backend, including the torch ones, recording that spelling. Both directions are the same table
read the other way round, so the pair costs one function more than the one-way version.

The split of duties across the arguments is what makes ONE config key work for eight backends:
`source` is passed per CALL by the code reporting (only it knows which engine produced the names
and which split it is reporting), while `target` and `renames` live on the `@configurable`
object, which a top-level `metric_names:` key broadcasts into every runnable at once.

**Consequences.**

- Parsing recognises the canonical `<split>/<name>` form whatever the source vocabulary, so
  translation is idempotent and a loop that already names things canonically may declare its real
  `source` and pass through untouched. Without that, a helper applied at two seams would produce
  `train/train/loss`.
- Rendering does NOT invert the base-name aliases: translating into Keras gives `val_accuracy`,
  not `val_sparse_categorical_accuracy`. Those four spellings are what `compile()` expanded
  `"accuracy"` into — compile-time artefacts, not names anyone writes — so a vocabulary decides
  how the SPLIT is spelled and the base name stays canonical.
- `epoch` and `step` are axis keys and pass through: prefixing `epoch` with `train/` would invent
  a second series for the x-value of every other one.
- An unknown framework raises rather than falling back to the canonical vocabulary — a fallback
  renames nothing, which is indistinguishable from a config typo.
- The renaming happens on the way to a TRACKER, never to the dict the engine hands its own
  callbacks: `keras_tracking_callback` translates its copy while `logs` stays untouched, because
  a wired `EarlyStopping(monitor="val_loss")` must keep finding the name Keras gave it.
- Like `metric_slots`, this sits at the TOP LEVEL and imports nothing but the standard library
  plus confluid, so the framework-free backends can use it. UNLIKE `metric_slots` it IS
  re-exported from the package root: it is on the path metrics take to a tracker, and a YAML has
  to be able to name it.

**Example.**

```python
from matrainer import MetricNames

names = MetricNames()                                    # canonical out, any engine in
names({"loss": 0.5, "val_loss": 0.4}, source="keras")    # {'train/loss': 0.5, 'val/loss': 0.4}
names({"accuracy": 0.9}, source="fastai")                # {'val/accuracy': 0.9} — NOT train
MetricNames(target="keras")({"val/loss": 0.4})           # {'val_loss': 0.4}
```

**What you may change.** Add a vocabulary when a new engine arrives (one `MetricVocabulary` entry
plus its name in the `Framework` Literal — the runtime tuple is `get_args`, so nothing else
changes). Add an alias when an engine auto-expands a base name the way `compile()` does. Do NOT
invert the aliases for rendering, do not drop the canonical-form check in `parse_metric_name`
(idempotence depends on it), and do not move the translation into `Tracker.log_metrics`: the
tracker cannot know which engine or which split produced a bare key, which is the whole reason
`source` is a per-call argument.

## 7. A metric an owned loop cannot average gets a confusion-matrix accumulator (2026-08-04)

**Context.** A backend whose engine owns the training loop (Keras, Lightning, a
`transformers.Trainer`) gets stateful metrics for free: the engine calls `update()` per batch and
`compute()` once at the end. A backend whose loop *we* own — bare torch, TensorFlow, JAX, MLX —
sees one batch at a time, and the natural shape for a metric is `fn(logits, targets) -> float`
scored per batch and averaged over the epoch.

That shape is very nearly right for accuracy and quietly wrong for anything computed per class.
Measured on 10 classes at batch 32:

| | global macro-F1 | batch-averaged |
| --- | --- | --- |
| accuracy ≈ 0.9 | 0.9182 | 0.8849 |
| accuracy ≈ 0.3 | 0.3486 | 0.3032 |

Three to five points, always downward, and worse the more classes a batch is missing — a class
absent from a batch scores 0 for that batch and drags the mean. In a cross-framework comparison
that is not noise: it is a systematic penalty on whichever engines happen to own their own loop,
which is the one thing such a comparison must not introduce.

**Decision.** `matrainer/metric_functions.py` holds `zero_confusion` / `accumulate_confusion` /
`macro_f1`. A loop keeps ONE `C×C` integer accumulator per split, adds each batch to it, and
reduces once at the end — the same thing `torchmetrics` does internally, which is what makes the
two engines' numbers comparable. The accumulator's size is set by the class count, not the
dataset, so this costs nothing to run per batch.

Consumers keep the per-batch protocol for the metrics it suits and PARTITION by name: a name in
the confusion set is accumulated, everything else is scored per batch and averaged as before.

**Consequences.**

- It sits at the TOP LEVEL, not under `matrainer.torch`, for the `metric_slots` reason: that
  subpackage requires torch and the callers here are deliberately framework-free. numpy is the
  only import, and every engine that needs this already produces arrays numpy can read.
- Not re-exported from the package root, again like `metric_slots`: the root is the *tracking*
  surface and nothing here touches a tracker.
- `argmax` over the last axis, so raw logits and probabilities give the same answer and no
  backend has to declare which it produces.
- An out-of-range target id is an ERROR, not a silent wrap: the flat `truth * C + prediction`
  index would fold a stray id into another class's counts, the totals would still add up, and the
  metric would still look plausible. The usual cause is an accumulator sized from a stale
  `num_classes`.
- SINGLE-LABEL only. A multi-hot row has no single cell to land in, so a consumer must DROP a
  confusion metric on a multi-label run rather than approximate one — a number under a name that
  means something else is worse than a gap, because nothing downstream can tell.
- Counting is `np.bincount` on the flattened index rather than `np.add.at`, which is the obvious
  spelling and roughly an order of magnitude slower — measurable when it runs once per batch.

**Example.**

```python
from matrainer.metric_functions import accumulate_confusion, macro_f1, zero_confusion

matrix = zero_confusion(num_classes)
for logits, targets in batches:
    accumulate_confusion(matrix, logits, targets)   # in place, returns the matrix
macro_f1(matrix)                                    # == sklearn == torchmetrics, exactly
```

**The per-pixel additions (2026-08-05) are that invitation taken up.** A second task needed the
same accumulator for `mean_iou` / `overall_accuracy` / `iou_per_class`, and each is a pure
reduction over the matrix already being kept — no second structure, no second accumulate path.
Two things they added that the classification set did not need:

- **`accumulate_confusion(..., ignore_index=)`**, for the void class a per-pixel task carries. The
  drop happens BEFORE the range check, because both sentinels in the wild (`-100`, `255`) are
  deliberately outside `[0, C)` and checking first would reject every batch that has one.
- **A `nan` for a class present nowhere**, rather than the `0` `macro_f1` contributes. The two are
  not inconsistent: F1's zero matches what sklearn and torchmetrics report for a class with no
  support, and IoU's `nan` matches what `MulticlassJaccardIndex(average=None)` reports for the
  same case. Each follows the library its number has to agree with.

The bias measurement is worse here than for macro-F1 and differently shaped, which is worth
knowing before anyone "simplifies" a loop back to averaging: batch-averaged mean IoU is not merely
low, it MOVES WITH THE BATCH SIZE — 10 classes at accuracy 0.7 against an exact 0.5922 reads
0.6164 at batch 4 and 0.5798 at batch 32. So the property the tests pin is BATCH-SIZE
INDEPENDENCE rather than a direction, and that is the check to reach for when reviewing any
hand-written loop's metric.

**What you may change.** Add a metric that derives from a confusion matrix (per-class recall,
Cohen's kappa, a weighted F1) — the accumulator is already there and a new reduction is a pure
function over it. Do NOT widen the accumulator to multi-label by inventing a per-label matrix
here: that is a different structure (C independent 2×2 tables), and if it is needed it is a new
function with its own name. Do not make `macro_f1` configurable into micro/weighted by adding a
parameter that changes what the returned number MEANS under one name — add the reduction beside
it instead.


## 8. A sweep's tracker-side halves are matrainer's; its matrix is not (2026-08-06)

### Context

A *sweep* runs one config under every backend a task project has, into one tracker, so the runs
become comparable. The first task project to build one wrote a ~290-line `examples/sweep/
sweep_report.py` beside its driver: naming the experiment, reading MLflow and CSV back, folding the
bookkeeping columns, rendering the table. The second copied it. Both were told to, deliberately —
one duplication is cheaper than a premature abstraction, and the workspace backlog recorded the
trigger to fix it: **the third consumer.**

When the third arrived, the copies were compared by AST with docstrings stripped: **byte-identical.**
What differed was prose — the line wrapping and which metric names the comments happened to cite.

### Decision

Lift the two tracker-side halves to `matrainer.sweep` and leave everything else where it is.

The cut is by WHAT THE CODE QUERIES, and both sides of it are sharp:

| stays in the task project | moved to matrainer |
|---|---|
| which cells the matrix has | naming the experiment `<prefix>_<N>` |
| which engine a row pins, and why | reading MLflow / CSV back |
| the config overlay and its metric alignment | which columns fold away |
| probing what is installed, isolating failures | rendering the table |

### Consequences

* **It belongs here by the same argument that moved `metric_slots` and `metric_names`.** Both halves
  are queries against a TRACKER, which is the thing this package abstracts, and the vocabulary the
  report folds on (`epoch` / `runtime` / `total_flos`, Lightning's `_step` suffix) is
  `matrainer.metric_names`' own.
* **`python -m matrainer.sweep`, not a console script.** This package deliberately ships no binary —
  the workspace retired its CLI — and a module main is neither a binary nor an entry-point group.
* **The two constants became PARAMETERS.** A `./sweep_logs` path and a `http://127.0.0.1:5005`
  MLflow URI are fine as an example script's module constants and wrong in a library, so they are
  defaulted arguments (`save_dir` / `tracking_uri`) with the old values as the defaults.
* **`QUERYABLE` became load-bearing.** It was a documentation-only tuple in both copies while
  `collect` branched on string literals; it is now what `collect` reads, so the advertised set and
  the implemented one cannot disagree.
* The consuming drivers each dropped ~290 lines and one test file; what they kept — `all_backends.yaml`
  and `run_sweep.sh` — is exactly the half that differs between tasks, which is why the split holds.

### Example

```python
from matrainer.sweep import collect, next_experiment_name, render

name = next_experiment_name("mlflow", "Thu_Aug_6_2026")   # -> "Thu_Aug_6_2026_3"
rows, note = collect("mlflow", name)
print(render(rows, name, "mlflow", note))
```

### What you may change

Which trackers can be read (`aim` and `tensorboard` are genuinely queryable and are not implemented
yet), and what counts as bookkeeping. What you may not do is teach it about a TASK — a metric this
module knows by name is a metric a task project cannot rename.

**Amendment (2026-08-07, user decision).** The DRIVER joined the tracker-side halves: three
near-identical per-project bash drivers carried the same probe / subprocess / isolate / report
loop, so `matrainer.sweep` became a package — `report.py` (this record's original scope,
verbatim), `spec.py` (the declarative matrix a project ships as `examples/sweep/sweep.yaml`)
and `runner.py` (the machinery). The boundary this record draws is UNCHANGED and now enforced
structurally: everything task-specific — which backends, what each requires, which row pins an
engine, how a tracker is started — is DATA in the project's spec, never code here. The command
surface is the `matrainer` console script's `sweep` group (a scoped, user-authorized exception
to the no-binary rule — see the AGENTS mandate) and the unchanged module main; the default
per-cell command is the generic `matrainer train`.

```yaml
# a project's examples/sweep/sweep.yaml — the matrix as data
config: all_backends.yaml
frameworks:
  - {name: lightning, requires: [pytorch_lightning], models: [zoo, dncnn]}
  - {name: keras, requires: [keras, tensorflow], env: {KERAS_BACKEND: tensorflow}, models: [unet]}
models:
  - name: zoo                       # first = the config's default, no flag
  - {name: dncnn, flags: [--model, dncnn]}
trackers: [mlflow, csv]
tracker_setup: {mlflow: [aisland, mlflow, start]}
```

---

## 9. The shared trainer machinery moved up: one API base, one concrete base per framework (2026-08-06)

**Context.** Record 1 pushed all training code into the consuming projects, and the projects then
did the anticipated thing: four tasks each grew a merged train/evaluate/test/predict runnable per
engine — up to ten engines per project — deliberately deferring a shared base "until the fleet
exists to design it against". By 2026-08-06 the fleet existed, and the duplication was measured
rather than felt: two projects' Lightning modules 61% byte-identical, the tracker plumbing
(`_live_trackers`) and the two `@output` properties existing in **28 copies** each, the trainer
construction byte-identical between two projects, ~1,060 near-identical CLI lines. Every mandate
protecting the fleet's behavioural agreement (one name per quantity, per-split metric slots,
reset-at-epoch-start) had to be enforced by per-project AST drift tests because no code was shared.

**Decision (user-authorized reversal of record 1's scope).** Matrainer owns the shared trainer
machinery, in a strict three-layer shape:

1. **`matrainer.runnable.TrainerRunnable`** — ONE abstract, framework-free base defining the
   NEUTRAL half of the API: the neutral constructor slots (stored verbatim), `run()` →
   `run_entrypoint`, the four `@entrypoint`-marked abstract capabilities, tracker/prepare/
   prediction plumbing, and the neutral hooks (`initialize`, `_prediction_output`,
   `_reportable`, `_report_nonscalars`). It stores NO model/loss/metrics slot — deliberately.
2. **A FAMILY base per engine family pins the framework types** —
   `matrainer.torch.runnable.TorchTrainerRunnable` serves every engine that trains a
   `torch.nn.Module` against torchmetrics (lightning, the bare loop, fastai, huggingface):
   `model: Lazy[nn.Module]`, `loss_fn: Optional[Lazy[nn.Module]]`, the metric quartet as
   `MetricsSpec`, `_targets -> Tensor`, the typed default hooks, the metric builders, the
   sidecar write and the `trained_model -> nn.Module` output. A keras family base mirrors it
   with keras types. This layer exists so that nothing user-facing is typed `Any`: a type the
   neutral base cannot name lives exactly one layer down, where it can be.
3. **One machinery base per framework** fills the loop in, each self-contained in its
   extras-gated subpackage. **The abstraction stops at the API — there is no abstract trainer
   OBJECT**: the `pl.Trainer` slot lives only in the lightning base, the epoch loop only in
   the torch one, the `Learner` only in fastai's. Machinery bases share module *functions*
   (`flow_trackers`, `write_collection_sidecar`, the `report_*` fan-outs), never code between
   each other.
4. **A consuming project's task base** implements the hooks once for its whole engine fleet;
   the concrete class combines task base + framework base and keeps its own FULL flat
   `__init__` (the confluid introspection surface).

**Consequences.**
- Record 1's "removing matrainer keeps a consumer training" invariant is dead by design; what
  replaces it is *"a task's behaviour is written once per hook, not once per engine"* — the
  behavioural-agreement mandates are now enforced by shared code plus matrainer's own base tests,
  and a project's drift tests shrink to "inherits the base and does not shadow it".
- The light core is UNCHANGED: `import matrainer` still imports no framework; every family base
  lives behind its extra.
- The four projects now break together when a base changes — which is the point (one fix, four
  consumers), and why base changes ship with the consumers updated in the same workspace change.

**Example.** A consuming project's concrete runnable, whole:

```python
@configurable(task="classification", role="trainer", framework="torch")
class LightningClassifier(ClassificationTorchHooks, ClassificationBase, LightningRunnable):
    def __init__(self, model, train_set=None, ..., num_classes=None, label_map=None):
        super().__init__(model=model, train_set=train_set, ...)   # shared slots, stored verbatim
        self.num_classes = num_classes                            # task-specific state
        self.label_map = label_map
```

**What you may change.** New hooks (added with a safe default so every family base keeps
working), new family bases, and base behaviour — shipped together with the consumer updates.
What you may *not* do: introduce an abstract trainer object above the family bases, share code
*between* family bases except via module functions, grow task semantics (what a loss/metric/label
IS for a task) into any base, or reintroduce a matrainer CLI binary.

**Amendment (2026-08-07).** The second machinery base landed: `matrainer.torch.loop.TorchLoopRunnable`,
the bare-torch epoch written in the open. Its one design addition to the shape above is the
BATCH-hook seam — `_train_batch_loss` / `_eval_batch` / `_predict_batch` carry the
read-forward-loss default (input → forward → loss → metric update), so a task whose model
computes its loss inside `forward` overrides exactly three methods instead of the loop, and
`_eval_batch` returning `None` declares a loss-less eval split (the base then omits that
split's `<split>/loss` key rather than recording a fake zero). The batch shape is a machinery
ctor param (`collate_fn`), because it is a property of the MODEL's contract, not a config key.

```python
class TorchDetectorLike(DetectionHooks, TorchLoopRunnable):        # loss-inside-forward
    def _train_batch_loss(self, batch):                            # the ONLY loop override
        images, targets = self._to_device(batch)
        return sum(self.model(images, targets).values())           # the summed loss dict
    def _eval_batch(self, batch, collection):
        images, targets = self._to_device(batch)
        collection.update(self.model(images), targets)
        return None                                                # eval mode has no loss
```

**Amendment (2026-08-07, second).** The third machinery base landed:
`matrainer.fastai.learner.FastaiRunnable`, the `Learner` member. Its design additions: the
metric slots stay the NAME lists a fastai config wires (`_build_fit/_eval_metrics` are
deliberate no-ops — fastai metrics resolve at Learner assembly, alias-first through the
`_metric_aliases()` hook, union-on-fit / narrow-on-eval through `_learner_metrics`), the
fastai-shaped sentinels fall through to the task hooks (`_default_fastai_loss` →
`_default_loss`), and the Learner assembly is a set of overridable seams
(`_learner_model` / `_learner_loss` / `_learner_metric_objects` / `_learner_callbacks` /
`_place`) — which is what lets a loss-inside-forward task keep its no-op-loss bridge as five
small hook overrides instead of its own fit loop.

```python
class FastaiDetectorLike(DetectionHooks, FastaiRunnable):          # loss-inside-forward
    def _learner_model(self):
        return NativeAdapter(super()._learner_model())             # train fwd -> summed loss dict
    def _learner_loss(self):
        return passthrough_loss                                    # the model already computed it
    def _learner_metric_objects(self, split):
        return []                                                  # scored via a callback instead
```

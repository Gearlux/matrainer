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

**Decision.** Matrainer's core is an **experiment-tracking library**: no training loop, no
trainer/evaluator class and no CLI live in the light core. A class that trains reads like a standard
example for its framework and reaches for Matrainer solely to track the run. The runnable protocol
(`TorchRunner`, `ProgressReporting`, the `entrypoint` method-annotation mechanism), the workflow
combinators, and `DatasetProcessor` moved to `recordstream`; the generic runner is
`recordstream run <config>` binding a top-level `runnable:` key. Record 9 later added the shared
trainer machinery ABOVE this core, in extras-gated framework subpackages, and record 10 the factory
behind the `matrainer` console script; the core's contract below is unchanged.

**Consequences.**
- The tracking half stays usable on its own: a plain training class can use any tracker through a
  framework hookup (the example below) and keep training with the callback deleted. If a change makes the *tracking*
  modules load-bearing for training, it's in the wrong layer.
- `import matrainer` imports no torch/lightning/mlflow (see record 2). The light core is
  `confluid`/`liquifai`/`loggair`/`coolname`/`numpy`.
- A `training_step`, a `run()` loop or command-line wiring is rejected in review when it lands in the
  tracking core (`matrainer.tracking` and the framework hookups); the machinery has its homes in the
  framework subpackages (record 9) and `matrainer.cli` / `matrainer.console` (record 10).

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
styles are all in scope. What you may *not* add to the tracking core: a training/evaluation loop, a
step function, a `run()` dispatcher, or a `DatasetProcessor`/workflow combinator (those live in
`recordstream`); trainer machinery belongs in a framework subpackage's base (record 9) and command-line
wiring in the CLI factory (record 10).

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
  `!class:` / `!ref:` tags that only its own resolver can load; embedding that inside the
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
The live-server measurement is recorded beside `ADDRESSABLE_SCHEMES` in
`tracking/clearml_backend.py`.

---

### Amendment (2026-10-16): a remote-only run keeps its manifest beside its checkpoints

**Context.** The manifest was written to the first tracker's `log_dir`, else `runs/<experiment>/<run>` under the
launch directory. A run tracked on a server alone (a remote-only backend has no `log_dir`) therefore landed in
two folders: the trainer's checkpoints and metrics in its `output_dir`, the manifest, config and package freeze in a
second tree named after the experiment — and nothing tied the two together (measured on a real RF-DETR run: the
checkpoint folder held an empty `hparams.yaml`, and the model's size and machine were only in the other tree).

**Decision.** `log_run_context` builds the manifest in the first tracker's `log_dir`; when that tracker keeps none,
in the runnable's `output_dir`; only without either in `runs/<experiment>/<run>`. `capture_log_file` copies the log
file into `<manifest folder>/logs/` as well as uploading it. Rejected: a remote-only tracker reporting its
`save_dir` as `log_dir` — `log_dir` means "this backend's own local tree", and `log_manifest` skips uploading the
files inside it, so the manifest would never reach the server; a per-trainer manifest built by each consumer — the
rule would be written once per trainer.

**Consequences.** A trainer with an `output_dir` and only remote tracking writes ONE folder per run. A run with a
local tracker is unchanged. The copied log is not a manifest artifact, so it is not uploaded twice.

**Example.**

```python
manifest = log_run_context([server_tracker], runnable=trainer)   # trainer.output_dir = "runs/project/nano"
manifest.path   # runs/project/nano/manifest.yaml, beside checkpoint_best_total.pth
```

**What you may change.** Which runnable attribute names the output folder (today `output_dir`); the copy's
folder name. Not: the manifest skipping a local tracker's folder, or a second folder for a run that has one.

## 4. Backends are separately installable: tolerate, declare, select (2026-08-01)

> **Status: all three layers ship.** Tolerance was already there; discovery + `[tool.aisland]
> trackers` + the workspace `aisland tracker` command landed 2026-08-01. `AimTracker` and
> `ClearMLTracker` were the first backends added under the new rules — each one new file plus one
> extra, with no edit to any existing file, which is the claim this record makes.
>
> **Amendment (2026-10-23):** the declaration is now `[tool.aisland.selection.tracker]` (and `.framework`), which
> this project DEFINES with its `question`; the workspace command is `aisland workspace configure tracker`. The
> `[tool.aisland] trackers = [...]` spelling below is the 2026-08-01 one.

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
- Every existing import keeps working. A consumer doing
  `from matrainer.lightning import scalar_metrics` needed no change.
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
* **`python -m matrainer.sweep` is a module main.** It is neither a binary nor an entry-point group of
  its own; the `matrainer` console script's `sweep` group (see the amendment below) dispatches to the
  same code.
* **The two constants became PARAMETERS.** A `./sweep_logs` path and a `http://127.0.0.1:5005`
  MLflow URI are fine as an example script's module constants and wrong in a library, so they are
  defaulted arguments (`save_dir` / `tracking_uri`) with the old values as the defaults.
* **`QUERYABLE` became load-bearing.** It was a documentation-only tuple in both copies while
  `collect` branched on string literals; it is now what `collect` reads, so the advertised set and
  the implemented one cannot disagree.
* The consuming drivers each dropped ~290 lines and one test file; what they kept — `all_backends.yaml`
  and the matrix spec (`examples/sweep/sweep.yaml`, below) — is exactly the half that differs between
  tasks, which is why the split holds.

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
machinery, in a strict three-layer shape (layers 1-3 below), under a fourth layer that a consuming
project owns:

1. **`matrainer.runnable.TrainerRunnable`** — ONE abstract, framework-free base defining the
   NEUTRAL half of the API: the neutral constructor slots (stored verbatim), `run()` →
   `run_entrypoint`, the four `@entrypoint`-marked abstract capabilities, tracker/prepare/
   prediction plumbing, and the neutral hooks (`initialize`, `_prediction_output`,
   `_reportable`, `_report_nonscalars`). It stores NO model/loss/metrics slot — deliberately.
2. **A FAMILY base per engine family pins the framework types** —
   `matrainer.torch.runnable.TorchTrainerRunnable` serves every engine that trains a
   `torch.nn.Module` against torchmetrics (lightning, the bare loop, fastai, huggingface):
   `model: Partial[nn.Module]`, `loss_fn: Optional[Partial[nn.Module]]`, the metric quartet as
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
    def __init__(self, model: Optional[Mandatory[Partial[nn.Module]]] = None, train_set=None, ...,
                 num_classes=None, label_map=None):
        super().__init__(model=model, train_set=train_set, ...)   # shared slots, stored verbatim
        self.num_classes = num_classes                            # task-specific state
        self.label_map = label_map
```

The `model` slot is `Mandatory` but defaulted: confluid still reports it as required to form-specs
and MCP schemas, while the `None` default keeps the class zero-arg constructible. An empty slot
fails at the start of `run()` with a clear error (`TrainerRunnable._require_model`).

**What you may change.** New hooks (added with a safe default so every family base keeps
working), new family bases, and base behaviour — shipped together with the consumer updates.
What you may *not* do: introduce an abstract trainer object above the family bases, share code
*between* family bases except via module functions, or grow task semantics (what a loss/metric/label
IS for a task) into any base.

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

**Amendment (2026-08-07, third — P5).** The KERAS FAMILY landed, and it is the first family
with its own FAMILY base beside the machinery: `matrainer.keras.runnable.KerasTrainerRunnable`
pins the keras types (the `Partial[keras.Model]` slot, the compile-slot aliases, the `_to_arrays`
batch hook, the sentinel hooks, the UNION-as-family-property) under TWO machinery SIBLINGS —
`matrainer.keras.trainer.KerasRunnable` (the engine owns the loop: compile/fit, the History
read-back, the `_before_fit`/`_fit_scores` seams) and `matrainer.tensorflow.loop.
TensorFlowRunnable` (we own the loop: tf.data, traced steps, per-split CLONED metric objects,
the hand-driven CallbackList, the SavedModel export, the engine guard). The design additions
the consolidation forced: the compile seams (`_compiled_loss` may be a positional LIST — a
detection spec's loss is not a slot), `_before_fit` (the tracking callback's `on_train_end`
closes the run, so a tracker-touching artifact write after `fit()` returns arrives at an
ENDED run), and the metric CLONE (a live metric wired into a per-split spec is a stateful
accumulator three splits must not share — one consumer's TF backend shipped exactly that).
Verified behaviour-preserving at the strongest level available: the two seeded spec-engine
cells reproduced their canonical losses BIT-EXACTLY through the new bases
(keras 2.30023455619812, tensorflow 2.3113211567203202 — the keras digit predates the sixth
amendment's `shuffle=False` fix, which moved it).

```python
class KerasSegmenterLike(SegKerasHooks, KerasRunnable):            # compile/fit member
    def _default_keras_loss(self): return SparseCategoricalCrossentropy(from_logits=True)

class TFSegmenterLike(SegKerasHooks, TensorFlowRunnable):          # native-loop sibling
    pass                                                            # same hooks, other machinery
```

**Amendment (2026-08-08, fourth — P6, huggingface).** The `transformers.Trainer` machinery
landed: `matrainer.huggingface.trainer.HuggingFaceRunnable`, on the TORCH family base — the
member that writes least while swapping most, because the engine owns somebody else's whole
training program (epoch, optimizer, scheduler, accumulation, mixed precision, distribution).
What the machinery therefore is, is ASSEMBLY: the deferred `TrainingArguments`/`Trainer`
slots with STRING dotted targets (importing the machinery never imports transformers), the
run-time argument FLOOR (a config-pinned key wins — the workspace `TrainingArguments` rule,
now written once), the `log_history` REPLAY as the tracker bridge (`report_to="none"` on
purpose — Trainer auto-enables every importable tracker library), and the record-loader
predict walk. The design additions the consolidation forced: the `_trainer_factory()` seam
(one task's model refuses `labels=`, so its loss rides a `compute_loss` subclass that must
still yield to a config-replaced Trainer), the `_metrics_hook(split)` seam returning `None`
by default (one task scores OUTSIDE the Trainer through the fleet's torchmetrics and wires
no hook at all), and the no-op metric builders with the family restore (the fastai
arrangement, reused verbatim). Verified behaviour-preserving at the strongest level
available: the seeded detection cell reproduced its canonical numbers BIT-EXACTLY through
the new base (train/loss 40.41484375, val/map_50 0.046988025307655334).

```python
class HFRestorerLike(RestorationBase, HuggingFaceRunnable):        # the loss-slot member
    def _trainer_factory(self):
        loss_fn = flow(self.loss_fn)
        class _T(transformers.Trainer):                            # Swin2SR refuses labels=
            def compute_loss(self, model, inputs, **kw):
                return loss_fn(model(inputs["pixel_values"]), inputs.pop("labels"))
        return _T

class HFDetectorLike(DetectionBase, HuggingFaceRunnable):          # the torchmetrics member
    def _build_fit_metrics(self, include_train=True):              # restore the family builders
        TorchTrainerRunnable._build_fit_metrics(self, include_train=False)
    def _evaluate_split(self, source, split):                      # score OUTSIDE the Trainer
        return self._decoded_scores(source, self.val_metrics if split == "val" else self.test_metrics)
```

**Amendment (2026-08-08, fifth — P6, jax + mlx).** The two OWNED-LOOP SIBLINGS landed and the
migration is COMPLETE — every per-project backend now runs on a shared machinery base.
`matrainer.jax.loop.JaxLoopRunnable` and `matrainer.mlx.loop.MlxLoopRunnable` share one hook
surface (the batch boundary, the sentinel hooks, the metric STATE hooks, the fit hooks) under
each engine's own loop; what the consolidation forced into the DESIGN is the state-hook
triple — `_new_split_state` / `_score_batch` / `_split_scores` — because metric ACCUMULATION
is a task vocabulary (a confusion matrix here, a squared-error total there, nothing at all
for a task that scores after the fit), while the "accumulate per split, reduce ONCE" rule is
the engine-independent part the driver owns. The epoch drivers SPLAT the batch and unpack a
bare-loss step, so detection's 4-arity dense-map steps drive the same loop. Verified
behaviour-preserving at the strongest level available: the seeded detection cells reproduced
their canonical losses BIT-EXACTLY through the new bases (jax 2.1331496040026345, mlx
1.953120619058609), and the per-pixel task's losses matched to the printed digit (jax
0.6379/0.7427 vs the recorded 0.638/0.743; mlx 0.7888/0.7511 + mIoU 0.351 exactly).

```python
class JaxRestorerLike(RestorationBase, JaxLoopRunnable):           # accumulate-and-reduce
    def _new_split_state(self, split): return zero_squared_error()
    def _score_batch(self, state, outputs, arrays, split):
        accumulate_squared_error(state, np.asarray(outputs), np.asarray(arrays[1]))
    def _split_scores(self, state, split, step=None):
        return {"psnr": psnr(state, self.data_range)}

class MlxDetectorLike(DetectionBase, MlxLoopRunnable):             # 4-arity, bare-loss step
    def _make_step(self):
        loss_fn = self._loss_fn()
        def compute(model, x, y_cls, y_box, y_ctr):
            return loss_fn(model(x), y_cls, y_box, y_ctr)          # no aux — driver unpacks
        return nn.value_and_grad(self.model, compute)
```

**Amendment (2026-08-08, sixth — `KerasRunnable` passes `shuffle=False`; batch order must not
read ambient RNG).** The first full-fleet MLflow sweep caught a seeded keras cell landing a
different final loss under a live `MlflowTracker` than under `csv` or no tracker — both arms
individually deterministic, so it read as a seed leak. The bisect (run under both trackers with
the real CLI path instrumented): initial weights, the sequence's row order, and the first
batch's bytes were IDENTICAL at `fit` entry; all four RNG streams (python/numpy/torch/mps) were
identical entering `model.fit`; after `on_train_begin`, python's global `random` alone diverged;
and the tensors actually entering `train_step` 0 differed. Mechanism: Keras's `PyDatasetAdapter`
permutes BATCH order via the global `random.shuffle` when `fit(shuffle=True)` (the default), and
mlflow's telemetry client draws one `random.randint` at run creation
(`mlflow/telemetry/client.py::_get_config`) — inside the tracking callback's `on_train_begin`,
i.e. after the seed, before the first shuffle draw — shifting the permutation. The fix is the
machinery's, not a tracker workaround: `model.fit(..., shuffle=False)`, because the sequence
already OWNS row order (a private seeded `np.random.default_rng`, reshuffled every epoch via
`on_epoch_end`, which Keras drives) — the engine's second, ambient-state shuffle bought nothing
and made training a function of how many draws any in-process library had consumed. Verified:
with the fix, the seeded detection cell is byte-identical across `csv` and `mlflow` (every loss
component and AP scalar) and across reruns. Consequence: keras cells' trajectories CHANGED —
seeded numbers recorded before this amendment are not bit-comparable across it (the detection
fleet's canonical moved from 2.30023455619812 to 2.285114049911499). Pinned by
`tests/test_keras_family.py::test_the_fit_never_lets_keras_shuffle_batch_order`.


**Amendment (2026-10-14, seventh — `_before_run`, and a setting on every torch trainer without a constructor
change).** On a Mac, torch keeps the Apple GPU's cached memory, which is the machine's memory: an RF-DETR
segmentation smoke run held 11 GB of an 18 GB laptop and swapped (8 min 42 s an epoch); with the allocator's LOW
watermark at 0.5 it held 9.1-9.5 GB and the whole run took 2 min 45 s. torch reads that value from
`PYTORCH_MPS_LOW_WATERMARK_RATIO` when the GPU is first used, so it must be set before the model is built. Two
decisions follow. The neutral `TrainerRunnable` gains a `_before_run()` hook, no-op by default, that EVERY `run()`
calls first (the API base's, the owned loop's, fastai's) — the one place a family can act before `initialize`.
And the setting, `mps_low_watermark`, is a BODY slot of `TorchTrainerRunnable`, not a constructor parameter: a
constructor parameter would have meant the same new argument in every concrete trainer's flat `__init__` (about
sixteen classes in four consuming projects), while confluid configures a body slot from a config or the command line
and lists it in the form-specs the same way. A hard memory LIMIT was measured and rejected:
`torch.mps.set_per_process_memory_fraction(0.6)` stopped the run with `MPS backend out of memory`.

```yaml
runnable: !class:your.project.Trainer
  mps_low_watermark: 0.5     # or on the command line: runnable.mps_low_watermark=0.5
```

What you may change: what a family does in `_before_run`. Not: calling it after `initialize`, or turning the
setting into a constructor parameter of the base (every consumer's flat constructor would have to follow).

---

## 10. The trainer CLI is a factory; a project CLI is its FRAMEWORKS table (2026-08-07)

### Context

Every training project grows the same command line: three verbs (`train`/`evaluate`/`predict`)
that force a `task` onto a `runnable:`-rooted config, a `--framework` flag naming which scope
block activates, a `frameworks` listing command, and two pre-steps that must run BEFORE the
config loads (the keras engine pin, the registration import). Four projects carried that as
~280 near-identical lines each — measured, the modules differed in their FRAMEWORKS table,
their verb help texts, and nothing else that was not a docstring — and every behavioural fix
(the taskless-verb guard, the pin-before-registration order) had to be applied four times.
Their package roots duplicated a second machine: the PEP 562 lazy-name resolver that keeps
`import <project>` framework-free.

### Decision

`matrainer.cli` is the FACTORY: `Framework` (description + registrations — `None`, one module,
or a tuple that makes a multi-module coupling readable in the table), `default_dispatch` (the
materialize-and-force-task path, with the taskless-fit guard as the default — a silent
`hasattr` would let an evaluate verb TRAIN), `make_trainer_app(name, description, frameworks,
verb_docs, dispatch)` and `trainer_main(app, frameworks)` running the established pre-step
order. A project's `cli.py` shrinks to its DATA — the table and the verb texts — plus two
calls. The generic `matrainer` console script is the same factory invoked WITHOUT a table
(`frameworks=None` skips the listing command), which is exactly the difference between a
project-agnostic runner and a project CLI. The lazy-name resolver is `matrainer._exports.
lazy_exports(__name__, table)`; each root keeps its `_BACKEND_NAMES` table.

### Consequences

The order (`requested_framework` → `pin_keras_backend` → `register_frameworks` →
`normalize_framework_flag` → `app.run()`) is pinned ONCE, in matrainer's own suite, instead of
per project — each arrow was a real bug when reversed. A behavioural extension lands
everywhere at once (one project's taskless-fit guard reached the other three CLIs in the lift). `dispatch=` is
the extension point for a project needing more than the default. The cost is that a project's
CLI behaviour is no longer readable in its own module — the factory's docstrings carry the
rationale, and the project module says so.

### Example

```python
from matrainer.cli import Framework, make_trainer_app, trainer_main

FRAMEWORKS = {"lightning": Framework("the reference backend", "myproject.torch_task")}
app = make_trainer_app(name="myproject", description="…", frameworks=FRAMEWORKS,
                       verb_docs={"train": "Train a …"})

def main() -> None:          # [project.scripts] myproject = "myproject.cli:main"
    trainer_main(app, FRAMEWORKS)
```

### What you may change

Add pre-steps, verbs or `Framework` fields freely — every consumer inherits them. What you may
not do is teach the factory a TASK (a verb doc naming one belongs in the project's
`verb_docs=`) or reorder the pre-steps without re-measuring the two orderings the docstrings
record.

## 11. A logger adapter must project `save_dir`, or checkpoints land in the CWD (2026-08-10)

### Context

`TrackerLogger` adapts a matrainer `Tracker` to the `pytorch_lightning.loggers.Logger`
interface. It implemented the parts a reader thinks of as "logging" — `name`, `version`,
`experiment`, `log_metrics`, `log_hyperparams`, `save`, `finalize` — and left `save_dir` to the
base class, which returns `None`. That looks harmless: nothing in the metric path reads it.

But `save_dir` is not only a logging concern. `ModelCheckpoint.__resolve_ckpt_dir` reads it
FIRST and falls back to `Trainer.default_root_dir` — the process CWD — when it is `None`:

```python
if trainer.loggers[0].save_dir is not None:
    save_dir = trainer.loggers[0].save_dir
else:
    save_dir = trainer.default_root_dir
ckpt_path = os.path.join(save_dir, str(name), version, "checkpoints")
```

So the adapter's `name`/`version` (the tracker's experiment and run name) were composed against
the launch directory. A sweep names its experiment by date, so a run launched from a project
root wrote `<repo>/Sun_Aug_9_2026_1/lightning_convnet/checkpoints/` INTO the repository. It
stayed invisible because every file inside happened to match a `checkpoints/` ignore rule —
`git status --ignored` collapsed the whole tree to one `!!` line, and the directory itself was
never ignored at all. One non-checkpoint artifact would have made it committable.

### Decision

`TrackerLogger.save_dir` returns `self.tracker.save_dir`. Every `Tracker` backend already
defaults it to `./runs` (`tracking/base.py`), so checkpoints land beside the metrics the same
tracker writes, under one directory a project can ignore with one line.

The property is a PROJECTION, never a second source of truth: there is no separate
checkpoint-root knob to keep in sync, and overriding the tracker's `save_dir` moves both.

### Consequences

* A whole run — metrics, checkpoints, manifest, artifacts — lives under `runs/<experiment>/<run>/`.
* The workspace mandate "Run Output Lands In A Gitignored `runs/`" is enforceable, because the
  default no longer depends on where the operator happened to `cd` first.
* An adapter over a Lightning interface must be checked for members read by things OTHER than
  the obvious caller. `save_dir` is read by a callback, not by the logging path, which is why
  a complete-looking adapter was missing it.

### Example

```python
tracker = CsvTracker(experiment_name="mnist", run_name="convnet")   # save_dir="./runs"
trainer = pl.Trainer(logger=TrackerLogger(tracker), callbacks=[ModelCheckpoint()])
# checkpoints -> ./runs/mnist/convnet/checkpoints/   (not <cwd>/mnist/convnet/checkpoints/)
```

### What you may change

The tracker's `save_dir` — that is the knob, and both metrics and checkpoints follow it. What
you may not do is reintroduce a checkpoint root that is independent of it, or drop the property
on the grounds that the metric path does not read it. Pinned by
`tests/lightning/test_trackers.py::test_checkpoints_land_under_the_trackers_save_dir_not_the_launch_directory`,
which asserts against Lightning's REAL resolution rather than a restatement of the rule.

## 12. The checkpoint loader reads generic formats and learns no library's own (2026-09-27)

### Context

`matrainer.torch.load_lightning_checkpoint` loads weights into a model from three generic
shapes: a Lightning `.ckpt` (its `model.` prefix stripped), an Ultralytics `.pt` (a live module
under `model`), and a plain state dict. When the wrapper's keys do not overlap the file's, it
retargets the load to a named child. Missing and unexpected keys are logged at INFO;
`strict=True` makes a mismatch raise.

Model libraries also write their OWN training files. One detection library's
`{model, optimizer, lr_scheduler, epoch, args}` file reads, through this function, as a
five-key state dict. Measured: it landed 1 of 465 tensors, with only the INFO lines to show
for it, while that library's own loader landed all 465.

### Decision

A library's own checkpoint format is loaded by the model wrapper that owns that library,
through the library's own loader. The wrapper recognises the file by its shape and routes it;
only the generic shapes reach this function. This function learns no per-library format, and a
load that matches zero keys stays an INFO line rather than an error (user ruling, 2026-09-27).

### Consequences

* The function's scope stays three generic shapes. It does not grow a format table that every
  new model library would have to extend.
* The routing lives in the wrapper, the one place that knows both the file and the library.
* A caller that needs a mismatch to be fatal passes `strict=True`.
* No test pins the zero-match behaviour yet.

### Example

```python
from matrainer.torch import load_lightning_checkpoint

class DetectorWrapper(nn.Module):
    def solidify(self) -> None:
        if is_library_checkpoint(self.checkpoint_path):    # the library's own shape
            self.detector = library.load(self.checkpoint_path)
        else:                                              # .ckpt / .pt / plain state dict
            load_lightning_checkpoint(self, self.checkpoint_path)
```

### What you may change

A fourth GENERIC shape (one no single library owns), and better diagnostics on a poor match.
What you may not do is add a library-specific branch here, or make a zero-match load raise by
default.

## 13. A model package is a folder of a flow graph and its weights, run as one sealed op (2026-10-12)

### Context

A trained checkpoint answers correctly only on inputs made the way its training inputs were made —
a detector trained on spectrogram pictures needs the same FFT, the same grey scale, the same
picture size — and its raw outputs (pixel boxes) mean something only after the steps that project
them back into the caller's units. Both sides already existed as a recordstream flow graph, but
every consumer that ran the checkpoint rebuilt that graph by hand around a bare checkpoint path:
the same includes, the same orientation override, the same class list, three times over for one
model. Measured on one consumer: its review graph and the package built from the same files gave
the same table on 81 captures, box for box — and the same graph read from the workspace a day
later, after an edit of the picture size, scored 48 fewer boxes right, with nothing in the run
telling that the picture had changed under the checkpoint.

### Decision

A **model package** is a folder: `package.yaml` (the entry graph, every graph file and weights
file with its SHA-256, the class names, the versions of the distributions the graph's classes
come from), plus `graphs/` and `weights/` when they are copied in. `PackageModel` (a runnable)
writes it; `ModelPackage` (an op) runs it.

* **The graph stays the file its author wrote.** The weights are not written into it:
  `package.yaml` maps each predictor slot (a dotted key, `flow.detect.op.model.checkpoint_path`)
  to its file, and the op fills the slot with the absolute path when it loads the graph. Copied
  graph files keep their layout below the folder holding them all, so every relative `include:`
  resolves inside the package unchanged — no file is rewritten.
* **Loading is confluid's own rule.** The op loads `{include: [entry graph], <slot>: <weights>,
  **overrides}`: document order, the last spec wins. No new spelling, no precedence of its own.
* **Copy or reference, per kind.** `graphs` and `weight_files` each take `copy` (frozen: a later
  edit of the source changes nothing) or `reference` (relative paths: the package follows the
  sources). Freezing is the point of a package; referencing is what a person iterating on the
  graph wants — both are legitimate, so it is a setting, not a rule.
* **A sealed op, not a `Subgraph` subclass.** A step-through tracer rebuilds a subgraph through its
  constructor with `steps=` (its current inside); a package has no such constructor argument, and
  giving it one would make the package's graph editable in place, which is what a package exists
  to prevent. The op holds a private `Subgraph` and delegates `consumes` / `produces` / `flags`
  to it, so chain checks still see the package's own declarations.
* **In matrainer, framework-free.** A package is the hand-off between a training run and its
  consumers — the same place as the run manifest, `class_names.json` and the checkpoint policies —
  and it needs only confluid, recordstream and pyyaml, all core dependencies. It reuses
  `save_class_names` and the manifest's `file_digest` / `command_line` / `contract_roots`.

Rejected: a pickled pipeline object (unreadable, tied to the class layout of the moment it was
written); an exported network (ONNX, TorchScript) — that is the compilation project's job and
leaves out the pre- and postprocessing a recordstream graph expresses; writing the weights path
into a rewritten copy of the graph (a second file per package to keep in step with its source).

### Consequences

* Any pipeline runs a model as one step with one path; a settings form needs two fields (the
  package, the overrides) whatever the model.
* A copied package keeps giving the answers it gave when it was made; a referencing one does not.
* A package is only as right as the graph it was made from: package a checkpoint from a graph
  that has since drifted from the one it was trained with and the package freezes the drift.
* A file a graph reaches other than by `include:` (a path in a setting) is not collected; an
  absolute include stays absolute, so it is not frozen.
* The version list over-names: a namespace several distributions install into lists them all.

### Example

```yaml
# package.yaml — the writer
runnable: !class:matrainer.model_package.PackageModel
  graph: config/model.yaml
  weights: {flow.detect.op.model.checkpoint_path: ./runs/a/checkpoint_best_total.pth}
  output_dir: ./runs/a/package
```

```python
from matrainer.model_package import ModelPackage

op = ModelPackage(path="runs/a/package", overrides={"flow.detect.op.device": "mps"})
record = op(record)          # the package's graph, its weights, the override applied last
```

### What you may change

New fields in `package.yaml` that a reader may ignore (bump `PACKAGE_FORMAT` when a field changes
meaning); a transport form (a zip of the folder); collecting more kinds of referenced files. What
you may not do is give the op a precedence rule of its own, rewrite the author's graph files, or
make it a `Subgraph` whose inside a tracer can replace.

**Amendment (2026-10-13).** The op's loading moved into `load_package_graph(path, overrides, owner=)`,
the ONE loader: a compiler that reads a package loads exactly the graph the op runs, through the same
call, and names itself in the refusals (`owner`). It returns the `request` it gave `confluid.load` — the
entry graph included, each weight slot, the overrides — so a reader needing the same document at
another stage (its markers with their file locations) loads the same thing. A compiled form of a package
is a compiler's own folder under `<package>/compiled/<name>/` by default; `PackageModel` replacing the
package deletes it, which is right — it was of the old weights — and this module never reads or writes
that folder.

---

## 14. A resilient wrapper, not a resilient backend (2026-10-13)

### Context

A tracker that talks to a remote server puts the network inside the training loop: a call that
hangs holds the loop, a call that fails loses its value, and a server that is down for ten minutes
of a ten-hour run should cost nothing. The same four answers — a time limit per call, a queue that
replays what the network lost in order, the bytes of a deleted artifact kept for that replay, one
broken endpoint turned off without the others — apply to every remote backend, and none of them is
about any backend's library.

### Decision

`ResilientTracker` (`tracking/resilient.py`) is a `Tracker` that holds ONE tracker and calls it
through a thread with a time limit (`run_with_timeout`). A failure is classified once, in the
wrapper: a network failure (`is_network_error`: a timeout, a connection error, or a message saying
so — client libraries wrap socket errors in their own types) queues the call; a 5xx
(`is_server_error`) turns that method off for the run; anything else is dropped with a warning. A
new call made while the queue is not empty goes BEHIND it, so values keep their order. After
`max_retries` failures in a row no call is attempted until `finalize`, which retries the queue and
then ends the wrapped run. Starting the run is the wrapped tracker's first `.experiment` access,
done once with `start_timeout`; if it fails, the wrapper is off for the run (nothing queued could
be replayed without a run). It is a composite like `TrackerGroup`, inside the `Tracker`
abstraction — record 2's "no view/adapter layer" is about adapting things that are NOT trackers,
and still holds.

Rejected: resilience inside each backend (it would be written six times and differ); resilience in
`TrackerGroup` (a group's job is fan-out, and a single tracker needs it too).

### Consequences

- The wrapper sees only what the wrapped call raises, so it turns the wrapped tracker's
  `raise_errors` on (amendment below); a backend that ignores the setting gets the time limits but
  nothing queued.
- A call that timed out keeps running on its thread and may still land, so a replay can record the
  same value twice. Accepted: deduplicating would need per-call identities on the server side.
- The wrapper owns no identity: name, run name, `save_dir`, tags and `log_dir` are the wrapped
  tracker's, so `TrackerLogger.save_dir` (record 11) still places checkpoints under `runs/`, and the
  artifact backups land in the same run directory.
- `finalize` may wait `max_retries × finalize_retry_delay` plus the calls' own limits at run end.

**Amendment (2026-10-14).** Every tracker takes `raise_errors`, off by default, and the wrapper
turns it on for the tracker it wraps on first use (`ResilientTracker.inner`). Off, a failed backend
call is a debug line and the run goes on, exactly as before: a bare tracker must never stop
training over a value. On, the same debug line is written and the error raised. A call that loops
over names (MLflow's parameters, metrics and tags, Aim's and ClearML's metrics) tries every name and
raises the first failure, and a call made of parts (`log_manifest`: the parameters, the files, each
native slot) runs every part first (`Tracker._run_all`) — the wrapper drops a refused call whole, so
a refused parameter must not cost the manifest its upload. Three failures stay quiet even when on:
Aim's starting tags (a failed start turns the wrapper off for the whole run), W&B's reference
attachment (record 3: the artifact is the record, the reference an optimisation) and MLflow's
system-metrics shutdown (the run must still be ended). `is_server_error` reads the status the error
carries — its `status_code`, its response's, or MLflow's `get_http_status_code()` — and, without
one, only a message that writes a 5xx status (`503 Service Unavailable`): MLflow refuses a changed
parameter with a 400 whose message holds the new value, and a value of 500 turned parameter logging
off for the rest of the run (measured). Rejected: raising by default (every bare run would stop on
a network error that costs one value today).

### Example

```python
from matrainer import MlflowTracker, ResilientTracker

tracker = ResilientTracker(tracker=MlflowTracker(experiment_name="exp"), timeout=5.0)
tracker.log_metrics({"val/loss": 0.4}, step=3)   # queued if the server is unreachable
assert tracker.inner.raise_errors                # turned on by the wrapper's first call
tracker.finalize()                               # the queue sent, then the run ended
```

### What you may change

The classification words, the defaults, a new `Tracker` method (route it through `_call` with its
time limit). What you may not do: let the wrapper raise into a training loop (a missing `tracker:`
is the one configuration error it raises), call the wrapped tracker outside `run_with_timeout`, or
let a new call overtake a queued one; turn `raise_errors` on by default, or let a backend's
`raise_errors` stop the names or parts after a failed one.

# Marainer

**Experiment-tracking library** for machine-learning runs — parameters, git provenance, log
files, config artifacts, figures, metric values, and system metrics — logged to whichever backend
you use (MLflow, TensorBoard, CSV, Weights & Biases), across whichever training platform you run
(PyTorch Lightning, TensorFlow/Keras, Ultralytics, a bare training loop).

Marainer holds **no training loop, no trainer/evaluator class, and no CLI.** A class that trains
reads like a standard example for its framework and lives in *your* project; it reaches for Marainer
only to *track* the run. The guiding invariant: **removing Marainer should cost only a couple of
line changes** — the run keeps training (with loggair logging), it just loses the rich tracking.

## Installation

```bash
pip install git+https://github.com/Gearlux/marainer.git@main
```

The core install is light — `confluid` / `liquifai` / `loggair` / `coolname` / `numpy` only.
`import marainer` pulls in **no** torch, pytorch-lightning, mlflow, or wandb. Everything heavy is
behind an extra and lazily imported the first time you use its backend:

| Extra | Pulls in | Enables |
|---|---|---|
| `marainer[torch]` | `torch`, `torchmetrics` | `marainer.torch` — checkpoint loading + the metric helpers |
| `marainer[lightning]` | torch, `pytorch-lightning`, `torchmetrics`, `tabulate`, `matplotlib` | `marainer.lightning` — Lightning tracking callbacks |
| `marainer[mlflow]` | `mlflow`, `psutil`, `pyobjc` (macOS) | `MlflowTracker` + system metrics |
| `marainer[tensorboard]` | torch (`torch.utils.tensorboard`) | `TensorBoardTracker` |
| `marainer[wandb]` | `wandb` | `WandbTracker` |
| `marainer[aim]` | `aim` | `AimTracker` |
| `marainer[clearml]` | `clearml` | `ClearMLTracker` |
| `marainer[keras]` | `keras` | `marainer.keras.keras_tracking_callback` |
| `marainer[tensorflow]` | `marainer[keras]`, `tensorflow` | the same helper, with the TensorFlow engine underneath |
| `marainer[all]` | lightning + mlflow + tensorboard + wandb | everything |

A missing backend library raises one clear error — e.g. `pip install marainer[mlflow]` — never a
bare `ModuleNotFoundError`.

## The `Tracker` — the one abstraction

`marainer.Tracker` (in `marainer.tracking.base`) mirrors the `pytorch_lightning.loggers.Logger`
shape so it drops into any framework's logger slot, plus a few extensions:

```
name / version / experiment                    # identity + the lazily-built native handle
log_hyperparams(params)                         # config params
log_metrics(metrics, step=None)                 # scalar metrics over time
set_tags(tags) / log_artifact(path, dir=None)   # tags + file artifacts
log_figure(name, figure, step=None)             # matplotlib / array figures
save() / finalize(status) / log_dir             # lifecycle
```

Backends live one-per-file, each importing its library **lazily inside its methods** and every
one **zero-arg constructible** (the native handle is built on first `.experiment` access):

| Backend | Library | Notes |
|---|---|---|
| `CsvTracker` | stdlib only | the always-available default — `metrics.csv` + `hparams.yaml` + artifact copies |
| `MlflowTracker` | `mlflow` | run lifecycle via `MlflowClient`; optional `log_system_metrics=True` |
| `TensorBoardTracker` | `torch.utils.tensorboard` | artifacts copied into the run dir |
| `AimTracker` | `aim` | local `.aim` repository; `aim up` serves the UI, no server to operate |
| `ClearMLTracker` | `clearml` | a ClearML task; `offline=True` records to disk with no server and no credentials |
| `WandbTracker` | `wandb` | minimal, mock-tested |

**Backends are discovered, not listed.** Adding one is a *single new file* — drop
`marainer/tracking/<name>_backend.py` in with a `Tracker` subclass and it resolves by name from
both `marainer.tracking` and `marainer`, appears in `dir()` and `__all__`, and needs no import
line, no `__all__` entry, and no registry edit. `discover_backends()` returns the map if you want
it. A backend shipped by a *different* distribution is discovered through the
`confluid.configurables` entry point instead, with no change here at all.

`TrackerGroup([...])` fans one call out to several backends behind the same surface.

```python
from marainer import CsvTracker, MlflowTracker, TrackerGroup

tracker = TrackerGroup([CsvTracker(), MlflowTracker(experiment_name="drone-classify")])
tracker.log_hyperparams({"lr": 1e-3, "batch_size": 32})
for step, loss in enumerate(losses):
    tracker.log_metrics({"train/loss": loss}, step=step)
tracker.finalize("FINISHED")
```

Trackers are `@configurable`, so they also wire straight from YAML.

## Choosing a backend

The `Tracker` surface is the same everywhere, so the choice is about **where the results live** and
**what it costs to run**, not about the API. Marainer's own selection criterion for what it
implements in-tree is *free and self-hostable* — you should be able to `pip install`, start a server
on your own machine, and see your runs, with no account and no licence key.

**Free, self-hosted, no account:**

| Backend | Library | Licence | Local UI | Status |
|---|---|---|---|---|
| `CsvTracker` | stdlib only | — | none (plain files) | ✅ shipped, the default |
| `MlflowTracker` | `mlflow` | Apache-2.0 | `mlflow server` | ✅ shipped |
| `TensorBoardTracker` | `torch.utils.tensorboard` | Apache-2.0 | `tensorboard --logdir` | ✅ shipped |
| `AimTracker` | `aim` | Apache-2.0 | `aim up` — no server infrastructure | ✅ shipped |
| `ClearMLTracker` | `clearml` | Apache-2.0 (self-hosted server via docker-compose) | full web UI — or `offline=True` for no server at all | ✅ shipped |

Other free and self-hostable options, **not** implemented here — the `Tracker` ABC makes any of them
a single new file if you need one (see [architecture record 4](docs/architecture.md)):
`dvclive` (git-native experiment versioning), `trackio` (a Weights & Biases-API-compatible local
dashboard), `sacred` + Omniboard (needs a MongoDB), and `determined` (Apache-2.0, but it is a cluster
scheduler — heavy if all you want is tracking).

**Hosted services with a free tier** — usable, but the server is not something you can run yourself
for free, so a fresh checkout is not steered toward them:

| Service | Status | Note |
|---|---|---|
| Weights & Biases | ✅ `WandbTracker` shipped | free personal tier; self-hosting is the enterprise offering. `wandb offline` writes local run directories, but reading them back needs a server. |
| Neptune | not implemented | free individual tier, hosted; self-host is enterprise. |
| Comet | not implemented | free individual/academic tier, hosted; self-host is enterprise. Their Apache-2.0 *Opik* is LLM evaluation, not experiment tracking. |

Check current terms before relying on any hosted tier — they change more often than this file does.

**Rule of thumb.** Start on `CsvTracker` (works with zero extras). Add `MlflowTracker` when you want
a model registry and a queryable run history; add `AimTracker` when you mainly want to *compare*
many runs quickly and do not want to operate a server; add `ClearMLTracker` when you want dataset
and orchestration surfaces in the same place. `TensorBoardTracker` complements rather than replaces
any of them — it is a visualizer, not a metadata store. `TrackerGroup` means you never have to pick
just one.

## Automatic provenance

`log_run_context` is the **one call at the start of a run**. It fills a [run manifest](#the-run-manifest)
— environment, code state, configuration, hyperparameters — and projects it onto every tracker,
plus the `git.commit` tag and the config tree as `config/` artifacts. `capture_log_file` is its
end-of-run counterpart: it uploads the loggair output file and finishes the manifest.

```python
from marainer import CsvTracker, MlflowTracker, log_run_context
from marainer.tracking import capture_log_file

trackers = [CsvTracker(), MlflowTracker(experiment_name="drone-classify")]

manifest = log_run_context(trackers, runnable=my_configured_runnable)
# ... run training ...
manifest.record_output("epoch=3.ckpt", monitor="val/acc", value=0.91)
capture_log_file(trackers, manifest=manifest, status="success")
```

Hand it the configured object and it derives the hyperparameters and dumps the resolved config —
don't hand-build a params dict for something it can read. `params=` is still there for values that
live outside the config, and it wins over anything derived:

```python
manifest = log_run_context(trackers, params={"lr": 1e-3}, runnable=runnable)
```

| Argument | Effect |
|---|---|
| `runnable` | derive hyperparameters + dump the resolved config to `config/final.yaml` |
| `params` | explicit values, merged **over** the derived ones |
| `manifest` | reuse a manifest you built yourself instead of one built from the first tracker |
| `capture_code` | set `False` to skip the per-repository git probe (~1s in a large workspace) |

It returns the manifest so you can keep recording; with no trackers *and* no manifest it returns
`None` and writes nothing. Don't hand-roll git/config/log-file logging elsewhere — these are the
sanctioned helpers.

## The run manifest

`RunManifest` is the one record of everything *static* about a run — identity, command line, code
state, configuration, hyperparameters, environment, datasets, models, outputs. It is the **union**:
every fact lands here whichever backend you use, and a backend's native slot is a projection of it,
never an alternative. That is what keeps a run tracked to a server comparable with the same run
tracked to a local directory.

```python
from marainer import CsvTracker, RunManifest

tracker = CsvTracker(experiment_name="drone-classify")
manifest = RunManifest(
    run_dir=tracker.log_dir,
    experiment_name=tracker.experiment_name,
    run_name=tracker.run_name,   # share the tracker's identity — otherwise both draw their own slug
    task="fit",
)

manifest.capture_environment()      # argv, cwd, interpreter, accelerator, environment variables
manifest.capture_code()             # every editable install: commit, dirty, branch, remote
manifest.capture_config(runnable)   # config tree by digest + the resolved dump + hyperparameters

manifest.record_dataset(kind="hf", name="org/set", split="train", count=12_000)
manifest.record_model(source="zoo", name="efficientvit_b0", revision="v1")
manifest.record_output("epoch=3.ckpt", monitor="val/acc", value=0.91)

tracker.log_manifest(manifest)      # push it to the backend
manifest.finish("success")
```

Two files land side by side and always agree, because they are written together:
`manifest.yaml` (what you read) and `manifest.json` (what tooling reads).

```yaml
manifest_version: 1
status: success
run:
  experiment_name: drone-classify
  run_name: run_petite_pudu
  task: fit
  started_at: '2026-08-01T13:28:59.760546+00:00'
  stopped_at: '2026-08-01T13:31:02.117903+00:00'
command: {}
code: []
config: {}
params:
  lr: 0.001
  batch_size: 32
env: {}
datasets:
- kind: hf
  name: org/set
  revision: null
  split: train
  count: 12000
  digest: null
models:
- source: zoo
  name: efficientvit_b0
  revision: v1
  digest: null
outputs:
- kind: checkpoint
  path: epoch=3.ckpt
  monitor: val/acc
  value: 0.91
```

**It survives an interrupted run.** Every call above re-emits both files atomically, so
crash-safety is a *write policy*, not a shutdown hook — kill a run mid-epoch and the manifest is
already on disk with everything gathered so far. `SIGINT` / `SIGTERM` / `atexit` only stamp the
final `status` (`running` → `success` / `failed` / `interrupted`), and the signal handler chains to
whatever handler was there before, so your Ctrl-C still behaves like Ctrl-C. Pass
`handle_interrupts=False` if your application owns signal handling.

| Section | Kind | Written with |
|---|---|---|
| `run` | mapping | the constructor (`experiment_name` / `run_name` / `task`) + `set("run", …)` |
| `command`, `config`, `params`, `env` | mapping | `set(section, {...})` — merges, never replaces |
| `code`, `datasets`, `models`, `outputs` | list | `add(section, {...})`, or `record_dataset` / `record_model` / `record_output` |

A mistyped section name raises rather than silently creating a key, and `set` on a list section (or
`add` on a mapping one) tells you which to use. `run_dir=None` derives
`runs/<experiment_name>/<run_name>`; the directory is created on first write, not at construction.

### What the three `capture_*` calls collect

**`capture_environment()`** — `argv`, cwd, interpreter, platform, and the accelerator, read from
the framework *already loaded* in the process (marainer never imports torch to find out). Plus the
run-relevant environment variables, selected by prefix (`MLFLOW_`, `CUDA_`, `HF_`, `WANDB_`, …):

```python
manifest.capture_environment(env_prefixes=["PROJECT_"], env_names=["DATA_ROOT"])
```

**Credentials are redacted.** A name containing `TOKEN` / `KEY` / `SECRET` / `PASSWORD` / `AUTH`
keeps its name and loses its value — the manifest is uploaded as a run artifact, so a captured
`WANDB_API_KEY` would be a secret published to every backend. That a token *was* set is itself
provenance, so the key stays:

```yaml
env:
  variables:
    CUDA_VISIBLE_DEVICES: '0'
    MLFLOW_TRACKING_URI: http://localhost:5005
    WANDB_API_KEY: '***'
```

**`capture_code()`** — one record per repository whose code the run can execute, discovered from
editable-install metadata (PEP 610) plus the working directory, deduplicated by git toplevel. A
single `git.commit` tag describes only where the shell was standing; a run's behaviour is set by
every editable package on `sys.path`:

```yaml
code:
- {name: confluid,  path: /w/confluid,  commit: 3f9c0e09…, dirty: false, branch: dev/main, remote: …}
- {name: marainer,  path: /w/marainer,  commit: db120840…, dirty: true,  branch: dev/main, remote: …}
```

`dirty` is `null`, not `false`, when the probe itself failed — "unknown" and "clean" are very
different when you are trying to reproduce a run.

**`capture_config(runnable)`** — the entrypoint config and its transitively `include:`-d tree, each
by path *and* SHA-256, so a later reader can tell whether the file still on disk is the one the run
used. The *resolved* configuration is dumped to `config/final.yaml` beside the manifest and
referenced the same way — it keeps confluid's `!class:` / `!ref` tags, which would make the manifest
itself unreadable by a generic YAML tool. Scalar hyperparameters are flattened into `params`.
Called without a `runnable`, it records the input side only.

### Pushing it to a backend

`tracker.log_manifest(manifest)` flushes the manifest, logs both files as artifacts, and pushes the
`params` section through `log_hyperparams` — the one native slot *every* backend has. So the whole
record reaches every backend for free. A file already inside the tracker's own `log_dir` is skipped
rather than copied next to itself.

Backends with more native slots additionally write there — never *instead*, which is what keeps runs
comparable across backends:

| Backend | Also writes |
|---|---|
| `MlflowTracker` | the `mlflow.source.*` standard tags (name, type, git commit / branch / repoURL / dirty), and datasets as real **dataset inputs** via `log_inputs`, so MLflow links runs that share one |
| `TensorBoardTracker` | the manifest as YAML in the **Text** tab (`add_text`) — TensorBoard has no artifact store, so a copied file would never be visible |
| `ClearMLTracker` | the manifest in the **Configuration** tab (`connect_configuration`), readable and diffable between tasks; datasets and models as named **Hyperparameter** sections ClearML can compare across runs |
| `CsvTracker`, `AimTracker`, `WandbTracker` | the default: both files plus `log_hyperparams` |

**The single-commit slot may come back empty.** MLflow has one field for a source commit while the
manifest holds a list. The narrowing is explicit — the repository containing the cwd, else the one
containing the entry script, else a lone record — and yields *nothing* when none applies, rather
than picking arbitrarily. An empty field sends you to the manifest; a wrong one just looks right.

## Three ways to hook tracking into PyTorch Lightning

`marainer.lightning` (behind `[lightning]`) is **optional tracking helpers**, not a platform. Your
`LightningModule` stays a standard one; pick whichever hookup fits — the choice is yours:

**1. Plain functions** — call `log_run_context` / `capture_log_file` from your own run wrapper
(shown above). Zero Lightning coupling.

**2. A `pl.Callback`** — `TrackingCallback` logs provenance on fit/test start and the log file on
end:

```python
import pytorch_lightning as pl
from marainer import CsvTracker
from marainer.lightning import TrackingCallback

trainer = pl.Trainer(
    max_epochs=10,
    callbacks=[TrackingCallback([CsvTracker()], params={"lr": 1e-3})],
)
```

**3. The `TrackerLogger` bridge** — hand a Marainer `Tracker` to `Trainer(logger=…)` so it receives
per-step metrics like any pl logger:

```python
from marainer.lightning import TrackerLogger
trainer = pl.Trainer(logger=TrackerLogger(MlflowTracker(experiment_name="exp")))
```

Metrics may equally stay on stock `pytorch_lightning.loggers.*` — Marainer **uses** pl loggers,
never reimplements them.

`marainer.lightning` also re-exports the metric utilities (`build_metric_collection`,
`MetricPreset`, `coerce_metric_value`, `scalar_metrics`) — they LIVE in `marainer.torch` since
2026-08-01, because none of them imports `pytorch_lightning`; both spellings work. It ships the
progress bridge
(`LightningProgressCallback` / `attach_lightning_progress`, mapping Lightning's per-batch hooks onto
`recordstream.runnable.ProgressReporting`), the system-metric callbacks
(`MlflowSystemMetricsCallback` / `MPSMetricsCallback` + `activate_for_lightning`), and
`build_checkpoint_callbacks` (translates the checkpoint policies below into `ModelCheckpoint`s).

For Keras, `marainer.keras.keras_tracking_callback` is the analogue. It requires `keras`, not
TensorFlow: Keras 3 is standalone and picks its own compute backend, so a consumer training
on the torch backend is never asked to install a framework it did not choose.

## Framework-neutral checkpoint policies

Declare *what* to checkpoint once; each framework translates it. `marainer.checkpoints` provides
zero-arg `@configurable` carriers:

| Policy | Meaning |
|---|---|
| `SaveLast()` | keep `last.ckpt`, refreshed every epoch |
| `SaveEveryEpoch(every_n_epochs=1)` | snapshot on a cadence |
| `SaveBest(monitor, mode, top_k=1)` | keep the best `top_k` by a monitored metric |

```python
from marainer.checkpoints import SaveBest, SaveLast
from marainer.lightning import build_checkpoint_callbacks

callbacks = build_checkpoint_callbacks([SaveLast(), SaveBest(monitor="val/acc", mode="max")])
trainer = pl.Trainer(callbacks=callbacks)   # -> pytorch_lightning ModelCheckpoint(s)
```

No policy ⇒ the framework default. Add a policy here; translate it in the framework helper.

## System metrics (`marainer.tracking.system`, framework-free)

- `enable_mlflow_system_metrics()` — the process-wide MLflow enable for `mlflow.start_run` clients.
- `start_system_metrics_monitor()` — CPU / RAM / disk (and NVIDIA GPU when `pynvml` is present).
- `MpsMetricsSampler` — an Apple-Silicon GPU sampler with a plain `start()` / `stop()`, pushing nine
  `system/mps_*` scalars via `MlflowClient.log_metric`. It reads utilisation from an inlined IOKit
  `PerformanceStatistics` query (the no-sudo private API the `stats` menu-bar app uses); `torch` and
  `mlflow` are imported lazily.

A non-Lightning platform calls these directly; the Lightning `pl.Callback` wrappers live in
`marainer.lightning.callbacks`. Tune the cadence with
`MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL` (seconds, default 10) or disable via
`MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false`.

## `marainer.torch` — torch-only, Lightning-free helpers (`marainer[torch]`)

- `load_lightning_checkpoint(model, path, strict=False)` — loads a Lightning `.ckpt` (strips the
  `model.` prefix), an Ultralytics `.pt` (pulls the live module's `state_dict`), or a plain
  state-dict; auto-retargets to a named child when keys match one. `strict=False` tolerates a
  re-headed `num_classes`.
- `build_metric_collection(metrics, **dims)` — normalize any metrics spec (a mapping, a list, a
  single metric, an already-built `MetricCollection`) into one `MetricCollection` the caller clones
  per split, flowing each template with the sizing kwarg (`num_classes=` / `num_labels=`).
- `coerce_metric_value(value)` / `scalar_metrics(raw)` — JSON-safe coercion of a torchmetrics
  `compute()` result; `scalar_metrics` keeps only the scalars, so an `N×N` confusion matrix is
  dropped rather than tabulated.
- `MetricPreset(category=…, task=…)` — instantiate every compatible torchmetrics metric in a
  category, for a config that wants "all the classification metrics" without naming them.

```python
from marainer.torch import build_metric_collection, scalar_metrics

base = build_metric_collection({"accuracy": LazyClass(MulticlassAccuracy)}, num_classes=10)
train_metrics, val_metrics = base.clone(prefix="train/"), base.clone(prefix="val/")
...
final = scalar_metrics(val_metrics.compute())     # {"val/accuracy": 0.94}
```

It all sits outside `marainer.lightning` so a non-Lightning platform — an Ultralytics loop, or a
hand-written torch training loop — uses it via `marainer[torch]` without pulling
`pytorch_lightning`. The metric helpers moved here on 2026-08-01 for exactly that reason: they
import only confluid, torch and torchmetrics, so their old home described their first consumer
rather than their dependencies. `marainer.lightning` re-exports all four, unchanged.

## Random run names

`marainer.generate_run_name(prefix="train")` returns a friendly `adjective_noun` slug (`coolname`,
offline) — the default `Tracker.run_name`, and what a consumer's merged class uses
(`generate_run_name(prefix=task)`).

## What lives elsewhere

The **runnable protocol** and **pipeline running** are *not* in Marainer — they moved to
[RecordStream](https://github.com/Gearlux/recordstream):

- `recordstream.runnable` — `TorchRunner`, `ProgressReporting`, and the `entrypoint` /
  `runnable_entrypoints` method-annotation mechanism a merged train+eval class uses to declare its
  `fit`/`evaluate`/`test`/`predict` capabilities.
- `recordstream.workflow` — the `Sequence` / `Conditional` / `Switch` combinators + predicates.
- `recordstream.processing` — `DatasetProcessor`.
- `recordstream.ensure_record_dataset` / `RecordSource` — normalise a wired dataset slot into a
  map-style `Dataset` of records.
- `recordstream.outputs` — the prediction-output contracts (`ClassificationOutput` /
  `DetectionOutput` / `SegmentationOutput`) and their torch builders, beside the sink that reads
  them.
- `recordstream.predictions` — the `PredictionsSink` protocol + `ClassificationPredictionsSink`.
- `recordstream.inverse_frequency_weights` / `class_counts` — class-balance weights, a statistic
  over the labels. What a framework then does with the vector (`torch.nn`'s `weight=`, Keras's
  `class_weight`) is the consuming runnable's business.
- `recordstream run <config.yaml>` — the generic runner. A config roots at a top-level `runnable:` key
  plus a `task:`; consumers may also ship their own CLI (`sonair lightning train`, `raidar evaluate`, …).

There is no `marainer.runner` / `marainer.progress` / `marainer.workflow` / `marainer.processing`
and no marainer binary.

## Examples

`examples/standalone_tracking.py` — a torch-free run of the `Tracker` surface end to end
(hyperparams → metrics loop → figure → provenance → finalize) against `CsvTracker`, so you can see
the artifacts land without any framework installed:

```bash
python examples/standalone_tracking.py
```

## Dependencies

| Package | Purpose |
|---|---|
| `confluid` | YAML configuration + dependency injection (`@configurable` Trackers/policies) |
| `liquifai` | provenance's config-include resolution |
| `loggair` | structured logging + the active-log-file handle `capture_log_file` uploads |
| `coolname` | random run names |
| `pyyaml` | the run manifest's readable half (`manifest.yaml`) |
| `numpy` | the predictions sink |

Everything else — torch, pytorch-lightning, torchmetrics, mlflow, tensorboard, wandb, tensorflow —
is optional and behind an extra.

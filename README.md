# Matrainer

**Experiment-tracking library** for machine-learning runs — parameters, git provenance, log
files, config artifacts, figures, metric values, and system metrics — logged to whichever backend
you use (MLflow, TensorBoard, CSV, Weights & Biases), across whichever training platform you run
(PyTorch Lightning, TensorFlow/Keras, Ultralytics, a bare training loop).

Matrainer holds **no training loop, no trainer/evaluator class, and no CLI.** A class that trains
reads like a standard example for its framework and lives in *your* project; it reaches for Matrainer
only to *track* the run. The guiding invariant: **removing Matrainer should cost only a couple of
line changes** — the run keeps training (with loggair logging), it just loses the rich tracking.

## The name

**Matrainer** is a portmanteau of **mariner** — the one who keeps the log, recording every heading,
every reading and every condition of the voyage — and **trainer**, the thing being logged. That is
the whole job description: it does not steer the ship, it writes down where it went. A mariner's log
is also why the [run manifest](#the-run-manifest) is the *union* of everything known about a run
rather than whatever the current backend happens to have a field for — a logbook that only recorded
what this port cares about would be useless at the next one.

## Installation

```bash
pip install git+https://github.com/Gearlux/matrainer.git@main
```

The core install is light — `confluid` / `liquifai` / `loggair` / `coolname` / `numpy` only.
`import matrainer` pulls in **no** torch, pytorch-lightning, mlflow, or wandb. Everything heavy is
behind an extra and lazily imported the first time you use its backend:

| Extra | Pulls in | Enables |
|---|---|---|
| `matrainer[torch]` | `torch`, `torchmetrics` | `matrainer.torch` — checkpoint loading, the metric helpers, `resolve_device` |
| `matrainer[lightning]` | torch, `pytorch-lightning`, `torchmetrics`, `tabulate`, `matplotlib` | `matrainer.lightning` — Lightning tracking callbacks |
| `matrainer[mlflow]` | `mlflow`, `psutil`, `pyobjc` (macOS) | `MlflowTracker` + system metrics |
| `matrainer[tensorboard]` | torch (`torch.utils.tensorboard`) | `TensorBoardTracker` |
| `matrainer[wandb]` | `wandb` | `WandbTracker` |
| `matrainer[aim]` | `aim` | `AimTracker` |
| `matrainer[clearml]` | `clearml` | `ClearMLTracker` |
| `matrainer[fastai]` | torch, `fastai` (+ two measured upper bounds — see `pyproject.toml`) | `matrainer.fastai.FastaiRunnable` |
| `matrainer[keras]` | `keras` | `matrainer.keras.keras_tracking_callback` |
| `matrainer[plot]` | `matplotlib` | `render_matrix_as_figure` / `render_vector_as_figure` — a confusion matrix as a heatmap, a per-class metric as a bar chart, for `log_figure` |
| `matrainer[tensorflow]` | `matrainer[keras]`, `tensorflow` | the same helper, with the TensorFlow engine underneath |
| `matrainer[all]` | every extra above | everything |

A missing backend library raises one clear error — e.g. `pip install matrainer[mlflow]` — never a
bare `ModuleNotFoundError`.

## The trainer API — one abstract base, one concrete base per framework

Since 2026-08 matrainer also ships the shared TRAINER machinery a task project builds its merged
train/evaluate/test/predict runnables on. Three layers, each in its own extras-gated home:

| Layer | Class | What it owns |
|---|---|---|
| Neutral API (`matrainer.runnable`, light core) | `TrainerRunnable` | the universal constructor slots, `run()` dispatch through the four `@entrypoint`-marked capabilities, tracker/prepare/prediction plumbing, the neutral task hooks |
| torch family (`matrainer.torch`, `[torch]`) | `TorchTrainerRunnable` | the TYPED slots (`model: Lazy[nn.Module]`, the `MetricsSpec` quartet), the sentinel resolution, the per-split metric builders, the sidecar write, `trained_model` |
| Lightning machinery (`matrainer.lightning`, `[lightning]`) | `LightningRunnable` | the loaders, the deferred `pl.Trainer` slot with its tracking hookup, the standard step functions, the four entry points |
| bare-torch machinery (`matrainer.torch.loop`, `[torch]`) | `TorchLoopRunnable` | the hand-written epoch (`zero_grad`/`backward`/`step`), device resolution, the per-split scoring, a plain-`state_dict` checkpoint, per-epoch history, and the plain-function tracking hookup |
| fastai machinery (`matrainer.fastai`, `[fastai]`) | `FastaiRunnable` | the `DataLoaders`/`Learner` assembly as deferred slots, the positional batch tuple built on the task's `_targets` hook, the Recorder read-back into canonical names, alias-first metric-name resolution, and the plain-function tracking hookup |
| Keras family (`matrainer.keras.runnable`, `[keras]`) | `KerasTrainerRunnable` | the TYPED slots (`Lazy[keras.Model]`, the `KerasLoss`/`KerasMetric`/`KerasOptimizer` aliases), the `_to_arrays` batch hook, the keras sentinel hooks, the metric UNION, the `RecordSequence` boundary, the tracking-callback plumbing |
| Keras compile/fit machinery (`matrainer.keras.trainer`, `[keras]`) | `KerasRunnable` | build + compile (with the `_compiled_loss` / `_compiled_optimizer` / `_check_compiled_metrics` seams), the `History` read-back, the `_before_fit` / `_fit_scores` fit hooks, the four entry points over `keras.Model`'s own `fit`/`evaluate`/`predict` |
| native-TensorFlow machinery (`matrainer.tensorflow.loop`, `[tensorflow]`) | `TensorFlowRunnable` | the `tf.data` pipeline, traced `GradientTape` steps (XLA via `jit_compile`), per-split CLONED metric objects, the hand-driven `CallbackList`, the SavedModel export, the engine guard |

The abstraction deliberately **stops at the API** — there is no abstract trainer *object*: the
`pl.Trainer` slot exists only in the Lightning base, the open-coded epoch only in the loop
base, the `Learner` only in the fastai one, and the compile step only in the Keras one (whose
two machinery bases are SIBLINGS sharing the keras family layer). The loop base
adds three BATCH hooks (`_train_batch_loss` / `_eval_batch` / `_predict_batch`) with the
read-forward-loss default, so a task whose model computes its loss inside `forward` (a native
detection contract) overrides exactly those three; `_eval_batch` may return `None` to declare
"this split has no loss", and the split then reports its metrics alone. A consuming project writes its TASK once as a
pair of hook mixins and composes a concrete class per engine:

```python
from matrainer.lightning import LightningRunnable

@configurable(task="mytask", role="trainer", framework="torch")
class LightningMyTask(MyTaskTorchHooks, LightningRunnable):
    def __init__(self, model, train_set=None, ..., my_task_knob=1.0):
        super().__init__(model=model, train_set=train_set, ...)   # shared slots, stored verbatim
        self.my_task_knob = my_task_knob                          # task-specific state
```

The hooks a task implements: `initialize` (resolve dimensions/labels, flow the model, build the
metrics), `_targets` (the batch read the loss expects), `_default_loss` / `_default_metrics` /
`_metric_flow_dims` (the sentinel defaults and their sizing kwargs), `_prediction_output` (the
sink contract), `_reportable` / `_report_nonscalars` (reporting filters), `_after_fit` (epilogue).
Everything else — tracker wiring, epoch reporting, sidecars, entry points — is inherited.

## The `Tracker` — the one abstraction

`matrainer.Tracker` (in `matrainer.tracking.base`) mirrors the `pytorch_lightning.loggers.Logger`
shape so it drops into any framework's logger slot, plus a few extensions:

```
name / version / experiment                    # identity + the lazily-built native handle
log_hyperparams(params)                         # config params
log_metrics(metrics, step=None)                 # scalar metrics over time
set_tags(tags) / log_artifact(path, dir=None)   # tags + file artifacts
log_figure(name, figure, step=None)             # matplotlib / array figures
log_text(name, text, step=None)                 # anything that is neither (a rendered table)
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
`matrainer/tracking/<name>_backend.py` in with a `Tracker` subclass and it resolves by name from
both `matrainer.tracking` and `matrainer`, appears in `dir()` and `__all__`, and needs no import
line, no `__all__` entry, and no registry edit. `discover_backends()` returns the map if you want
it. A backend shipped by a *different* distribution is discovered through the
`confluid.configurables` entry point instead, with no change here at all.

`TrackerGroup([...])` fans one call out to several backends behind the same surface.

```python
from matrainer import CsvTracker, MlflowTracker, TrackerGroup

tracker = TrackerGroup([CsvTracker(), MlflowTracker(experiment_name="drone-classify")])
tracker.log_hyperparams({"lr": 1e-3, "batch_size": 32})
for step, loss in enumerate(losses):
    tracker.log_metrics({"train/loss": loss}, step=step)
tracker.finalize("FINISHED")
```

Trackers are `@configurable`, so they also wire straight from YAML.

## Choosing a backend

The `Tracker` surface is the same everywhere, so the choice is about **where the results live** and
**what it costs to run**, not about the API. Matrainer's own selection criterion for what it
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
end-of-run counterpart, and it is where a run ENDS: it uploads the loggair output file, finishes
the manifest, and finalizes every tracker — in that order, because a backend with an explicit run
lifecycle stops accepting writes once the run is closed.

That last part is deliberately not left to the caller. A run whose tracker is never finalized
stays `RUNNING` in the UI with no end time and therefore no duration, while its manifest on disk
says `success` — the same run, described two ways. Pass `finalize=False` if you reach this more
than once per run and your backend cannot take a second ending (MLflow can; `WandbTracker`
cannot, since its `finish()` drops the run handle).

```python
from matrainer import CsvTracker, MlflowTracker, log_run_context
from matrainer.tracking import capture_log_file

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
from matrainer import CsvTracker, RunManifest

tracker = CsvTracker(experiment_name="drone-classify")
manifest = RunManifest(
    run_dir=tracker.log_dir,
    experiment_name=tracker.experiment_name,
    run_name=tracker.run_name,   # share the tracker's identity — otherwise both draw their own slug
    task="fit",
)

manifest.capture_environment()      # argv, a pasteable command line, accelerator, env variables
manifest.capture_code()             # every editable install: commit, dirty, branch, remote
manifest.capture_config(runnable)   # config tree by digest + the resolved dump + hyperparameters
manifest.capture_packages()         # every installed distribution -> environment/requirements.txt
manifest.capture_datasets(runnable) # every dataset the configuration identifies
manifest.capture_models(runnable)   # every weight artifact the configuration loads

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
command:
  argv: [/w/.venv/lib/.../recordstream/cli.py, run, train.yaml]
  cwd: /w/project
  executable: /w/.venv/bin/python
  line: cd /w/project && /w/.venv/bin/python -m recordstream.cli run train.yaml
code: []
config: {}
params:
  lr: 0.001
  batch_size: 32
env:
  packages: {path: environment/requirements.txt, sha256: 85231b59…, count: 730, tool: uv}
datasets:
- kind: hf
  name: ylecun/mnist
  uri: hf://datasets/ylecun/mnist?split=train
  url: https://huggingface.co/datasets/ylecun/mnist/viewer/default/train
  revision: null
  split: train
  count: null
  digest: null
  slot: train_set
models:
- kind: hf
  name: timm/efficientvit_b0.r224_in1k
  uri: hf://models/timm/efficientvit_b0.r224_in1k
  url: https://huggingface.co/timm/efficientvit_b0.r224_in1k
  role: pretrained
  slot: model
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

Two files sit *beside* the manifest rather than inside it — `config/final.yaml` and
`environment/requirements.txt` — because each is another library's format, and the manifest
references them by path and digest. `manifest.artifacts()` lists them, and `log_manifest` uploads
them with the manifest, so a remote-only backend is never left holding a reference to a file that
exists on one machine.

A mistyped section name raises rather than silently creating a key, and `set` on a list section (or
`add` on a mapping one) tells you which to use. `run_dir=None` derives
`runs/<experiment_name>/<run_name>`; the directory is created on first write, not at construction.

### What the `capture_*` calls collect

**`capture_environment()`** — `argv`, cwd, interpreter, platform, and the accelerator, read from
the framework *already loaded* in the process (matrainer never imports torch to find out). Plus the
run-relevant environment variables, selected by prefix (`MLFLOW_`, `CUDA_`, `HF_`, `WANDB_`, …):

```python
manifest.capture_environment(env_prefixes=["PROJECT_"], env_names=["DATA_ROOT"])
```

It also records **`command.line`** — the invocation as a shell snippet you can paste to run it
again, `cd` included so a config referenced by a relative path resolves:

```yaml
command:
  line: cd /w/project && /w/.venv/bin/python -m recordstream.cli run train.yaml --framework torch
```

`argv` alone cannot be pasted back: a run launched with `python -m` leaves the module's *file* path
in `argv[0]`, which does not re-execute. Both are kept, because `argv` is what tooling parses. An
invocation with no honest reconstruction — an interactive session, or `python -c`, whose code never
reaches `sys.argv` — leaves `line` out rather than guessing at one.

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
- {name: matrainer,  path: /w/matrainer,  commit: db120840…, dirty: true,  branch: dev/main, remote: …}
```

`dirty` is `null`, not `false`, when the probe itself failed — "unknown" and "clean" are very
different when you are trying to reproduce a run.

**`capture_config(runnable)`** — the entrypoint config and its transitively `include:`-d tree, each
by path *and* SHA-256, so a later reader can tell whether the file still on disk is the one the run
used. The *resolved* configuration is dumped to `config/final.yaml` beside the manifest and
referenced the same way — it keeps confluid's `!class:` / `!ref` tags, which would make the manifest
itself unreadable by a generic YAML tool. Scalar hyperparameters are flattened into `params`.
Called without a `runnable`, it records the input side only.

**`capture_packages()`** — every installed distribution and its version, written to
`environment/requirements.txt` beside the manifest and referenced by path, digest, line count and
the program that produced it. The `code` section covers the packages you are *editing*; this covers
the several hundred wheels around them, which set a run's behaviour just as firmly.

```yaml
env:
  packages: {path: environment/requirements.txt, sha256: 85231b59…, count: 730, tool: uv}
```

`uv pip freeze` is used when `uv` is on `PATH` and `pip freeze` otherwise — purely a speed choice
(0.06 s against 0.96 s on a 730-package environment, and this runs on every tracked run), and
nothing is lost by it: the two disagree only on how they spell an editable install, and the commit,
branch and *dirty* flag for every editable are already in `code`. If neither program works the call
is a no-op — an environment that cannot be listed is not a reason to fail a run.

**`capture_datasets(runnable)`** — every dataset the configuration identifies. It walks the wired
object graph for sources that can name the data they read and records one entry each, deduplicated,
so a source reached through a stream or a split view is recorded once under the outer slot:

```yaml
datasets:
- {kind: hf, name: ylecun/mnist, uri: 'hf://datasets/ylecun/mnist?split=train',
   url: 'https://huggingface.co/datasets/ylecun/mnist/viewer/default/train',
   split: train, slot: train_set}
```

A **local** file also carries its `size` and SHA-256 `digest`, so a later reader can tell whether
the checkpoint still on disk is the one the run loaded — the same question `capture_config` answers
for the YAML tree. A remote reference (`hf://`, `s3://`) carries neither: it is identified by its
URI and revision, and fetching gigabytes to fingerprint it would be absurd. A *directory* carries
neither either — a SavedModel or an imagefolder has no single content hash. Hashing runs at roughly
1.8 GB/s, and files above `MAX_DIGEST_BYTES` (2 GiB) keep their size and skip the digest.

`kind`, `name` and `split` are derived from the URI rather than read off the source, so this needs
no knowledge of any particular kind of dataset and a new source type needs no change here. `uri` is
the canonical handle two runs are compared on; `url` is a link you can open, and is `null` for data
with no web page. A run whose sources carry no identifier records nothing, which is not an error.

**`capture_models(runnable)`** — the same walk for weights. A model wrapper reports the artifacts it
loads (`matrainer.uri.model_refs`), and **one wrapper may report several**, because a backbone
fine-tuned from a checkpoint used two:

```yaml
models:
- {kind: hf, name: timm/efficientvit_b0.r224_in1k, uri: 'hf://models/timm/efficientvit_b0.r224_in1k',
   url: 'https://huggingface.co/timm/efficientvit_b0.r224_in1k', role: pretrained, slot: model}
- {kind: file, name: /w/runs/epoch=3.ckpt, uri: 'file:///w/runs/epoch%3D3.ckpt',
   url: null, role: checkpoint, slot: model}
```

`role` says what each artifact contributed — `pretrained` for weights from outside the run,
`checkpoint` for weights a previous run wrote — because the two are not interchangeable to someone
trying to reproduce the result. A model trained from scratch loaded no artifact and records
nothing; the architecture it used is configuration, and is already in `params`.

**Making your own model wrapper report** is one property, and there is nothing to register:

```python
from matrainer.uri import ModelRef, file_model_ref, hf_model_ref

class MyModel:
    @property
    def model_refs(self) -> list[ModelRef]:
        refs = []
        if self.pretrained:
            refs.append(hf_model_ref(self.repo_id))            # hf://models/<id> + the Hub link
        if self.checkpoint_path:
            refs.append(file_model_ref(self.checkpoint_path))  # file:// + role="checkpoint"
        return refs
```

Read stored configuration only — the property is asked of wrappers that may never run, and must
never build or download anything to answer.

### Portable paths

A manifest full of absolute paths is only readable on the machine that wrote it. Set any of
**`PROJECT_ROOT`**, **`DATA_ROOT`** or **`MODEL_ROOT`** and every path under one is recorded
against it instead:

```yaml
command:
  cwd: $PROJECT_ROOT/sonair
  line: cd "$PROJECT_ROOT/sonair" && "$PROJECT_ROOT/.venv/bin/python" -m recordstream.cli run cfg.yaml
code:
- {name: confluid, path: $PROJECT_ROOT/confluid, commit: 3f9c0e09…, dirty: false}
datasets:
- {kind: file, name: $DATA_ROOT/imagefolder, uri: 'file://$DATA_ROOT/imagefolder'}
env:
  roots: {PROJECT_ROOT: /Users/gert/taidal, DATA_ROOT: /Volumes/Store}   # recorded verbatim
```

The whole document is contracted on the way out — the command line, the config tree, the code
checkouts, `file://` dataset and model URIs, output paths — so nothing has to remember to do it,
and a section added later is portable for free. Only `env.roots` keeps its absolute values: it is
the expansion table, and contracting a definition into a reference to itself would destroy it.

**`os.path.expandvars` is the exact inverse.** A reader with the same variables set gets the
original strings back, and one without them has `env.roots` to resolve by hand — so nothing is
lost either way. The substitution matches on a path *boundary*, so `/store` never swallows the
start of `/storefront`, and the most specific root wins when one nests inside another. Set none of
the three and nothing changes.

The pasteable `command.line` is contracted too, and stays pasteable: a word carrying a variable is
**double**-quoted rather than single-quoted, because a single-quoted `$PROJECT_ROOT` is a literal
dollar sign and the line would look for a directory that does not exist.

### Pushing it to a backend

`tracker.log_manifest(manifest)` flushes the manifest, logs both files *and its side artifacts*
(`config/final.yaml`, `environment/requirements.txt`) as artifacts, and pushes the `params` section
through `log_hyperparams` — the one native slot *every* backend has. So the whole record reaches
every backend for free. A file already inside the tracker's own `log_dir` is skipped rather than
copied next to itself.

Backends with more native slots additionally write there — never *instead*, which is what keeps runs
comparable across backends:

| Backend | Also writes |
|---|---|
| `MlflowTracker` | the `mlflow.source.*` standard tags (name, type, git commit / branch / repoURL / dirty); datasets as real **dataset inputs** and models as **logged-model inputs** via `log_inputs`, so MLflow links every run that shares one |
| `TensorBoardTracker` | the manifest as YAML in the **Text** tab (`add_text`) — TensorBoard has no artifact store, so a copied file would never be visible |
| `ClearMLTracker` | the manifest in the **Configuration** tab (`connect_configuration`), readable and diffable between tasks; datasets and models as named **Hyperparameter** sections ClearML can compare across runs; and — **with a server** — each model as a first-class `InputModel`, which puts it in the task's Models tab and on ClearML's lineage graph |
| `AimTracker` | each section as a structured **run value** (`run["code"]`, `run["env"]`, `run["datasets"]`, `run["models"]`, …), which Aim's UI filters and compares across runs rather than leaving inside a file — and that is Aim's *most* native slot: it has sequences (metrics, text, images) but no artifact, dataset or model entity to register into |
| `WandbTracker` | datasets and models as used **Artifacts** (`use_artifact`, typed `dataset` / `model`), which is what puts the run on W&B's lineage graph; an addressable URI also becomes an artifact **reference**, recording where the bytes live without uploading them |
| `CsvTracker` | the default: both files plus `log_hyperparams` |

**A server-only slot degrades rather than replacing.** ClearML's `InputModel` needs a server — measured, it dies in offline mode with `AttributeError: 'NoneType' object has no attribute 'response'` — so `offline=True`, the mode that needs no server and no credentials at all, skips it once with a debug note and keeps the hyperparameter section, which is then the only projection it has. That is why the section stays rather than being replaced by the richer slot.

**Two things a native slot cannot do for itself**, both measured rather than assumed. MLflow does
*not* deduplicate a logged model the way it deduplicates a dataset input by digest — and
`log_manifest` runs at least twice per run — so the tracker remembers what it has already linked;
without that, every model would appear twice. And both libraries restrict entity *names* to
character sets that exclude the `/`, `:` and `.` in every real model or dataset identifier
(`timm/resnet50.a1_in1k` is rejected outright by each), so the name is a sanitised label and the
exact identifier travels in the entity's tags or metadata.

**The single-commit slot may come back empty.** MLflow has one field for a source commit while the
manifest holds a list. The narrowing is explicit — the repository containing the cwd, else the one
containing the entry script, else a lone record — and yields *nothing* when none applies, rather
than picking arbitrarily. An empty field sends you to the manifest; a wrong one just looks right.

## Three ways to hook tracking into PyTorch Lightning

`matrainer.lightning` (behind `[lightning]`) is **optional tracking helpers**, not a platform. Your
`LightningModule` stays a standard one; pick whichever hookup fits — the choice is yours:

**1. Plain functions** — call `log_run_context` / `capture_log_file` from your own run wrapper
(shown above). Zero Lightning coupling.

**2. A `pl.Callback`** — `TrackingCallback` captures the run context on fit/test start and
finishes the manifest (plus the log file) on end:

```python
import pytorch_lightning as pl
from matrainer import CsvTracker
from matrainer.lightning import TrackingCallback

trainer = pl.Trainer(
    max_epochs=10,
    callbacks=[TrackingCallback([CsvTracker()], runnable=my_configured_runnable)],
)
```

Pass `runnable=` and it derives the hyperparameters and dumps the resolved config, exactly as
`log_run_context` does — `params=` remains for values outside the config and wins over anything
derived. The manifest is on the callback as `.manifest` from fit/test start onward, and a run
that raises is recorded as `failed` (or `interrupted` on Ctrl-C) rather than left to look
successful.

**3. The `TrackerLogger` bridge** — hand a Matrainer `Tracker` to `Trainer(logger=…)` so it receives
per-step metrics like any pl logger:

```python
from matrainer.lightning import TrackerLogger
trainer = pl.Trainer(logger=TrackerLogger(MlflowTracker(experiment_name="exp")))
```

Metrics may equally stay on stock `pytorch_lightning.loggers.*` — Matrainer **uses** pl loggers,
never reimplements them.

`matrainer.lightning` also re-exports the metric utilities (`build_metric_collection`,
`MetricPreset`, `coerce_metric_value`, `scalar_metrics`) — they LIVE in `matrainer.torch` since
2026-08-01, because none of them imports `pytorch_lightning`; both spellings work. It ships the
progress bridge
(`LightningProgressCallback` / `attach_lightning_progress`, mapping Lightning's per-batch hooks onto
`recordstream.runnable.ProgressReporting`), the system-metric callbacks
(`MlflowSystemMetricsCallback` / `MPSMetricsCallback` + `activate_for_lightning`), and
`build_checkpoint_callbacks` (translates the checkpoint policies below into `ModelCheckpoint`s).

For Keras, `matrainer.keras.keras_tracking_callback` is the analogue. It requires `keras`, not
TensorFlow: Keras 3 is standalone and picks its own compute backend, so a consumer training
on the torch backend is never asked to install a framework it did not choose.

## Framework-neutral checkpoint policies

Declare *what* to checkpoint once; each framework translates it. `matrainer.checkpoints` provides
zero-arg `@configurable` carriers:

| Policy | Meaning |
|---|---|
| `SaveLast()` | keep `last.ckpt`, refreshed every epoch |
| `SaveEveryEpoch(every_n_epochs=1)` | snapshot on a cadence |
| `SaveBest(monitor, mode, top_k=1)` | keep the best `top_k` by a monitored metric |

```python
from matrainer.checkpoints import SaveBest, SaveLast
from matrainer.lightning import build_checkpoint_callbacks

callbacks = build_checkpoint_callbacks([SaveLast(), SaveBest(monitor="val/acc", mode="max")])
trainer = pl.Trainer(callbacks=callbacks)   # -> pytorch_lightning ModelCheckpoint(s)
```

No policy ⇒ the framework default. Add a policy here; translate it in the framework helper.

## System metrics (`matrainer.tracking.system`, framework-free)

- `enable_mlflow_system_metrics()` — the process-wide MLflow enable for `mlflow.start_run` clients.
- `start_system_metrics_monitor()` — CPU / RAM / disk (and NVIDIA GPU when `pynvml` is present).
- `MpsMetricsSampler` — an Apple-Silicon GPU sampler with a plain `start()` / `stop()`, pushing nine
  `system/mps_*` scalars via `MlflowClient.log_metric`. It reads utilisation from an inlined IOKit
  `PerformanceStatistics` query (the no-sudo private API the `stats` menu-bar app uses); `torch` and
  `mlflow` are imported lazily.

A non-Lightning platform calls these directly; the Lightning `pl.Callback` wrappers live in
`matrainer.lightning.callbacks`. Tune the cadence with
`MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL` (seconds, default 10) or disable via
`MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false`.

## Per-split metric slots (`matrainer.metric_slots`, no extra needed)

A training runnable typically takes `metrics` (what **every** split measures) plus
`train_metrics` / `val_metrics` / `test_metrics` (what **that** split measures on top), so an
expensive or split-specific metric can be added to validation alone without restating the shared
set beside it. How the two combine is the same everywhere; *what a metric is* is not — so the rule
lives here as two pure functions and is applied to whichever vocabulary your engine uses (a
torchmetrics template mapping, a list of `keras.metrics` objects, a list of fastai function names).

```python
from matrainer.metric_slots import merge_metric_specs, union_metric_specs

# merge — an engine whose loop you own: each split gets its own set
merge_metric_specs({"acc": A}, {"confusion": C})   # {"acc": A, "confusion": C}   add
merge_metric_specs({"acc": A}, {"acc": A2})        # {"acc": A2}                  shadow
merge_metric_specs(["accuracy"], ["top_k"])        # ["accuracy", "top_k"]        concatenate
merge_metric_specs({"acc": A}, None)               # {"acc": A}   None = "not chosen"

# union — an engine that compiles ONE metric list for several phases (keras, transformers)
union_metric_specs(shared, train_spec, val_spec, test_spec)
```

Two mappings merge by key (a same-named entry in the split **shadows** the shared one, which is
how a split overrides a template rather than adding to it); two sequences concatenate; anything
else — a ready-made collection, a mapping against a list — means the split **replaces**, because
there is no key-wise way to combine them and silently ignoring the split's own spec would be worse.
`union_metric_specs` is for an engine that cannot be told to narrow (`keras.Model.compile` owns one
list): a metric asked for on one split is then computed on the others too, which is noise a reader
ignores — the alternative, dropping it, is a wired metric that silently reports nothing.

The module imports **nothing but the standard library**, so a deliberately framework-free training
loop can use it without pulling torch. That is also why it sits at the top level rather than under
`matrainer.torch`.

## Metrics a hand-written loop cannot average (`matrainer.metric_functions`, no extra needed)

If your engine owns the loop, you get stateful metrics for free — it accumulates across batches
and reduces once. If **you** own the loop you see one batch at a time, and the obvious move is to
score each batch and average. For accuracy that is very nearly right. For a macro-averaged F1 it
is not:

| | global macro-F1 | batch-averaged |
| --- | --- | --- |
| accuracy ≈ 0.9, 10 classes, batch 32 | 0.9182 | 0.8849 |
| accuracy ≈ 0.3, 10 classes, batch 32 | 0.3486 | 0.3032 |

Three to five points, always downward, and worse the more classes a batch is missing — a class
absent from a batch scores 0 for that batch and drags the mean. Accumulate a confusion matrix
instead: one `C×C` integer array whose size does not grow with the dataset.

```python
from matrainer.metric_functions import accumulate_confusion, macro_f1, zero_confusion

matrix = zero_confusion(num_classes)
for logits, targets in batches:            # logits are argmaxed; raw or softmaxed, same answer
    accumulate_confusion(matrix, logits, targets)
print(macro_f1(matrix))
```

`macro_f1` matches `sklearn.metrics.f1_score(..., average="macro", zero_division=0)` and
`torchmetrics.classification.MulticlassF1Score(average="macro")` exactly — which is the point: a
number from your loop is comparable with one from a framework that computed it for you. numpy is
the only import, so a deliberately framework-free loop can use it. Single-label targets only; a
multi-hot row has no single cell to land in.

## What a metric is CALLED (`matrainer.metric_names`, no extra needed)

Every tracker records the key it is handed, verbatim. So a run recorded under one engine and the
same run under another land in different columns: Keras writes `loss` / `val_loss`, fastai
`train_loss` / `valid_loss` plus **unprefixed validation** metrics, a `transformers.Trainer`
`train_*` / `eval_*`, a `LightningModule` whatever it logs. `MetricNames` translates between those
vocabularies, in either direction:

```python
from matrainer import MetricNames

names = MetricNames()                                     # default target: matrainer's own
names({"loss": 0.5, "val_loss": 0.4}, source="keras")     # {'train/loss': 0.5, 'val/loss': 0.4}
names({"accuracy": 0.9}, source="fastai")                 # {'val/accuracy': 0.9}  ← see below
names({"eval_accuracy": 0.9}, source="huggingface")       # {'val/accuracy': 0.9}

MetricNames(target="keras")({"val/loss": 0.4})            # {'val_loss': 0.4}  ...and back out
```

The canonical vocabulary is `<split>/<name>` with `split` one of `train` / `val` / `test` — what
`build_metric_collection(...).clone(prefix="val/")` already produces, and what TensorBoard and
ClearML group their curves on. `epoch` and `step` pass through untouched: they are the x-axis, not
a measurement. Translation is idempotent, so a loop that already names things canonically can
declare its real `source` and pass through unchanged.

**`source` is the caller's, `target` and `renames` are config's.** Only the code doing the
reporting knows which engine produced the names — and whether an unprefixed `accuracy` was
measured on train (Keras) or on validation (fastai's `Recorder` computes its metrics on the
validation pass and prefixes none of them). Everything a *user* chooses is on the object, so one
YAML key can reach several differently-sourced backends and mean the right thing in each:

```yaml
metric_names: !class:matrainer.MetricNames
  target: keras                       # record this run in Keras spelling instead
  renames: {top_k_accuracy: accuracy_top5}   # align a metric this module cannot know about
```

`renames` applies to the parsed **base** name, after the source's own aliases resolve
(`sparse_categorical_accuracy` → `accuracy`) and before the target's spelling is applied, so one
entry reads the same whatever the two vocabularies are.

Where it is already applied for you: `keras_tracking_callback` (`source="keras"`) and
`TrackerLogger` (`source="lightning"`) both take a `metric_names=` argument and default to the
canonical translation. A hand-written loop calls `as_metric_names(self.metric_names)` — one line
that turns an unwired slot, a live object or a deferred YAML marker into a usable translator.

What is **not** translated is what the engine hands its own callbacks: a
`keras.callbacks.EarlyStopping(monitor="val_loss")` must keep finding the name Keras gave it, so
only the copy on its way to a tracker is renamed.

## `matrainer.torch` — torch-only, Lightning-free helpers (`matrainer[torch]`)

- `load_lightning_checkpoint(model, path, strict=False)` — loads a Lightning `.ckpt` (strips the
  `model.` prefix), an Ultralytics `.pt` (pulls the live module's `state_dict`), or a plain
  state-dict; auto-retargets to a named child when keys match one. `strict=False` tolerates a
  re-headed `num_classes`.
- `build_metric_collection(metrics, **dims)` — normalize any metrics spec (a mapping, a list, a
  single metric, an already-built `MetricCollection`) into one `MetricCollection` the caller clones
  per split, flowing each template with the sizing kwarg (`num_classes=` / `num_labels=` /
  `data_range=`). The kwarg is applied **per template**, not blanket: a template only receives a
  dimension its constructor DECLARES, and never one it has already PINNED — so the injection is a
  floor a config can override, and a metric that takes no such argument can sit beside ones that do.
  (Three of torchmetrics' seven full-reference image metrics take `data_range`; the blanket form made
  wiring one of the other four break the whole collection.)
- `coerce_metric_value(value)` / `scalar_metrics(raw)` — JSON-safe coercion of a torchmetrics
  `compute()` result; `scalar_metrics` keeps only the scalars, so an `N×N` confusion matrix is
  dropped — those channels hold one number per name, and so is a `[C]` per-class vector. Report
  such a metric separately with `matrainer.tracking.render_matrix_as_table` /
  `render_vector_as_table` (below).
- `MetricPreset(category=…, task=…)` — instantiate every compatible torchmetrics metric in a
  category, for a config that wants "all the classification metrics" without naming them.
- `resolve_device(choice="auto")` + the closed `Device` literal (`"auto"`/`"cpu"`/`"cuda"`/`"mps"`)
  — where the run executes. `"auto"` prefers CUDA, then Apple's MPS, then CPU; anything else is
  taken at its word, so an operator who wrote `cpu` to compare against a GPU run gets CPU, and one
  who wrote `cuda` on a machine without it gets torch's own error instead of a silent fallback that
  would make the benchmark meaningless. A `Trainer` makes this choice for you; a hand-written loop
  has to make it itself, and every such loop was making it identically.

```python
from matrainer.torch import build_metric_collection, resolve_device, scalar_metrics

device = resolve_device("auto")                   # cuda -> mps -> cpu, whichever is present

base = build_metric_collection({"accuracy": LazyClass(MulticlassAccuracy)}, num_classes=10)
train_metrics, val_metrics = base.clone(prefix="train/"), base.clone(prefix="val/")
...
final = scalar_metrics(val_metrics.compute())     # {"val/accuracy": 0.94}
```

It all sits outside `matrainer.lightning` so a non-Lightning platform — an Ultralytics loop, or a
hand-written torch training loop — uses it via `matrainer[torch]` without pulling
`pytorch_lightning`. The metric helpers moved here on 2026-08-01 for exactly that reason: they
import only confluid, torch and torchmetrics, so their old home described their first consumer
rather than their dependencies. `matrainer.lightning` re-exports all four, unchanged.

### Reporting a metric that is not a single number

`scalar_metrics` drops what a `{name: float}` channel cannot hold. Renderers put such a value back,
and a channel gives each a home. There are **two shapes**, each with a table form and a figure form:

| shape | table (`log_text`) | figure (`log_figure`) |
|---|---|---|
| square `N×N` — a **confusion matrix** | `render_matrix_as_table` | `render_matrix_as_figure` (heatmap) |
| length-`C` — a **per-class metric** (`average=None`) | `render_vector_as_table` | `render_vector_as_figure` (bar chart) |

The table form needs only `tabulate` and is readable in a plain log, so it is what a consumer sends
by default; the figure form needs matplotlib (`matrainer[plot]`) on every eval epoch, so it is
opt-in. All four are framework-free — a value only needs `.tolist()`, or `.detach().cpu().tolist()`
— with their libraries imported lazily so the module stays in the light core.

**The two shapes get different renderings on purpose.** A per-class IoU is not a degenerate matrix:
it has one axis of meaning, its values share a scale, and what a reader wants from it is a *ranking*
— which class the model is worst at. A heatmap of a `1×C` strip answers that badly, so the vector
renderers sort **worst-first** and draw the mean as a reference line.

```python
from matrainer.tracking import (
    per_class_vectors, render_matrix_as_table, render_vector_as_figure, render_vector_as_table,
)

values = val_metrics.compute()          # scalars, vectors and matrices, together

table = render_matrix_as_table("val/confusion", values["val/confusion"], ["cat", "dog"])
tracker.log_text("val/confusion", table, step=epoch)

# `per_class_vectors` finds the VECTORS by SHAPE, so a runnable needs no naming convention
for payload in per_class_vectors(values, class_names=["pet", "background", "boundary"]):
    table = render_vector_as_table(payload["name"], payload["values"], payload["class_names"])
    logger.info(table)
    tracker.log_text(payload["name"], table, step=epoch)
    # opt in when a picture is worth it
    tracker.log_figure(
        payload["name"],
        render_vector_as_figure(payload["name"], payload["values"], payload["class_names"]),
        step=epoch,
    )
```

```
val/iou_per_class (mean 0.7509):

| class      |   iou_per_class |
|------------|-----------------|
| boundary   |          0.4827 |
| pet        |          0.8549 |
| background |          0.9150 |
```

`per_class_vectors` selects **by shape, not by metric name** (a flat numeric sequence of length ≥ 2):
a runnable emits scalars, vectors and matrices in one mapping with no knowledge of which is which,
so a name-based rule would need a convention nothing enforces. Scalars and matrices fall through,
which is what lets the same mapping be passed to both selection rules.

`render_matrix_as_figure` builds a `matplotlib.figure.Figure` **directly**, never through
`pyplot` — a pyplot figure is registered in a process-global manager and lives until someone
calls `pyplot.close`, so a value rendered once an epoch would leak a figure per epoch in a caller
with no reason to know pyplot has global state.

`log_text(name, text, step=None)` is the third content channel beside `log_metrics` (numbers)
and `log_figure` (images). Every backend has it: unlike `log_figure`, whose default is a skip,
its default is real work — a `.md` file written into the backend's own `log_dir` (or uploaded
via `log_artifact` when there is none), so a caller never has to ask whether this backend can
hold a table. Where a backend has a native channel it uses that INSTEAD:

| Backend | `log_text` lands in |
|---|---|
| MLflow | `client.log_text` → `text/<name>_<step>.md`, previewed inline in the UI |
| TensorBoard | `add_text` → the **Text** tab, with `step` driving the slider |
| Aim | `aim.Text` tracked as a sequence, so every step is kept |
| ClearML | an artifact named `text/<name>_<step>.md` — *not* `report_text`, which writes to the console stream where a per-epoch table is unusable |
| W&B | the inherited `.md` file, which W&B syncs from the run directory (it has no text primitive) |
| CSV | the inherited `.md` file under the run directory |

Note the contrast with `log_manifest`, where a native slot is written *in addition* to the
artifact: there the artifact is a union guarantee that keeps two differently-tracked runs
comparable, whereas here the text *is* the payload, so one durable home per backend is the whole
requirement. `step` is part of the filename rather than a version of one file, so a value logged
once an epoch keeps its history on a backend whose native channel has no step axis.

## Random run names

`matrainer.generate_run_name(prefix="train")` returns a friendly `adjective_noun` slug (`coolname`,
offline) — the default `Tracker.run_name`.

`run_prefix(task)` supplies the prefix a runnable's task implies, so the two compose into the one
line a merged train/evaluate/predict class writes:

```python
from matrainer import generate_run_name
from matrainer.run_names import run_prefix

generate_run_name(prefix=run_prefix("fit"))       # "train_funny_rabbit"
run_prefix("test"), run_prefix("evaluate")        # ("evaluate", "evaluate")
```

The mapping is deliberately many-to-one: `test` and `evaluate` are two entry points but one *kind*
of run, and a name that said `test_…` for one and `evaluate_…` for the other would split one
concept across two prefixes in every tracking backend. An unrecognised verb — including one a
future runnable adds — is an evaluation rather than an error, because a run name is a label and
refusing to produce one is never the right answer.

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

A package root keeps its core framework-free with the same library — the PEP 562 resolver is
`matrainer._exports.lazy_exports`; the name table stays the project's:

```python
_BACKEND_NAMES = {"LightningThing": ("myproject.lightning_thing", "lightning")}
__all__ = [*_BACKEND_NAMES]
__getattr__, __dir__ = lazy_exports(__name__, _BACKEND_NAMES)
```

Touching a name whose backend is missing raises an `ImportError` naming the exact extra; an
empty extra (`""`) marks a core name whose failure passes through untranslated.

## Running and reading a SWEEP (`matrainer sweep`, no extra needed)

A *sweep* is one config run under many backends and/or many trackers, so the runs become
comparable. A consuming project ships only its MATRIX as a declarative spec
(`examples/sweep/sweep.yaml`); the machinery — probing what is installed, one subprocess per
cell, experiment naming from the tracker, failure isolation, the report — is matrainer's:

```bash
matrainer sweep backends examples/sweep/sweep.yaml               # frameworks × models, one tracker
matrainer sweep trackers examples/sweep/sweep.yaml               # every tracker, one framework
matrainer sweep all      examples/sweep/sweep.yaml               # frameworks × models × trackers
matrainer sweep backends examples/sweep/sweep.yaml --dry-run     # print the commands, run nothing
#   … --only lightning,torch --epochs 1 --tracker csv            # the common narrowing flags
```

The spec names the confluid overlay every cell runs, the backend rows (with `requires:`
find_spec probes and per-row `env:` pins), the optional model-variant dimension, the trackers,
and any per-tracker setup command:

```yaml
config: all_backends.yaml
epochs: 5
frameworks:
  - {name: lightning, requires: [pytorch_lightning], models: [zoo, dncnn]}
  - {name: keras, requires: [keras, tensorflow], env: {KERAS_BACKEND: tensorflow}, models: [dncnn]}
models:
  - name: zoo                            # first = the config's default, appends no flag
  - {name: dncnn, flags: [--model, dncnn]}
trackers: [mlflow, csv]
tracker_setup: {mlflow: [aisland, mlflow, start]}
```

Each cell runs through the **generic runner** — `matrainer train <config> --framework F
--tracker T --experiment_name E --run_name R` — which works standalone too, on any
`runnable:`-rooted config whose class references are module-qualified:

```bash
matrainer train    path/to/train_config.yaml --framework torch --max_epochs 1
matrainer evaluate path/to/train_config.yaml
matrainer predict  path/to/train_config.yaml
```

The tracker-side halves keep their original surface (`python -m matrainer.sweep …` works
everywhere the console script does):

```bash
# name THIS run of the sweep: <prefix>_<N>, N one past the highest already recorded
matrainer sweep next-experiment mlflow "$(date '+%a_%b_%-d_%Y')"

# read it back — one row per run
matrainer sweep report mlflow Thu_Aug_6_2026_1
matrainer sweep report csv    Thu_Aug_6_2026_1 --save-dir ./sweep_logs --all
```

```
| run              | status   |  sec |   val/psnr |   val/mse |   train/loss |
|------------------|----------|------|------------|-----------|--------------|
| huggingface_zoo  | FINISHED |  249 |    27.4394 |    0.0018 |       0.0486 |
| lightning_dncnn  | FINISHED |   61 |    20.5510 |    0.0088 |       0.0791 |
```

**The count comes from the TRACKER, not a local file.** So a wiped tracker restarts at 1, and a
second checkout of the same project does not collide with the first — a counter file would keep
counting a history that is no longer there.

**Bookkeeping columns are folded away, and never silently.** A `transformers.Trainer` records its
timings, throughput and FLOP count in the same `log_history` as the loss; Lightning logs `epoch`.
All worth keeping in the tracker, all noise in a table whose job is to compare a dozen runs on the
same few numbers. The footer says how many went, *which*, and the command that brings them back:

```
11 runs in csv experiment 'Thu_Aug_6_2026_1'
4 columns hidden — bookkeeping, within-epoch series (epoch, train/loss_step, …) — see them with:
  python -m matrainer.sweep report csv Thu_Aug_6_2026_1 --all
```

Two reasons, named separately, because `train/loss_step` is a genuine result rather than
bookkeeping — Lightning's per-*batch* series of a quantity the table already shows per epoch, which
a one-row-per-run table cannot render and only one engine emits.

**Only `mlflow` and `csv` can be read from here.** A tracker is a write surface first: an offline
ClearML task is a directory waiting to be uploaded and W&B needs its server, so those say where to
open them instead of showing an empty table — which would read as "the sweep recorded nothing".

The table is the **union** of every metric any run reported, validation first. A metric one backend
computes and the others do not earns its own column with the rest blank; the alternative is a table
that silently drops whatever does not fit its widest complete row. Non-scalar values (a per-class
vector, a confusion matrix) never reach it at all — they cannot go through a `{name: float}` channel,
so a runnable sends them to the trackers' text channel and you read them in the tracker's own UI.

## What lives elsewhere

The **runnable protocol** and **pipeline running** are *not* in Matrainer — they moved to
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
  plus a `task:`; consumers may also ship their own CLI wrapping it.

There is no `matrainer.runner` / `matrainer.progress` / `matrainer.workflow` / `matrainer.processing`
and no matrainer binary.

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

Everything else — torch, pytorch-lightning, torchmetrics, mlflow, tensorboard, wandb, aim,
clearml, tensorflow, matplotlib — is optional and behind an extra.

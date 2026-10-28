# The run manifest

What produced a run — captured automatically, written as one document, and projected into every backend's native slots.

Runnable twin: [`examples/run_manifest.py`](../examples/run_manifest.py) — a manifest written by hand and read back, portable paths included.

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

`config/final.yaml` is **reload-faithful** for the torch stack — and it reloads in a **cold**
process: the dump names every class by its importable dotted path (`dump(qualified=True)`), so
`confluid.load("config/final.yaml")` reconstructs the runnable with no registrations and no prior
imports. `matrainer.torch` additionally registers document spellings (confluid's
`register_dump_spelling`) for the values a resolved run holds that YAML cannot natively carry — a small `torch.Tensor` (a class-weights vector, a loss's `weight`)
dumps as its values + dtype and reloads to the exact tensor, and a `torchmetrics.Metric` /
`MetricCollection` dumps as its configuration and reloads to an equivalent, computing object. A
tensor above 65 536 elements keeps the explicit placeholder instead — model-scale weights belong
to checkpoints, not YAML.

A mistyped section name raises rather than silently creating a key, and `set` on a list section (or
`add` on a mapping one) tells you which to use. `run_dir=None` derives
`runs/<experiment_name>/<run_name>`; the directory is created on first write, not at construction.

**Where `log_run_context` writes it.** Beside the first tracker's own files when that tracker keeps a
local folder (`log_dir`: CSV, TensorBoard, MLflow's local store). A tracker that only sends to a server
keeps none, and then the manifest goes into the runnable's own `output_dir` — the folder holding its
checkpoints — so one folder holds the whole run:

```
<output_dir>/
  checkpoint_best_total.pth  metrics.csv  …        # the trainer's own files
  manifest.yaml  manifest.json                      # the run manifest
  config/final.yaml  environment/requirements.txt   # its side files
  logs/<the run's log>                              # copied by capture_log_file
```

Without either, it is `runs/<experiment_name>/<run_name>`. `capture_log_file` also copies the run's log
file into `logs/` of that folder (the trackers still receive it as an artifact).

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
  cwd: $PROJECT_ROOT/my_project
  line: cd "$PROJECT_ROOT/my_project" && "$PROJECT_ROOT/.venv/bin/python" -m recordstream.cli run cfg.yaml
code:
- {name: confluid, path: $PROJECT_ROOT/confluid, commit: 3f9c0e09…, dirty: false}
datasets:
- {kind: file, name: $DATA_ROOT/imagefolder, uri: 'file://$DATA_ROOT/imagefolder'}
env:
  roots: {PROJECT_ROOT: /home/me/work, DATA_ROOT: /Volumes/Store}   # recorded verbatim
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

**MLflow renames the keys it would otherwise refuse.** A parameter, metric or tag *key* must hold
only `A-Za-z0-9`, `_`, `-`, `.`, `:`, ` ` or `/`, must read as a canonical path, and must fit in 250
characters. Names reaching a tracker come from a training framework's own log dict, from a user's tags
and from the flat configuration view the manifest's `params` section is built from, and none of those is
bound by MLflow's rules. So `MlflowTracker` maps every key through `mlflow_safe_name` first:

| logged | reaches MLflow as |
|---|---|
| `RFDETRDetector.train_set.Target.kwargs[source].Target.target` | `RFDETRDetector.train_set.Target.kwargs.source.Target.target` |
| `callbacks[0].patience` | `callbacks.0.patience` |
| `model<v2>`, `val/acc@1` | `model_v2_`, `val/acc_1` |
| `datasets[$DATA_ROOT/set/]` | `datasets._DATA_ROOT_set_` (not a canonical path, so its slashes go) |
| a key over 250 characters | its tail, behind an 8-character digest |
| `RFDETRDetector.epochs`, `a:b-c d/e_f` | unchanged — a key MLflow accepts is never renamed |

Without it a single bracketed key cost the run its **whole** `params` section: `log_hyperparams`
raised, `log_manifest` raised with it, and under `ResilientTracker` the manifest was dropped with
`Dropping log_manifest: …` at WARNING. Only this backend does it — the restriction is MLflow's own,
and every other backend records names as written. The function is public, so a tracker elsewhere that
speaks MLflow's dialect can hold itself to the same rule:

```python
from matrainer.tracking.mlflow_backend import mlflow_safe_name   # imports no mlflow
```

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

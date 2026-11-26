# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versioning is
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0a3] — 2026-11-26

### Added
- **`matrainer.tracking.ending_the_run(trackers, manifest)`** — a context manager around a run's work: work that
  finishes ends the run as `success`, work that raises ends it as `failed` (`interrupted` for a Ctrl-C) and the error
  is raised again. Both go through `capture_log_file`, so a failed run keeps its log file
  ([`docs/run-manifest.md`](https://github.com/Gearlux/matrainer/blob/main/docs/run-manifest.md)).

### Fixed
- **A run that raises is ended on its trackers** — the torch loop, fastai, Hugging Face, JAX and MLX trainers,
  `KerasRunnable` and `TensorFlowRunnable` ended a run only on their last line, so a training that raised stayed "running" on the tracker for
  ever, without its log file. They now end it as `failed` or `interrupted`.
- **A run that fails on a Mac ends** — the Apple-GPU sampler's thread is a daemon its exit hook stops and joins. As a
  non-daemon thread it kept the process alive for ever when a run raised before its tracker was finalized: Python
  joins such threads before it runs exit hooks.

### Changed
- **matrainer requires liquifai 0.4.0 and confluid 0.5.0** (`liquifai>=0.4.0`, was `>=0.3.0`; `confluid>=0.5.0`, was
  `>=0.4.1`): every verb passes `script_command(config_keys=...)`, new in liquifai 0.4.0, which itself requires
  confluid 0.5.0.
- **The licence is now the PolyForm Noncommercial License 1.0.0** (`LICENSE`, `pyproject.toml`): free for personal, academic and other noncommercial use; commercial use needs a separate licence from the author. It was MIT; the versions released under MIT keep that licence.
- **`<project> evaluate cfg.yaml --dump-config` shows the task the verb runs** — `task: test` (`predict` for
  `predict`), where it showed the config's own `task:`. Each verb adds `runnable.task` to the configuration when the
  runnable's class takes a `task`; a runnable without one gets nothing added, and the other verbs still refuse it.

### Documentation
- **Four docs pages gained the runnable example that proves them** — `examples/metrics.py` (the metric-spec
  rules, the batch-averaging error measured at ~5 points of macro-F1, the name translation),
  `examples/run_manifest.py` (a manifest written and read back, `$DATA_ROOT` contraction included),
  `examples/project_cli.py` (an app from a frameworks table, the `--framework` rewrite, the lazy package
  root) and `examples/model_package.py` (a package written `copy` and `reference`, read back, its weights in
  the slot). Those four run with no training framework; `examples/lightning_tracking.py` adds a real
  one-epoch fit showing a `ModelCheckpoint` with no `dirpath` landing under the tracker's `save_dir`
  rather than the launch directory, the logged metrics arriving, and a run that raises recorded as
  `failed`, and `examples/torch_helpers.py` loads a checkpoint in each of its three on-disk shapes, has a
  state dict refused by `TorchModelFile`, and shows `resolve_device('cpu')` giving CPU on a GPU machine.
  `examples/trainer_api_loop.py` and `examples/trainer_api_lightning.py` compose ONE task mixin onto two
  machinery bases, which is the four-layer claim made checkable: 19 constructor slots come from the shared
  layers, the loop base adds `checkpoint_dir` / `device` / `grad_clip`, and the Lightning base adds none.
  `examples/sweep.py` drives a four-row sweep — one backend absent, one cell dying — through to the folded
  comparison table, with the spec's `command:` key replacing the per-cell training with a stub, so the real
  driver, the failure isolation, the tracker-counted experiment name and the report all run in well under
  a second. CI runs them all, and nine of the ten pages link to a twin; `backends.md` says it has none by
  design, because comparing what each backend costs to run needs a server per backend.
- **The README is a quickstart and each topic is its own page.** It was 1339 lines; it is now install, a
  first run with its output, how to configure a tracker, a common-tasks table and an index — and the ten
  feature sections moved, unchanged, to `docs/`: `tracking.md`, `backends.md`, `run-manifest.md`,
  `trainer-api.md`, `lightning.md`, `metrics.md`, `torch.md`, `model-packages.md`, `cli.md`, `sweeps.md`.
  Nothing was dropped, so a link into a README section now goes to that topic's page.

### Tracking
- **The MLflow tracking server is a workspace service of this package** (`aisland mlflow start`), beside
  Aim and ClearML: `mlflow server` over a SQLite store and an artifact folder in the service's data folder,
  on port 5005, ready once `/health` answers. Without `mlflow` installed it names the command that adds it to
  the `tracker` selection.
- **`MlflowTracker` logs parameter, metric and tag names MLflow accepts** — every key goes through the new
  public `matrainer.tracking.mlflow_backend.mlflow_safe_name`: `[key]` becomes `.key` (so the config walk's
  `kwargs[source].Target.target` lands as `kwargs.source.Target.target` and `callbacks[0].patience` as
  `callbacks.0.patience`), any other character outside `[A-Za-z0-9_\-. :/]` becomes `_`, a name that is not
  canonical as a path loses its slashes, and one over 250 characters keeps its tail behind a digest. A name
  MLflow already accepts is unchanged. Previously one bracketed key made `log_hyperparams` — and the
  `log_manifest` that projects it — raise, so the run lost its whole `params` section.
- **A run tracked only on a server keeps its manifest beside its checkpoints**: when the first tracker keeps
  no local folder, `log_run_context` writes `manifest.yaml`, `config/final.yaml` and
  `environment/requirements.txt` into the runnable's `output_dir` instead of `runs/<experiment>/<run>`.
- **`capture_log_file` copies the run's log into the run folder's `logs/`**, beside the manifest.

## [0.1.0a2] — 2026-10-15

### Trainers
- **`mps_low_watermark`** on every torch trainer (`TorchTrainerRunnable`: the Lightning, owned-loop, fastai and
  Hugging Face bases) and **`matrainer.torch.apply_mps_low_watermark`** — on Apple's GPU, torch gives its cached
  memory back above that share of the GPU's working set (`PYTORCH_MPS_LOW_WATERMARK_RATIO`, set before the model is
  built). A config or the command line sets it (`runnable.mps_low_watermark=0.5`); unset leaves torch's default.

### Tracking
- **`ResilientTracker`** — wraps one tracker so a slow or unreachable tracking server never stops a
  run or loses a value: a time limit per call (an artifact's grows with its size), a call the
  network lost queued and replayed in order, a queued artifact's bytes kept (and backed up under
  `runs/<experiment>/<run>/pending_artifacts/`) so a deleted file still uploads, a 5xx turning off
  only that kind of call (the status read from the error, never a 500 in its message), and a run
  that cannot start turning tracking off instead of stopping training. It turns the wrapped
  tracker's `raise_errors` on, so a failed MLflow, W&B, Aim or ClearML call is queued and replayed.
- **`raise_errors`** on every tracker, off by default: a failed backend call is logged at debug
  level and the run goes on, as before; on, the error is raised as well. A call that sends several
  values tries every one and raises the first failure; `log_manifest` sends every part before it
  raises.

### Model packages
- **`matrainer.model_package`** — a model's whole inference graph (a flow graph: preprocessing, the
  predictor, postprocessing), its weights and its class names in one folder. `PackageModel` (a
  runnable) writes one, copying or referencing the graph files and the weights files separately;
  `package.yaml` lists every file with its SHA-256 and the installed versions of the graph's
  distributions. `ModelPackage` (an op) runs a package as one step, with `overrides` into its graph
  applied last. `read_package` returns the package file as plain data. `load_package_graph` loads a
  package's graph the one way the op runs it — for another reader of packages, such as a compiler —
  and returns a `LoadedPackage` holding the document it gave `confluid.load`, so another stage of the
  same document can be loaded.

### Dependencies
- `recordstream>=0.1.0a3` (core, `[keras]`, `[jax]`) — the first release with the flow subgraph the
  model packages run; `confluid>=0.4.1`, which that recordstream requires.

## [0.1.0a1] — 2026-09-27

First public pre-release. The surface it ships:

### Tracking
- **`Tracker`** — one abstract base mirroring the PyTorch Lightning logger shape
  (`log_hyperparams`, `log_metrics`, `save`, `finalize`) plus `set_tags`, `log_artifact`,
  `log_figure`, `log_text` and `log_manifest`. `TrackerGroup` fans a call out to several.
- **Six backends**, each in its own extra and imported lazily, so `import matrainer` loads none
  of their libraries: `CsvTracker` (standard library only, always available), `MlflowTracker`
  (`[mlflow]`), `TensorBoardTracker` (`[tensorboard]`), `WandbTracker` (`[wandb]`), `AimTracker`
  (`[aim]`) and `ClearMLTracker` (`[clearml]`). A missing library raises one error naming the
  extra to install. Backends are discovered from the package, so a new one is one new file.
- **The run manifest** (`RunManifest`, `log_run_context`) — one record per run holding everything
  known about it: the command line to rerun it, every editable package's git commit, the installed
  packages, the resolved config, the datasets and models it used (with size and SHA-256 for local
  files), and its terminal status. Written as YAML and JSON after every change, so an interrupted
  run still leaves a complete record. Every backend receives it as an artifact; backends with a
  native home for a fact (MLflow dataset and model inputs, W&B artifacts, ClearML input models,
  Aim run values) also get it there. Secrets in environment variables are redacted. Paths under
  `$PROJECT_ROOT` / `$DATA_ROOT` / `$MODEL_ROOT` are recorded relative to them.
- **Provenance** — git commit (with `-dirty`), the config files a run included, and the log file,
  uploaded to every tracker; `capture_log_file` ends the run on every tracker.
- **System metrics** (`matrainer.tracking.system`) — CPU, memory and disk through MLflow, and an
  Apple-Silicon GPU sampler that needs no `sudo`.
- **Rendering** — confusion matrices and per-class vectors as markdown tables or figures
  (`[plot]` for the figures).

### Metrics
- `MetricNames` translates metric keys between engine vocabularies (`val_loss`, `valid_loss`,
  `eval_loss`, `val/loss`), so one quantity lands in one column whichever engine produced it.
- `matrainer.metric_slots` merges per-split metric choices; `matrainer.metric_functions` computes
  macro-F1, accuracy and IoU from a confusion matrix accumulated over a whole split, for loops
  that see one batch at a time. Both are standard-library/numpy only.

### Trainer API
- `TrainerRunnable` — the framework-neutral base of a train / evaluate / test / predict runnable.
- Per-framework bases behind extras: `LightningRunnable` (`[lightning]`), `TorchLoopRunnable`
  (`[torch]`), `FastaiRunnable` (`[fastai]`), `KerasRunnable` (`[keras]`), `TensorFlowRunnable`
  (`[tensorflow]`), `HuggingFaceRunnable` (`[huggingface]`), `JaxLoopRunnable` (`[jax]`) and
  `MlxLoopRunnable` (`[mlx]`).
- Framework-neutral checkpoint policies (`SaveLast`, `SaveEveryEpoch`, `SaveBest`) and their
  Lightning translation.
- `matrainer.torch` — `load_lightning_checkpoint` (Lightning `.ckpt`, Ultralytics `.pt`, plain
  state dict), `TorchModelFile` (a whole pickled model as a model slot), `resolve_device`, and
  the torchmetrics helpers.
- Random human-readable run names (`generate_run_name`).

### Command line
- `matrainer.cli` builds a project's trainer CLI (`train` / `evaluate` / `predict`,
  `--framework`) from a table of frameworks.
- The `matrainer` console script: the same three verbs over any runnable config, plus
  `matrainer sweep backends | trackers | all <sweep.yaml>` to run one config under every backend
  and tracker and report the results side by side.

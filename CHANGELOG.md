# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versioning is
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Tracking
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

# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versioning is
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Model packages
- **`matrainer.model_package`** — a model's whole inference graph (a flow graph: preprocessing, the
  predictor, postprocessing), its weights and its class names in one folder. `PackageModel` (a
  runnable) writes one, copying or referencing the graph files and the weights files separately;
  `package.yaml` lists every file with its SHA-256 and the installed versions of the graph's
  distributions. `ModelPackage` (an op) runs a package as one step, with `overrides` into its graph
  applied last. `read_package` returns the package file as plain data.

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

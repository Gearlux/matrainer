# Marainer

Project-agnostic training and processing pipeline orchestrator built on PyTorch Lightning, Confluid, and DataFlux.

Marainer provides a fully YAML-configurable training loop with multi-backend logging (TensorBoard, CSV, MLflow), flexible data pipelines (DataFlux), and a comprehensive metrics system (torchmetrics).

## Installation

```bash
pip install git+https://github.com/Gearlux/marainer.git@main
```

Tab completion: `marainer --install-completion` (bash/zsh/fish auto-detected from `$SHELL`).

### Development

```bash
git clone https://github.com/Gearlux/marainer.git
cd marainer
pip install -e ".[dev]"
```

## Quick Start

### 1. Define a Model

Any `nn.Module` works. Use `@configurable` to make constructor parameters controllable from YAML:

```python
# my_project/model.py
import torch
import torch.nn as nn
from confluid import configurable

@configurable
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10, input_channels: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))
```

### 2. Write a YAML Config

```yaml
# train.yaml
experiment_name: mnist_digits
run_name: baseline

num_classes: 10

model: !class:my_project.model.SimpleCNN()
  input_channels: 1

optimizer: !lazy:torch.optim.Adam
  lr: 0.001

metrics:
  - !class:torchmetrics.classification.MulticlassAccuracy()
  - !class:torchmetrics.classification.MulticlassF1Score()

# !clone: creates independent deep copies (torchmetrics requires separate state)
train_metrics: !class:torchmetrics.MetricCollection()
  metrics: !clone:metrics
  prefix: "train/"

val_metrics: !class:torchmetrics.MetricCollection()
  metrics: !clone:metrics
  prefix: "val/"

train_set: !class:dataflux.core.Flux()
  source: !class:dataflux.sources.HuggingFaceSource()
    path: mnist
    split: train
    input_feature: image
    target_feature: label
  ops:
    - !class:dataflux.ops.ToTensorOp()
      normalize: true

val_set: !class:dataflux.core.Flux()
  source: !class:dataflux.sources.HuggingFaceSource()
    path: mnist
    split: test
    input_feature: image
    target_feature: label
  ops:
    - !class:dataflux.ops.ToTensorOp()
      normalize: true

max_epochs: 10
batch_size: 32
num_workers: 4
```

### 3. Train

```bash
marainer train train.yaml
```

Marainer will:
- Instantiate the model, optimizer, data pipelines, and metrics from YAML
- Create separate train and validation metric collections
- Run the training loop via PyTorch Lightning
- Log hyperparameters (config params, git hash, config script) and metrics to all configured loggers

---

## The `train` Command

The `train` command is Marainer's primary entry point. It orchestrates the full lifecycle of a training run.

### Usage

```bash
marainer train <config.yaml>
```

### What Happens

1. **Config loading** -- Confluid parses the YAML file, resolving `!class:` tags, `!ref:` references, environment variables (`${VAR}`), and scope overrides.

2. **Trainer construction** -- A `Trainer` instance is created with all parameters from the YAML. Confluid's broadcasting pushes root-level scalars (like `num_classes`, `batch_size`, `max_epochs`) into matching constructor parameters across all configured objects.

3. **Logger setup** -- Configured logging backends (TensorBoard, CSV, MLflow) are attached to the Lightning Trainer.

4. **Hyperparameter logging** -- All configurable parameters, the git commit hash, and the config script path are logged to every attached logger.

5. **Training execution** -- `Trainer.run()` materializes all deferred objects and starts the Lightning training loop.

### Configuration Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `experiment_name` | `str` | `"marainer_run"` | MLflow experiment name |
| `run_name` | `str` | `"train"` | MLflow run name |
| `model` | `!class:` | *required* | Neural network model |
| `optimizer` | `!class:` | `Adam(lr=1e-3)` | Optimizer |
| `loss_fn` | `!class:` | `CrossEntropyLoss()` | Loss function (post-construction) |
| `train_metrics` | `!class:` | `None` | Train MetricCollection (post-construction) |
| `val_metrics` | `!class:` | `None` | Validation MetricCollection (post-construction) |
| `test_metrics` | `!class:` | `None` | Test MetricCollection (post-construction) |
| `train_set` | `!class:` | *required* | Training dataset (Flux pipeline or Dataset) |
| `val_set` | `!class:` | `None` | Validation dataset (used during `fit`) |
| `test_set` | `!class:` | `None` | Held-out test dataset. When set, `lightning_trainer.test()` runs after `fit()` |
| `train_loader` | `!class:` | `DataLoader` | Training DataLoader config |
| `val_loader` | `!class:` | `DataLoader` | Validation DataLoader config |
| `test_loader` | `!class:` | `DataLoader` | Test DataLoader config |
| `loggers` | `list[!class:]` | `None` | Logging backends (TensorBoard, CSV, MLflow) |
| `lightning` | `!class:` | `L.Trainer` | PyTorch Lightning Trainer config |
| `batch_size` | `int` | -- | Broadcast to DataLoaders |
| `num_workers` | `int` | -- | Broadcast to DataLoaders |
| `max_epochs` | `int` | -- | Broadcast to Lightning Trainer |

### Dataset-derived model dimensions (`!lazy:`)

The trainer is task-agnostic but supports deriving model dimensions from the
dataset at run time. Declare the `model` (and/or metrics) as `!lazy:` and omit a
dimension the dataset can supply; `LightningTrainer.run()` flows the dataset
first, reads its `derived_dimensions() -> dict` hook (a duck-typed protocol — a
classification dataset returns `{"num_classes": N}` by walking its targets), and
injects only the **unset** keys when it constructs those deferred nodes:

```yaml
model: !lazy:sonair.timm_classifier.TimmModel   # num_classes omitted
  model_name: efficientvit_b0
val_metrics: !lazy:torchmetrics.MetricCollection
  metrics:
    top1: !class:torchmetrics.classification.MulticlassAccuracy { top_k: 1 }
train_set: !class:dataflux.sources.HuggingFaceSource  # provides derived_dimensions()
  path: ylecun/mnist
  split: train
```

marainer never names a specific dimension (no hardcoded `num_classes`) — keys
come entirely from the dataset, and an explicitly-set value always wins over a
derived one. Datasets that don't implement `derived_dimensions()` (e.g.
detection/segmentation today) are untouched. This is the optimizer pattern
(`flow(self.optimizer, params=self.parameters())`) generalised; see
`marainer.dimensions`.

### Configurable Loss Function

The loss function defaults to `nn.CrossEntropyLoss()` but can be overridden in YAML via Confluid's post-construction attribute setting:

```yaml
# Use a different loss function
loss_fn: !class:torch.nn.MSELoss()

# Or with parameters
loss_fn: !class:torch.nn.CrossEntropyLoss()
  label_smoothing: 0.1
```

Any `torch.nn` loss module works. No code changes required.

### Configurable Optimizer

```yaml
# SGD with momentum
optimizer: !lazy:torch.optim.SGD
  lr: 0.01
  momentum: 0.9

# AdamW
optimizer: !lazy:torch.optim.AdamW
  lr: 0.001
  weight_decay: 0.01
```

> **Use `!lazy:` (not `!class:`) for `optimizer`, `train_loader` / `val_loader` / `test_loader`, and `lightning`.** The trainer holds these as deferred body slots and builds them itself at run time — the optimizer with `params=model.parameters()`, the loaders with `dataset=…`, the Lightning Trainer after auto-`ModelCheckpoint` / logger wiring. A bare `!class:` value for one of these is materialized eagerly on assignment (an optimizer built with no `params` would crash); Marainer auto-defers a misconfigured `!class:` here and logs a warning, but wire it `!lazy:` to be explicit.

### Lightning Trainer Configuration

Pass any `pytorch_lightning.Trainer` argument:

```yaml
lightning: !lazy:pytorch_lightning.Trainer
  max_epochs: 50
  accelerator: gpu
  devices: 1
  precision: 16-mixed
  gradient_clip_val: 1.0
```

Or use root-level broadcasting for common parameters:

```yaml
max_epochs: 50       # broadcast into Lightning Trainer
batch_size: 64       # broadcast into DataLoaders
num_workers: 8       # broadcast into DataLoaders
```

---

## The `process` / `download` / `convert` Commands — `DatasetProcessor`

For non-training workflows — converting datasets, windowing signals, running a
pure DataFlux pipeline from a source to a sink — Marainer ships
`marainer.processing.DatasetProcessor`. It's the generic `processor:` object the
`process`, `download`, and `convert` verbs dispatch via `.run()` (the three
verbs are semantic aliases; they all bind the object under the top-level
`processor:` key).

```yaml
# convert.yaml — run with: marainer convert convert.yaml
import:
  - dataflux.core
  - dataflux.storage.hdf5

processor: !class:marainer.processing.DatasetProcessor
  show_progress: true            # optional rich.progress bar
  flux: !class:Flux
    source: !class:SomeSource { ... }
    ops: [ ... ]
  sink: !class:HDF5Sink
    path: ./out.h5
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `flux` | `dataflux.core.Flux` | — | The pipeline (source + ops) to execute. |
| `sink` | sink \| `None` | `None` | Terminal sink; when absent, samples are materialized to an in-memory list. |
| `show_progress` | `bool` | `False` | Wrap iteration in a `rich.progress` bar (total from `len(flux.source)` when available). |
| `progress_desc` | `str` | `"DatasetProcessor"` | Progress-bar label. |

> `DatasetProcessor` lives in Marainer (not a domain package) because it is the
> workspace's generic data-pipeline runner and depends only on DataFlux +
> Confluid. It was moved here from `waivefront.processing`.

---

## Metrics System

Marainer integrates with [torchmetrics](https://torchmetrics.readthedocs.io/) to provide flexible, YAML-configurable metrics. Metrics are automatically:

- **Split** into independent train and validation collections (required by torchmetrics for correct state tracking)
- **Prefixed** with `train/` and `val/` for unambiguous logging
- **Logged** to the Lightning progress bar and all configured loggers at epoch end
- **Moved** to the correct device (CPU/GPU) by Lightning

### Approach 1: Explicit Metrics

Define metrics as a list and use `!clone:` to create independent train/val collections:

```yaml
num_classes: 10   # broadcast to all metrics

metrics:
  - !class:torchmetrics.classification.MulticlassAccuracy()
  - !class:torchmetrics.classification.MulticlassF1Score()
      average: weighted
  - !class:torchmetrics.classification.MulticlassPrecision()
  - !class:torchmetrics.classification.MulticlassRecall()

train_metrics: !class:torchmetrics.MetricCollection()
  metrics: !clone:metrics
  prefix: "train/"

val_metrics: !class:torchmetrics.MetricCollection()
  metrics: !clone:metrics
  prefix: "val/"
```

Shared parameters like `num_classes` are **broadcast** from the YAML root into each metric's constructor automatically by Confluid. You can override per-metric (e.g., `average: weighted` above overrides the default).

The `!clone:` tag deep-copies the referenced `metrics` list, giving each MetricCollection independent metric instances (required by torchmetrics for correct state tracking).

### Approach 2: Inline MetricCollection with !clone:

Define a single MetricCollection and clone it with different prefixes:

```yaml
num_classes: 10

metrics: !class:torchmetrics.MetricCollection()
  metrics:
    - !class:torchmetrics.classification.MulticlassAccuracy()
    - !class:torchmetrics.classification.MulticlassF1Score()

train_metrics: !clone:metrics
  prefix: "train/"

val_metrics: !clone:metrics
  prefix: "val/"
```

### MetricPreset Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `category` | `str` | `"classification"` | torchmetrics sub-module: `classification`, `regression`, `segmentation`, `detection`, `audio`, `image`, `text`, `clustering`, `retrieval`, `nominal`, `video`, `multimodal` |
| `task` | `str` | `""` (all) | Task variant filter. For classification: `"binary"`, `"multiclass"`, `"multilabel"`. Filters by class name prefix. Empty string loads all variants. |
| `exclude` | `list[str]` | `None` | Metric class names to skip (merged with built-in non-scalar exclusions) |
| `num_classes` | `int` | `None` | Number of classes (passed to metrics that accept it) |
| `num_labels` | `int` | `None` | Number of labels (for multilabel tasks) |
| `average` | `str` | `"macro"` | Averaging strategy: `"macro"`, `"micro"`, `"weighted"` |

Only parameters that a metric's `__init__` explicitly accepts are passed. Metrics whose required parameters cannot be satisfied are skipped with a warning.

### Task-Specific Examples

All examples below use the `!clone:` pattern. Define metrics once, then create independent train/val collections.

#### Binary Classification

```yaml
metrics:
  - !class:torchmetrics.classification.BinaryAccuracy()
  - !class:torchmetrics.classification.BinaryF1Score()
  - !class:torchmetrics.classification.BinaryAUROC()

train_metrics: !class:torchmetrics.MetricCollection()
  metrics: !clone:metrics
  prefix: "train/"

val_metrics: !class:torchmetrics.MetricCollection()
  metrics: !clone:metrics
  prefix: "val/"
```

#### Multiclass Classification (MNIST, CIFAR, etc.)

```yaml
num_classes: 10

metrics:
  - !class:torchmetrics.classification.MulticlassAccuracy()
  - !class:torchmetrics.classification.MulticlassF1Score()
  - !class:torchmetrics.classification.MulticlassPrecision()

train_metrics: !class:torchmetrics.MetricCollection()
  metrics: !clone:metrics
  prefix: "train/"

val_metrics: !class:torchmetrics.MetricCollection()
  metrics: !clone:metrics
  prefix: "val/"
```

#### Regression

```yaml
metrics:
  - !class:torchmetrics.regression.MeanSquaredError()
  - !class:torchmetrics.regression.MeanAbsoluteError()
  - !class:torchmetrics.regression.R2Score()

train_metrics: !class:torchmetrics.MetricCollection()
  metrics: !clone:metrics
  prefix: "train/"

val_metrics: !class:torchmetrics.MetricCollection()
  metrics: !clone:metrics
  prefix: "val/"
```

### Log Output

With metrics configured, the Lightning progress bar and MLflow will show:

```
Epoch 2: 100%|####| 63/63 [00:05] train/loss=0.234 val/loss=0.312
                                   train/MulticlassAccuracy=0.931
                                   val/MulticlassAccuracy=0.918
                                   train/MulticlassF1Score=0.928
                                   val/MulticlassF1Score=0.915
```

### Limitations

- **Object detection metrics** (e.g., `MeanAveragePrecision`) expect dict-based inputs and are not compatible with the standard `(preds, target)` call pattern. Use a custom Trainer subclass for detection tasks.
- **Non-scalar metrics** (ConfusionMatrix, ROC, PrecisionRecallCurve) are excluded by default from presets because `self.log_dict()` requires scalar values. You can still use them with explicit metrics if you override `training_step`/`validation_step`.

---

## Confluid Integration

Marainer leverages [Confluid](https://github.com/Gearlux/confluid) for all configuration and dependency injection.

### Key Concepts

- **`!class:module.Class()`** -- Instantiates a class immediately (Instance). Parameters are indented below.
- **`!class:module.Class`** -- Deferred class. Stays as a `Class` object until explicitly `flow()`'d at runtime.
- **`!clone:ref`** -- Deep-copies a referenced value, producing independent instances. Essential for torchmetrics which requires separate state for train vs validation.
- **Broadcasting** -- Root-level scalar values are automatically injected into any class constructor that accepts a matching parameter name. This is how `num_classes: 10` reaches both the model and all metrics without repetition.
- **`@configurable`** -- Decorator that enables post-construction attribute setting. Any YAML key that doesn't match a constructor parameter is set as an attribute (e.g., `loss_fn`, `train_metrics`, `val_metrics`).

### How Broadcasting Works

```yaml
num_classes: 10        # Root-level scalar
batch_size: 32         # Root-level scalar

model: !class:my_project.SimpleCNN()    # __init__ accepts num_classes -> gets 10
  input_channels: 1                      # Explicit param (not broadcast)

metrics:
  - !class:torchmetrics.classification.MulticlassAccuracy()  # accepts num_classes -> gets 10
```

Explicit values always take precedence over broadcast values.

---

## DataFlux Integration

Training and validation datasets are defined as [DataFlux](https://github.com/Gearlux/dataflux) `Flux` pipelines:

```yaml
train_set: !class:dataflux.core.Flux()
  source: !class:dataflux.sources.HuggingFaceSource()
    path: mnist
    split: train
    input_feature: image
    target_feature: label
    count: 1000                    # Optional: limit samples
  ops:
    - !class:dataflux.ops.numpy.RescaleOp()
      in_min: 0
      in_max: 255
    - !class:dataflux.ops.ToTensorOp()
      normalize: false
```

DataFlux pipelines support functional composition with operations like normalization, tensor conversion, augmentation, and more. See the DataFlux documentation for available sources and operations.

### Train / Val / Test Splits

Marainer treats `val_set` and `test_set` as distinct signals: `val_set` is used per-epoch during `fit()` for model selection, and `test_set` is evaluated once via `lightning_trainer.test()` after `fit()` completes.

The recommended pattern carves the training split of the source into a disjoint train/val view via DataFlux's `DatasetSplit`, and reserves the source's own test split for `test_set`:

```yaml
hf_train: !class:dataflux.sources.HuggingFaceSource()
  path: mnist
  split: train
  input_feature: image
  target_feature: label

hf_test: !class:dataflux.sources.HuggingFaceSource()
  path: mnist
  split: test
  input_feature: image
  target_feature: label

train_set: !class:dataflux.core.Flux()
  source: !class:dataflux.sources.DatasetSplit()
    source: !ref:hf_train
    split: train
    val_fraction: 0.1
    seed: 42

val_set: !class:dataflux.core.Flux()
  source: !class:dataflux.sources.DatasetSplit()
    source: !ref:hf_train
    split: val
    val_fraction: 0.1
    seed: 42

test_set: !class:dataflux.core.Flux()
  source: !ref:hf_test
```

`DatasetSplit` also supports explicit `start`/`end` ranges and works with any indexable source. See DataFlux's README for the full reference and the HuggingFace native slicing alternative.

---

## Logging

Marainer supports multiple logging backends via PyTorch Lightning's logger system. Configure any combination of TensorBoard, CSV, and MLflow loggers under the `lightning:` block:

```yaml
lightning: !lazy:pytorch_lightning.Trainer
  max_epochs: 10
  accelerator: auto
  logger:
    - !class:pytorch_lightning.loggers.TensorBoardLogger()
        save_dir: "."
        name: "tb_logs"
    - !class:pytorch_lightning.loggers.CSVLogger()
        save_dir: "."
        name: "csv_logs"
    - !class:pytorch_lightning.loggers.MLFlowLogger()
        experiment_name: !ref:experiment_name
        run_name: !ref:run_name
```

> **Important:** `logger:` lives **inside** the `lightning:` block — this is the single kwarg that PyTorch Lightning's `Trainer(logger=…)` constructor consumes. A top-level `loggers:` key on the marainer trainer is a no-op (no such kwarg exists; Lightning silently falls back to its default `./lightning_logs/`). The contract is pinned by `tests/test_loggers.py::TestYamlLoggers::test_yaml_loggers_reach_lightning_trainer`.

All configured loggers automatically receive:

- **Hyperparameters** -- All configurable parameters from the YAML config
- **Git hash** -- Current commit hash for reproducibility
- **Config script** -- Path to the YAML config file used
- **YAML config files** -- The full include tree (main YAML + every transitively `include:`-d file) is copied/uploaded as `config/<name>.yaml` to every logger so a run is reproducible from its artifacts alone
- **Metrics** -- Loss and all configured torchmetrics logged per epoch via `self.log()`

When `logger:` is omitted, Lightning uses its default logger (TensorBoardLogger writing to `./lightning_logs/version_N/`).

### Default checkpointing

Marainer auto-injects a `ModelCheckpoint` callback before training starts (so a fresh YAML always produces something to load from):

- `save_last=True` → writes `last.ckpt` after every epoch
- `save_top_k=1` → keeps the single best checkpoint by the first val metric
- `monitor` / `mode` resolved in this order:
    1. First key in `val_metrics.metrics` (its `higher_is_better` attribute decides `mode`)
    2. `val/loss` (`mode=min`) when no val metrics are wired but a `val_set` is present
    3. `train/loss` (`mode=min`) when no `val_set` is wired

To opt out, supply your own `ModelCheckpoint` (any `monitor`) under `lightning.callbacks` or set `enable_checkpointing: false` on the Lightning trainer. Either choice is respected; no second callback is added.

### Per-logger archive policy (`log_all_checkpoints`)

Each Lightning logger can independently opt into archiving every checkpoint by using the matching subclass from `marainer.loggers`:

```yaml
lightning: !lazy:pytorch_lightning.Trainer
  max_epochs: 10
  logger:
    - !class:marainer.loggers.MarainerMLFlowLogger
        tracking_uri: file:./mlruns
        experiment_name: !ref:experiment_name
        log_all_checkpoints: true     # MLflow stores every epoch's checkpoint
    - !class:marainer.loggers.MarainerTensorBoardLogger
        save_dir: "."
        name: "tb_logs"
        # log_all_checkpoints omitted → defaults to false (last + best only)
```

When **any** wired logger has `log_all_checkpoints: true`, marainer escalates the auto-injected `ModelCheckpoint` to `save_top_k=-1` so every checkpoint is persisted to disk. For MLflow specifically, the flag also propagates into Lightning's own `log_model="all"` so checkpoints are uploaded as MLflow model artifacts. Sticking with `!class:pytorch_lightning.loggers.TensorBoardLogger` (no flag) still works — the trainer treats it as `log_all_checkpoints=False`.

### TensorBoard

```yaml
lightning: !lazy:pytorch_lightning.Trainer
  max_epochs: 10
  logger:
    - !class:pytorch_lightning.loggers.TensorBoardLogger()
        save_dir: "."
        name: "tb_logs"
```

View results: `tensorboard --logdir tb_logs`

### CSV

```yaml
lightning: !lazy:pytorch_lightning.Trainer
  max_epochs: 10
  logger:
    - !class:pytorch_lightning.loggers.CSVLogger()
        save_dir: "."
        name: "csv_logs"
```

Metrics are written to `csv_logs/<name>/version_N/metrics.csv`, hyperparameters to `hparams.yaml`.

### MLflow

```yaml
experiment_name: mnist_experiment
run_name: baseline_v2

lightning: !lazy:pytorch_lightning.Trainer
  max_epochs: 10
  logger:
    - !class:pytorch_lightning.loggers.MLFlowLogger()
        experiment_name: !ref:experiment_name
        run_name: !ref:run_name
```

Use `!ref:` to reuse root-level `experiment_name` and `run_name` values. View results: `mlflow ui`

### System metrics

When an `MLFlowLogger` is wired, `LightningTrainer.run()` automatically
attaches an MLflow `SystemMetricsMonitor` to the Lightning logger's run
via a callback (`MlflowSystemMetricsCallback`). Without this, the global
`mlflow.system_metrics.enable_system_metrics_logging()` flag is silently
ineffective for Lightning runs — `pytorch_lightning.loggers.MLFlowLogger`
uses `MlflowClient.create_run()` directly and bypasses the fluent
`mlflow.start_run()` hook the global flag relies on.

The monitor records CPU / RAM / disk usage via `psutil` at the configured
sampling interval (default 10 s). NVIDIA GPU metrics are picked up
automatically when `pynvml` is installed.

On Apple-Silicon hosts the trainer additionally attaches an
`MPSMetricsCallback` that runs a background daemon thread between
`on_fit_start` and `on_fit_end` (and the matching `test_*` hooks),
sampling MPS GPU stats every `sample_interval_s` seconds and pushing
them straight to MLflow via `MlflowClient.log_metric`. The cadence is
decoupled from training-step pacing so it matches MLflow's built-in
CPU / RAM / disk monitor.

Three per-process scalars come from `torch.mps`:

- `system/mps_allocated_mb` — current process allocation
- `system/mps_driver_mb` — driver-side allocation (includes caches)
- `system/mps_recommended_max_mb` — Apple's soft cap

The other six come from an inline IOKit reader
(`marainer.system_metrics._read_mps_performance_statistics`) which
queries the `IOAccelerator` service's `PerformanceStatistics` dict
via `pyobjc` — the same no-sudo, private-API source used by the
`stats` macOS menu-bar app and `macmon`:

- `system/mps_device_util_pct` — overall GPU busy %
- `system/mps_renderer_util_pct` — render-stage busy %
- `system/mps_tiler_util_pct` — tile-binning-stage busy %
- `system/mps_in_use_system_memory_mb` — system-level GPU memory in use
- `system/mps_in_use_system_memory_driver_mb` — driver-side share of the above
- `system/mps_alloc_system_memory_mb` — total system memory reserved by the GPU driver

**Sampling cadence:** default 10 s. Override workspace-wide via
`MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL` (same env var MLflow's
built-in monitor honours — so all system metrics tick at the same
rate) or per-run by instantiating the callback directly with
`sample_interval_s=...`. `pyobjc-core` + `pyobjc-framework-Cocoa` are
declared under `sys_platform == "darwin"` so Linux / CUDA environments
skip the IOKit reader cleanly. If Apple renames keys across macOS
releases (it has happened — see
[exelban/stats#2130](https://github.com/exelban/stats/issues/2130))
any missing key is skipped silently rather than failing the sample.

**Thread lifecycle:** the sampler runs in a **non-daemon** thread and
also registers an `atexit` hook to signal stop. Together these prevent
the thread from accessing `torch.mps` / IOKit / `MlflowClient` C state
during interpreter teardown, which would segfault on macOS.

The same behaviour applies to subclasses (`LightningClassificationTrainer`,
`LightningSegmentationTrainer`, `LightningDetectionTrainer`).
`raidar_ultralytics.UltralyticsTrainer` / `UltralyticsEvaluator` use the
global `enable_system_metrics_logging()` path (ultralytics' MLflow
integration calls `mlflow.start_run()` so the flag is sufficient there)
and don't get the per-epoch MPS callback.

Opt-out per trainer:

```yaml
trainer: !class:marainer.trainer.LightningTrainer
  log_system_metrics: false   # disable for this run
  # ... (other trainer settings unchanged)
```

Workspace-wide tuning via `.env`:

```bash
# Sampling interval in seconds (default: 10)
MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL=30
# Disable from the environment (overrides the trainer's default-on behaviour)
MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false
```

NVIDIA GPU metrics are picked up automatically when `pynvml` is installed
(not a marainer dep — install in the consuming project if you need it).

---

## Examples

The `examples/` directory contains ready-to-run configurations:

| File | Description |
|---|---|
| `mnist_train_minimal.yaml` | Minimal MNIST training (no metrics) |
| `mnist_train.yaml` | MNIST with data preprocessing |
| `mnist_train_metrics.yaml` | MNIST with explicit metrics (Accuracy, F1, Precision, Recall) |
| `mnist_train_preset.yaml` | MNIST with MetricPreset (all multiclass metrics) |
| `mnist_train_logging.yaml` | MNIST with TensorBoard, CSV, and MLflow loggers |
| `mnist_model.py` | Simple CNN model for MNIST |

Run any example from the `examples/` directory:

```bash
cd examples
marainer train mnist_train_metrics.yaml
```

---

## Project Architecture

```
marainer/
  __init__.py       # Package init, optimizer registrations
  trainer.py        # Trainer (LightningModule) with configurable metrics
  metrics.py        # MetricPreset for category-based metric discovery
  cli.py            # CLI entry point (Liquify app)
examples/
  mnist_model.py    # Example CNN model
  mnist_train*.yaml # Example configurations
tests/
  test_metrics.py             # Metric system tests (27 tests)
  test_training_scenarios.py  # Training integration tests
  test_unpack_batch.py        # Batch handling tests
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `pytorch-lightning` | Training loop, device management, logging |
| `torch` | Neural networks, tensors, optimizers |
| `torchmetrics` | Flexible metric computation |
| `confluid` | YAML configuration and dependency injection |
| `liquifai` | CLI framework with automatic DI |
| `dataflux` | Data pipeline abstraction |
| `logflow` | Structured logging |
| `mlflow` | Experiment tracking |
| `tensorboard` | TensorBoard logging backend |

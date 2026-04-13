# Marainer

Project-agnostic training and processing pipeline orchestrator built on PyTorch Lightning, Confluid, and DataFlux.

Marainer provides a fully YAML-configurable training loop with integrated experiment tracking (MLflow), flexible data pipelines (DataFlux), and a comprehensive metrics system (torchmetrics).

## Installation

```bash
pip install git+https://github.com/Gearlux/marainer.git@main
```

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

optimizer: !class:torch.optim.Adam
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
- Log all parameters and metrics to MLflow

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

3. **MLflow setup** -- An experiment is created (or reused) with the configured `experiment_name`, and a new run starts with `run_name`.

4. **Parameter logging** -- All configurable parameters are extracted and logged to MLflow for full traceability.

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
| `train_set` | `!class:` | *required* | Training dataset (Flux pipeline or Dataset) |
| `val_set` | `!class:` | `None` | Validation dataset |
| `train_loader` | `!class:` | `DataLoader` | Training DataLoader config |
| `val_loader` | `!class:` | `DataLoader` | Validation DataLoader config |
| `lightning` | `!class:` | `L.Trainer` | PyTorch Lightning Trainer config |
| `batch_size` | `int` | -- | Broadcast to DataLoaders |
| `num_workers` | `int` | -- | Broadcast to DataLoaders |
| `max_epochs` | `int` | -- | Broadcast to Lightning Trainer |

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
optimizer: !class:torch.optim.SGD
  lr: 0.01
  momentum: 0.9

# AdamW
optimizer: !class:torch.optim.AdamW
  lr: 0.001
  weight_decay: 0.01
```

### Lightning Trainer Configuration

Pass any `pytorch_lightning.Trainer` argument:

```yaml
lightning: !class:pytorch_lightning.Trainer()
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

## Metrics System

Marainer integrates with [torchmetrics](https://torchmetrics.readthedocs.io/) to provide flexible, YAML-configurable metrics. Metrics are automatically:

- **Split** into independent train and validation collections (required by torchmetrics for correct state tracking)
- **Prefixed** with `train/` and `val/` for unambiguous logging
- **Logged** to both the Lightning progress bar and MLflow at epoch end
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
    - !class:dataflux.ops.numpy.NormalizeOp()
      max_value: 255
    - !class:dataflux.ops.ToTensorOp()
      normalize: false
```

DataFlux pipelines support functional composition with operations like normalization, tensor conversion, augmentation, and more. See the DataFlux documentation for available sources and operations.

---

## MLflow Integration

Every training run is automatically tracked in MLflow:

- **Parameters** -- All configurable parameters (model architecture, optimizer settings, data pipeline config) are logged
- **Metrics** -- Loss and all configured torchmetrics are logged per epoch
- **Experiment organization** -- Runs are grouped by `experiment_name`

```yaml
experiment_name: mnist_experiment    # MLflow experiment
run_name: baseline_v2               # MLflow run name
```

View results with:

```bash
mlflow ui
```

---

## Examples

The `examples/` directory contains ready-to-run configurations:

| File | Description |
|---|---|
| `mnist_train_minimal.yaml` | Minimal MNIST training (no metrics) |
| `mnist_train.yaml` | MNIST with data preprocessing |
| `mnist_train_metrics.yaml` | MNIST with explicit metrics (Accuracy, F1, Precision, Recall) |
| `mnist_train_preset.yaml` | MNIST with MetricPreset (all multiclass metrics) |
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
  cli.py            # CLI entry point (Liquify app) with MLflow orchestration
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
| `liquify` | CLI framework with automatic DI |
| `dataflux` | Data pipeline abstraction |
| `logflow` | Structured logging |
| `mlflow` | Experiment tracking |

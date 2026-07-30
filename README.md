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
| `marainer[torch]` | `torch` | `marainer.torch` — checkpoint loading (`load_lightning_checkpoint`) |
| `marainer[lightning]` | torch, `pytorch-lightning`, `torchmetrics`, `tabulate`, `matplotlib` | `marainer.lightning` — Lightning tracking callbacks + metric helpers |
| `marainer[mlflow]` | `mlflow`, `psutil`, `pyobjc` (macOS) | `MlflowTracker` + system metrics |
| `marainer[tensorboard]` | torch (`torch.utils.tensorboard`) | `TensorBoardTracker` |
| `marainer[wandb]` | `wandb` | `WandbTracker` |
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

Four backends live one-per-file, each importing its library **lazily inside its methods** and every
one **zero-arg constructible** (the native handle is built on first `.experiment` access):

| Backend | Library | Notes |
|---|---|---|
| `CsvTracker` | stdlib only | the always-available default — `metrics.csv` + `hparams.yaml` + artifact copies |
| `MlflowTracker` | `mlflow` | run lifecycle via `MlflowClient`; optional `log_system_metrics=True` |
| `TensorBoardTracker` | `torch.utils.tensorboard` | artifacts copied into the run dir |
| `WandbTracker` | `wandb` | minimal, mock-tested |

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

## Automatic provenance

`marainer.log_run_context(trackers, params)` stamps everything you want reproducible in one call —
so a run is rebuildable from its artifacts alone:

- **hyperparameters** — `params`
- **git commit** — the `git.commit` tag (`+ "-dirty"` on uncommitted changes)
- **config artifacts** — the transitively-`include:`-d YAML tree, uploaded under `config/`

`marainer.tracking.capture_log_file(trackers)` uploads the active loggair output file
(`loggair.get_active_config()["log_file"]`) as a `logs/` artifact at run end. Don't hand-roll
git/config/log-file logging elsewhere — these are the sanctioned helpers.

```python
from marainer import log_run_context
from marainer.tracking import capture_log_file

log_run_context([tracker], params={"lr": 1e-3, "model": "efficientvit_b0"})
# ... run training ...
capture_log_file([tracker])
```

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

`marainer.lightning` also ships the metric utilities (`build_metric_collection`, `MetricPreset`,
`coerce_metric_value`, `scalar_metrics`), the progress bridge
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

## `marainer.torch` — torch-only, Lightning-free helper (`marainer[torch]`)

- `load_lightning_checkpoint(model, path, strict=False)` — loads a Lightning `.ckpt` (strips the
  `model.` prefix), an Ultralytics `.pt` (pulls the live module's `state_dict`), or a plain
  state-dict; auto-retargets to a named child when keys match one. `strict=False` tolerates a
  re-headed `num_classes`.

It sits outside `marainer.lightning` so a non-Lightning platform (an Ultralytics loop) can read a
checkpoint back via `marainer[torch]` without pulling `pytorch_lightning`.

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
| `numpy` | the predictions sink |

Everything else — torch, pytorch-lightning, torchmetrics, mlflow, tensorboard, wandb, tensorflow —
is optional and behind an extra.

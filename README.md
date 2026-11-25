# Matrainer

**Experiment tracking** for machine-learning runs — parameters, git provenance, log
files, config artifacts, figures, metric values, and system metrics — logged to whichever backend
you use (MLflow, TensorBoard, CSV, Weights & Biases, Aim, ClearML), across whichever training platform you run
(PyTorch Lightning, TensorFlow/Keras, Ultralytics, a bare training loop).

Tracking is the core and it stands alone: `import matrainer` loads no training framework and no
tracker library, and a class that trains can reach for Matrainer only to *track* the run. Matrainer
also ships the shared trainer machinery a task project builds its runnables on — one abstract API
base and one concrete base per framework ([the trainer API](https://github.com/Gearlux/matrainer/blob/main/docs/trainer-api.md)) — and a
`matrainer` console script, the generic runner and the sweep driver. Every framework and tracker
library sits behind an extra.

## The name

**Matrainer** is a portmanteau of **mariner** — the one who keeps the log, recording every heading,
every reading and every condition of the voyage — and **trainer**, the thing being logged. That is
the whole job description: it does not steer the ship, it writes down where it went. A mariner's log
is also why the [run manifest](https://github.com/Gearlux/matrainer/blob/main/docs/run-manifest.md) is the *union* of everything known about a
run rather than whatever the current backend happens to have a field for — a logbook that only
recorded what this port cares about would be useless at the next one.

## Installation

```bash
pip install matrainer
```

matrainer and its `recordstream` dependency are on PyPI as pre-releases; pip installs them without
`--pre`, because no final release exists yet. Leave `--pre` off: it lets pip take pre-releases of *every*
dependency too, and `pip install --pre "matrainer[all]"` then fails outright building an `aim` alpha.

The core install carries **no** training framework and **no** tracker library: `import matrainer`
pulls in no torch, pytorch-lightning, tensorflow, mlflow or wandb. It is not small on disk —
`recordstream`, a core dependency, brings its own data stack (pandas, pyarrow, scipy, opencv,
`datasets`), about 65 packages in all. Every framework and tracker is behind an extra and lazily
imported the first time you use its backend:

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
| `matrainer[huggingface]` | torch, `transformers`, `accelerate` | `matrainer.huggingface.HuggingFaceRunnable` |
| `matrainer[jax]` | `flax`, `optax`, `recordstream[keras]` | `matrainer.jax.JaxLoopRunnable` |
| `matrainer[mlx]` | `mlx` (Apple silicon marker) | `matrainer.mlx.MlxLoopRunnable` |
| `matrainer[keras]` | `keras` | `matrainer.keras.keras_tracking_callback` |
| `matrainer[plot]` | `matplotlib` | `render_matrix_as_figure` / `render_vector_as_figure` — a confusion matrix as a heatmap, a per-class metric as a bar chart, for `log_figure` |
| `matrainer[tensorflow]` | `matrainer[keras]`, `tensorflow` | the same helper, with the TensorFlow engine underneath |
| `matrainer[all]` | `lightning`, `mlflow`, `tensorboard`, `wandb`, `aim`, `clearml`, `keras`, `plot` | every tracker, Lightning and Keras (the other training frameworks stay separate extras) |

**Python 3.13:** the Aim tracker's library (`aim`) publishes no Python 3.13 packages yet, so
`matrainer[aim]` — and `matrainer[all]`, which includes it — install on Python 3.12 only. Every other
extra installs on both.

Install one with `pip install "matrainer[mlflow]"` (several: `"matrainer[lightning,mlflow]"`).
A missing backend library raises one clear error naming the extra to install — never a bare
`ModuleNotFoundError`.


## A first run

`examples/standalone_tracking.py` runs the whole `Tracker` surface — hyperparameters, a metrics
loop, a figure, provenance, finalize — against `CsvTracker`, with no training framework installed:

```bash
python examples/standalone_tracking.py
```

```
Tracked run written to: /tmp/matrainer_demo_uwc4_npl/standalone-demo/demo_wild_robin
  environment/requirements.txt
  hparams.json
  manifest.json
  manifest.yaml
  metrics.csv
  tags.json
```

That is a tracked run: the values you logged (`metrics.csv`, `hparams.json`, `tags.json`) beside the
record of what produced them — `manifest.yaml` and the installed-package freeze. Point the same code
at another backend by swapping the tracker; nothing else changes.

## Configuring it

A tracker is a [confluid](https://github.com/Gearlux/confluid) `@configurable`, so a run's config
wires it by name and every constructor argument is a key:

```yaml
trackers:
  - !class:matrainer.MlflowTracker
    experiment_name: drone-classify
    tracking_uri: https://mlflow.example.org
```

In code, the same thing is `MlflowTracker(experiment_name="drone-classify")`. Wrap it in
`!class:matrainer.ResilientTracker` so a dropped connection queues the call instead of stopping the
run, and list several trackers to log to all of them at once — see
[tracking a run](https://github.com/Gearlux/matrainer/blob/main/docs/tracking.md).

## Common tasks

| I want to | Go to |
|---|---|
| log a run from my own loop or platform | [Tracking a run](https://github.com/Gearlux/matrainer/blob/main/docs/tracking.md) |
| choose between MLflow, TensorBoard, CSV, W&B, Aim and ClearML | [Choosing a backend](https://github.com/Gearlux/matrainer/blob/main/docs/backends.md) |
| know what produced a run, months later | [The run manifest](https://github.com/Gearlux/matrainer/blob/main/docs/run-manifest.md) |
| track a PyTorch Lightning run | [Tracking a Lightning run](https://github.com/Gearlux/matrainer/blob/main/docs/lightning.md) |
| build my project's runnable on the shared bases | [The trainer API](https://github.com/Gearlux/matrainer/blob/main/docs/trainer-api.md) |
| name, average or report my metrics | [Metrics](https://github.com/Gearlux/matrainer/blob/main/docs/metrics.md) |
| report a confusion matrix or a per-class vector | [Metrics](https://github.com/Gearlux/matrainer/blob/main/docs/metrics.md#reporting-a-metric-that-is-not-a-single-number) |
| load a checkpoint, pick a device | [`matrainer.torch`](https://github.com/Gearlux/matrainer/blob/main/docs/torch.md) |
| hand a trained model to another tool | [Model packages](https://github.com/Gearlux/matrainer/blob/main/docs/model-packages.md) |
| give my project a `train`/`evaluate`/`predict` CLI | [Building a project CLI](https://github.com/Gearlux/matrainer/blob/main/docs/cli.md) |
| run one config across every framework and compare | [Running and reading a sweep](https://github.com/Gearlux/matrainer/blob/main/docs/sweeps.md) |

## Documentation

| Page | What it covers |
|---|---|
| [Tracking a run](https://github.com/Gearlux/matrainer/blob/main/docs/tracking.md) | the `Tracker` surface, `ResilientTracker`, run names, system metrics |
| [Choosing a backend](https://github.com/Gearlux/matrainer/blob/main/docs/backends.md) | what each backend costs to run, and the three shipped as services |
| [The run manifest](https://github.com/Gearlux/matrainer/blob/main/docs/run-manifest.md) | automatic provenance, what the `capture_*` calls collect, portable paths, native slots |
| [The trainer API](https://github.com/Gearlux/matrainer/blob/main/docs/trainer-api.md) | the four layers, and the checkpoint policies each framework translates |
| [Tracking a Lightning run](https://github.com/Gearlux/matrainer/blob/main/docs/lightning.md) | the three hookup styles and when to use each |
| [Metrics](https://github.com/Gearlux/matrainer/blob/main/docs/metrics.md) | per-split slots, the metrics a loop cannot average, metric names, non-scalar reporting |
| [`matrainer.torch`](https://github.com/Gearlux/matrainer/blob/main/docs/torch.md) | checkpoint loading, `resolve_device`, the dump spellings |
| [Model packages](https://github.com/Gearlux/matrainer/blob/main/docs/model-packages.md) | a model as a folder: graph, weights, class names |
| [Building a project CLI](https://github.com/Gearlux/matrainer/blob/main/docs/cli.md) | the CLI factory and a project's frameworks table |
| [Running and reading a sweep](https://github.com/Gearlux/matrainer/blob/main/docs/sweeps.md) | the sweep driver, its isolation and its read-back |

Why a mechanism exists, rather than how to use it: [`docs/architecture.md`](https://github.com/Gearlux/matrainer/blob/main/docs/architecture.md).

## Environment variables

| Variable | Effect |
|---|---|
| `KERAS_BACKEND` | Keras 3 backend; `matrainer.tensorflow` and `--framework tensorflow` default it to `tensorflow` |
| `MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL` | seconds between system-metric samples (default 10) |
| `PROJECT_ROOT`, `DATA_ROOT`, `MODEL_ROOT` | machine roots; paths under them are recorded as `$NAME/...` in the manifest |
| `AIM_REPO` | the Aim repository, written by the Aim service and read as `$AIM_REPO` by run configs |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO`, `PYTORCH_MPS_LOW_WATERMARK_RATIO` | Apple-GPU memory limits; set them in the shell before the process starts, or let the device options set them |

`env.complete` lists them.


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

There is no `matrainer.runner` / `matrainer.progress` / `matrainer.workflow` / `matrainer.processing`.
The one binary, `matrainer`, is the generic runner and the sweep driver described above.


## Dependencies

| Package | Purpose |
|---|---|
| `confluid` | YAML configuration + dependency injection (`@configurable` Trackers/policies) |
| `liquifai` | provenance's config-include resolution |
| `loggair` | structured logging + the active-log-file handle `capture_log_file` uploads |
| `coolname` | random run names |
| `pyyaml` | the run manifest's readable half (`manifest.yaml`) |
| `recordstream` | the record model the trainer API runs on, and the label-map format run artifacts use |
| `numpy` | the confusion-matrix metrics (`matrainer.metric_functions`) |

Everything else — torch, pytorch-lightning, torchmetrics, mlflow, tensorboard, wandb, aim,
clearml, tensorflow, matplotlib — is optional and behind an extra.

## License

[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0) (the `LICENSE` file): free for personal, academic and other noncommercial use. Commercial use needs a separate licence from the author.

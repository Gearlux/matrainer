# Tracking a run

The `Tracker` abstraction, the wrapper that survives a dropped server, run naming, and the system metrics every backend gets for free.

Runnable twin: [`examples/standalone_tracking.py`](../examples/standalone_tracking.py) — hyperparameters, a metrics loop, a figure, provenance and finalize, with no training framework installed.

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

**A failed call never stops the run.** A backend call the server refused or never answered is
logged at debug level and the call returns, so a tracking outage costs a value, not a training run.
Set `raise_errors` on a tracker to have the error raised as well, after its debug line:

```yaml
trackers:
  - !class:matrainer.MlflowTracker
    tracking_uri: https://mlflow.example.org
    raise_errors: true      # default false; a failed call now raises into the training loop
```

A call that sends several values (MLflow sends each parameter, metric and tag on a request of its
own) still tries every one and raises the first failure, and `log_manifest` sends every part — the
files, the parameters, each native slot — before it raises. `ResilientTracker` (below) turns
`raise_errors` on for the tracker it wraps: what a call raises is what the wrapper can queue.

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

### A tracking server that drops out — `ResilientTracker`

A tracker that talks to a remote server makes the training loop wait on the network. Wrap it in
`ResilientTracker` and a slow or unreachable server never stops a run or loses a value:

```yaml
trackers:
  - !class:matrainer.ResilientTracker
    timeout: 5.0                 # seconds per call; an artifact's limit grows with its size
    tracker: !class:matrainer.MlflowTracker
      experiment_name: drone-classify
      tracking_uri: https://mlflow.example.org
```

| What happens | What the wrapper does |
|---|---|
| a call hangs or the connection fails | after `timeout` s it queues the call; the next call replays the queue first, in order |
| `max_retries` (10) failures in a row | stops calling the server, only queues; `finalize` makes `max_retries` more tries |
| an artifact's file is deleted after logging | its bytes were read when it was logged, so it still uploads under its name |
| the process dies before the queue is sent | a queued artifact has a copy under `runs/<experiment>/<run>/pending_artifacts/` |
| the server answers 5xx on one kind of call | that kind is off for the run; every other call goes on (the status is read from the error, never a 500 in its message) |
| the server refuses a call for another reason | the call is dropped with a warning |
| the run cannot be started (`start_timeout`, 30 s) | tracking is off for this run; training carries on |

What the log shows when the network drops for a while:

```
WARNING Network error in log_metrics, queued it (1/10): log_metrics timed out after 5 s
INFO    Sent 3 queued tracking call(s) (0 still queued).
```

| Setting | Default | Meaning |
|---|---|---|
| `tracker` | — | the tracker to protect (required to run) |
| `timeout` | `5.0` | seconds per metric / parameter / tag call (figures and text: at least 30 s) |
| `start_timeout` | `30.0` | seconds to start the run |
| `artifact_timeout` | `None` | a fixed upload limit; `None` = `max(timeout, 60 + size_MB / artifact_upload_rate_mbps)` |
| `artifact_upload_rate_mbps` | `5.0` | the slowest upload rate the scaled limit allows for |
| `max_retries` | `10` | failures in a row before only queueing; tries at `finalize` |
| `finalize_retry_delay` | `1.0` | seconds between `finalize`'s tries |
| `backup_artifacts` | `True` | write the on-disk copy of a queued artifact |

**The wrapper turns the wrapped tracker's `raise_errors` on** the first time it uses it — a tracker
keeps a failed call to itself by default, which would leave nothing to queue — so the configuration
above needs nothing more. A replayed call is sent whole: a call that timed out may still land later,
and a call that names several values may have sent some before the network dropped, so a replay can
record a metric point twice (MLflow accepts a parameter re-sent with the same value). Rationale:
[architecture record 14](architecture.md).

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

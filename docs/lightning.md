# Tracking a PyTorch Lightning run

Three ways to hook a tracker into Lightning, and when to reach for each.

Runnable twin: [`examples/lightning_tracking.py`](../examples/lightning_tracking.py) — a one-epoch fit on synthetic tensors that shows the checkpoint following the tracker's `save_dir`, the metrics arriving, and a run that raises recorded as `failed`.

## Three ways to hook tracking into PyTorch Lightning

`matrainer.lightning` (behind `[lightning]`) is **optional tracking helpers**, not a platform. Your
`LightningModule` stays a standard one; pick whichever hookup fits — the choice is yours:

**1. Plain functions** — call `log_run_context` / `capture_log_file` from your own run wrapper (as
[automatic provenance](run-manifest.md#automatic-provenance) shows). Zero Lightning coupling.

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

The bridge also decides where **checkpoints** land. `ModelCheckpoint` reads its directory from
the first logger's `save_dir`, so `TrackerLogger` reports the tracker's — meaning a whole run
lands under one root:

```
<save_dir>/<experiment_name>/<run_name>/              # metrics, manifest, artifacts
<save_dir>/<experiment_name>/<run_name>/checkpoints/  # ModelCheckpoint, with no dirpath set
```

`save_dir` defaults to `./runs` on every backend, and that is the only knob — move it and the
checkpoints follow:

```python
trainer = pl.Trainer(logger=TrackerLogger(CsvTracker(save_dir="/data/runs")))
```

Set `ModelCheckpoint(dirpath=…)` explicitly only when a later evaluate/predict config must
NAME the path (a default run name is a random slug, so nothing can address it afterwards) —
and point it inside `runs/` too, e.g. `./runs/checkpoints/<name>`.

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

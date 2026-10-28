# Running and reading a sweep

One subprocess per cell, isolated failures, and a table read back from the tracker.

## Running and reading a SWEEP (`matrainer sweep`, no extra needed)

A *sweep* is one config run under many backends and/or many trackers, so the runs become
comparable. A consuming project ships only its MATRIX as a declarative spec
(`examples/sweep/sweep.yaml`); the machinery — probing what is installed, one subprocess per
cell, experiment naming from the tracker, failure isolation, the report — is matrainer's:

```bash
matrainer sweep backends examples/sweep/sweep.yaml               # frameworks × models, one tracker
matrainer sweep trackers examples/sweep/sweep.yaml               # every tracker, one framework
matrainer sweep all      examples/sweep/sweep.yaml               # frameworks × models × trackers
matrainer sweep backends examples/sweep/sweep.yaml --dry-run     # print the commands, run nothing
#   … --only lightning,torch --epochs 1 --tracker csv            # the common narrowing flags
```

The spec names the confluid overlay every cell runs, the backend rows (with `requires:`
find_spec probes and per-row `env:` pins), the optional model-variant dimension, the trackers,
and any per-tracker setup command:

```yaml
config: all_backends.yaml
epochs: 5
frameworks:
  - {name: lightning, requires: [pytorch_lightning], models: [zoo, dncnn]}
  - {name: keras, requires: [keras, tensorflow], env: {KERAS_BACKEND: tensorflow}, models: [dncnn]}
models:
  - name: zoo                            # first = the config's default, appends no flag
  - {name: dncnn, flags: [--model, dncnn]}
trackers: [mlflow, csv]
tracker_setup: {mlflow: [aisland, mlflow, start]}
```

Each cell runs through the **generic runner** — `matrainer train <config> --framework F
--tracker T --experiment_name E --run_name R` — which works standalone too, on any
`runnable:`-rooted config whose class references are module-qualified:

```bash
matrainer train    path/to/train_config.yaml --framework torch --max_epochs 1
matrainer evaluate path/to/train_config.yaml
matrainer predict  path/to/train_config.yaml
```

The tracker-side halves keep their original surface (`python -m matrainer.sweep …` works
everywhere the console script does):

```bash
# name THIS run of the sweep: <prefix>_<N>, N one past the highest already recorded
matrainer sweep next-experiment mlflow "$(date '+%a_%b_%-d_%Y')"

# read it back — one row per run
matrainer sweep report mlflow Thu_Aug_6_2026_1
matrainer sweep report csv    Thu_Aug_6_2026_1 --save-dir ./sweep_logs --all
```

```
| run              | status   |  sec |   val/psnr |   val/mse |   train/loss |
|------------------|----------|------|------------|-----------|--------------|
| huggingface_zoo  | FINISHED |  249 |    27.4394 |    0.0018 |       0.0486 |
| lightning_dncnn  | FINISHED |   61 |    20.5510 |    0.0088 |       0.0791 |
```

**The count comes from the TRACKER, not a local file.** So a wiped tracker restarts at 1, and a
second checkout of the same project does not collide with the first — a counter file would keep
counting a history that is no longer there.

**Bookkeeping columns are folded away, and never silently.** A `transformers.Trainer` records its
timings, throughput and FLOP count in the same `log_history` as the loss; Lightning logs `epoch`.
All worth keeping in the tracker, all noise in a table whose job is to compare a dozen runs on the
same few numbers. The footer says how many went, *which*, and the command that brings them back:

```
11 runs in csv experiment 'Thu_Aug_6_2026_1'
4 columns hidden — bookkeeping, within-epoch series (epoch, train/loss_step, …) — see them with:
  python -m matrainer.sweep report csv Thu_Aug_6_2026_1 --all
```

Two reasons, named separately, because `train/loss_step` is a genuine result rather than
bookkeeping — Lightning's per-*batch* series of a quantity the table already shows per epoch, which
a one-row-per-run table cannot render and only one engine emits.

**Only `mlflow` and `csv` can be read from here.** A tracker is a write surface first: an offline
ClearML task is a directory waiting to be uploaded and W&B needs its server, so those say where to
open them instead of showing an empty table — which would read as "the sweep recorded nothing".

The table is the **union** of every metric any run reported, validation first. A metric one backend
computes and the others do not earns its own column with the rest blank; the alternative is a table
that silently drops whatever does not fit its widest complete row. Non-scalar values (a per-class
vector, a confusion matrix) never reach it at all — they cannot go through a `{name: float}` channel,
so a runnable sends them to the trackers' text channel and you read them in the tracker's own UI.

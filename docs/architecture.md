# Marainer Architecture Decisions

Short decision records for the mechanisms whose *shape* isn't obvious from the code alone.
Each record: **Context → Decision → Consequences → Example → What you may change**. User-facing
"how do I use it" lives in the [README](../README.md); this file is the *why*.

Per the workspace "Docs Never Name Dependent Projects" mandate, these records describe generic
"consuming projects" / "a training platform", never a specific downstream package by name.

---

## 1. Marainer tracks; it does not train (2026-07)

**Context.** Marainer used to be a training *orchestrator*: a `LightningTrainer` / `LightningEvaluator`
pair with `training_step` / `validation_step` / `test_step`, a `run()` loop, a five-verb CLI, and
`pytorch_lightning` / `torch` hard-imported even in modules that only logged a git hash. Two problems
compounded. (a) Every consumer inherited a heavy base whose training loop it usually overrode anyway,
so the "shared" loop was a delegation maze, not a simplification. (b) A project that only wanted the
*tracking* (log params, git hash, config artifacts, system metrics) still paid the full Lightning
import cost and couldn't be removed without gutting the training code.

**Decision.** Marainer is an **experiment-tracking library only** — no training loop, no
trainer/evaluator class, no CLI. A class that trains reads like a standard example for its framework
and lives in the *consuming* project; it reaches for Marainer solely to track the run. The runnable
protocol (`TorchRunner`, `ProgressReporting`, the `entrypoint` method-annotation mechanism), the
workflow combinators, and `DatasetProcessor` moved to `sampleflux`; the generic runner is
`sampleflux run <config>` binding a top-level `runnable:` key.

**Consequences.**
- The guiding invariant is testable: **removing Marainer costs a consumer only a couple of line
  changes** — it keeps training (with loggair logging), losing only the rich tracking. If a change
  makes Marainer load-bearing for *training*, it's in the wrong project.
- `import marainer` imports no torch/lightning/mlflow (see record 2). The light core is
  `confluid`/`liquifai`/`loggair`/`coolname`/`numpy`.
- There is exactly one thing to reintroduce-and-reject in review: a `training_step`, a `run()` loop,
  or a `marainer.cli`. None belong here.

**Example.** A consumer's standard `LightningModule` opts into tracking without inheriting anything
from Marainer:

```python
import pytorch_lightning as pl
from marainer import CsvTracker, MlflowTracker
from marainer.lightning import TrackingCallback

trainer = pl.Trainer(
    max_epochs=10,
    callbacks=[TrackingCallback([CsvTracker(), MlflowTracker(experiment_name="exp")])],
)
trainer.fit(MyStandardLightningModule(), datamodule=dm)   # delete the callback -> still trains
```

**What you may change.** New tracking helpers, backends, provenance sources, and framework hookup
styles are all in scope. What you may *not* add back: a training/evaluation loop, a step function, a
`run()` dispatcher, a `DatasetProcessor`/workflow combinator (those live in `sampleflux`), or a CLI.

---

## 2. The `Tracker` ABC + one backend per file, lazily imported (2026-07)

**Context.** Tracking must reach several logger libraries (MLflow, TensorBoard, CSV, Weights &
Biases) that a given install rarely all has. Importing all of them at package import — the old shape
— made a light `import marainer` drag in mlflow/wandb and fail on a machine that has neither, even
for a consumer that only wanted the CSV backend. Consumers also want ONE surface they can hand to any
framework's logger slot, not four bespoke logging call sites.

**Decision.** A single `Tracker` ABC (`tracking/base.py`) mirrors the
`pytorch_lightning.loggers.Logger` shape (`name`/`version`/`experiment`/`log_hyperparams`/
`log_metrics`/`save`/`finalize`) plus the extensions `set_tags`/`log_artifact`/`log_figure`/
`log_dir`. `TrackerGroup` fans out to a list behind the same surface. Each backend lives
**one-per-file** and imports its library **lazily inside its methods** via
`marainer._imports.require(module, extra)`; every backend is **zero-arg constructible**, building its
native handle on first `.experiment` access. The Tracker is the whole abstraction — no wrapper layer
sits over it (an earlier `as_tracker` / `_LoggerTrackerView` machinery was removed as redundant).

**Consequences.**
- `import marainer` / `import marainer.tracking` imports **none** of torch/mlflow/tensorboard/wandb;
  a `tests/test_light_import.py` pins that contract. A missing library surfaces as one actionable
  error (`pip install marainer[mlflow]`), never a bare `ModuleNotFoundError` at import time.
- `CsvTracker` is stdlib-only, so it is the always-available default and the backend every test can
  exercise without mocking.
- Zero-arg construction is what lets Trackers be `@configurable` and wire from YAML like every other
  workspace primitive (the workspace lazy-init rule).
- A framework hookup (a `pl.Callback`, a `TrackerLogger` bridge, a Keras callback) is written ONCE
  against `Tracker`, so all four backends work through any of the three hookup styles for free.

**Example.** Adding a backend is a new file that stays import-safe:

```python
# marainer/tracking/newthing_backend.py
from marainer._imports import require
from marainer.tracking.base import Tracker

class NewThingTracker(Tracker):
    @property
    def experiment(self):
        if self._exp is None:
            newthing = require("newthing", "newthing")   # lazy: import + extra-name for the error
            self._exp = newthing.Run(self.experiment_name)
        return self._exp
```

**What you may change.** Add or extend backends and Tracker methods freely. What you may not do:
import a backend's library at module top level, add a required constructor arg, or reintroduce a
view/adapter layer over `Tracker`.

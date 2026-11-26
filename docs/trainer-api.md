# The trainer API

One neutral base, one concrete base per framework — and the checkpoint policies every one of them translates.

Runnable twins, one task on two engines: [`examples/trainer_api_loop.py`](../examples/trainer_api_loop.py) composes it onto the bare-torch base, [`examples/trainer_api_lightning.py`](../examples/trainer_api_lightning.py) onto the Lightning one. The task mixin is identical in both; 19 constructor slots come from the shared layers, the loop base adds `checkpoint_dir` / `device` / `grad_clip`, and the Lightning base adds none.

## The trainer API — one abstract base, one concrete base per framework

Since 2026-08 matrainer also ships the shared TRAINER machinery a task project builds its merged
train/evaluate/test/predict runnables on. Three layers, each in its own extras-gated home:

| Layer | Class | What it owns |
|---|---|---|
| Neutral API (`matrainer.runnable`, light core) | `TrainerRunnable` | the universal constructor slots, `run()` dispatch through the four `@entrypoint`-marked capabilities, tracker/prepare/prediction plumbing, the neutral task hooks |
| torch family (`matrainer.torch`, `[torch]`) | `TorchTrainerRunnable` | the TYPED slots (`model: Partial[nn.Module]`, the `MetricsSpec` quartet), the sentinel resolution, the per-split metric builders, the sidecar write, `trained_model` |
| Lightning machinery (`matrainer.lightning`, `[lightning]`) | `LightningRunnable` | the loaders, the deferred `pl.Trainer` slot with its tracking hookup, the standard step functions, the four entry points |
| bare-torch machinery (`matrainer.torch.loop`, `[torch]`) | `TorchLoopRunnable` | the hand-written epoch (`zero_grad`/`backward`/`step`), device resolution, the per-split scoring, a plain-`state_dict` checkpoint, per-epoch history, and the plain-function tracking hookup |
| fastai machinery (`matrainer.fastai`, `[fastai]`) | `FastaiRunnable` | the `DataLoaders`/`Learner` assembly as deferred slots, the positional batch tuple built on the task's `_targets` hook, the Recorder read-back into canonical names, alias-first metric-name resolution, and the plain-function tracking hookup |
| Keras family (`matrainer.keras.runnable`, `[keras]`) | `KerasTrainerRunnable` | the TYPED slots (`Partial[keras.Model]`, the `KerasLoss`/`KerasMetric`/`KerasOptimizer` aliases), the `_to_arrays` batch hook, the keras sentinel hooks, the metric UNION, the `RecordSequence` boundary, the tracking-callback plumbing |
| Keras compile/fit machinery (`matrainer.keras.trainer`, `[keras]`) | `KerasRunnable` | build + compile (with the `_compiled_loss` / `_compiled_optimizer` / `_check_compiled_metrics` seams), the `History` read-back, the `_before_fit` / `_fit_scores` fit hooks, the four entry points over `keras.Model`'s own `fit`/`evaluate`/`predict` |
| native-TensorFlow machinery (`matrainer.tensorflow.loop`, `[tensorflow]`) | `TensorFlowRunnable` | the `tf.data` pipeline, traced `GradientTape` steps (XLA via `jit_compile`), per-split CLONED metric objects, the hand-driven `CallbackList`, the SavedModel export, the engine guard |
| HuggingFace machinery (`matrainer.huggingface`, `[huggingface]`) | `HuggingFaceRunnable` | the deferred `TrainingArguments`/`Trainer` slots (string-dotted — no module-level transformers import), the run-time argument FLOOR (a config-pinned key wins), the `Trainer`-subclass seam, the `log_history` replay onto the trackers, the record-loader predict walk, and the plain-function tracking hookup |
| JAX machinery (`matrainer.jax`, `[jax]`) | `JaxLoopRunnable` | the jit-ONCE step builders (`nnx.value_and_grad` with the outputs as an aux), the splat-tolerant epoch driver, the per-epoch reshuffle, name-resolved optax losses/FACTORY optimizers, the metric STATE hooks (`_new_split_state`/`_score_batch`/`_split_scores` — one accumulator per split, reduced once), and the plain-function tracking hookup |
| MLX machinery (`matrainer.mlx`, `[mlx]`, Apple silicon) | `MlxLoopRunnable` | the same hook surface under MLX's three facts: the per-step `mx.eval` over parameters + optimizer state + loss, train/eval as module STATE flipped at the loop boundaries, and CLASS-shipped optimizers (instantiated where the jax helper calls a factory) |

The abstraction deliberately **stops at the API** — there is no abstract trainer *object*: the
`pl.Trainer` slot exists only in the Lightning base, the open-coded epoch only in the loop
base, the `Learner` only in the fastai one, the compile step only in the Keras one (whose
two machinery bases are SIBLINGS sharing the keras family layer), and the
`transformers.Trainer` assembly only in the HuggingFace one — the member that writes least
while swapping most, because that engine owns the whole training program. The loop base
adds three BATCH hooks (`_train_batch_loss` / `_eval_batch` / `_predict_batch`) with the
read-forward-loss default, so a task whose model computes its loss inside `forward` (a native
detection contract) overrides exactly those three; `_eval_batch` may return `None` to declare
"this split has no loss", and the split then reports its metrics alone. A consuming project writes its TASK once as a
pair of hook mixins and composes a concrete class per engine:

```python
from matrainer.lightning import LightningRunnable

@configurable(task="mytask", role="trainer", framework="torch")
class LightningMyTask(MyTaskTorchHooks, LightningRunnable):
    def __init__(self, model, train_set=None, ..., my_task_knob=1.0):
        super().__init__(model=model, train_set=train_set, ...)   # shared slots, stored verbatim
        self.my_task_knob = my_task_knob                          # task-specific state
```

The hooks a task implements: `initialize` (resolve dimensions/labels, flow the model, build the
metrics), `_targets` (the batch read the loss expects), `_default_loss` / `_default_metrics` /
`_metric_flow_dims` (the sentinel defaults and their sizing kwargs), `_prediction_output` (the
sink contract), `_reportable` / `_report_nonscalars` (reporting filters), `_after_fit` (epilogue).
Everything else — tracker wiring, epoch reporting, sidecars, entry points — is inherited.

## Framework-neutral checkpoint policies

Declare *what* to checkpoint once; each framework translates it. `matrainer.checkpoints` provides
zero-arg `@configurable` carriers:

| Policy | Meaning |
|---|---|
| `SaveLast()` | keep `last.ckpt`, refreshed every epoch |
| `SaveEveryEpoch(every_n_epochs=1)` | snapshot on a cadence |
| `SaveBest(monitor, mode, top_k=1)` | keep the best `top_k` by a monitored metric |

```python
from matrainer.checkpoints import SaveBest, SaveLast
from matrainer.lightning import build_checkpoint_callbacks

callbacks = build_checkpoint_callbacks([SaveLast(), SaveBest(monitor="val/acc", mode="max")])
trainer = pl.Trainer(callbacks=callbacks)   # -> pytorch_lightning ModelCheckpoint(s)
```

No policy ⇒ the framework default. Add a policy here; translate it in the framework helper.

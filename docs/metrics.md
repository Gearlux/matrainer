# Metrics

Per-split metric slots, the metrics a hand-written loop cannot average, what a metric is called, and how to report one that is not a single number.

Runnable twin: [`examples/metrics.py`](../examples/metrics.py) — the metric-spec rules, the batch-averaging error measured, and the name translation.

## Per-split metric slots (`matrainer.metric_slots`, no extra needed)

A training runnable typically takes `metrics` (what **every** split measures) plus
`train_metrics` / `val_metrics` / `test_metrics` (what **that** split measures on top), so an
expensive or split-specific metric can be added to validation alone without restating the shared
set beside it. How the two combine is the same everywhere; *what a metric is* is not — so the rule
lives here as two pure functions and is applied to whichever vocabulary your engine uses (a
torchmetrics template mapping, a list of `keras.metrics` objects, a list of fastai function names).

```python
from matrainer.metric_slots import merge_metric_specs, union_metric_specs

# merge — an engine whose loop you own: each split gets its own set
merge_metric_specs({"acc": A}, {"confusion": C})   # {"acc": A, "confusion": C}   add
merge_metric_specs({"acc": A}, {"acc": A2})        # {"acc": A2}                  shadow
merge_metric_specs(["accuracy"], ["top_k"])        # ["accuracy", "top_k"]        concatenate
merge_metric_specs({"acc": A}, None)               # {"acc": A}   None = "not chosen"

# union — an engine that compiles ONE metric list for several phases (keras, transformers)
union_metric_specs(shared, train_spec, val_spec, test_spec)
```

Two mappings merge by key (a same-named entry in the split **shadows** the shared one, which is
how a split overrides a template rather than adding to it); two sequences concatenate; anything
else — a ready-made collection, a mapping against a list — means the split **replaces**, because
there is no key-wise way to combine them and silently ignoring the split's own spec would be worse.
`union_metric_specs` is for an engine that cannot be told to narrow (`keras.Model.compile` owns one
list): a metric asked for on one split is then computed on the others too, which is noise a reader
ignores — the alternative, dropping it, is a wired metric that silently reports nothing.

The module imports **nothing but the standard library**, so a deliberately framework-free training
loop can use it without pulling torch. That is also why it sits at the top level rather than under
`matrainer.torch`.

## Metrics a hand-written loop cannot average (`matrainer.metric_functions`, no extra needed)

If your engine owns the loop, you get stateful metrics for free — it accumulates across batches
and reduces once. If **you** own the loop you see one batch at a time, and the obvious move is to
score each batch and average. For accuracy that is very nearly right. For a macro-averaged F1 it
is not:

| | global macro-F1 | batch-averaged |
| --- | --- | --- |
| accuracy ≈ 0.9, 10 classes, batch 32 | 0.9182 | 0.8849 |
| accuracy ≈ 0.3, 10 classes, batch 32 | 0.3486 | 0.3032 |

Three to five points, always downward, and worse the more classes a batch is missing — a class
absent from a batch scores 0 for that batch and drags the mean. Accumulate a confusion matrix
instead: one `C×C` integer array whose size does not grow with the dataset.

```python
from matrainer.metric_functions import accumulate_confusion, macro_f1, zero_confusion

matrix = zero_confusion(num_classes)
for logits, targets in batches:            # logits are argmaxed; raw or softmaxed, same answer
    accumulate_confusion(matrix, logits, targets)
print(macro_f1(matrix))
```

`macro_f1` matches `sklearn.metrics.f1_score(..., average="macro", zero_division=0)` and
`torchmetrics.classification.MulticlassF1Score(average="macro")` exactly — which is the point: a
number from your loop is comparable with one from a framework that computed it for you. numpy is
the only import, so a deliberately framework-free loop can use it. Single-label targets only; a
multi-hot row has no single cell to land in.

## What a metric is CALLED (`matrainer.metric_names`, no extra needed)

Every tracker records the key it is handed, verbatim. So a run recorded under one engine and the
same run under another land in different columns: Keras writes `loss` / `val_loss`, fastai
`train_loss` / `valid_loss` plus **unprefixed validation** metrics, a `transformers.Trainer`
`train_*` / `eval_*`, a `LightningModule` whatever it logs. `MetricNames` translates between those
vocabularies, in either direction:

```python
from matrainer import MetricNames

names = MetricNames()                                     # default target: matrainer's own
names({"loss": 0.5, "val_loss": 0.4}, source="keras")     # {'train/loss': 0.5, 'val/loss': 0.4}
names({"accuracy": 0.9}, source="fastai")                 # {'val/accuracy': 0.9}  ← see below
names({"eval_accuracy": 0.9}, source="huggingface")       # {'val/accuracy': 0.9}

MetricNames(target="keras")({"val/loss": 0.4})            # {'val_loss': 0.4}  ...and back out
```

The canonical vocabulary is `<split>/<name>` with `split` one of `train` / `val` / `test` — what
`build_metric_collection(...).clone(prefix="val/")` already produces, and what TensorBoard and
ClearML group their curves on. `epoch` and `step` pass through untouched: they are the x-axis, not
a measurement. Translation is idempotent, so a loop that already names things canonically can
declare its real `source` and pass through unchanged.

**`source` is the caller's, `target` and `renames` are config's.** Only the code doing the
reporting knows which engine produced the names — and whether an unprefixed `accuracy` was
measured on train (Keras) or on validation (fastai's `Recorder` computes its metrics on the
validation pass and prefixes none of them). Everything a *user* chooses is on the object, so one
YAML key can reach several differently-sourced backends and mean the right thing in each:

```yaml
metric_names: !class:matrainer.MetricNames
  target: keras                       # record this run in Keras spelling instead
  renames: {top_k_accuracy: accuracy_top5}   # align a metric this module cannot know about
```

`renames` applies to the parsed **base** name, after the source's own aliases resolve
(`sparse_categorical_accuracy` → `accuracy`) and before the target's spelling is applied, so one
entry reads the same whatever the two vocabularies are.

Where it is already applied for you: `keras_tracking_callback` (`source="keras"`) and
`TrackerLogger` (`source="lightning"`) both take a `metric_names=` argument and default to the
canonical translation. A hand-written loop calls `as_metric_names(self.metric_names)` — one line
that turns an unwired slot, a live object or a deferred YAML marker into a usable translator.

What is **not** translated is what the engine hands its own callbacks: a
`keras.callbacks.EarlyStopping(monitor="val_loss")` must keep finding the name Keras gave it, so
only the copy on its way to a tracker is renamed.

### Reporting a metric that is not a single number

`scalar_metrics` drops what a `{name: float}` channel cannot hold. Renderers put such a value back,
and a channel gives each a home. There are **two shapes**, each with a table form and a figure form:

| shape | table (`log_text`) | figure (`log_figure`) |
|---|---|---|
| square `N×N` — a **confusion matrix** | `render_matrix_as_table` | `render_matrix_as_figure` (heatmap) |
| length-`C` — a **per-class metric** (`average=None`) | `render_vector_as_table` | `render_vector_as_figure` (bar chart) |

The table form needs only `tabulate` and is readable in a plain log, so it is what a consumer sends
by default; the figure form needs matplotlib (`matrainer[plot]`) on every eval epoch, so it is
opt-in. All four are framework-free — a value only needs `.tolist()`, or `.detach().cpu().tolist()`
— with their libraries imported lazily so the module stays in the light core.

**The two shapes get different renderings on purpose.** A per-class IoU is not a degenerate matrix:
it has one axis of meaning, its values share a scale, and what a reader wants from it is a *ranking*
— which class the model is worst at. A heatmap of a `1×C` strip answers that badly, so the vector
renderers sort **worst-first** and draw the mean as a reference line.

```python
from matrainer.tracking import (
    per_class_vectors, render_matrix_as_table, render_vector_as_figure, render_vector_as_table,
)

values = val_metrics.compute()          # scalars, vectors and matrices, together

table = render_matrix_as_table("val/confusion", values["val/confusion"], ["cat", "dog"])
tracker.log_text("val/confusion", table, step=epoch)

# `per_class_vectors` finds the VECTORS by SHAPE, so a runnable needs no naming convention
for payload in per_class_vectors(values, class_names=["pet", "background", "boundary"]):
    table = render_vector_as_table(payload["name"], payload["values"], payload["class_names"])
    logger.info(table)
    tracker.log_text(payload["name"], table, step=epoch)
    # opt in when a picture is worth it
    tracker.log_figure(
        payload["name"],
        render_vector_as_figure(payload["name"], payload["values"], payload["class_names"]),
        step=epoch,
    )
```

```
val/iou_per_class (mean 0.7509):

| class      |   iou_per_class |
|------------|-----------------|
| boundary   |          0.4827 |
| pet        |          0.8549 |
| background |          0.9150 |
```

`per_class_vectors` selects **by shape, not by metric name** (a flat numeric sequence of length ≥ 2):
a runnable emits scalars, vectors and matrices in one mapping with no knowledge of which is which,
so a name-based rule would need a convention nothing enforces. Scalars and matrices fall through,
which is what lets the same mapping be passed to both selection rules.

`render_matrix_as_figure` builds a `matplotlib.figure.Figure` **directly**, never through
`pyplot` — a pyplot figure is registered in a process-global manager and lives until someone
calls `pyplot.close`, so a value rendered once an epoch would leak a figure per epoch in a caller
with no reason to know pyplot has global state.

`log_text(name, text, step=None)` is the third content channel beside `log_metrics` (numbers)
and `log_figure` (images). Every backend has it: unlike `log_figure`, whose default is a skip,
its default is real work — a `.md` file written into the backend's own `log_dir` (or uploaded
via `log_artifact` when there is none), so a caller never has to ask whether this backend can
hold a table. Where a backend has a native channel it uses that INSTEAD:

| Backend | `log_text` lands in |
|---|---|
| MLflow | `client.log_text` → `text/<name>_<step>.md`, previewed inline in the UI |
| TensorBoard | `add_text` → the **Text** tab, with `step` driving the slider |
| Aim | `aim.Text` tracked as a sequence, so every step is kept |
| ClearML | an artifact named `text/<name>_<step>.md` — *not* `report_text`, which writes to the console stream where a per-epoch table is unusable |
| W&B | the inherited `.md` file, which W&B syncs from the run directory (it has no text primitive) |
| CSV | the inherited `.md` file under the run directory |

Note the contrast with `log_manifest`, where a native slot is written *in addition* to the
artifact: there the artifact is a union guarantee that keeps two differently-tracked runs
comparable, whereas here the text *is* the payload, so one durable home per backend is the whole
requirement. `step` is part of the filename rather than a version of one file, so a value logged
once an epoch keeps its history on a backend whose native channel has no step axis.

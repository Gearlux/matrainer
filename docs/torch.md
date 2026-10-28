# `matrainer.torch`

The torch-only, Lightning-free helpers: checkpoint loading, device resolution and the dump spellings.

Runnable twin: [`examples/torch_helpers.py`](../examples/torch_helpers.py) — the three checkpoint shapes loaded through one call, a state dict refused by `TorchModelFile`, `cpu` taken at its word, and the per-template sizing kwarg.

## `matrainer.torch` — torch-only, Lightning-free helpers (`matrainer[torch]`)

- `load_lightning_checkpoint(model, path, strict=False)` — loads a Lightning `.ckpt` (strips the
  `model.` prefix), an Ultralytics `.pt` (pulls the live module's `state_dict`), or a plain
  state-dict; auto-retargets to a named child when keys match one. `strict=False` tolerates a
  re-headed `num_classes`.
- `TorchModelFile(file=…)` — the WHOLE-MODEL counterpart: loads a complete saved model
  (`torch.save(model, path)`) lazily from one file and uses it exactly as saved. Registered
  `role="model"`, so canvas/discovery surfaces offer it as an object node docking into any torch
  model slot. A file holding a dict (a state dict, a training checkpoint) is refused with the fix
  named — those carry no architecture and load through a wrapper's `checkpoint_path` instead. Two
  inherent caveats: the pickle stores an import *reference* to the model's class (the defining
  package must be installed where the file loads), and un-pickling executes code from the file —
  load only files you trust.
- `build_metric_collection(metrics, **dims)` — normalize any metrics spec (a mapping, a list, a
  single metric, an already-built `MetricCollection`) into one `MetricCollection` the caller clones
  per split, flowing each template with the sizing kwarg (`num_classes=` / `num_labels=` /
  `data_range=`). The kwarg is applied **per template**, not blanket: a template only receives a
  dimension its constructor DECLARES, and never one it has already PINNED — so the injection is a
  floor a config can override, and a metric that takes no such argument can sit beside ones that do.
  (Three of torchmetrics' seven full-reference image metrics take `data_range`; the blanket form made
  wiring one of the other four break the whole collection.)
- `coerce_metric_value(value)` / `scalar_metrics(raw)` — JSON-safe coercion of a torchmetrics
  `compute()` result; `scalar_metrics` keeps only the scalars, so an `N×N` confusion matrix is
  dropped — those channels hold one number per name, and so is a `[C]` per-class vector. Report
  such a metric separately with `matrainer.tracking.render_matrix_as_table` /
  `render_vector_as_table` — see
  [reporting a metric that is not a single number](metrics.md#reporting-a-metric-that-is-not-a-single-number).
- `MetricPreset(category=…, task=…)` — instantiate every compatible torchmetrics metric in a
  category, for a config that wants "all the classification metrics" without naming them.
- `resolve_device(choice="auto")` + the closed `Device` literal (`"auto"`/`"cpu"`/`"cuda"`/`"mps"`)
  — where the run executes. `"auto"` prefers CUDA, then Apple's MPS, then CPU; anything else is
  taken at its word, so an operator who wrote `cpu` to compare against a GPU run gets CPU, and one
  who wrote `cuda` on a machine without it gets torch's own error instead of a silent fallback that
  would make the benchmark meaningless. A `Trainer` makes this choice for you; a hand-written loop
  has to make it itself, and every such loop was making it identically.
- `apply_mps_low_watermark(value)` — on Apple's GPU, torch keeps the memory it has cached, and on a
  Mac that memory is the machine's: a training run can swap the laptop. Every torch trainer
  (Lightning, the owned loop, fastai, Hugging Face) takes `mps_low_watermark` and hands it to torch
  before the model is built, so torch gives its cache back above that share of the GPU's working
  set. Measured on an 18 GB M3 Pro (a segmentation smoke run, batch 1): unset, 11 GB, swapping, an
  8 min 42 s epoch; `0.5`, 9.1-9.5 GB and 2 min 45 s for the whole run.

  ```yaml
  runnable: !class:your.project.Trainer
    mps_low_watermark: 0.5      # unset: torch's default (1.4); ignored off Apple's GPU
  ```

  ```bash
  recordstream run cfg.yaml runnable.mps_low_watermark=0.5
  ```

  It is `PYTORCH_MPS_LOW_WATERMARK_RATIO`: a value already in the environment wins (the log says
  so), a value above the high watermark (`PYTORCH_MPS_HIGH_WATERMARK_RATIO`, torch's default 1.7)
  is refused naming both, and a process that already used the GPU gets a warning — torch read the
  variable then. Not a memory LIMIT: `torch.mps.set_per_process_memory_fraction(0.6)` stopped the
  same run with `MPS backend out of memory`.

```python
from matrainer.torch import build_metric_collection, resolve_device, scalar_metrics

device = resolve_device("auto")                   # cuda -> mps -> cpu, whichever is present

base = build_metric_collection({"accuracy": PartialClass(MulticlassAccuracy)}, num_classes=10)
train_metrics, val_metrics = base.clone(prefix="train/"), base.clone(prefix="val/")
...
final = scalar_metrics(val_metrics.compute())     # {"val/accuracy": 0.94}
```

It all sits outside `matrainer.lightning` so a non-Lightning platform — an Ultralytics loop, or a
hand-written torch training loop — uses it via `matrainer[torch]` without pulling
`pytorch_lightning`. The metric helpers moved here on 2026-08-01 for exactly that reason: they
import only confluid, torch and torchmetrics, so their old home described their first consumer
rather than their dependencies. `matrainer.lightning` re-exports all four, unchanged.

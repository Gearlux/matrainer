# Model packages

A trained model as a folder another tool can run: its graph, its weights and its class names.

## Model packages (`matrainer.model_package`, no extra needed)

A checkpoint alone does not say how to run it: the model answers correctly only on inputs made the
way its training inputs were made, and its raw outputs need the steps that turn them back into the
caller's terms. A **model package** is one folder holding that whole inference graph — a
recordstream flow graph: preprocessing, the predictor, postprocessing — beside the weights it loads
and the class names, so any pipeline runs the model as ONE step.

Write a package with the `PackageModel` runnable (`recordstream run package.yaml`):

```yaml
runnable: !class:matrainer.model_package.PackageModel
  graph: config/model.yaml                 # a flow graph: `flow:` + `outputs:`, includes allowed
  weights:                                 # predictor slot (a dotted key into the graph) -> its file
    flow.detect.op.model.checkpoint_path: ./runs/a/checkpoint_best_total.pth
  graphs: copy                             # copy | reference
  weight_files: copy                       # copy | reference
  class_names: [bird, cat, dog]            # optional, id order
  description: the detector of run a
  output_dir: ./runs/a/package
```

```
runs/a/package/
  package.yaml        # the entry graph, every graph and weights file with its sha256, the class
                      # names, the installed versions of the distributions the graph's classes come from
  graphs/...          # graphs: copy — the graph and every file it includes, in their own layout
  weights/...         # weight_files: copy — each weights file
  class_names.json    # recordstream's LabelMap format (as save_class_names writes it)
```

| setting | `copy` | `reference` |
|---|---|---|
| `graphs` | the package holds the graph files: an edit of the source later changes nothing | the package records their paths relative to itself and follows every edit |
| `weight_files` | the package holds the weights | the package records the weights file's path |

Running the writer again replaces the package; a folder holding anything else is refused untouched
(`PackageModel: runs/a holds files and is not a package (it has no package.yaml)`).

Run a package with the `ModelPackage` op — one step in any flow graph or stream:

```yaml
flow:
  model:
    op: !class:matrainer.model_package.ModelPackage
      path: $PROJECT_ROOT/runs/a/package
      overrides:                           # dotted keys into the package's graph, applied LAST
        flow.detect.op.device: mps
        flow.detect.op.model.score_threshold: 0.3
  score:
    op: ...
    from: model
```

The op reads the package on first use (its constructor reads nothing), fills each weights slot with
the file's absolute path, and applies `overrides` after both — confluid's one rule, the last spec
wins — so any setting of the package's graph is reachable without editing the package. It is one
sealed step: a step-through tracer sees `model`, not the package's inner steps. A worker process
gets the op without its loaded graph and reads the package itself. `read_package(path)` returns
`package.yaml` as plain data. Rationale: [`docs/architecture.md`](architecture.md) record 13.

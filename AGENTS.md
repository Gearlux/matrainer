# Matrainer Mandates

Rules only. The reasons are in [`docs/architecture.md`](docs/architecture.md) (cited as §N) and
the module docstrings; usage is in the [README](README.md). Root `AGENTS.md` rules are not
repeated here.

## Current state

Experiment tracking plus the workspace's shared trainer machinery (user-authorized 2026-08-06,
§9). On PyPI as pre-release `0.1.0a1`; the repo is private.

| area | modules |
|---|---|
| tracking | `tracking/`: `Tracker` + `TrackerGroup` (`base.py`); `ResilientTracker` (`resilient.py`); six backends, one file each (csv, mlflow, tensorboard, wandb, aim, clearml); `manifest.py`, `provenance.py`, `render.py`, `system.py` |
| trainer API | `TrainerRunnable` (`runnable.py`) → family bases `TorchTrainerRunnable`, `KerasTrainerRunnable` → machinery bases `LightningRunnable`, `TorchLoopRunnable`, `FastaiRunnable`, `KerasRunnable`, `TensorFlowRunnable`, `HuggingFaceRunnable`, `JaxLoopRunnable`, `MlxLoopRunnable` |
| framework-free | `metric_slots`, `metric_names`, `metric_functions`, `uri`, `run_names`, `checkpoints`, `artifacts`, `model_package` |
| `matrainer.torch` | `load_lightning_checkpoint`, `TorchModelFile`, metric helpers, `resolve_device`, dump spellings |
| CLI | `cli.py` (factory), `console.py` (the `matrainer` binary), `_exports.py`, `sweep/` |

Every framework and tracker library is behind an extra, imported lazily via `_imports.require`.

**Who owns what:** recordstream owns records, collate and batch read-back, dataset
normalization, loader slots, prediction contracts and sink, the class-weight statistic, the
runnable protocol and `DatasetProcessor` (README "What lives elsewhere"). Consuming projects own
task semantics. A collate or batch reader written here belongs in recordstream.

## Tracking

- `ResilientTracker` (`tracking/resilient.py`) is a composite `Tracker` like `TrackerGroup`, not a
  layer over the abstraction (the next rule's "no wrapper layer" means no adapter of NON-trackers).
  Every wrapped call goes through `run_with_timeout`; a new call never overtakes a queued one; the
  wrapper never raises into a loop except for a missing `tracker:`; its name, run name, `save_dir`
  and `log_dir` are the wrapped tracker's (`tests/tracking/test_resilient.py`). §14.
- `Tracker` is the whole abstraction; no wrapper layer over it. Backends are zero-arg, build their
  handle on first `.experiment`, and import their library inside methods via `require(module,
  extra)`. `import matrainer` loads no tracker or framework (`tests/test_light_import.py`). §2.
- Backends are DISCOVERED (`discover_backends()` + PEP 562 `__getattr__`), never listed in an
  import list or `__all__`; a new backend is one new file
  (`tests/tracking/test_discovery.py::test_a_new_backend_file_needs_no_edit_to_an_existing_file`).
  `[tool.aisland] trackers` stays a sibling of `frameworks`. In-tree backends are free and
  self-hostable. §4.
- Keep all three hookup styles (functions, `TrackingCallback`, `TrackerLogger`); never reimplement
  a `pytorch_lightning` logger. `TrackingCallback` / `keras_tracking_callback` take `runnable=`;
  `TrackingCallback` stamps `failed`/`interrupted` from `on_exception`.
- Git, config-tree and log-file provenance go only through `log_run_context` / `capture_log_file`.
- `log_text`: the base writes a `.md` file; a backend with a native text channel uses it INSTEAD
  (ClearML `upload_artifact`, never `report_text`; W&B inherits the file). `step` is part of the
  filename (`tests/tracking/test_base.py`).
- Renderers build `matplotlib.figure.Figure`, never `pyplot`
  (`tests/tracking/test_render.py::test_the_figure_is_not_registered_with_pyplot`, twin in
  `test_render_vector.py`). Per-class vectors keep their own renderer pair (worst-first);
  `per_class_vectors` selects by SHAPE, never name. `tabulate` gets
  floats + `floatfmt` (`test_render_vector.py::test_the_values_keep_the_precision_the_column_advertises`).
  Keep the blank line after a table title (`test_render.py::test_a_blank_line_separates_the_title_from_the_table`).
- Checkpoint policies are declarative carriers; each framework subpackage translates them
  (`build_checkpoint_callbacks`). System metrics stay framework-free in `tracking.system`.
- Run names: `generate_run_name(prefix=run_prefix(task))`, never a bespoke timestamp or uuid;
  `run_prefix` is many-to-one and never raises (`tests/test_run_names.py`).
- `matrainer.artifacts` delegates to `LabelMap.save`/`.load` and is a no-op on `None`
  (`tests/test_artifacts.py`).

## Run manifest (§3)

- Every fact lands in the manifest; a native slot is written ALSO, never instead. The base
  `log_manifest` projects `params` via `log_hyperparams`; an override calls `super()` first.
- `log_run_context` is the one entry and captures datasets, models and packages itself; never a
  per-consumer `record_dataset`/`record_model` call. `kind`/`name` come from the URI, so capture
  holds no source-specific code. A capture failure is a debug log, never a raise.
- `capture_log_file` finalizes every tracker LAST, after the pushes; `finalize=False` is the
  escape hatch; a backend never finalizes itself
  (`tests/tracking/test_provenance.py::test_the_run_is_ended_AFTER_the_artifacts_are_pushed`,
  `::test_capture_log_file_can_upload_without_ending_the_run`; consumer-side
  `test_the_terminal_call_is_the_one_that_ends_the_run`).
- `primary_code()` returns `None` rather than `entries[0]`
  (`tests/tracking/test_manifest.py::test_primary_code_is_none_when_nothing_ties_the_run_to_a_repository`).
- `capture_code` returns a LIST. `environment_snapshot` keeps a `SECRET_MARKERS` name, redacts the
  value. `accelerator_state` reads `sys.modules.get("torch")`, never imports, and stays separate
  from `resolve_device`. `config_params` is the one hyperparameter derivation.
- `configured_slots` (objects) stays distinct from `config_params` (scalars) and reads `_modules`
  children, detected structurally, never via `isinstance`. Fakes must reproduce the diversion
  (`tests/tracking/test_provenance.py::test_a_child_object_a_framework_diverts_is_still_a_slot`).
- `matrainer/uri.py` is top level, stdlib-only, not re-exported
  (`tests/test_model_uri.py::test_the_module_imports_nothing_but_the_standard_library`).
  `model_refs` is a LIST of `ModelRef`s with a `role` (datasets expose two strings; keep the
  asymmetry). An empty list is an answer (`refs is not None`). Asking never builds: flow only a
  deferred marker whose class could answer (`declares_key`, never `hasattr(cls, "model")`; the
  never-built group in `tests/test_model_uri.py`). `hf_model_ref` is the one `hf://models/`
  spelling; a zoo without a web page reports no url.
- Portable roots are contracted once over the whole document (`to_dict` and `section`), never at a
  capture site; `env.roots` is skipped; boundary match, longest root wins; a word with a variable
  is double-quoted (`tests/tracking/test_provenance.py::test_expandvars_is_the_exact_inverse`).
  Keep the autouse `tests/conftest.py::no_ambient_portable_roots`.
- `command.argv` and `command.line` both stay; `line` is omitted for `UNREPRODUCIBLE_ARGV0`.
- `local_file_facts`: size always, digest up to `MAX_DIGEST_BYTES`, nothing for remote URIs or
  directories, computed before contraction.
- Side files from `RunManifest.artifacts()` travel with the manifest; `_write_artifact` is the one
  side-artifact write. Both manifest files are rewritten atomically after EVERY mutation. The
  YAML is tag-free; the resolved config is its own `config/final.yaml`. `pyyaml` stays a core dep.
- Torch dump spellings (`torch/spellings.py`): each recipe gets a dump → load → equivalence test;
  tensors decline above `TENSOR_DUMP_CAP`; metrics never use the generic slot walk
  (`tests/test_dump_spellings.py`).

## Model packages (§13)

- A package is a FOLDER: `package.yaml` + `graphs/` / `weights/` when copied + `class_names.json`
  (via `save_class_names`). The author's graph files are copied byte for byte in their own layout,
  never rewritten; the weights path is never written into a graph — `package.yaml` maps the slot
  (`tests/test_model_package.py::test_a_copied_package_holds_every_graph_file_the_weights_and_the_class_names`).
- `ModelPackage` loads `{include: [graph], <slot>: <weights>, **overrides}` through `confluid.load`:
  the one document-order rule, overrides last. Never a precedence or a spelling of its own
  (`::test_overrides_are_applied_last`, `::test_an_override_naming_an_included_block_reaches_it`).
- `graphs` and `weight_files` stay two separate `copy | reference` settings (user choice
  2026-10-12: both must be possible) (`::test_graphs_and_weights_are_copied_or_referenced_independently`).
- `ModelPackage` is a sealed op holding a private `Subgraph`, NOT a `Subgraph` subclass (a tracer
  rebuilds subgraphs with `steps=`); its constructor reads nothing, it reloads when `path` or
  `overrides` change, and it pickles without the loaded graph
  (`::test_constructing_the_op_reads_nothing`, `::test_a_changed_override_takes_effect_on_the_next_record`,
  `::test_an_op_that_has_run_pickles_without_its_loaded_graph`).
- The writer replaces a package and refuses any other non-empty folder untouched; it writes into
  `<output_dir>.partial` and moves it into place last
  (`::test_a_folder_that_is_not_a_package_is_never_written_into`). `package.yaml` carries the
  generated-file header (`::test_the_package_file_carries_the_generator_header`).
- `load_package_graph(path, overrides, owner=)` is the ONE loader of a package: the op calls it, and so
  does torpaido's compiler (`owner="CompilePackage"`), so the loading rule is written once. Its
  `request` is what `confluid.load` was given (the entry graph included, the weight slots, the
  overrides), for a reader that needs the same document at another stage
  (`::test_load_package_graph_gives_the_graph_the_op_runs`, `::test_load_package_graph_names_its_caller_in_a_refusal`).
- A compiled form of a package (torpaido's `compiled.yaml` + artifact) lives in `<package>/compiled/<name>/`
  by default; `PackageModel` replacing the package deletes them, as intended — they were of the old weights
  (§13 amendment). matrainer never reads or writes that folder.

## Backend behaviour (measured; the docstrings say why)

- Lineage projections are rule 2 (also, never instead). Test fakes ENFORCE the real library's name
  and scheme rules.
- MLflow: keep the `_logged_models` memo; names are sanitised labels; read lineage back with
  `client.get_run(id).inputs.model_inputs`, not `search_runs`
  (`tests/tracking/test_mlflow_backend.py::test_a_second_log_manifest_does_not_duplicate_the_models`).
- W&B: `_add_reference` keeps its own `try`; `expandvars` the URI first; only
  `REFERENCEABLE_SCHEMES` become references
  (`tests/tracking/test_wandb_backend.py::test_an_unresolvable_reference_still_reaches_the_lineage_graph`).
- ClearML: `auto_connect_frameworks=False`
  (`tests/tracking/test_clearml_backend.py::test_frameworks_are_never_auto_connected`). Only
  `ADDRESSABLE_SCHEMES` reach `import_model`; offline skips it once; the memo is written before
  `connect`; the models section stays. Tests restore `CLEARML_OFFLINE_MODE` via `monkeypatch`.
  `clearml.Dataset` lineage is deliberately not built (decided 2026-08-03, do not reopen) unless
  ClearML addresses `hf://` or datasets move to `s3://`. §3.
- Aim: `log_hyperparams` merges by hand; `set_tags` resolves the run first
  (`tests/tracking/test_aim_backend.py::test_set_tags_before_the_run_exists_does_not_double_add`);
  `log_dir` is `None`; `capture_terminal_logs=False`. Run values are the lineage ceiling (no fake
  registry). An empty `iter_runs()` right after a run is expected.

## Trainer machinery (§9)

- Four layers: neutral `TrainerRunnable` (no model/loss/metric slot) → family base naming the
  framework types → machinery base → the consumer's concrete class. Nothing user-facing is `Any`.
- No abstract trainer OBJECT; machinery bases share module functions only (`flow_trackers`,
  `write_collection_sidecar`, `report_matrices`/`report_vectors`).
- A concrete class keeps its FULL flat `__init__` (no `**kwargs`) and stores task state before
  `super().__init__(...)`.
- `model` is `Optional[Mandatory[Partial[T]]] = None`; `_require_model()` raises at the top of
  every `run()`, after that base's own precondition
  (`tests/test_runnable.py::test_an_empty_model_slot_raises_a_clear_error_at_run`,
  `::test_a_runnable_without_a_model_slot_is_untouched_by_the_guard`).
- A machinery hook never shares a name with a machinery method that consumers shadow-scan.
- The no-op metric builders (fastai, huggingface, keras family, jax, mlx) stay no-ops; a
  torchmetrics-scoring consumer restores the family builders. Every base's `evaluate()` scores
  VAL; the huggingface, jax and mlx bases also record eval runs.
- `KerasRunnable` fits with `shuffle=False`
  (`tests/test_keras_family.py::test_the_fit_never_lets_keras_shuffle_batch_order`); a
  tracker-touching write goes in `_before_fit`. `TensorFlowRunnable` clones per-split metric
  objects, traces steps over locals (never `self`), builds the optimizer outside the graph.
  `HuggingFaceRunnable`: `report_to="none"` + `log_history` replay, a config-pinned
  `TrainingArguments` key beats the floor, no `train_metrics` param, `metric_key_prefix="test"`,
  no module-level transformers import. Jax jits each step once; mlx `mx.eval`s parameters,
  optimizer state and loss every step. `FastaiRunnable` keeps `from fastai.callback import
  schedule`.
- A base change ships with every consumer updated in the same workspace change.

## Metrics

- `metric_slots`: top level, stdlib-only, not re-exported; no consumer re-derives the merge
  (`tests/test_metric_slots.py::test_the_module_imports_nothing_but_the_standard_library`).
- `metric_names` (§6): idempotent parse; no alias inversion; `epoch`/`step` pass through; an
  unknown framework raises; rename the tracker's copy, never the engine's `logs`; never inside
  `Tracker.log_metrics`. It IS re-exported. A hand-written loop calls
  `as_metric_names(self.metric_names)` (`tests/test_metric_names.py`).
- `metric_functions` (§7): top level, numpy only, not re-exported. Out-of-range targets raise;
  single-label only; `np.bincount`; an absent class's IoU is `nan`; `ignore_index` drops before
  the range check. Tests pin agreement with sklearn/torchmetrics and batch-size independence
  (`tests/test_metric_functions.py`).
- `build_metric_collection` sizes only templates that DECLARE the dimension (`inspect.signature`,
  never `confluid.accepts_key`) and have not pinned it (`tests/test_metrics.py::TestSizingKwargs`).

## `matrainer.torch`, `matrainer.lightning`, `matrainer.keras`

- `matrainer.torch` never imports `pytorch_lightning` *(unpinned here)* and is no grab bag: data
  helpers go to recordstream, a framework's training convention to the consuming runnable.
  `matrainer.lightning` keeps re-exporting the four metric helpers. §5.
- `load_lightning_checkpoint` learns no library's own format, and a zero-match load stays an INFO
  line, never an error (user ruling 2026-09-27, do not re-litigate). §12. *(unpinned)*
- `TorchModelFile`: `role="model"`, `framework="torch"`, no task; refuses a dict-holding file
  naming the `checkpoint_path` fix; own entry point (`tests/test_model_file.py`).
- `resolve_device` takes an explicit choice at its word
  (`tests/test_device.py::test_an_explicit_choice_is_taken_at_its_word`).
- `TrackerLogger.save_dir` returns `self.tracker.save_dir`, which places Lightning checkpoints
  under `runs/` (root "Run Output Lands In A Gitignored `runs/`"). Never drop it; never add an
  independent checkpoint root
  (`tests/lightning/test_trackers.py::test_checkpoints_land_under_the_trackers_save_dir_not_the_launch_directory`). §11.
- An extra names the API it imports: `[keras]` is keras, not tensorflow. Import keras only via
  `require("keras", "keras", feature=…, via="recordstream.keras")`; never a bare `import keras`
  (`tests/test_keras.py::test_the_callback_builds_with_no_KERAS_BACKEND_chosen`,
  `::test_a_missing_keras_still_names_keras_not_the_shim`).

## CLI and sweep

- `matrainer.cli` is the factory (§10). A project's `cli.py` is its FRAMEWORKS table plus two
  calls; the factory never learns a task (`verb_docs=`). Pre-step order `requested_framework` →
  `pin_keras_backend` → `register_frameworks` → `normalize_framework_flag` → `app.run()`
  (`tests/test_cli.py::test_the_pre_step_order_is_the_established_one`). `--framework` is
  rewritten to `--scope framework=…`, never appended. A runnable with no `task` runs for `train`
  only; other verbs raise (`::test_a_taskless_runnable_trains_as_is_and_refuses_the_other_verbs`).
  Package roots use `lazy_exports`; an empty extra marks a core name (`tests/test_exports.py`).
- ONE binary, the `matrainer` console script (user-authorized 2026-08-07, do not re-litigate):
  `train|evaluate|predict <config.yaml>` plus `sweep`, built from `make_trainer_app(frameworks=None)`.
  It uses `materialize_runnable`, never a bare `flow()`, and strips `--registrations` before
  liquifai. No task verbs, FRAMEWORKS table, registrations knowledge, `download`/`convert`/
  `process` verbs or `matrainer.commands` plugin group (`tests/test_console.py`).
- Sweep (§8): driver, naming and read-back live here (`matrainer sweep` and
  `python -m matrainer.sweep` are one `main`); the matrix, engine quirks and metric
  alignment stay in the project's `examples/sweep/`. One subprocess per cell, row `env` scoped to
  it, failures isolated, report before the non-zero exit. Numbering counts from the tracker.
  `QUERYABLE` is what `collect` reads. `save_dir` moves written logs too. Folded columns are
  hidden, never dropped (`--all`); `render` takes a `note` for a tracker it cannot read.
  (`tests/test_sweep.py`, `tests/test_sweep_runner.py`)

## Packaging, release, tests

- "Light core" means framework-free, not small: never describe it as a short package list. A heavy
  library goes behind an extra plus a lazy `require(...)`.
- Releases use the hand-written `release.yml` (scaffold owns only `ci.yml`/`Jenkinsfile*`), with
  recordstream's discipline: an extra may name only a PyPI package (the release resolves every
  extra); every workspace spec on matrainer, extras included, reads `>=0.1.0a1` (aisland
  `tests/test_workspace_dependency_specs.py`) until a final `0.1.0`, then all change in one pass;
  floors state what the suite runs against; README links are absolute
  (`tests/test_docs_links.py`); the newest CHANGELOG entry is the version (`tests/test_packaging.py`).
- CI runs `mypy .` here with only its extras and stricter flags: list every extra-gated import CI
  lacks in `[[tool.mypy.overrides]] ignore_missing_imports`, and reproduce CI in a throwaway
  Python 3.12 venv before pushing.
- Keep `addopts = "--cov-branch"`; never delete `branch = true` to fix a combine error
  (`tests/test_coverage_config.py`).

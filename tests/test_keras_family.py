"""The Keras family bases — `KerasTrainerRunnable` + the two machinery siblings.

What this pins is the INHERITED wiring the task projects' fleet tests exempt their concretes
from showing: the sentinel resolution via the keras task hooks, the union-as-family-property,
the deliberate no-op metric builders, the default batch shape, the tracking callback joining
wired trackers, the compile step's place in `run()`, the artifact prologue's ordering (BEFORE
the fit — the tracking callback's `on_train_end` closes the run out), the harmonized
`evaluate`-scores-val rule, and — on the TensorFlow side — the per-split metric CLONE, the
outside-the-graph `optimizer.build`, and the per-epoch reshuffle.

Structure-level where behaviour needs an engine (the four task projects run the real fits in
their own suites, per engine); behavioural where keras-on-any-engine suffices.
"""

from __future__ import annotations

import inspect
from typing import Any, List, Optional

import pytest

keras = pytest.importorskip("keras")

from confluid import Lazy  # noqa: E402

from matrainer.keras.runnable import KerasLoss, KerasMetric, KerasTrainerRunnable  # noqa: E402
from matrainer.keras.trainer import KerasRunnable  # noqa: E402

# --------------------------------------------------------------------------- #
# The family base
# --------------------------------------------------------------------------- #


class _Hooked(KerasTrainerRunnable):
    """A minimal task-hooks stand-in over the family base."""

    def _default_keras_loss(self) -> Optional[Lazy[KerasLoss]]:
        return keras.losses.MeanAbsoluteError()

    def _default_keras_metrics(self) -> Optional[List[Lazy[KerasMetric]]]:
        return [keras.metrics.MeanSquaredError(name="mse")]

    # The abstract entry points, never driven here.
    def initialize(self) -> None:  # pragma: no cover - structure only
        pass

    def fit(self) -> None:  # pragma: no cover
        pass

    def evaluate(self) -> None:  # pragma: no cover
        pass

    def test(self) -> None:  # pragma: no cover
        pass

    def predict(self) -> None:  # pragma: no cover
        pass


def test_the_sentinels_resolve_via_the_task_hooks_idempotently() -> None:
    """`None` means "not chosen"; the hooks fill it ONCE and a wired value is never replaced."""
    runnable = _Hooked()
    assert runnable.loss is None and runnable.metrics is None
    runnable._resolve_task_defaults()
    assert isinstance(runnable.loss, keras.losses.MeanAbsoluteError)
    assert runnable.metrics is not None and [m.name for m in runnable.metrics] == ["mse"]

    chosen = keras.losses.MeanSquaredError()
    wired = _Hooked()
    wired.loss = chosen
    wired._resolve_task_defaults()
    assert wired.loss is chosen, "an explicitly wired loss must never be replaced"


def test_an_unanswered_hook_leaves_the_sentinel_for_a_later_ask() -> None:
    """A task hook may need a dimension `initialize` has not resolved yet — `None` stays."""

    class _Unanswered(_Hooked):
        def _default_keras_loss(self) -> Optional[Lazy[KerasLoss]]:
            return None

        def _default_keras_metrics(self) -> Optional[List[Lazy[KerasMetric]]]:
            return None

    runnable = _Unanswered()
    runnable._resolve_task_defaults()
    assert runnable.loss is None and runnable.metrics is None


def test_the_compiled_metric_list_is_the_union_of_the_four_slots() -> None:
    """`compile()` owns ONE list, so the four slots UNION — the family property."""
    runnable = _Hooked()
    shared, val_only = keras.metrics.MeanSquaredError(name="shared"), keras.metrics.MeanSquaredError(name="val_only")
    runnable.metrics = [shared]
    runnable.val_metrics = [val_only]
    assert list(runnable._compiled_metrics()) == [shared, val_only]


def test_the_metric_builders_are_deliberate_noops() -> None:
    """Keras metrics resolve at compile / step build — there is no prefixed-collection state."""
    runnable = _Hooked()
    runnable.metrics = [keras.metrics.MeanSquaredError()]
    runnable._build_fit_metrics()
    runnable._build_eval_metrics()
    assert isinstance(runnable.metrics, list), "the builders must not transform the specs"


def test_the_default_batch_shape_is_the_image_to_image_pair() -> None:
    import numpy as np

    runnable = _Hooked()
    batch = {"image": np.ones((4, 8, 8, 3), dtype="uint8"), "target": np.zeros((4, 8, 8, 3), dtype="uint8")}
    x, y = runnable._to_arrays(batch)
    assert x.dtype.name == "float32" and y.dtype.name == "float32"
    (x_only,) = runnable._to_arrays(batch, with_targets=False)
    assert x_only.shape == (4, 8, 8, 3)


def test_a_tracking_callback_joins_wired_trackers(tmp_path: Any) -> None:
    """`runnable=self` rides the family's `_callbacks` — the one hyperparameter derivation."""
    from matrainer.tracking.csv_backend import CsvTracker

    runnable = _Hooked()
    assert runnable._callbacks() == [], "no trackers, no callback"
    runnable.trackers = CsvTracker(save_dir=str(tmp_path))
    callbacks = runnable._callbacks()
    assert len(callbacks) == 1 and "KerasTracking" in type(callbacks[0]).__name__
    assert "runnable=self" in inspect.getsource(KerasTrainerRunnable._callbacks)


# --------------------------------------------------------------------------- #
# The compile/fit machinery
# --------------------------------------------------------------------------- #
def test_run_compiles_between_the_tasks_initialize_and_the_dispatch() -> None:
    body = inspect.getsource(KerasRunnable.run).split('"""')[-1]
    assert body.index("initialize()") < body.index("_compile()") < body.index("run_entrypoint")


def test_the_artifact_prologue_runs_before_the_fit() -> None:
    """The tracking callback's `on_train_end` closes the run out, so a tracker-touching write
    sequenced after `model.fit` returns would arrive at an ENDED run."""
    body = inspect.getsource(KerasRunnable.fit)
    assert body.index("_before_fit()") < body.index("model.fit(")


def test_the_fit_never_lets_keras_shuffle_batch_order() -> None:
    """`model.fit(shuffle=False)` — the sequence owns row order (private seeded RNG, reshuffled
    per epoch via `on_epoch_end`, which Keras drives). Keras's PyDataset shuffle permutes BATCH
    order through the global `random` module, so a seeded run's trajectory would depend on how
    many draws ambient code consumed first — measured: mlflow's telemetry client draws one
    `random.randint` at run creation, which alone moved a seeded cell's final loss."""
    body = inspect.getsource(KerasRunnable.fit)
    assert "shuffle=False" in body[body.index("model.fit(") :]


def test_evaluate_scores_the_validation_split() -> None:
    """The fleet harmonization: `evaluate` used to delegate to `test()` in every keras backend."""
    assert '_evaluate_split(self.val_set, "val")' in inspect.getsource(KerasRunnable.evaluate)
    assert '_evaluate_split(self.test_set, "test")' in inspect.getsource(KerasRunnable.test)


def test_the_compile_seams_are_overridable() -> None:
    """A task whose loss is not a slot (the detection spec's positional list) overrides
    `_compiled_loss`; one that translates knobs overrides `_compiled_optimizer`."""
    source = inspect.getsource(KerasRunnable._compile)
    assert "_compiled_optimizer()" in source and "_compiled_loss()" in source
    assert "_check_compiled_metrics(" in source


# --------------------------------------------------------------------------- #
# The TensorFlow machinery (structure-level: the engine binds per process)
# --------------------------------------------------------------------------- #
tf = pytest.importorskip("tensorflow")

from matrainer.tensorflow.loop import TensorFlowRunnable  # noqa: E402


def test_a_live_metric_is_cloned_per_split() -> None:
    """A metric object is a stateful ACCUMULATOR — three splits sharing one instance is silent
    cross-contamination (one project's TF backend shipped exactly that for a month)."""

    class _TfHooked(TensorFlowRunnable):
        def initialize(self) -> None:  # pragma: no cover - structure only
            pass

    runnable = _TfHooked()
    live = keras.metrics.MeanSquaredError(name="mse")
    clone = runnable._as_metric(live)
    assert clone is not live and type(clone) is type(live) and clone.name == "mse"


def test_the_optimizer_builds_outside_the_graph_and_the_sequence_reshuffles() -> None:
    body = inspect.getsource(TensorFlowRunnable.fit)
    assert body.index("_optimizer.build(") < body.index("on_train_begin()")
    assert "sequence.on_epoch_end()" in body, "forgetting the reshuffle is invisible"
    assert body.index("_before_fit()") < body.index("_run_epoch(")


def test_the_steps_close_over_locals_not_self() -> None:
    """Tracing captures the closure — `self` inside a traced fn drags the whole runnable in."""
    for builder in (TensorFlowRunnable._train_step_fn, TensorFlowRunnable._eval_step_fn):
        # The body AFTER the decorator line — the decorator itself may read self.jit_compile.
        after_decorator = inspect.getsource(builder).split("@tf.function", 1)[1]
        inner = after_decorator.split("def ", 1)[1]
        assert "self." not in inner, f"{builder} captures self inside the traced step"


def test_evaluate_scores_the_validation_split_here_too() -> None:
    assert '_evaluate_split(self.val_set, "val")' in inspect.getsource(TensorFlowRunnable.evaluate)


def test_the_epoch_splats_the_batch() -> None:
    """`step(*batch)` — a task whose batch is not a pair (dense detection maps) drives the same
    loop with its own step arity."""
    assert "step(*batch)" in inspect.getsource(TensorFlowRunnable._run_epoch)

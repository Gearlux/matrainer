"""``TensorFlowRunnable`` — the native-TensorFlow machinery base of the trainer API.

The hand-written member of the Keras family (``docs/architecture.md`` record 9): a
``tf.data.Dataset`` over the family's ``RecordSequence``, ``tf.GradientTape`` steps under
``@tf.function`` (which is what makes ``jit_compile``/XLA a knob at all), the epoch loop
``fit()`` would have run, a ``keras.callbacks.CallbackList`` driven by hand, and ``_export`` —
the SavedModel, the one run artifact that is TensorFlow's and not Keras's. It SHARES the
family base (:class:`matrainer.keras.runnable.KerasTrainerRunnable` — the keras types, the
``_to_arrays`` hook, the sequence, the sentinels) with the compile/fit member and inherits
NOTHING from it: the two machinery bases are siblings, exactly as the Lightning and bare-torch
bases are.

What owning the loop buys over ``compile()``, and this base therefore does differently:

* **Per-split metric OBJECTS** — each traced step closes over its own split's list
  (:meth:`_split_metrics`, ``merge_metric_specs`` narrowing properly where ``compile()`` can
  only union), and a LIVE metric wired into a spec is CLONED per split through Keras's own
  ``get_config()``/``from_config()`` — a metric object is a stateful ACCUMULATOR, so splits
  with overlapping specs must not share instances (measured: the clone reports 0.0 where the
  original, after an update, reports 1.0; one project's TF backend shipped the shared-instance
  bug for a month because nothing raises).
* **The two ordering rules a traced loop imposes**, both measured: ``optimizer.build(...)``
  runs OUTSIDE the graph (Keras builds slot variables on the first ``apply_gradients``, and
  creating variables during a RETRACE — which a short final batch triggers — is an error), and
  the step functions are built ONCE and reused across epochs (a fresh trace per epoch retraces
  the whole graph).
* **``sequence.on_epoch_end()`` once per epoch** — the per-epoch reshuffle a ``fit()`` would
  have done. Forgetting it is INVISIBLE: the run trains every epoch on the same row order.

The engine is REQUIRED, not preferred: :meth:`_require_tensorflow_backend` raises unless Keras
is bound to TensorFlow — a torch-backed ``keras.Model`` cannot be trained with a
``GradientTape``. This module is itself the lowest layer that imports the engine, so it
``setdefault``s ``KERAS_BACKEND=tensorflow`` BEFORE anything reads it (the same ordering rule
every project's ``_tensorflow.py`` shim encodes: an import sorter must not be able to put a
keras import above the default).

Tracking is the family's: the tracking callback rides the ``CallbackList``, so provenance,
epoch logs and the run lifecycle arrive exactly as they do under ``model.fit``.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")  # BEFORE tensorflow/keras load — see the docstring

from pathlib import Path  # noqa: E402
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union, cast  # noqa: E402

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from confluid import Lazy, configurable, flow  # noqa: E402
from loggair import get_logger  # noqa: E402
from recordstream import PredictionsSink, RecordSource  # noqa: E402
from recordstream.keras import keras, keras_backend  # noqa: E402
from recordstream.runnable import RunnableTask, run_entrypoint  # noqa: E402

from matrainer.artifacts import write_metrics_sidecar  # noqa: E402
from matrainer.keras.runnable import KerasMetric, KerasOptimizer, KerasTrainerRunnable  # noqa: E402
from matrainer.metric_names import MetricNames, as_metric_names  # noqa: E402
from matrainer.metric_slots import merge_metric_specs  # noqa: E402
from matrainer.tracking.base import Tracker  # noqa: E402

logger = get_logger(__name__)

__all__ = ["TensorFlowNetwork", "TensorFlowRunnable"]

#: What the model slot admits: a ``keras.Model``, or a raw ``tf.Module`` speaking the same
#: ``__call__`` contract. NOT a union with a common ancestor — a ``keras.Model`` descends
#: ``KerasAutoTrackable`` and is not a ``tf.Module``; their only shared base is private.
TensorFlowNetwork = Union[keras.Model, tf.Module]


def _output_signature(batch: Any) -> Any:
    """``tf.TensorSpec``s for one materialized batch — what ``from_generator`` needs declared."""
    arrays = batch if isinstance(batch, tuple) else (batch,)
    return tuple(
        tf.TensorSpec(shape=(None, *np.asarray(a).shape[1:]), dtype=tf.as_dtype(np.asarray(a).dtype)) for a in arrays
    )


def _as_optimizer(value: Any) -> Any:
    """A live Keras optimizer from whatever the slot holds — instance, string alias, bare
    class, or a deferred marker. The bare-class branch is tested BEFORE ``flow`` because
    ``flow()`` on a bare registered class would INSTANTIATE it through ``materialize``."""
    if isinstance(value, type):
        return value()
    built = flow(value)
    if isinstance(built, str):
        return keras.optimizers.get(built)
    return built


def _call_network(network: Any, x: Any, training: bool) -> Any:
    """One forward pass, tolerant of a network whose ``__call__`` lacks ``training=``.

    A raw ``tf.Module`` need not declare the kwarg (train/eval may be module state or simply
    absent); the ``except TypeError`` fallback is traced once and costs nothing after.
    """
    try:
        return network(x, training=training)
    except TypeError:
        return network(x)


# `@configurable(validate=False)` for the same reason as the other bases: the mark makes
# confluid's body-slot scans see THIS layer's slots in a concrete subclass's MRO.
@configurable(validate=False)
class TensorFlowRunnable(KerasTrainerRunnable):
    """A native tf.data + GradientTape train/evaluate/test/predict runnable, task hooks open.

    Args (on top of the family base's):
        jit_compile: Compile the traced steps with XLA.
        prefetch: ``dataset.prefetch(AUTOTUNE)`` between the sequence and the loop.
        export_dir: Where ``fit()`` writes the SavedModel. ``None`` skips the export.
    """

    def __init__(
        self,
        *,
        model: Optional[Lazy[TensorFlowNetwork]] = None,
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        task: RunnableTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 32,
        jit_compile: bool = False,
        prefetch: bool = True,
        export_dir: Optional[str] = None,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        callbacks: Optional[List[Lazy["keras.callbacks.Callback"]]] = None,
        experiment_name: str = "matrainer",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
        optimizer: Lazy[KerasOptimizer] = "adam",
        loss: Optional[Lazy[Any]] = None,
        metrics: Optional[List[Lazy[KerasMetric]]] = None,
        train_metrics: Optional[List[Lazy[KerasMetric]]] = None,
        val_metrics: Optional[List[Lazy[KerasMetric]]] = None,
        test_metrics: Optional[List[Lazy[KerasMetric]]] = None,
        metric_names: Optional[Lazy[MetricNames]] = None,
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            batch_size=batch_size,
            trackers=trackers,
            callbacks=callbacks,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            metric_names=metric_names,
        )
        self.jit_compile = bool(jit_compile)
        self.prefetch = bool(prefetch)
        self.export_dir = export_dir
        self._optimizer: Any = None
        self._loss_fn: Any = None
        self._metric_objects: Dict[str, List[Any]] = {"train": [], "val": [], "test": []}

    # ---- run dispatch ----------------------------------------------------------------------- #
    def run(self) -> None:
        self._require_tensorflow_backend()
        self.initialize()
        self._build_live_objects()
        # The `@entrypoint` markers (inherited from the API base) ARE the dispatch table.
        run_entrypoint(self, self.task)

    @staticmethod
    def _require_tensorflow_backend() -> None:
        """This loop is raw TensorFlow — ``tf.GradientTape`` cannot train a torch-backed model.

        Keras 3 binds ONE engine per process, at import; a wrong binding is an operator-level
        fact (re-run with ``KERAS_BACKEND=tensorflow`` in a FRESH process), so the message says
        exactly that instead of failing tensors-deep inside the first step.
        """
        backend = keras_backend()
        if backend != "tensorflow":
            raise RuntimeError(
                f"this backend needs Keras bound to the TensorFlow engine, but KERAS_BACKEND is "
                f"{backend!r}. Re-run in a fresh process with KERAS_BACKEND=tensorflow (Keras 3 "
                "binds one engine per process, at import)."
            )

    # ---- live objects ----------------------------------------------------------------------- #
    def _build_live_objects(self) -> None:
        """Resolve the compile-slot values into the live objects the traced steps close over."""
        self._resolve_task_defaults()
        self.model = flow(self.model)
        self._optimizer = _as_optimizer(self.optimizer)
        self._loss_fn = flow(self.loss)
        self._metric_objects = {split: self._split_metrics(split) for split in ("train", "val", "test")}

    def _split_metrics(self, split: str) -> List[Any]:
        """One split's OWN metric objects — narrowed properly, which is what owning the loop buys.

        The combination rule is :func:`matrainer.metric_slots.merge_metric_specs`, shared with
        every backend and task; each entry then resolves through :meth:`_as_metric`.
        """
        spec = merge_metric_specs(self.metrics, getattr(self, f"{split}_metrics")) or []
        return [self._as_metric(m) for m in spec]

    def _as_metric(self, value: Any) -> Any:
        """A live Keras metric from whatever a spec entry holds — string alias, class, live
        instance, or a deferred marker. Overridable: a task with its own alias vocabulary (a
        contextual ``"accuracy"``) resolves it here before delegating to this default.

        A LIVE instance is CLONED through Keras's own ``get_config()``/``from_config()`` —
        a metric object is a stateful accumulator, and three splits sharing one instance is a
        silent cross-contamination (measured: the clone reports 0.0 where the original, after
        an update, reports 1.0). A metric that cannot round-trip its config is shared with a
        debug note rather than refused — a custom metric without ``get_config`` still works
        for the common case of appearing in ONE split's spec.
        """
        if isinstance(value, type):
            return value()
        built = flow(value)
        if isinstance(built, str):
            return keras.metrics.get(built)
        if isinstance(built, keras.metrics.Metric):
            try:
                return type(built).from_config(built.get_config())
            except (TypeError, ValueError, NotImplementedError):
                logger.debug(f"metric {type(built).__name__} cannot be cloned; sharing the instance across splits.")
        return built

    # ---- the tf.data boundary --------------------------------------------------------------- #
    def _dataset(self, source: Any, shuffle: bool = False, with_targets: bool = True) -> Tuple[Any, Any]:
        """``(tf.data.Dataset, sequence)`` over one split — the sequence rides along because the
        loop must call its ``on_epoch_end()`` (the reshuffle ``fit()`` would have done) and read
        its length for the callback params."""
        sequence = self._sequence(source, shuffle=shuffle, with_targets=with_targets)
        dataset = tf.data.Dataset.from_generator(
            lambda: (sequence[i] for i in range(len(sequence))),
            output_signature=_output_signature(sequence[0]),
        )
        return (dataset.prefetch(tf.data.AUTOTUNE) if self.prefetch else dataset), sequence

    # ---- the traced steps (built ONCE — a fresh trace per epoch retraces the graph) --------- #
    def _train_step_fn(self) -> Callable[..., Any]:
        """One training step: forward under the tape, loss, apply, update the train metrics."""
        network, optimizer, loss_fn = self.model, self._optimizer, self._loss_fn
        metrics = self._metric_objects["train"]

        @tf.function(jit_compile=True if self.jit_compile else None)
        def train_step(x: Any, y: Any) -> Any:
            with tf.GradientTape() as tape:
                logits = _call_network(network, x, training=True)
                loss = loss_fn(y, logits)
            variables = network.trainable_variables
            optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
            for metric in metrics:
                metric.update_state(y, logits)
            return loss

        return cast(Callable[..., Any], train_step)

    def _eval_step_fn(self, split: str = "val") -> Callable[..., Any]:
        """One evaluation step: forward, loss, update that split's metrics. No tape."""
        network, loss_fn = self.model, self._loss_fn
        metrics = self._metric_objects[split]

        @tf.function(jit_compile=True if self.jit_compile else None)
        def eval_step(x: Any, y: Any) -> Any:
            logits = _call_network(network, x, training=False)
            for metric in metrics:
                metric.update_state(y, logits)
            return loss_fn(y, logits)

        return cast(Callable[..., Any], eval_step)

    def _predict_step_fn(self) -> Callable[..., Any]:
        network = self.model

        @tf.function(jit_compile=True if self.jit_compile else None)
        def predict_step(x: Any) -> Any:
            return _call_network(network, x, training=False)

        return cast(Callable[..., Any], predict_step)

    # ---- the loop ---------------------------------------------------------------------------- #
    def _reset_metrics(self, split: str) -> None:
        for metric in self._metric_objects.get(split, []):
            metric.reset_state()

    def _metric_results(self, split: str) -> Dict[str, float]:
        return {metric.name: float(metric.result()) for metric in self._metric_objects.get(split, [])}

    def _run_epoch(
        self, dataset: Any, step: Callable[..., Any], callbacks: Any = None, split: str = "train"
    ) -> Dict[str, float]:
        """One pass over a dataset: mean loss plus that split's metric results, UNPREFIXED.

        ``step(*batch)`` rather than ``step(x, y)`` so a task whose batch is not a pair (dense
        detection maps beside the input) drives the same loop with its own step arity.
        """
        self._reset_metrics(split)
        total, batches = 0.0, 0
        for index, batch in enumerate(dataset):
            total += float(step(*batch) if isinstance(batch, tuple) else step(batch))
            batches += 1
            if callbacks is not None:
                callbacks.on_train_batch_end(index, {"loss": total / batches})
        return {"loss": total / batches if batches else float("nan"), **self._metric_results(split)}

    # ---- fit hooks (the same seams as the compile/fit sibling) ------------------------------- #
    def _before_fit(self) -> None:
        """Task prologue after ``on_train_begin`` — no-op by default; see the sibling base."""

    def _fit_scores(self, final: Dict[str, float]) -> Dict[str, float]:
        """Extend the last epoch's values with task-computed scores — identity by default."""
        return final

    # ---- the four entry points (markers inherited from the API base) ------------------------- #
    def fit(self) -> None:
        logger.info(f"{self.log_label} tensorflow training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError(f"{type(self).__name__}.fit() requires a 'train_set'.")
        train_data, sequence = self._dataset(self.train_set, shuffle=True)
        val_data = self._dataset(self.val_set)[0] if self.val_set is not None else None
        train_step = self._train_step_fn()
        eval_step = self._eval_step_fn("val")
        # OUTSIDE the graph: Keras builds slot variables on the first `apply_gradients`, and
        # creating variables during a RETRACE — which a short final batch triggers — is an error.
        self._optimizer.build(list(self.model.trainable_variables))

        # The model is passed UNCONDITIONALLY: the History callback (add_history=True) assigns
        # `model.history = self` and crashes on None — and a raw tf.Module accepts the
        # attribute assignment exactly as a keras.Model does (measured: a tf.nn spec module
        # trained under this CallbackList for months in two projects).
        callbacks = keras.callbacks.CallbackList(self._callbacks(), add_history=True, model=self.model)
        callbacks.set_params({"epochs": self.max_epochs, "steps": len(sequence), "verbose": 0})
        callbacks.on_train_begin()
        self._before_fit()

        logs: Dict[str, float] = {}
        for epoch in range(self.max_epochs):
            callbacks.on_epoch_begin(epoch)
            logs = self._run_epoch(train_data, train_step, callbacks)
            if val_data is not None:
                logs.update({f"val_{k}": v for k, v in self._run_epoch(val_data, eval_step, split="val").items()})
            callbacks.on_epoch_end(epoch, logs)
            # The per-epoch reshuffle `fit()` would have done. Forgetting it is INVISIBLE —
            # the run trains every epoch on the same row order.
            sequence.on_epoch_end()
            for key, value in logs.items():
                self._history.setdefault(key, []).append(float(value))
            logger.info(f"epoch {epoch + 1}/{self.max_epochs}: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items()))
        callbacks.on_train_end(logs)

        final = self._fit_scores({k: v[-1] for k, v in self._history.items()})
        self._computed_metrics = as_metric_names(self.metric_names)(final, source="keras")
        logger.info(f"Training complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        self._export()

    def evaluate(self) -> None:
        """Score the VALIDATION split — the ``evaluate`` / ``test`` split is the only difference."""
        if self.val_set is None:
            raise ValueError(f"{type(self).__name__}.evaluate() requires a 'val_set'.")
        self._computed_metrics = self._evaluate_split(self.val_set, "val")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    def test(self) -> None:
        if self.test_set is None:
            raise ValueError(f"{type(self).__name__}.test() requires a 'test_set'.")
        self._computed_metrics = self._evaluate_split(self.test_set, "test")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    def _evaluate_split(self, source: Any, split: Literal["val", "test"]) -> Dict[str, float]:
        dataset, _ = self._dataset(source)
        results = self._run_epoch(dataset, self._eval_step_fn(split), split=split)
        computed = as_metric_names(self.metric_names)(results, source="keras", split=split)
        logger.info(f"Evaluation complete: {computed}")
        return computed

    def predict(self) -> None:
        """Stream per-record predictions to the sink, pairing each batch with its metadata."""
        if self.test_set is None:
            raise ValueError(f"{type(self).__name__}.predict() requires a 'test_set'.")
        self.predictions_sink = flow(self.predictions_sink)
        predict_step = self._predict_step_fn()
        sequence = self._sequence(self.test_set, with_targets=False)
        for batch in sequence.batches():
            # Re-derived from the RECORD batch (not indexed off the sequence): the loop needs
            # the record dict for the metadata pairing, and `_to_arrays` is cheap.
            x = self._to_arrays(batch, with_targets=False)[0]
            preds = self._prediction_output(np.asarray(predict_step(tf.convert_to_tensor(x))))
            if self.predictions_sink is not None:
                self._write_predictions(preds, batch)
        if self.predictions_sink is not None and hasattr(self.predictions_sink, "close"):
            cast(PredictionsSink, self.predictions_sink).close()
        logger.info("Prediction complete.")

    # ---- the SavedModel ---------------------------------------------------------------------- #
    def _export(self) -> None:
        """Write the SavedModel — the one run artifact that is TensorFlow's and not Keras's.

        A ``keras.Model`` exports itself; a raw ``tf.Module``'s ``__call__`` is untraced Python
        and cannot be serialized, so the fallback traces an explicit signature at the input
        shape :meth:`_input_shape` discovers.
        """
        if not self.export_dir:
            return
        directory = Path(self.export_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        exporter = getattr(self.model, "export", None)
        if callable(exporter):
            exporter(str(directory))
        else:
            signature = tf.TensorSpec(shape=(None, *self._input_shape()), dtype=tf.float32)
            traced = tf.function(lambda x: _call_network(self.model, x, training=False)).get_concrete_function(
                signature
            )
            tf.saved_model.save(self.model, str(directory), signatures=traced)
        logger.info(f"Exported SavedModel to {directory}")

    def _input_shape(self) -> Tuple[int, ...]:
        """The per-record input shape, for the exported signature — declared, else measured."""
        declared = getattr(self.model, "input_shape", None)
        if declared:
            return tuple(int(d) for d in declared if d is not None)[-3:] or tuple(int(d) for d in declared[1:])
        source = self.test_set if self.test_set is not None else self.train_set
        if source is None:
            raise RuntimeError("cannot derive the export signature: no input_shape and no dataset to measure.")
        batch = self._sequence(source, with_targets=False)[0]
        return tuple(np.asarray(batch[0]).shape[1:])

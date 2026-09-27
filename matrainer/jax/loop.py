"""``JaxLoopRunnable`` — the jitted JAX/optax machinery base of the trainer API.

The owned-loop member for the FUNCTIONAL engine (``docs/architecture.md`` record 9): JAX
brings function transformations and expects you to write the loop yourself, so this base
writes it once — the step builders traced with ``nnx.jit`` ONCE and reused across epochs (a
fresh jit per epoch retraces the whole graph and ends up slower than the eager loop it
replaced, the same trap ``tf.function`` has), the epoch driver with the per-epoch reshuffle a
``fit()`` would have done for you, the plain-function tracking hookup, and the per-epoch
callback fan-out (JAX ships no callback protocol, so a callback is a plain
``callback(epoch, logs)`` callable — the plainest thing that works).

Three engine facts the whole loop encodes:

* **The step is a PURE FUNCTION of ``(model, batch)``** — ``nnx.value_and_grad``
  differentiates it and ``nnx.Optimizer`` applies the update in place; the default train step
  returns its outputs as an AUX beside the loss (``has_aux=True``), so the training metrics
  are scored from the very tensor the gradient was taken through — no second forward pass,
  and no eval-mode re-run that would measure a different network (dropout/batch norm).
* **Losses and optimizers are NAMES, not ``!class:`` tags** — optax ships plain FUNCTIONS
  and FACTORIES, so there is no class for a tag to resolve or for confluid's registry to tag;
  a named optimizer is resolved and CALLED with the ``learning_rate`` knob, which stays the
  one place an LR is written.
* **The epoch driver SPLATS the batch** (``step(model, ..., *arrays)``), so a task whose
  batch is not a pair (dense detection maps) drives the same loop with its own step arity —
  and a step may return a bare loss instead of a ``(loss, outputs)`` pair.

Metric ACCUMULATION is a task vocabulary, not an engine one, so the loop exposes it as three
state hooks (:meth:`_new_split_state` / :meth:`_score_batch` / :meth:`_split_scores`) — a
task keeps ONE accumulator per split and reduces ONCE per epoch (the
``matrainer.metric_functions`` rule: a per-batch average of a per-class metric is
systematically wrong, and moves with the batch size).
"""

from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from confluid import Partial, configurable, flow, output
from flax import nnx
from loggair import get_logger
from recordstream import PredictionsSink, RecordSource, batch_values
from recordstream.keras import RecordSequence
from recordstream.runnable import RunnableTask

from matrainer.artifacts import write_metrics_sidecar
from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.runnable import TrainerRunnable
from matrainer.tracking.base import Tracker
from matrainer.tracking.provenance import capture_log_file, log_run_context

logger = get_logger(__name__)

#: A JAX loss: the NAME of an ``optax.losses`` function, or a callable — optax ships
#: functions, so there is nothing for a ``!class:`` tag to resolve.
JaxLoss = Union[str, Callable[..., Any]]

#: A JAX metric: a NAME the task's vocabulary resolves, or a callable.
JaxMetric = Union[str, Callable[..., Any]]

#: A JAX optimizer: the NAME of an ``optax`` FACTORY (``"adam"`` / ``"adamw"`` / ``"sgd"``),
#: or a callable returning a ``GradientTransformation``.
JaxOptimizer = Union[str, Callable[..., Any]]

#: A plain callable invoked once per epoch with the epoch index and that epoch's logs — this
#: loop is ours, so there is no framework callback CLASS to name.
EpochCallback = Callable[[int, Dict[str, float]], None]

__all__ = [
    "EpochCallback",
    "JaxLoopRunnable",
    "JaxLoss",
    "JaxMetric",
    "JaxOptimizer",
    "as_optax_loss",
    "as_optax_optimizer",
]


def as_optax_loss(value: JaxLoss) -> Callable[..., Any]:
    """An optax loss from a callable or its NAME in ``optax.losses`` (or ``optax``).

    The string form is not sugar: optax's losses are plain FUNCTIONS, so there is no class
    for a ``!class:`` tag to name or for confluid's registry to tag — the fastai-names
    situation exactly.
    """
    built = flow(value)
    if callable(built):
        return cast(Callable[..., Any], built)
    loss = getattr(optax.losses, str(built), None) or getattr(optax, str(built), None)
    if loss is None:
        raise ValueError(
            f"Unknown optax loss {built!r}. Names come from `optax.losses` — e.g. "
            "softmax_cross_entropy_with_integer_labels, sigmoid_binary_cross_entropy."
        )
    return cast(Callable[..., Any], loss)


def as_optax_optimizer(value: JaxOptimizer, learning_rate: float) -> Any:
    """An ``optax.GradientTransformation`` from a callable or its NAME in ``optax``.

    A NAME is resolved and CALLED with the learning rate, because optax optimizers are
    FACTORIES (``optax.adam(1e-3)``) rather than classes — which is why the MLX sibling
    INSTANTIATES where this one calls.
    """
    built = flow(value)
    if callable(built) and not isinstance(built, str):
        return built(learning_rate) if not hasattr(built, "update") else built
    factory = getattr(optax, str(built), None)
    if factory is None:
        raise ValueError(f"Unknown optax optimizer {built!r}. Names come from `optax` — e.g. adam, adamw, sgd.")
    return factory(learning_rate)


# `@configurable(validate=False)` for the same reason as the other bases: the mark makes
# confluid's body-slot scans see THIS layer's slots in a concrete subclass's MRO. The concrete
# class's flat signature stays the validated config surface.
@configurable(validate=False)
class JaxLoopRunnable(TrainerRunnable):
    """A jitted JAX/optax train/evaluate/test/predict runnable, task hooks left open.

    Args (on top of the neutral base's):
        batch_size: Rows per batch (the ``RecordSequence`` boundary).
        learning_rate: Peak LR, passed to the named optax FACTORY.
        optimizer: An ``optax`` factory NAME or a callable returning a
            ``GradientTransformation``.
        jit: Trace the step functions with ``nnx.jit``. On by default — that IS the JAX
            story; turn it off to read a Python traceback from inside a step.
        loss: An ``optax.losses`` NAME or a callable — a sentinel (``None`` = the task's
            :meth:`_default_jax_loss` answers once the task has peeked its data).
        metrics / train_metrics / val_metrics / test_metrics: metric NAMES (or callables) in
            the task's vocabulary — ALL THREE splits are served, because this loop is ours
            and the train pass scores the logits its gradient step already produced.
        callbacks: ``callback(epoch, logs)`` callables, invoked after each epoch.

    Optional OVERRIDE slots (each already defaulted; a same-named top-level YAML key replaces
    one via broadcasting): ``val_set``, ``predictions_sink``, ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        *,
        model: Optional[Partial[nnx.Module]] = None,
        train_set: Optional[Partial[RecordSource]] = None,
        test_set: Optional[Partial[RecordSource]] = None,
        task: RunnableTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        optimizer: JaxOptimizer = "adam",
        jit: bool = True,
        loss: Optional[JaxLoss] = None,
        # `Any`: the metric vocabulary is the TASK's (name lists here, unlike the torch
        # family's torchmetrics specs); the concrete ctor declares the task's real type.
        metrics: Any = None,
        train_metrics: Any = None,
        val_metrics: Any = None,
        test_metrics: Any = None,
        trackers: Optional[Union[Partial[Tracker], Iterable[Partial[Tracker]]]] = None,
        callbacks: Optional[List[Partial[EpochCallback]]] = None,
        experiment_name: str = "matrainer",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
        metric_names: Optional[Partial[MetricNames]] = None,
    ) -> None:
        super().__init__(
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            trackers=trackers,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            metric_names=metric_names,
        )
        # The slot keeps the consumer-visible type: concrete ctors mark the param Mandatory,
        # so a bare None only comes from the base's own zero-arg construction.
        self.model: Partial[nnx.Module] = cast(Partial[nnx.Module], model)
        self.batch_size = batch_size
        self.learning_rate = float(learning_rate)
        self.optimizer: JaxOptimizer = optimizer
        self.jit = bool(jit)
        # Sentinels: `None` means "not chosen" — resolved via the task hooks once the task's
        # initialize has peeked what it needs.
        self.loss: Optional[JaxLoss] = loss
        self.metrics: Any = metrics
        self.train_metrics: Any = train_metrics
        self.val_metrics: Any = val_metrics
        self.test_metrics: Any = test_metrics
        # Stored VERBATIM (lazy-init): entries may be deferred markers, flowed where used.
        self.callbacks: Optional[List[Partial[EpochCallback]]] = callbacks
        self._history: Dict[str, List[float]] = {}
        self._computed_metrics: Dict[str, float] = {}

    # ---- sentinel resolution + the deliberate no-op builders -------------------------------- #
    def _default_jax_loss(self) -> Optional[JaxLoss]:
        """The task's default loss NAME/callable. ``None`` leaves the sentinel unresolved."""
        return None

    def _default_jax_metrics(self) -> Optional[List[Any]]:
        """The task's default metric NAMES. ``None`` leaves the sentinel unresolved."""
        return None

    def _resolve_task_defaults(self) -> None:
        """Fill the loss / metrics sentinels via the task hooks. Idempotent."""
        if self.loss is None:
            self.loss = self._default_jax_loss()
        if self.metrics is None:
            self.metrics = self._default_jax_metrics()

    def _build_fit_metrics(self, include_train: bool = True) -> None:
        """Deliberate no-op: the metric slots hold NAMES the task's vocabulary resolves —
        there is no collection object to build (the fastai/huggingface arrangement)."""

    def _build_eval_metrics(self) -> None:
        """Deliberate no-op — see :meth:`_build_fit_metrics`."""

    # ---- the batch boundary ------------------------------------------------------------------ #
    def _to_arrays(self, batch: Any, with_targets: bool = True) -> Any:
        """One collated record batch → the numpy tuple the traced steps splat over.

        recordstream hands over NUMPY, which is exactly what ``jnp.asarray`` takes — this
        family needs no conversion layer at all. The default is the image-to-image float32
        pair (channels-LAST — a JAX config's preprocess chain stops before the CHW
        ``ToTensor``); a task with a dtype rule (int64 ids, an int64 mask, dense maps)
        overrides.
        """
        x = np.asarray(batch_values(batch, self.input_key), dtype="float32")
        if not with_targets:
            return (x,)
        return x, np.asarray(batch_values(batch, self.target_key), dtype="float32")

    def _sequence(self, source: Any, shuffle: bool = False, with_targets: bool = True) -> RecordSequence:
        """recordstream's batching adapter, wearing THIS task's batch shape.

        The same ``PyDataset`` the keras family uses — engine-neutral despite the module
        name; ``_to_arrays`` is handed to it exactly as a ``collate_fn`` is handed to a
        ``DataLoader``.
        """
        return RecordSequence(
            source,
            batch_size=self.batch_size,
            shuffle=shuffle,
            transform=partial(self._to_arrays, with_targets=with_targets),
        )

    # ---- the traced steps -------------------------------------------------------------------- #
    def _loss_fn(self) -> Callable[..., Any]:
        """The loss as a callable over ``(outputs, targets)`` returning a SCALAR.

        The default resolves the ``loss`` slot through :func:`as_optax_loss` and takes the
        mean — optax's losses are ELEMENTWISE, and the mean is what makes a named optax loss
        agree with its torch namesake. A task with its own rule (an ignore-index mask, a
        written-out spec loss) overrides. An unresolved sentinel asks the task hook, so a
        directly-driven instance works without ``initialize()``.
        """
        chosen = self.loss if self.loss is not None else self._default_jax_loss()
        if chosen is None:
            raise RuntimeError(
                f"{type(self).__name__} resolved no loss — wire `loss:` or implement _default_jax_loss()/_loss_fn()."
            )
        loss = as_optax_loss(chosen)

        def compute(outputs: Any, targets: Any) -> Any:
            return jnp.mean(loss(outputs, targets))

        return compute

    def _make_steps(self) -> Tuple[Callable[..., Any], Callable[..., Any]]:
        """Build the train and eval steps ONCE, and jit them ONCE.

        A fresh ``nnx.jit`` per epoch retraces the whole graph every time and ends up slower
        than the eager loop it replaced — the same trap ``tf.function`` has. The default pair
        is the ``(x, y)`` shape with the outputs returned as an AUX beside the loss; a task
        whose batch is not a pair overrides the builder with its own arity (the epoch driver
        splats, and accepts a bare-loss return).
        """
        loss_fn = self._loss_fn()

        def train_step(model: Any, optimizer: Any, x: Any, y: Any) -> Any:
            def forward(m: Any) -> Any:
                # `has_aux`: the outputs come back BESIDE the loss, so the training metrics
                # are scored from the very tensor the gradient was taken through — no second
                # forward pass, and no eval-mode re-run measuring a different network.
                outputs = m(x, training=True)
                return loss_fn(outputs, y), outputs

            (loss, outputs), grads = nnx.value_and_grad(forward, has_aux=True)(model)
            optimizer.update(model, grads)
            return loss, outputs

        def eval_step(model: Any, x: Any, y: Any) -> Any:
            outputs = model(x, training=False)
            return loss_fn(outputs, y), outputs

        if self.jit:
            return nnx.jit(train_step), nnx.jit(eval_step)
        return train_step, eval_step

    def _predict_step(self) -> Callable[..., Any]:
        """The forward-only step, built (and jitted) ONCE — a fresh trace per batch retraces."""

        def predict_step(model: Any, x: Any) -> Any:
            return model(x, training=False)

        return nnx.jit(predict_step) if self.jit else predict_step

    def _optimizer(self) -> Any:
        """An ``nnx.Optimizer`` over the resolved optax transformation."""
        return nnx.Optimizer(self.model, as_optax_optimizer(self.optimizer, self.learning_rate), wrt=nnx.Param)

    # ---- the metric state hooks (the task's accumulation vocabulary) ------------------------- #
    def _new_split_state(self, split: str) -> Any:
        """One split's fresh metric accumulator state — ``None`` (the default) scores nothing.

        Zeroed once per EPOCH by the driver, never per batch: a metric a hand-written loop
        cannot average (macro-F1, mean IoU, PSNR) accumulates across the whole pass and
        reduces once (``matrainer.metric_functions``).
        """
        return None

    def _score_batch(self, state: Any, outputs: Any, arrays: Tuple[Any, ...], split: str) -> None:
        """Fold ONE batch into the split's state. ``outputs`` is the step's aux (``None`` for
        a bare-loss step); ``arrays`` is the batch tuple the step was splatted with."""

    def _split_scores(self, state: Any, split: str, step: Optional[int] = None) -> Dict[str, float]:
        """Reduce a split's state into UNPREFIXED scalars, once per epoch/pass.

        The driver prefixes with ``<split>/``. A task with non-scalar values (a per-class
        vector) reports them to the trackers' text channel from HERE — ``step`` is the epoch
        for that channel's axis — and returns only the scalars.
        """
        return {}

    # ---- the epoch driver -------------------------------------------------------------------- #
    def _train_epoch(self, sequence: RecordSequence, train_step: Any, optimizer: Any, epoch: int) -> Dict[str, float]:
        """One training pass — loss averaged over batches, metrics accumulated then reduced."""
        state = self._new_split_state("train")
        total, batches = 0.0, 0
        for index in range(len(sequence)):
            arrays = tuple(jnp.asarray(a) for a in sequence[index])
            result = train_step(self.model, optimizer, *arrays)
            loss, outputs = result if isinstance(result, tuple) else (result, None)
            total += float(loss)
            batches += 1
            self._score_batch(state, outputs, arrays, "train")
        logs = {"train/loss": total / batches if batches else float("nan")}
        logs.update({f"train/{k}": v for k, v in self._split_scores(state, "train", step=epoch).items()})
        return logs

    def _eval_pass(
        self, sequence: RecordSequence, eval_step: Any, split: str, epoch: Optional[int] = None
    ) -> Dict[str, float]:
        """One evaluation pass over a split, returning canonical ``<split>/<name>`` scalars."""
        state = self._new_split_state(split)
        total, batches = 0.0, 0
        for index in range(len(sequence)):
            arrays = tuple(jnp.asarray(a) for a in sequence[index])
            result = eval_step(self.model, *arrays)
            loss, outputs = result if isinstance(result, tuple) else (result, None)
            total += float(loss)
            batches += 1
            self._score_batch(state, outputs, arrays, split)
        logs = {f"{split}/loss": total / batches if batches else float("nan")}
        logs.update({f"{split}/{k}": v for k, v in self._split_scores(state, split, step=epoch).items()})
        return logs

    # ---- the fit hooks ----------------------------------------------------------------------- #
    def _fit_scores(self, logs: Dict[str, float]) -> Dict[str, float]:
        """Extend the last epoch's logs with task-computed post-fit scores — identity by
        default. The detection task appends its numpy AP over the validation split here."""
        return logs

    def _fit_epilogue(self) -> None:
        """Task epilogue after training, before the sidecar/terminal capture — no-op by
        default. The classification task persists its class names here."""

    # ---- the four entry points --------------------------------------------------------------- #
    def fit(self) -> None:
        logger.info(f"{self.log_label} jax training: {self.experiment_name} ({self.run_name})")
        logger.info(f"JAX devices: {jax.devices()}")
        if self.train_set is None:
            raise ValueError(f"{type(self).__name__}.fit() requires a 'train_set'.")
        # Warm every wired split IN THIS PROCESS (fork safety) — idempotent over a split the
        # task's initialize already prepared.
        self.train_set = self._prepare(self.train_set)
        self.val_set = self._prepare(self.val_set)
        self.test_set = self._prepare(self.test_set)
        # Provenance through matrainer's plain FUNCTIONS — `runnable=self` so the
        # hyperparameters and the resolved config come from the ONE derivation.
        trackers = self._live_trackers()
        names = as_metric_names(self.metric_names)
        manifest = log_run_context(trackers, runnable=self)
        optimizer = self._optimizer()
        # ONCE — see `_make_steps`.
        train_step, eval_step = self._make_steps()
        train_seq = self._sequence(self.train_set, shuffle=True)
        val_seq = self._sequence(self.val_set) if self.val_set is not None else None

        logs: Dict[str, float] = {}
        for epoch in range(self.max_epochs):
            logs = self._train_epoch(train_seq, train_step, optimizer, epoch)
            if val_seq is not None:
                logs.update(self._eval_pass(val_seq, eval_step, "val", epoch))
            # A `fit()` would have reshuffled for us; a hand-written loop has to. Forgetting
            # it is INVISIBLE — training just runs on one fixed row order forever.
            train_seq.on_epoch_end()
            for callback in self.callbacks or []:
                flow(callback)(epoch, logs)
            translated = names(logs)
            for tracker in trackers:
                tracker.log_metrics(translated, step=epoch)
            for key, value in translated.items():
                self._history.setdefault(key, []).append(value)
            logger.info(f"epoch {epoch + 1}/{self.max_epochs}: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items()))

        self._computed_metrics = names(self._fit_scores(logs))
        logger.info(f"Training complete: {self._computed_metrics}")
        self._fit_epilogue()
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        # Stamps the manifest's terminal status — left unfinished it stays `running`, and
        # matrainer's atexit hook then calls a successful run `interrupted`.
        capture_log_file(trackers, manifest=manifest, status="success")

    def evaluate(self) -> None:
        """Score the VALIDATION split — the ``evaluate`` / ``test`` split is the only difference."""
        if self.val_set is None:
            raise ValueError(f"{type(self).__name__}.evaluate() requires a 'val_set'.")
        self.val_set = self._prepare(self.val_set)
        self._run_eval_task(self.val_set, "val")

    def test(self) -> None:
        if self.test_set is None:
            raise ValueError(f"{type(self).__name__}.test() requires a 'test_set'.")
        self.test_set = self._prepare(self.test_set)
        self._run_eval_task(self.test_set, "test")

    def _run_eval_task(self, source: Any, split: Literal["val", "test"]) -> None:
        """One eval entry point's body: score, put it on the trackers, record the run."""
        trackers = self._live_trackers()
        manifest = log_run_context(trackers, runnable=self)
        self._computed_metrics = self._evaluate_split(source, split)
        for tracker in trackers:
            tracker.log_metrics(self._computed_metrics)
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        capture_log_file(trackers, manifest=manifest, status="success")

    def _evaluate_split(self, source: Any, split: Literal["val", "test"]) -> Dict[str, float]:
        """One split through the eval pass, translated into the canonical vocabulary.

        This loop already writes canonical ``<split>/<name>`` keys, so the default
        translation is a pass-through unless a ``target:`` vocabulary is wired; a task
        scoring another way (detection's numpy AP) overrides.
        """
        _, eval_step = self._make_steps()
        return as_metric_names(self.metric_names)(self._eval_pass(self._sequence(source), eval_step, split))

    def predict(self) -> None:
        """Stream per-record predictions to the sink, pairing each batch with its metadata.

        Walks ``sequence.batches()`` — the COLLATED record batches — rather than the arrays,
        because the sink needs the metadata that lives beside the image and ``_to_arrays``
        has already dropped it. One walk, so output and metadata cannot desync.
        """
        if self.test_set is None:
            raise ValueError(f"{type(self).__name__}.predict() requires a 'test_set'.")
        self.test_set = self._prepare(self.test_set)
        self.predictions_sink = flow(self.predictions_sink)
        predict_step = self._predict_step()
        sequence = self._sequence(self.test_set, with_targets=False)
        for index, batch in enumerate(sequence.batches()):
            preds = self._prediction_output(self._predict_batch(predict_step, sequence[index][0]))
            if self.predictions_sink is not None:
                self._write_predictions(preds, batch)
        if self.predictions_sink is not None and hasattr(self.predictions_sink, "close"):
            cast(PredictionsSink, self.predictions_sink).close()
        logger.info("Prediction complete.")

    def _predict_batch(self, predict_step: Any, x: Any) -> Any:
        """ONE image batch → the raw model output the prediction contract wraps."""
        return np.asarray(predict_step(self.model, jnp.asarray(x)))

    # ---- @outputs ----------------------------------------------------------------------------- #
    @property
    @output
    def trained_model(self) -> Any:
        """The live trained model (post-run ``@output``). Raises while still deferred."""
        from confluid import Fluid

        if isinstance(self.model, Fluid):
            raise RuntimeError("trained_model is only available after run() — the model is still a deferred Fluid.")
        return self.model

    @property
    @output
    def computed_metrics(self) -> Dict[str, float]:
        """Final SCALAR metrics (post-run ``@output``). Raises before ``run()``."""
        if not self._computed_metrics:
            raise RuntimeError("computed_metrics is only available after run().")
        return dict(self._computed_metrics)

    @property
    @output
    def history(self) -> Dict[str, List[float]]:
        """Per-epoch metric history — what a framework's logger would have accumulated."""
        return {key: list(values) for key, values in self._history.items()}

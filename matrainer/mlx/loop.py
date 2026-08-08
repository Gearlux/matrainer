"""``MlxLoopRunnable`` — the MLX machinery base of the trainer API (Apple silicon only).

The owned-loop member for Apple's engine (``docs/architecture.md`` record 9), the jax
sibling in HOOK SURFACE — the same batch boundary, sentinel hooks, metric state hooks and
entry points — under a loop shaped by three MLX facts, each measured in the consuming
projects:

1. **Unified memory** — CPU and GPU share it on Apple silicon, so there is NO device
   transfer; this is the one machinery whose batch path does genuinely LESS work.
2. **Arrays are LAZY** — ``mx.eval`` forces computation, and omitting it does NOT raise: the
   graph just grows until memory runs out. So the loop evaluates once per step over the
   parameters, the optimizer state AND the loss, deliberately and visibly.
3. **Train/eval is module STATE, not a per-call argument** — ``network.train(bool)`` at the
   loop boundaries, because a ``training=False`` default on ``__call__`` would flip the
   network to eval on every forward, including the one inside ``nn.value_and_grad`` where
   there is no flag to pass, and batch norm would then never train.

Two asymmetries with the jax sibling, both engine facts: ``mlx.optimizers`` ships CLASSES
where optax ships factories (so :func:`as_mlx_optimizer` INSTANTIATES where the jax helper
CALLS), and there is no jit knob — ``nn.value_and_grad`` is built once per fit and MLX owns
its own compilation. The train objective returns its outputs BESIDE the loss (MLX
differentiates the FIRST element of a returned tuple — verified), so the training metrics
are scored from the very tensor the gradient was taken through.
"""

from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple, Union, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers
import numpy as np
from confluid import Lazy, configurable, flow, output
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

#: An MLX loss: the NAME of an ``mlx.nn.losses`` function, or a callable — MLX ships
#: functions, so there is nothing for a ``!class:`` tag to resolve.
MlxLoss = Union[str, Callable[..., Any]]

#: An MLX metric: a NAME the task's vocabulary resolves, or a callable.
MlxMetric = Union[str, Callable[..., Any]]

#: An MLX optimizer: the NAME of an ``mlx.optimizers`` CLASS (``"adam"`` / ``"Adam"``), the
#: class itself, or one already built.
MlxOptimizer = Union[str, Any]

#: A plain callable invoked once per epoch with the epoch index and that epoch's logs.
EpochCallback = Callable[[int, Dict[str, float]], None]

__all__ = [
    "EpochCallback",
    "MlxLoopRunnable",
    "MlxLoss",
    "MlxMetric",
    "MlxOptimizer",
    "as_mlx_loss",
    "as_mlx_optimizer",
]


def as_mlx_loss(value: MlxLoss) -> Callable[..., Any]:
    """An MLX loss from a callable or its NAME in ``mlx.nn.losses`` — the optax situation
    exactly: plain functions, nothing for a tag or a registry."""
    built = flow(value)
    if callable(built):
        return cast(Callable[..., Any], built)
    loss = getattr(nn.losses, str(built), None)
    if loss is None:
        raise ValueError(
            f"Unknown MLX loss {built!r}. Names come from `mlx.nn.losses` — e.g. "
            "cross_entropy, binary_cross_entropy, l1_loss, mse_loss, huber_loss."
        )
    return cast(Callable[..., Any], loss)


def as_mlx_optimizer(value: MlxOptimizer, learning_rate: float) -> Any:
    """A live ``mlx.optimizers`` optimizer from a class NAME, the class, or an instance.

    MLX ships CLASSES (``mlx.optimizers.Adam``) where optax ships factories — hence this
    INSTANTIATES where the jax twin calls. A lowercase name is capitalised on the way
    (``"adam"`` → ``Adam``) so the two siblings' configs read alike.
    """
    built = flow(value)
    if isinstance(built, type):
        return built(learning_rate=learning_rate)
    if not isinstance(built, str):
        return built  # already an optimizer instance
    cls = getattr(mlx.optimizers, built.capitalize(), None) or getattr(mlx.optimizers, built, None)
    if cls is None:
        raise ValueError(
            f"Unknown MLX optimizer {built!r}. Names come from `mlx.optimizers` — e.g. adam, adamw, sgd, lion."
        )
    return cls(learning_rate=learning_rate)


# `@configurable(validate=False)` for the same reason as the other bases: the mark makes
# confluid's body-slot scans see THIS layer's slots in a concrete subclass's MRO.
@configurable(validate=False)
class MlxLoopRunnable(TrainerRunnable):
    """An MLX train/evaluate/test/predict runnable, task hooks left open.

    Args (on top of the neutral base's):
        batch_size: Rows per batch (the ``RecordSequence`` boundary).
        learning_rate: Passed to the named optimizer CLASS.
        optimizer: An ``mlx.optimizers`` class NAME, the class, or an instance.
        loss: An ``mlx.nn.losses`` NAME or a callable — a sentinel (``None`` = the task's
            :meth:`_default_mlx_loss` answers once the task has peeked its data).
        metrics / train_metrics / val_metrics / test_metrics: metric NAMES (or callables) in
            the task's vocabulary — ALL THREE splits are served, because this loop is ours.
        callbacks: ``callback(epoch, logs)`` callables, invoked after each epoch.

    Optional OVERRIDE slots (each already defaulted; a same-named top-level YAML key replaces
    one via broadcasting): ``val_set``, ``predictions_sink``, ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        *,
        model: Optional[Lazy[nn.Module]] = None,
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        task: RunnableTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        optimizer: MlxOptimizer = "adam",
        loss: Optional[MlxLoss] = None,
        # `Any`: the metric vocabulary is the TASK's (name lists); the concrete ctor declares
        # the task's real type.
        metrics: Any = None,
        train_metrics: Any = None,
        val_metrics: Any = None,
        test_metrics: Any = None,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        callbacks: Optional[List[Lazy[EpochCallback]]] = None,
        experiment_name: str = "matrainer",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
        metric_names: Optional[Lazy[MetricNames]] = None,
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
        self.model: Lazy[nn.Module] = cast(Lazy[nn.Module], model)
        self.batch_size = batch_size
        self.learning_rate = float(learning_rate)
        self.optimizer: MlxOptimizer = optimizer
        self.loss: Optional[MlxLoss] = loss
        self.metrics: Any = metrics
        self.train_metrics: Any = train_metrics
        self.val_metrics: Any = val_metrics
        self.test_metrics: Any = test_metrics
        self.callbacks: Optional[List[Lazy[EpochCallback]]] = callbacks
        self._history: Dict[str, List[float]] = {}
        self._computed_metrics: Dict[str, float] = {}

    # ---- sentinel resolution + the deliberate no-op builders -------------------------------- #
    def _default_mlx_loss(self) -> Optional[MlxLoss]:
        """The task's default loss NAME/callable. ``None`` leaves the sentinel unresolved."""
        return None

    def _default_mlx_metrics(self) -> Optional[List[Any]]:
        """The task's default metric NAMES. ``None`` leaves the sentinel unresolved."""
        return None

    def _resolve_task_defaults(self) -> None:
        """Fill the loss / metrics sentinels via the task hooks. Idempotent."""
        if self.loss is None:
            self.loss = self._default_mlx_loss()
        if self.metrics is None:
            self.metrics = self._default_mlx_metrics()

    def _build_fit_metrics(self, include_train: bool = True) -> None:
        """Deliberate no-op: the metric slots hold NAMES the task's vocabulary resolves —
        there is no collection object to build (the jax/fastai/huggingface arrangement)."""

    def _build_eval_metrics(self) -> None:
        """Deliberate no-op — see :meth:`_build_fit_metrics`."""

    # ---- the batch boundary ------------------------------------------------------------------ #
    def _to_arrays(self, batch: Any, with_targets: bool = True) -> Any:
        """One collated record batch → the numpy tuple the step consumes.

        The default is the image-to-image float32 pair (channels-LAST — MLX is channels-last
        native, and unified memory means the arrays reach the GPU with no transfer at all); a
        task with a dtype rule (int32 ids — MLX indexes with int32 — dense maps) overrides.
        """
        x = np.asarray(batch_values(batch, self.input_key), dtype="float32")
        if not with_targets:
            return (x,)
        return x, np.asarray(batch_values(batch, self.target_key), dtype="float32")

    def _sequence(self, source: Any, shuffle: bool = False, with_targets: bool = True) -> RecordSequence:
        """recordstream's batching adapter, wearing THIS task's batch shape."""
        return RecordSequence(
            source,
            batch_size=self.batch_size,
            shuffle=shuffle,
            transform=partial(self._to_arrays, with_targets=with_targets),
        )

    # ---- the step ---------------------------------------------------------------------------- #
    def _loss_fn(self) -> Callable[..., Any]:
        """The loss as a callable over ``(outputs, targets)`` returning a SCALAR.

        The default resolves the ``loss`` slot through :func:`as_mlx_loss` and takes the mean
        (MLX's losses default to ``reduction="none"``); a task with its own rule (an
        ignore-index mask, a written-out spec loss) overrides. An unresolved sentinel asks the
        task hook, so a directly-driven instance works without ``initialize()``.
        """
        chosen = self.loss if self.loss is not None else self._default_mlx_loss()
        if chosen is None:
            raise RuntimeError(
                f"{type(self).__name__} resolved no loss — wire `loss:` or implement _default_mlx_loss()/_loss_fn()."
            )
        loss = as_mlx_loss(chosen)

        def compute(outputs: Any, targets: Any) -> Any:
            return mx.mean(loss(outputs, targets))

        return compute

    def _make_step(self) -> Callable[..., Any]:
        """The training step: a loss over ``(model, *batch)``, differentiated by
        ``nn.value_and_grad`` — built ONCE per fit.

        The default objective is the ``(x, y)`` shape returning ``(loss, outputs)`` — MLX
        differentiates the FIRST element of a returned tuple, so the outputs ride along and
        the training metrics are scored from the very tensor the gradient was taken through.
        A task whose batch is not a pair overrides with its own arity (the driver splats, and
        accepts a bare-loss objective).
        """
        loss_fn = self._loss_fn()

        def compute(model: Any, x: Any, y: Any) -> Any:
            outputs = model(x)
            return loss_fn(outputs, y), outputs

        return cast(Callable[..., Any], nn.value_and_grad(cast(nn.Module, self.model), compute))

    def _eval_batch(self, loss_fn: Callable[..., Any], network: Any, arrays: Tuple[Any, ...]) -> Any:
        """ONE eval batch → ``(loss, outputs)``. A non-pair task overrides for its arity."""
        outputs = network(arrays[0])
        return loss_fn(outputs, arrays[1]), outputs

    def _optimizer(self) -> Any:
        return as_mlx_optimizer(self.optimizer, self.learning_rate)

    # ---- the metric state hooks (the task's accumulation vocabulary) ------------------------- #
    def _new_split_state(self, split: str) -> Any:
        """One split's fresh metric accumulator state — ``None`` (the default) scores nothing."""
        return None

    def _score_batch(self, state: Any, outputs: Any, arrays: Tuple[Any, ...], split: str) -> None:
        """Fold ONE batch into the split's state. ``outputs`` is the objective's aux (``None``
        for a bare-loss objective); ``arrays`` is the batch tuple the step was splatted with."""

    def _split_scores(self, state: Any, split: str, step: Optional[int] = None) -> Dict[str, float]:
        """Reduce a split's state into UNPREFIXED scalars, once per epoch/pass."""
        return {}

    # ---- the epoch driver -------------------------------------------------------------------- #
    def _train_epoch(self, sequence: RecordSequence, step: Any, optimizer: Any, epoch: int) -> Dict[str, float]:
        """One training pass. Train/eval is module STATE, so the mode is set ONCE, here."""
        network = cast(nn.Module, self.model)
        network.train(True)
        state = self._new_split_state("train")
        total, batches = 0.0, 0
        for index in range(len(sequence)):
            arrays = tuple(mx.array(a) for a in sequence[index])
            result, grads = step(network, *arrays)
            loss, outputs = result if isinstance(result, tuple) else (result, None)
            optimizer.update(network, grads)
            # Lazy arrays: NOTHING has computed until here — evaluating the parameters, the
            # optimizer state and the loss per step keeps the graph from growing unboundedly,
            # and omitting it does NOT raise.
            mx.eval(network.parameters(), optimizer.state, loss)
            total += float(loss)
            batches += 1
            self._score_batch(state, outputs, arrays, "train")
        logs = {"train/loss": total / batches if batches else float("nan")}
        logs.update({f"train/{k}": v for k, v in self._split_scores(state, "train", step=epoch).items()})
        return logs

    def _eval_pass(self, sequence: RecordSequence, split: str, epoch: Optional[int] = None) -> Dict[str, float]:
        """One evaluation pass, with the network switched to eval MODE (not a per-call flag)."""
        network = cast(nn.Module, self.model)
        network.train(False)
        loss_fn = self._loss_fn()
        state = self._new_split_state(split)
        total, batches = 0.0, 0
        for index in range(len(sequence)):
            arrays = tuple(mx.array(a) for a in sequence[index])
            loss, outputs = self._eval_batch(loss_fn, network, arrays)
            mx.eval(outputs, loss)
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
        logger.info(f"{self.log_label} mlx training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError(f"{type(self).__name__}.fit() requires a 'train_set'.")
        self.train_set = self._prepare(self.train_set)
        self.val_set = self._prepare(self.val_set)
        self.test_set = self._prepare(self.test_set)
        trackers = self._live_trackers()
        names = as_metric_names(self.metric_names)
        manifest = log_run_context(trackers, runnable=self)
        optimizer = self._optimizer()
        step = self._make_step()
        train_seq = self._sequence(self.train_set, shuffle=True)
        val_seq = self._sequence(self.val_set) if self.val_set is not None else None

        logs: Dict[str, float] = {}
        for epoch in range(self.max_epochs):
            logs = self._train_epoch(train_seq, step, optimizer, epoch)
            if val_seq is not None:
                logs.update(self._eval_pass(val_seq, "val", epoch))
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
        """One split through the eval pass, translated into the canonical vocabulary."""
        return as_metric_names(self.metric_names)(self._eval_pass(self._sequence(source), split))

    def predict(self) -> None:
        """Stream per-record predictions to the sink, pairing each batch with its metadata."""
        if self.test_set is None:
            raise ValueError(f"{type(self).__name__}.predict() requires a 'test_set'.")
        self.test_set = self._prepare(self.test_set)
        self.predictions_sink = flow(self.predictions_sink)
        cast(nn.Module, self.model).train(False)  # module STATE, not a per-call flag
        sequence = self._sequence(self.test_set, with_targets=False)
        for index, batch in enumerate(sequence.batches()):
            preds = self._prediction_output(self._predict_batch(sequence[index][0]))
            if self.predictions_sink is not None:
                self._write_predictions(preds, batch)
        if self.predictions_sink is not None and hasattr(self.predictions_sink, "close"):
            cast(PredictionsSink, self.predictions_sink).close()
        logger.info("Prediction complete.")

    def _predict_batch(self, x: Any) -> Any:
        """ONE image batch → the raw model output the prediction contract wraps."""
        outputs = cast(nn.Module, self.model)(mx.array(x))
        mx.eval(outputs)
        return np.asarray(outputs)

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

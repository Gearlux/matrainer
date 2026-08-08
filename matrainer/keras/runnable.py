"""The Keras FAMILY trainer base — where the Keras types are pinned down.

``KerasTrainerRunnable`` is the keras twin of :class:`matrainer.torch.runnable.
TorchTrainerRunnable`: the layer between the framework-neutral API
(:class:`matrainer.runnable.TrainerRunnable`) and the per-framework machinery bases — the
compile/fit member (:class:`matrainer.keras.trainer.KerasRunnable`) and the native-TensorFlow
loop (``matrainer.tensorflow.loop.TensorFlowRunnable``), which are SIBLINGS sharing only this
layer. It exists so nothing user-facing is typed ``Any``: the model slot is
``Lazy[keras.Model]``, the compile slots are the :data:`KerasLoss` / :data:`KerasMetric` /
:data:`KerasOptimizer` aliases (a live object, the string alias Keras resolves, or a deferred
marker — spelled out because ``Any`` types nothing and gives ``to_pydantic`` nothing to build a
widget from).

What it owns beyond the types:

* **The batch boundary** — :meth:`_sequence` wires :class:`recordstream.keras.RecordSequence`
  (the engine's DataLoader half: row order, slicing, the per-epoch reshuffle) with
  :meth:`_to_arrays`, the task hook handed in exactly as a ``collate_fn`` is handed to a
  ``DataLoader``. The default is the image-to-image shape (float32 in, float32 target); a task
  with a different target read overrides the ONE method.
* **The keras-shaped sentinels** — ``loss`` / ``metrics`` default to ``None`` ("not chosen",
  never "no loss"), filled by :meth:`_resolve_task_defaults` from the task hooks
  :meth:`_default_keras_loss` / :meth:`_default_keras_metrics`. The hooks are keras-VOCABULARY
  hooks (they return keras objects), which is why they are not the torch family's.
* **The metric UNION as a family property** — :meth:`_compiled_metrics` is
  :func:`matrainer.metric_slots.union_metric_specs` over the four slots, because
  ``keras.Model.compile`` owns ONE metric list and reports it on train AND validation; the
  ``_build_*_metrics`` builders are deliberate NO-OPS for the same reason (keras metrics
  resolve at compile / step-build, not into prefixed collections). The sibling TensorFlow loop
  narrows per split instead — each traced step closes over its own metric objects — which is
  exactly the difference between owning the loop and handing it to ``fit()``.
* **The callback plumbing** — :meth:`_callbacks` flows the wired callbacks, creates a
  file-writing callback's parent directory (Keras opens the path without doing so, minutes
  into a fit), and appends :func:`matrainer.keras.keras_tracking_callback` when trackers are
  wired.

The engine itself stays the operator's choice: Keras 3 reads ``KERAS_BACKEND`` once, at
import, and this module imports keras THROUGH :mod:`recordstream.keras`, which defaults the
variable to the first engine actually installed.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union, cast

import numpy as np
from confluid import Fluid, Lazy, configurable, flow, output
from loggair import get_logger
from recordstream import RecordSource, batch_values
from recordstream.keras import RecordSequence, keras, keras_backend
from recordstream.runnable import RunnableTask

from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.metric_slots import union_metric_specs
from matrainer.runnable import TrainerRunnable
from matrainer.tracking.base import Tracker

logger = get_logger(__name__)

__all__ = ["KerasLoss", "KerasMetric", "KerasOptimizer", "KerasTrainerRunnable", "keras_backend"]

#: A Keras loss: a live object or the string alias (``"mae"``) Keras resolves. Wired from YAML
#: a nested ``!class:`` arrives as a deferred marker — the ``Lazy[...]`` on the slot declares
#: that — and is flowed at USE (compile / step build).
KerasLoss = Union[str, keras.losses.Loss]

#: A Keras metric, same three shapes.
KerasMetric = Union[str, keras.metrics.Metric]

#: A Keras optimizer, same three shapes.
KerasOptimizer = Union[str, keras.optimizers.Optimizer]


# `@configurable(validate=False)` for the same reason as the other bases: the mark is what makes
# confluid's body-slot scans see this layer's slots (`loss`, the metric quartet, `callbacks`) in
# a concrete subclass's MRO; the concrete class's flat signature stays the validated surface.
@configurable(validate=False)
class KerasTrainerRunnable(TrainerRunnable):
    """The Keras-family trainer base: typed slots + the keras half of the machinery.

    Args (on top of :class:`~matrainer.runnable.TrainerRunnable`'s):
        model: The task model — a live ``keras.Model`` or (typically) a deferred ``!lazy:``
            marker the task's ``initialize`` flows with its resolved dimensions.
        batch_size: Rows per batch, handed to the :class:`RecordSequence` adapter.
        optimizer: Keras optimizer — a string alias (``"adam"``) or an instance.
        loss: Keras loss; ``None`` asks the task hook (:meth:`_default_keras_loss`).
        metrics: Shared Keras metrics; ``None`` asks :meth:`_default_keras_metrics`.
        train_metrics / val_metrics / test_metrics: That split's OWN metrics. How they combine
            is the machinery base's fact: the compile/fit member computes the UNION
            (``compile()`` owns one list), the TensorFlow loop narrows per split.
        callbacks: ``keras.callbacks.Callback`` instances (may arrive as deferred markers).
    """

    def __init__(
        self,
        *,
        model: Optional[Lazy["keras.Model"]] = None,
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        task: RunnableTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 32,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        callbacks: Optional[List[Lazy["keras.callbacks.Callback"]]] = None,
        experiment_name: str = "matrainer",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
        optimizer: Lazy[KerasOptimizer] = "adam",
        loss: Optional[Lazy[KerasLoss]] = None,
        metrics: Optional[List[Lazy[KerasMetric]]] = None,
        train_metrics: Optional[List[Lazy[KerasMetric]]] = None,
        val_metrics: Optional[List[Lazy[KerasMetric]]] = None,
        test_metrics: Optional[List[Lazy[KerasMetric]]] = None,
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
        # `keras.Model`, not `Lazy[keras.Model]`: the DEFERRED-ness is declared once, on the
        # ctor param, which is what `lazy_param_names` / `to_pydantic` read. Restating it here
        # would make the attribute a union no method could call through.
        self.model: "keras.Model" = cast("keras.Model", model)
        self.batch_size = batch_size
        self.optimizer: Lazy[KerasOptimizer] = optimizer
        # Sentinels: a live default could not be told apart from a user's explicit choice, and
        # the right default may depend on a dimension only the task's `initialize` resolves.
        self.loss: Optional[Lazy[KerasLoss]] = loss
        # The ATTRIBUTES are `Any` (the torch family's precedent, and the sanctioned
        # strictly-needed case): the TYPED config surface is the ctor param above, and one
        # task's base guards `train_metrics` with a property the typed attribute would clash
        # with. The machinery reads them as the `List[Lazy[KerasMetric]]` the params declare.
        self.metrics: Any = metrics
        self.train_metrics: Any = train_metrics
        self.val_metrics: Any = val_metrics
        self.test_metrics: Any = test_metrics
        # Stored VERBATIM (lazy-init): entries may be deferred markers, flowed where used.
        self.callbacks: Optional[List[Lazy["keras.callbacks.Callback"]]] = callbacks
        self._history: Dict[str, List[float]] = {}
        self._computed_metrics: Dict[str, float] = {}

    # ---- the keras-typed task hooks -------------------------------------------------------- #
    def _default_keras_loss(self) -> Optional[Lazy[KerasLoss]]:
        """The task's default Keras loss, asked once ``initialize`` knows the target shape.

        ``None`` (the default) leaves the sentinel unresolved — legitimate while a needed
        dimension is still unknown; :meth:`_resolve_task_defaults` is idempotent and asks again.
        """
        return None

    def _default_keras_metrics(self) -> Optional[List[Lazy[KerasMetric]]]:
        """The task's default Keras metrics — same contract as :meth:`_default_keras_loss`."""
        return None

    def _to_arrays(self, batch: Any, with_targets: bool = True) -> Any:
        """One collated record batch → the numpy ``(x, y)`` the engine consumes — THE task hook.

        The Keras twin of the torch family's ``_targets``, handed to the sequence exactly as a
        ``collate_fn`` is handed to a ``DataLoader``. The default is the image-to-image shape
        (float32 input, float32 target — recordstream hands over numpy already, so nothing is
        converted twice); a task with a different target read (int64 ids, a multi-hot, dense
        detection maps) overrides this one method.
        """
        x = np.asarray(batch_values(batch, self.input_key), dtype="float32")
        if not with_targets:
            return (x,)
        return x, np.asarray(batch_values(batch, self.target_key), dtype="float32")

    # ---- sentinel resolution + the no-op builders ------------------------------------------ #
    def _resolve_task_defaults(self) -> None:
        """Fill the ``loss`` / ``metrics`` sentinels via the keras task hooks. Idempotent."""
        if self.loss is None:
            self.loss = self._default_keras_loss()
            if self.loss is not None:
                logger.debug(f"loss defaulted to {type(self.loss).__name__}.")
        if self.metrics is None:
            self.metrics = self._default_keras_metrics()

    def _build_fit_metrics(self, include_train: bool = True) -> None:
        """Deliberate NO-OP: keras metrics resolve at compile / step build, not here.

        The API base requires the builder so every task's ``initialize`` can call it; in this
        family there is no prefixed-collection state to build — the compile/fit member hands
        ``compile()`` the UNION (:meth:`_compiled_metrics`) and the TensorFlow loop builds its
        per-split metric OBJECTS when it builds its steps.
        """

    def _build_eval_metrics(self) -> None:
        """Deliberate NO-OP — see :meth:`_build_fit_metrics`."""

    def _compiled_metrics(self) -> Sequence[Lazy[KerasMetric]]:
        """The ONE metric list a compiled model can compute — the UNION of the four slots.

        ``keras.Model.compile`` takes a single list and reports it on training AND validation
        (prefixed ``val_``); there is no hook that narrows a metric to one split short of
        recompiling between phases, which would discard the optimizer state a fit is using. So
        a metric wired to ``val_metrics`` is computed on the training split too — widening is
        the honest failure (a metric appearing where it was not asked for is noise a reader
        ignores, where DROPPING it would be a wired metric that silently reports nothing).
        """
        return union_metric_specs(self.metrics, self.train_metrics, self.val_metrics, self.test_metrics) or []

    # ---- the batch boundary ----------------------------------------------------------------- #
    def _sequence(self, source: Any, shuffle: bool = False, with_targets: bool = True) -> RecordSequence:
        """recordstream's batching adapter, wearing THIS task's batch shape."""
        return RecordSequence(
            source,
            batch_size=self.batch_size,
            shuffle=shuffle,
            transform=lambda batch: self._to_arrays(batch, with_targets=with_targets),
        )

    # ---- callbacks + tracking --------------------------------------------------------------- #
    def _callbacks(self) -> List[Any]:
        """Wired Keras callbacks plus OPTIONAL matrainer tracking (a few lines; removable)."""
        callbacks = [flow(cb) for cb in (self.callbacks or [])]
        for callback in callbacks:
            # Keras's file-writing callbacks (`CSVLogger`, `ModelCheckpoint`) open their path
            # without creating its parent, so a perfectly reasonable `./keras_logs/run.csv`
            # dies with FileNotFoundError several minutes into a fit — after the data walk and
            # the model build.
            target = getattr(callback, "filename", None) or getattr(callback, "filepath", None)
            if isinstance(target, (str, Path)):
                Path(target).expanduser().parent.mkdir(parents=True, exist_ok=True)
        trackers = self._live_trackers()
        if trackers:
            from matrainer.keras import keras_tracking_callback

            # `runnable=self` rather than a params dict: matrainer derives the hyperparameters
            # from the flowed object graph AND dumps the resolved config.
            callbacks.append(
                keras_tracking_callback(trackers, runnable=self, metric_names=as_metric_names(self.metric_names))
            )
        return callbacks

    # ---- @outputs --------------------------------------------------------------------------- #
    @property
    @output
    def trained_model(self) -> Any:
        """The live trained model (post-run ``@output``). Raises while still deferred."""
        if isinstance(self.model, Fluid):
            raise RuntimeError("trained_model is only available after run() — the model is still deferred.")
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
        """Per-epoch metric history, as the engine's own records kept it."""
        return {key: list(values) for key, values in self._history.items()}

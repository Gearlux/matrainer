"""The trainer API — ONE abstract base every per-framework runnable base fills in.

This module reverses the former "matrainer is tracking-only" rule (user decision 2026-08-06;
``docs/architecture.md`` record 9): matrainer now owns the SHARED trainer machinery, because
four task projects had grown the same merged train/evaluate/test/predict runnable per engine
with 60%+ of each module byte-identical across projects. The shape of the consolidation:

* **This class defines the FRAMEWORK-NEUTRAL half of the API** — the universal constructor
  slots, ``run()`` dispatch, the four ``@entrypoint``-marked capabilities, tracker plumbing,
  prediction writing, and the neutral task hooks (:meth:`initialize`,
  :meth:`_prediction_output`, :meth:`_reportable`, :meth:`_report_nonscalars`).
* **A FAMILY base pins the framework types** — ``matrainer.torch.runnable.TorchTrainerRunnable``
  stores ``model: Lazy[nn.Module]`` / ``loss_fn: Optional[Lazy[nn.Module]]`` / the
  ``MetricsSpec`` quartet and declares the torch-typed hooks (``_targets -> Tensor``, the
  metric builders); the keras family base (`matrainer.keras.runnable.KerasTrainerRunnable`)
  does the same with keras types. NOTHING user-facing is
  typed ``Any`` — a slot whose type is a framework's lives in that framework family's base,
  which is the whole reason this base does not store one.
* **One concrete base per framework fills the machinery in** (``matrainer.lightning.
  LightningRunnable``, …) — each self-contained in its extras-gated subpackage. The
  abstraction STOPS at the API: there is deliberately no abstract trainer OBJECT, so the
  ``pl.Trainer`` slot lives only in the lightning base, the epoch loop only in the torch one.
* **A consumer's task base implements the hooks once** for its whole engine fleet, and the
  concrete class is ``class LightningClassifier(ClassificationTorchHooks, ClassificationBase,
  LightningRunnable)`` — keeping its own FULL flat ``__init__`` (the confluid introspection
  surface) that stores task-specific state and passes the shared slots up.

This module is FRAMEWORK-FREE and light-core-safe: it imports recordstream's runnable protocol
and matrainer's own tracking surface, never torch/keras/jax.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterable, List, Mapping, Optional, Union, cast

from confluid import Lazy, configurable
from loggair import get_logger
from recordstream import PredictionsSink, RecordSource, batch_metadata, per_record_predictions, prepare_record_dataset
from recordstream.runnable import ProgressReporting, RunnableTask, entrypoint, run_entrypoint

from matrainer.metric_names import MetricNames
from matrainer.run_names import generate_run_name, run_prefix
from matrainer.tracking.base import Tracker, flow_trackers

logger = get_logger(__name__)

__all__ = ["TrainerRunnable"]


# `@configurable` on an ABSTRACT base is the established pattern (`Tracker` is one): the mark is
# what makes confluid's `__init__`-BODY scans (broadcast accept-list, `lazy_param_names`,
# `to_pydantic`'s body-slot fields) walk this class in a concrete subclass's MRO — without it the
# shared slots below would silently vanish from every form-spec. `validate=False` because the
# keyword-only cooperative constructor is machinery, not a config surface (the CONCRETE class's
# flat signature is what gets validated and introspected).
@configurable(validate=False)
class TrainerRunnable(ProgressReporting, ABC):
    """The abstract trainer API: the framework-neutral slots + dispatch + neutral task hooks.

    Subclassing layers (see the module docstring): a framework FAMILY base adds the
    framework-TYPED slots and hooks; a framework base implements the four entry points with
    its engine's own machinery; a consumer's task base implements the hooks; the concrete
    class combines them and keeps the full flat constructor.

    ``ProgressReporting`` only — the ``TorchRunner`` autograd marker joins in the torch
    family base, because a keras/jax/mlx runnable does not carry it.

    Args (all stored VERBATIM — the lazy-init mandate: the constructor does no work):
        train_set: Training dataset / Stream (``task="fit"``).
        test_set: Held-out eval dataset / Stream (``task="test"`` / ``"predict"``; also
            tested after ``fit`` when set).
        task: Entry point :meth:`run` dispatches to.
        max_epochs: Ergonomic flat-config knob every engine translates.
        trackers: matrainer trackers — one or a list; THE run's output wiring (there is no
            ``logger:`` slot anywhere in the fleet).
        experiment_name: Tracking identity.
        run_name: Tracking identity; ``None`` draws ``generate_run_name(run_prefix(task))``.
        input_key: Record key carrying the model input.
        target_key: Record key carrying the target.
        metric_names: How recorded metrics are named (``matrainer.metric_names``).
    """

    #: Short project label the run methods prefix their log lines with ("sonair training: ...").
    #: A consumer's task base overrides it; the default names the library.
    log_label: ClassVar[str] = "matrainer"

    #: The task model slot — DECLARED here (the name is universal) but STORED and TYPED by the
    #: framework family base: `Lazy[nn.Module]` in the torch family, a keras model in that one.
    #: `Any` is the sanctioned strictly-needed case (workspace mandate): the neutral base cannot
    #: name a framework type, and a task base's torch-free half still reads/flows the slot.
    model: Any

    def __init__(
        self,
        *,
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        task: RunnableTask = "fit",
        max_epochs: int = 10,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        experiment_name: str = "matrainer",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
        metric_names: Optional[Lazy[MetricNames]] = None,
    ) -> None:
        # Cooperative FIRST: a torch-family concrete has `nn.Module`/`LightningModule` further
        # down the MRO, and torch forbids attribute assignment before its own __init__ ran.
        super().__init__()
        self.train_set: Optional[Lazy[RecordSource]] = train_set
        self.val_set: Optional[Lazy[RecordSource]] = None
        self.test_set: Optional[Lazy[RecordSource]] = test_set
        self.task: RunnableTask = task
        self.max_epochs = max_epochs
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix=run_prefix(task))
        self.input_key = str(input_key)
        self.target_key = str(target_key)
        self.predictions_sink: Optional[Lazy[PredictionsSink]] = None
        self.metrics_sidecar_path: Optional[str] = None
        # Stored VERBATIM: may hold a single tracker, a list, or deferred Fluid markers;
        # `_live_trackers()` normalizes + flows at run time, writing back.
        self.trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = trackers
        # Stored VERBATIM for the same reason: wired from YAML it arrives as a deferred marker,
        # built where the metrics are reported.
        self.metric_names: Optional[Lazy[MetricNames]] = metric_names

    # ---- run dispatch --------------------------------------------------------------------- #
    def run(self) -> None:
        self.initialize()
        # The `@entrypoint` markers below ARE the dispatch table — a hand-written
        # {task: method} dict restates them and drifts (recordstream `docs/runnable.md`).
        run_entrypoint(self, self.task)

    # ---- the four entry points (the API; a framework base fills them in) ------------------- #
    @entrypoint("fit", role="trainer", primary=True)
    @abstractmethod
    def fit(self) -> None:
        """Train (and, when a ``test_set`` is wired, test afterwards)."""
        raise NotImplementedError

    @entrypoint("evaluate", role="evaluator")
    @abstractmethod
    def evaluate(self) -> None:
        """Score the validation split."""
        raise NotImplementedError

    @entrypoint("test", role="evaluator", primary=True)
    @abstractmethod
    def test(self) -> None:
        """Score the held-out split."""
        raise NotImplementedError

    @entrypoint("predict", role="predictor", primary=True)
    @abstractmethod
    def predict(self) -> None:
        """Stream predictions over the ``test_set`` (into the wired sink, when any)."""
        raise NotImplementedError

    # ---- the neutral task hooks (a consumer's task base implements these) ------------------ #
    @abstractmethod
    def initialize(self) -> None:
        """Task setup before dispatch: resolve dimensions/labels, flow the model, build metrics.

        Entirely task-owned — the four tasks differ wildly here (a label walk, a mask walk, a
        Boxes walk, a trainable-parameters check) — and it typically ends by calling the
        family base's metric builders.
        """
        raise NotImplementedError

    @abstractmethod
    def _resolve_task_defaults(self) -> None:
        """Fill the loss / metrics sentinels via the task hooks. Idempotent.

        Declared here because every family provides it (a task base's neutral half calls it
        from ``initialize``); IMPLEMENTED by the family base, whose typed slots it fills.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_fit_metrics(self, include_train: bool = True) -> None:
        """Build every split's live metrics for a fit run, once (guarded).

        ``include_train=False`` serves a task with no train-split predictions to score.
        Declared here (every family builds per-split metrics from the shared + per-split
        specs); implemented by the family base in its own metrics vocabulary.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_eval_metrics(self) -> None:
        """Build ONLY the split this eval task scores, once (guarded). Family-implemented."""
        raise NotImplementedError

    def _prediction_output(self, preds: Any) -> Any:
        """Wrap raw model output in the task's prediction CONTRACT for a sink.

        Identity by default; a task returns ``classification_output`` /
        ``segmentation_output`` / ``restoration_output`` — the contract a
        ``PredictionsSink`` reads by key. (``Any``: the payload is whatever the engine
        emitted — a batch of tensors, a list of per-image dicts — polymorphic by design.)
        """
        return preds

    def _reportable(self, values: Mapping[str, Any]) -> Mapping[str, Any]:
        """Filter a computed metrics mapping before ANY reporting path reads it.

        Identity by default; a task with a pseudo-metric to strip (detection MAP's
        ``classes`` key — the observed ids, not a metric) applies its rule here, ONCE for
        every path (epoch report, sidecar, ``computed_metrics``).
        """
        return values

    def _report_nonscalars(self, values: Mapping[str, Any], step: Optional[int], trackers: List[Tracker]) -> None:
        """Report the task's non-scalar metrics (matrices / per-class vectors) to the trackers.

        No-op by default (a task whose metrics are all scalar — restoration — needs nothing);
        classification calls :func:`matrainer.tracking.render.report_matrices`, the per-pixel
        and detection tasks :func:`~matrainer.tracking.render.report_vectors`.
        """

    # ---- shared plumbing ------------------------------------------------------------------- #
    def _live_trackers(self) -> List[Tracker]:
        """Normalize and flow the wired trackers ONCE, writing the result back to the slot.

        Written back rather than returned fresh each call: ``flow()`` on a LIVE object
        returns that same object (idempotent), but a deferred ``Fluid`` would be REBUILT on
        every call — two MLflow runs for one training run.
        """
        self.trackers = flow_trackers(self.trackers)
        return cast(List[Tracker], self.trackers)

    def _prepare(self, source: Optional[Lazy[RecordSource]]) -> Optional[Any]:
        """Normalize a wired source's TYPE and its STATE, in THIS process (fork safety).

        Delegates to :func:`recordstream.prepare_record_dataset`; a method so a task can opt
        out or extend. ``None`` passes through, so an unwired split needs no guard.
        """
        return prepare_record_dataset(source)

    def _write_predictions(self, preds: Any, batch: Any) -> None:
        """Write ONE prediction per record to the wired sink, paired with that record's metadata.

        The model emits a BATCHED output while the sink contract is per-record —
        :func:`recordstream.per_record_predictions` slices it, and
        :func:`recordstream.batch_metadata` recovers each record's metadata columns.
        """
        per_record_meta = (
            batch_metadata(batch, exclude=(self.input_key, self.target_key)) if isinstance(batch, dict) else None
        )
        items = per_record_predictions(preds)
        for i, item in enumerate(items):
            m = per_record_meta[i] if per_record_meta and i < len(per_record_meta) else {}
            cast(PredictionsSink, self.predictions_sink).write(item, m)

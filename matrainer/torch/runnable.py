"""The torch FAMILY trainer base — where the framework types are pinned down.

``TorchTrainerRunnable`` is the layer between the framework-neutral API
(:class:`matrainer.runnable.TrainerRunnable`) and the per-framework machinery bases
(``matrainer.lightning.LightningRunnable``, the bare-torch loop, the fastai ``Learner`` base,
the ``transformers.Trainer`` one — every engine that trains a ``torch.nn.Module`` against
torchmetrics). It exists so nothing user-facing is typed ``Any``: the model slot is
``Lazy[nn.Module]``, the loss ``Optional[Lazy[nn.Module]]``, the four metric slots
``MetricsSpec``, ``_targets`` returns a ``Tensor`` — types the neutral base cannot name
without importing a framework. A keras family base mirrors this with keras types.

What it owns beyond the types: the torch-family halves of the hook vocabulary
(``_default_loss`` / ``_default_metrics`` / ``_metric_flow_dims`` / ``_targets``), the
per-split metric BUILDERS (`merge_metric_specs` → `build_metric_collection` → prefix clone,
plus the two `_metrics_built`-guarded convenience builders every task's ``initialize`` calls),
the sidecar write, the shared ``trained_model`` output, and the per-task
``__needs_autograd__`` override of the ``TorchRunner`` marker.
"""

from abc import abstractmethod
from typing import Any, Dict, Iterable, Optional, Union, cast

import torch
import torch.nn as nn
from confluid import Fluid, Lazy, configurable, output
from loggair import get_logger
from recordstream import RecordSource
from recordstream.items import Record
from recordstream.runnable import RunnableTask, TorchRunner
from torch import Tensor
from torchmetrics import Metric, MetricCollection

from matrainer.metric_names import MetricNames
from matrainer.metric_slots import merge_metric_specs
from matrainer.runnable import TrainerRunnable
from matrainer.torch.metrics import MetricsSpec, build_metric_collection, scalar_metrics, write_collection_sidecar
from matrainer.tracking.base import Tracker

logger = get_logger(__name__)

__all__ = ["TorchTrainerRunnable"]


# `@configurable(validate=False)` for the same reason as the API base: the mark is what makes
# confluid's body-slot scans see this layer's slots (`loss_fn`, the metric quartet) in a concrete
# subclass's MRO; the concrete class's flat signature stays the validated config surface.
@configurable(validate=False)
class TorchTrainerRunnable(TrainerRunnable, TorchRunner):
    """The torch-family trainer base: typed slots + the torchmetrics half of the machinery.

    Args (on top of :class:`~matrainer.runnable.TrainerRunnable`'s):
        model: The task model — a live ``nn.Module`` or (typically) a deferred ``!lazy:``
            marker the task's ``initialize`` flows with its resolved dimensions.
        metrics: Shared metric templates flowed per split (``None`` = the task default).
        train_metrics: The train split's OWN spec, merged over ``metrics``.
        val_metrics: The val split's OWN spec, merged over ``metrics``.
        test_metrics: The test split's OWN spec, merged over ``metrics``.
    """

    def __init__(
        self,
        *,
        model: Optional[Lazy[nn.Module]] = None,
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        task: RunnableTask = "fit",
        max_epochs: int = 10,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        experiment_name: str = "matrainer",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
        metrics: MetricsSpec = None,
        train_metrics: MetricsSpec = None,
        val_metrics: MetricsSpec = None,
        test_metrics: MetricsSpec = None,
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
        # The slot keeps the CONSUMER-visible type (`Lazy[nn.Module]`, as every concrete class
        # declared it): the concrete constructors mark the param `Mandatory`, so a bare `None`
        # only ever comes from the base's own zero-arg construction — hence the cast rather
        # than an `Optional` that would force a None-check on every downstream read.
        self.model: Lazy[nn.Module] = cast(Lazy[nn.Module], model)
        # Sentinel: `None` means "not chosen" — a live default could not be told apart from a
        # user's explicit choice; `_resolve_task_defaults()` fills it via the task hooks.
        # `Lazy[...]`: a `!class:` loss stays a deferred marker until the run flows it.
        self.loss_fn: Optional[Lazy[nn.Module]] = None
        # Per-split slots, each dual-purpose: they hold that split's metrics SPEC until the
        # builders below run, and the live prefixed `MetricCollection` after. The ATTRIBUTES are
        # `Any` deliberately (the sanctioned strictly-needed case): the spec→collection state
        # transition is unknowable to a type checker, and the TYPED config surface is the ctor
        # param (`MetricsSpec`). `_metrics_built` is what tells the two states apart — `is None`
        # cannot, once a slot may legitimately arrive carrying a spec (without it a second
        # `initialize()` re-prefixes `val/` into `val/val/`).
        self.metrics: Any = metrics
        self.train_metrics: Any = train_metrics
        self.val_metrics: Any = val_metrics
        self.test_metrics: Any = test_metrics
        self._metrics_built: bool = False

    #: The compute device batches move to — PROVIDED by the machinery base (Lightning's own
    #: ``device`` property; an owned loop exposes its resolved device under the same name),
    #: declared here so the task hooks (`_targets` moving tensors) type-check against the family.
    device: "torch.device"

    @property
    def __needs_autograd__(self) -> bool:  # type: ignore[override]  # dynamic per-task override of the marker
        """Only ``fit`` needs autograd; evaluate / test / predict are inference-only."""
        return self.task == "fit"

    # ---- the torch-typed task hooks ------------------------------------------------------- #
    def _targets(self, batch: Record) -> Tensor:
        """The target tensor the loss expects, read from a collated batch — THE task-shaped read.

        A classifier returns int64 ids / a float multi-hot, a segmenter an int64 mask, a
        restorer the clean image. NOT abstract, deliberately: a detection task reads its batch
        natively inside overridden step functions and never calls this — the default therefore
        raises with that explanation instead of forcing a dummy implementation there.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares no _targets() — either the task base must implement "
            "it, or this task overrides the step functions and never reads targets this way."
        )

    def _default_loss(self) -> Optional[nn.Module]:
        """The task's default loss, built once ``initialize`` knows the target shape.

        ``None`` (the default) means the task has no separate loss object — e.g. a detection
        model computes its loss inside ``forward``.
        """
        return None

    def _default_metrics(self) -> MetricsSpec:
        """The task's default shared metric spec (torchmetrics templates).

        ``None`` (the default) leaves the sentinel unresolved — legitimate while a needed
        dimension is still unknown; :meth:`_resolve_task_defaults` is idempotent and will be
        asked again.
        """
        return None

    def _metric_flow_dims(self) -> Dict[str, Union[int, float]]:
        """The sizing kwargs metric templates are flowed with (``num_classes`` / ``data_range``)."""
        return {}

    def _after_fit(self, trainer: Any, train_ds: Any, val_ds: Any, test_ds: Any) -> None:
        """Task epilogue between training and the post-fit test (no-op by default).

        The classification task persists its class names here (beside the checkpoints, onto
        the trackers). ``trainer`` is the ENGINE's own driver object — a ``pl.Trainer`` under
        the lightning base, the runnable itself under an owned loop — which is why the family
        base cannot type it more precisely than the machinery base's call site does.
        """

    # ---- sentinel resolution + metric building -------------------------------------------- #
    def _resolve_task_defaults(self) -> None:
        """Fill the ``loss_fn`` / ``metrics`` sentinels via the task hooks. Idempotent.

        A value the user DID wire is left untouched — that is what the ``None`` sentinel
        buys: "not chosen" is distinguishable from "chosen to be the default". Also called
        from the step functions so a directly-invoked step works without ``initialize()``.
        """
        if self.loss_fn is None:
            self.loss_fn = self._default_loss()
            if self.loss_fn is not None:
                logger.debug(f"loss_fn defaulted to {type(self.loss_fn).__name__}.")
        if self.metrics is None:
            self.metrics = self._default_metrics()

    def _build_split_metrics(self, spec: MetricsSpec, prefix: str) -> MetricCollection:
        """One split's live, prefixed collection — its own spec merged over the shared one.

        The combination rule is :func:`matrainer.metric_slots.merge_metric_specs`, shared with
        every backend and task so a config moving between them cannot change shape; the sizing
        dimension comes from the task's :meth:`_metric_flow_dims` and is applied per template
        (see ``matrainer.torch.metrics._sizing_kwargs``).
        """
        return build_metric_collection(merge_metric_specs(self.metrics, spec), **self._metric_flow_dims()).clone(
            prefix=prefix
        )

    def _build_fit_metrics(self, include_train: bool = True) -> None:
        """Build every split's collection for a fit run, once (`_metrics_built`-guarded).

        ``include_train=False`` serves a task with no train slot — a detection model in train
        mode returns only a loss dict, so there are no train-split predictions to score.
        """
        if self._metrics_built:
            return
        if include_train:
            self.train_metrics = self._build_split_metrics(self.train_metrics, "train/")
        self.val_metrics = self._build_split_metrics(self.val_metrics, "val/")
        self.test_metrics = self._build_split_metrics(self.test_metrics, "test/")
        self._metrics_built = True

    def _build_eval_metrics(self) -> None:
        """Build ONLY the split this eval task scores, once (`_metrics_built`-guarded)."""
        if self._metrics_built:
            return
        slot = "val_metrics" if self.task == "evaluate" else "test_metrics"
        prefix = "val/" if self.task == "evaluate" else "test/"
        setattr(self, slot, self._build_split_metrics(getattr(self, slot), prefix))
        self._metrics_built = True

    # ---- reporting plumbing ---------------------------------------------------------------- #
    def _write_metrics_sidecar(self, metrics: MetricsSpec) -> None:
        """Hand the run's final scalars to the metrics sidecar (matrainer owns where they land)."""
        write_collection_sidecar(
            self.metrics_sidecar_path,
            metrics,
            self.experiment_name,
            self.run_name,
            reportable=self._reportable,
        )

    def _computed_eval_metrics(self) -> Dict[str, float]:
        """The eval-task half of the ``computed_metrics`` output — shared across the family."""
        metrics = self.val_metrics if self.task == "evaluate" else self.test_metrics
        if not isinstance(metrics, (Metric, MetricCollection)):
            raise RuntimeError("computed_metrics is only available after run().")
        return scalar_metrics(self._reportable(metrics.compute()))

    # ---- @outputs -------------------------------------------------------------------------- #
    @property
    @output
    def trained_model(self) -> nn.Module:
        """The trained/live model (post-run ``@output``). Raises while still deferred."""
        if isinstance(self.model, Fluid):
            raise RuntimeError("trained_model is only available after run(): the model is still a deferred Fluid.")
        return cast(nn.Module, self.model)

    @property
    @output
    @abstractmethod
    def computed_metrics(self) -> Dict[str, float]:
        """Final SCALAR metrics (post-run ``@output``). Raises before ``run()``.

        Abstract at the family level because the FIT half differs per machinery base (a
        ``pl.Trainer`` exposes ``callback_metrics``, an owned loop keeps its own dict); the
        shared eval half is :meth:`_computed_eval_metrics`.
        """
        raise NotImplementedError

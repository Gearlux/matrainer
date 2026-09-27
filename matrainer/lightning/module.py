"""``LightningRunnable`` — the PyTorch-Lightning machinery base of the trainer API.

The Lightning member of the family (``docs/architecture.md`` record 9): it fills the
:class:`~matrainer.torch.runnable.TorchTrainerRunnable` API in with Lightning's own machinery —
a ``LightningModule`` with the standard step functions, the deferred ``pl.Trainer`` slot and its
tracking hookup, the loader triple, and the four entry points driving ``trainer.fit`` /
``.validate`` / ``.test`` / ``.predict``. Everything task-shaped stays a hook a consumer's task
base implements (``initialize`` / ``_targets`` / the defaults / ``_prediction_output`` /
``_report_nonscalars`` / ``_after_fit``); everything here was measured byte-identical across the
task projects before the 2026-08-06 consolidation.

A concrete consumer class subclasses ``(TaskTorchHooks, TaskBase, LightningRunnable)``, keeps
its own FULL flat ``__init__`` (the confluid introspection surface) and passes the shared slots
up — see the record's example. This class itself is deliberately NOT ``@configurable``: it is a
base, not a wireable runnable.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Union, cast

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Partial, PartialClass, configurable, flow, output
from loggair import get_logger
from recordstream import RecordSource, batch_tensor, collate_records, ensure_record_dataset
from recordstream.items import Record
from recordstream.loaders import loader_slots
from recordstream.runnable import RunnableTask
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric, MetricCollection

from matrainer.lightning.callbacks import TrackingCallback, activate_for_lightning, attach_lightning_progress
from matrainer.lightning.trackers import TrackerLogger
from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.torch.metrics import MetricsSpec, scalar_metrics
from matrainer.torch.runnable import TorchTrainerRunnable
from matrainer.tracking.base import Tracker

logger = get_logger(__name__)

__all__ = ["LightningRunnable"]


# `@configurable(validate=False)` for the same reason as the two bases above it: the mark makes
# confluid's body-slot scans see THIS layer's slots (`optimizer`, the loader triple, `lightning`,
# `log_system_metrics`) in a concrete subclass's MRO — without it they vanish from every
# form-spec. The concrete class's flat signature stays the validated config surface.
@configurable(validate=False)
class LightningRunnable(TorchTrainerRunnable, L.LightningModule):
    """A standard Lightning train/evaluate/test/predict runnable, task hooks left open.

    Args (on top of the family base's):
        batch_size: Rows per batch, baked into the three deferred loader slots.
        num_workers: Loader workers (``persistent_workers`` derives from it).
        persistent_workers: Override that derivation. ``None`` derives (``num_workers != 0``);
            an explicit value is for a HOST fact the derivation cannot see — macOS terminates
            persistent workers slowly, so a config there sets ``false`` and keeps its workers.
        collate_fn: The batch-shape choice baked into the loaders — ``collate_records``
            stacks (classification/segmentation/restoration), ``collate_list`` does not
            (detection's ``List[Tensor]`` contract). A machinery-level knob for the task
            base, not a config key; a config replaces a loader slot wholesale instead.

    Optional OVERRIDE slots — every one is already constructed with a sensible default in
    ``__init__`` (the task-shaped ones, ``loss_fn`` + the metric slots, resolve to this task's
    defaults in ``initialize()``), so nothing here NEEDS wiring; a same-named top-level YAML key
    replaces one via broadcasting when you want to. ``val_set`` is the one a config typically
    does wire (there is no sensible default validation split): ``val_set``, ``optimizer``,
    ``train/val/test_loader``, ``lightning``, ``loss_fn``, ``predictions_sink``,
    ``metrics_sidecar_path``, ``log_system_metrics``.
    """

    def __init__(
        self,
        *,
        model: Optional[Partial[nn.Module]] = None,
        train_set: Optional[Partial[RecordSource]] = None,
        test_set: Optional[Partial[RecordSource]] = None,
        task: RunnableTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 32,
        num_workers: int = 0,
        persistent_workers: Optional[bool] = None,
        trackers: Optional[Union[Partial[Tracker], Iterable[Partial[Tracker]]]] = None,
        experiment_name: str = "matrainer",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
        metrics: MetricsSpec = None,
        train_metrics: MetricsSpec = None,
        val_metrics: MetricsSpec = None,
        test_metrics: MetricsSpec = None,
        metric_names: Optional[Partial[MetricNames]] = None,
        collate_fn: Callable[[List[Record]], Record] = collate_records,
    ) -> None:
        super().__init__(
            model=model,
            train_set=train_set,
            test_set=test_set,
            task=task,
            max_epochs=max_epochs,
            trackers=trackers,
            experiment_name=experiment_name,
            run_name=run_name,
            input_key=input_key,
            target_key=target_key,
            metrics=metrics,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            metric_names=metric_names,
        )
        # Deferred runtime slots — `Partial[T]` DECLARES that the run methods flow this later,
        # usually with a kwarg that does not exist yet (`params=self.parameters()`,
        # `dataset=`). Deferral is enforced by the slot holding a `PartialClass` value; the
        # annotation gives `to_pydantic` a real type for form-specs/widgets.
        self.optimizer: Partial[Optimizer] = PartialClass(torch.optim.AdamW, lr=1e-4, weight_decay=0.05)
        slots = loader_slots(batch_size, num_workers, collate_fn=collate_fn, persistent_workers=persistent_workers)
        self.train_loader: Partial[DataLoader[Any]] = slots.train
        self.val_loader: Partial[DataLoader[Any]] = slots.val
        self.test_loader: Partial[DataLoader[Any]] = slots.test
        # No `logger=` here: the trackers ARE the metric sinks, adapted to Lightning's logger
        # interface in `_build_trainer` (a Tracker may still be a deferred Fluid at this
        # point, so the adapter is built at run time, not now).
        self.lightning: Partial[L.Trainer] = PartialClass(L.Trainer, max_epochs=max_epochs)
        self.log_system_metrics: bool = True

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, cast(nn.Module, self.model)(x))

    # ---- standard Lightning step functions ------------------------------------------------- #
    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        self._resolve_task_defaults()  # no-op once resolved; keeps a directly-called step working
        x = batch_tensor(batch, self.input_key, self.device)
        y = self._targets(batch)
        preds = self(x)
        loss: Tensor = cast(nn.Module, self.loss_fn)(preds, y)
        # TWO calls, one series each, and that is what keeps the name canonical: Lightning
        # appends `_step` / `_epoch` when ONE name carries both, so a single
        # `on_step=True, on_epoch=True` call recorded `train/loss_epoch` — a name no other
        # backend writes, in the column a cross-engine comparison uses. The per-step series is
        # still worth having (nothing else shows the loss moving WITHIN an epoch) and keeps
        # the progress bar's live number, so it gets its own explicit name instead.
        self.log("train/loss", loss, on_step=False, on_epoch=True, batch_size=x.shape[0])
        self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=x.shape[0])
        if isinstance(self.train_metrics, (Metric, MetricCollection)):
            # UPDATE only — the reduction happens once in `on_train_epoch_end`, exactly as it
            # does for the eval splits. Logging per-batch values with `on_step=True` was
            # measured wrong twice over: Lightning's `log_every_n_steps` DROPS a series
            # shorter than the interval, and what survives is ONE batch's value beside every
            # other backend's epoch aggregate.
            self.train_metrics.update(preds, y)
        return loss

    def on_train_epoch_start(self) -> None:
        # Reset at epoch START, matching the eval splits: the LAST epoch's state has to
        # survive the run, because `_write_metrics_sidecar` and the `computed_metrics`
        # @output both `compute()` after the trainer has already returned.
        if isinstance(self.train_metrics, (Metric, MetricCollection)):
            self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self._report_epoch_metrics(self.train_metrics)

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        self._resolve_task_defaults()  # no-op once resolved; keeps a directly-called step working
        x = batch_tensor(batch, self.input_key, self.device)
        y = self._targets(batch)
        preds = self(x)
        loss: Tensor = cast(nn.Module, self.loss_fn)(preds, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        if isinstance(self.val_metrics, (Metric, MetricCollection)):
            self.val_metrics.update(preds, y)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        x = batch_tensor(batch, self.input_key, self.device)
        preds = self(x)
        if isinstance(self.test_metrics, (Metric, MetricCollection)):
            self.test_metrics.update(preds, self._targets(batch))
        if self.predictions_sink is not None:
            self._write_predictions(self._prediction_output(preds), batch)
        return preds

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        """Predict a batch, wrapped in the task's prediction CONTRACT (`_prediction_output`).

        Returning the contract rather than raw logits is what makes the output consumable: a
        predictions sink reads its keys by name and cannot softmax on the caller's behalf —
        only the model knows whether its head already did.
        """
        out = self._prediction_output(self(batch_tensor(batch, self.input_key, self.device)))
        if self.predictions_sink is not None:
            self._write_predictions(out, batch)
        return out

    # ---- epoch-end metric reporting -------------------------------------------------------- #
    # An eval split's metrics are computed HERE rather than by handing the whole collection to
    # `log_dict`, because `log_dict` carries SCALARS only: Lightning reduces every logged
    # value through `convert_tensors_to_scalars`, which raises on an N×N confusion matrix or
    # a per-class vector. Which computed values are scalar is a property of the COMPUTED
    # value, so the split happens at compute time: scalars to `log_dict`, the rest through
    # the task's `_report_nonscalars` hook (matrices / vectors → the trackers' text channel).
    #
    # Reset happens at epoch START, not end, so the LAST epoch's state survives the run —
    # `_write_metrics_sidecar` and the `computed_metrics` @output both `compute()` after
    # `trainer.validate()` / `.test()` has returned.
    def on_validation_epoch_start(self) -> None:
        if isinstance(self.val_metrics, (Metric, MetricCollection)):
            self.val_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self._report_epoch_metrics(self.val_metrics)

    def on_test_epoch_start(self) -> None:
        if isinstance(self.test_metrics, (Metric, MetricCollection)):
            self.test_metrics.reset()

    def on_test_epoch_end(self) -> None:
        self._report_epoch_metrics(self.test_metrics)

    def _report_epoch_metrics(self, collection: MetricsSpec) -> None:
        """Compute one split's metrics once at epoch end: scalars logged, the rest reported.

        The task's `_reportable` filter applies FIRST (identity for most tasks), then the
        scalars go to `log_dict` and the same computed mapping goes to the task's
        `_report_nonscalars` hook. Nothing is reported during Lightning's sanity check — two
        batches through an untrained model produce a table nobody wants to read, and it
        would be the FIRST one in the log.
        """
        if not isinstance(collection, (Metric, MetricCollection)):
            return
        values = self._reportable(collection.compute())
        self.log_dict(scalar_metrics(values))
        if self._trainer is not None and self.trainer.sanity_checking:
            return
        step = self.current_epoch if self._trainer is not None else None
        self._report_nonscalars(values, step, self._live_trackers())

    def configure_optimizers(self) -> Any:
        return flow(self.optimizer, params=self.parameters())

    # ---- the four entry points (markers inherited from the API base) ----------------------- #
    def fit(self) -> None:
        logger.info(f"{self.log_label} training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError(f"{type(self).__name__}.fit() requires a 'train_set'.")
        train_ds = ensure_record_dataset(self.train_set)
        val_ds = ensure_record_dataset(self.val_set) if self.val_set is not None else None
        test_ds = ensure_record_dataset(self.test_set) if self.test_set is not None else None
        t_loader = flow(self.train_loader, dataset=train_ds)
        v_loader = flow(self.val_loader, dataset=val_ds) if val_ds is not None else None
        trainer = self._build_trainer()
        trainer.fit(model=self, train_dataloaders=t_loader, val_dataloaders=v_loader)
        logger.info("Training complete.")
        self._after_fit(trainer, train_ds, val_ds, test_ds)
        if test_ds is not None:
            trainer.test(model=self, dataloaders=flow(self.test_loader, dataset=test_ds))

    def evaluate(self) -> None:
        self._run_eval("validate", "val_set", "val_loader", "val_metrics")

    def test(self) -> None:
        self._run_eval("test", "test_set", "test_loader", "test_metrics")

    def predict(self) -> None:
        logger.info(f"{self.log_label} predicting: {self.experiment_name} ({self.run_name})")
        if self.test_set is None:
            raise ValueError(f"{type(self).__name__}.predict() requires a 'test_set'.")
        dataset = ensure_record_dataset(self.test_set)
        if self.predictions_sink is not None:
            self.predictions_sink = flow(self.predictions_sink)
        loader = flow(self.test_loader, dataset=dataset)
        trainer = self._build_trainer()
        trainer.predict(model=self, dataloaders=loader)
        if self.predictions_sink is not None and hasattr(self.predictions_sink, "close"):
            self.predictions_sink.close()

    def _run_eval(self, stage: str, dataset_slot: str, loader_slot: str, metrics_slot: str) -> None:
        logger.info(f"{self.log_label} evaluating ({stage}): {self.experiment_name} ({self.run_name})")
        source = getattr(self, dataset_slot)
        if source is None:
            raise ValueError(f"{type(self).__name__}.{stage}() requires a '{dataset_slot}'.")
        dataset = ensure_record_dataset(source)
        if self.predictions_sink is not None:
            self.predictions_sink = flow(self.predictions_sink)
        loader = flow(getattr(self, loader_slot), dataset=dataset)
        trainer = self._build_trainer()
        runner = trainer.validate if stage == "validate" else trainer.test
        runner(model=self, dataloaders=loader)
        if self.predictions_sink is not None and hasattr(self.predictions_sink, "close"):
            self.predictions_sink.close()
        self._write_metrics_sidecar(getattr(self, metrics_slot))

    def _build_trainer(self) -> L.Trainer:
        """Flow the ``pl.Trainer`` and attach the OPTIONAL matrainer tracking.

        The trackers ARE the Trainer's loggers: each is flowed (it may still be a deferred
        ``Fluid``) and adapted through ``TrackerLogger``, so Lightning's per-step metrics
        land in the same backends as the provenance and the manifest. There is no separate
        ``logger:`` slot — one wiring, one place a run's output goes.
        """
        trackers = self._live_trackers()
        names = as_metric_names(self.metric_names)
        adapters = [TrackerLogger(t, metric_names=names) for t in trackers]
        # A runtime kwarg wins over the stored one, so an explicitly wired `lightning:` block
        # keeps every knob it set and only gains the loggers.
        trainer = cast(L.Trainer, flow(self.lightning, logger=adapters or False))
        if trackers:
            # `runnable=self` rather than a params dict: matrainer derives the
            # hyperparameters from the flowed object graph AND dumps the resolved config, so
            # no two backends can answer "what were the hyperparameters?" differently.
            cb = TrackingCallback(trackers, runnable=self)
            trainer.callbacks.append(cb)  # type: ignore[attr-defined]
        attach_lightning_progress(trainer, self)
        if self.log_system_metrics:
            activate_for_lightning(trainer)
        return trainer

    # ---- @outputs -------------------------------------------------------------------------- #
    @property
    @output
    def computed_metrics(self) -> Dict[str, float]:
        """Final SCALAR metrics (post-run ``@output``). Raises before ``run()``."""
        if self.task == "fit":
            try:
                return scalar_metrics(dict(self.trainer.callback_metrics))
            except RuntimeError as exc:
                raise RuntimeError("computed_metrics is only available after run().") from exc
        return self._computed_eval_metrics()

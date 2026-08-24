"""``TorchLoopRunnable`` — the bare-torch machinery base of the trainer API.

The control-experiment member of the family (``docs/architecture.md`` record 9): the epoch
written out in the open — ``zero_grad`` / ``backward`` / ``step``, an explicit device, an
explicit ``model.train()`` / ``model.eval()``, an explicit ``torch.no_grad()`` on the eval
pass — with everything task-shaped left to the hooks a consumer's task base implements. Every
method here was measured near-identical across the four task projects' hand-written loops
before the consolidation; what differed per task is exactly the three batch hooks below
(:meth:`_train_batch_loss`, :meth:`_eval_batch`, :meth:`_predict_batch` — a detection task
overrides all three for the loss-dict-inside-``forward`` contract).

What a loop implies, and this base deliberately does NOT have: no framework loggers (tracking
goes through ``trackers:``, via matrainer's plain functions — the hookup style that exists for
loops with no callback protocol), no callback protocol, no distributed strategy. Multi-GPU is
out of scope ON PURPOSE — that is what the Lightning base is for, and a half-``Trainer`` here
would be a worse one.

The device CHOICE is :func:`matrainer.torch.device.resolve_device` (a ``Trainer`` picks an
accelerator for you; every hand-written loop picks identically), resolved at the top of
``run()`` and exposed as the read-only :attr:`device` property — the same name Lightning's own
property gives the task hooks, which is what lets ONE task-hooks mixin serve both machinery
bases.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union, cast

import torch
import torch.nn as nn
from confluid import Partial, PartialClass, configurable, flow, output
from loggair import get_logger
from recordstream import RecordSource, batch_tensor, collate_records
from recordstream.items import Record
from recordstream.loaders import loader_slots
from recordstream.runnable import RunnableTask, run_entrypoint
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Metric, MetricCollection

from matrainer.artifacts import write_metrics_sidecar
from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.torch.device import Device, resolve_device
from matrainer.torch.metrics import MetricsSpec, scalar_metrics
from matrainer.torch.runnable import TorchTrainerRunnable
from matrainer.tracking.base import Tracker

logger = get_logger(__name__)

__all__ = ["TorchLoopRunnable"]


# `@configurable(validate=False)` for the same reason as the other bases: the mark makes
# confluid's body-slot scans see THIS layer's slots (`optimizer`, the loader triple, the device
# knobs) in a concrete subclass's MRO. The concrete class's flat signature stays the validated
# config surface.
@configurable(validate=False)
class TorchLoopRunnable(TorchTrainerRunnable):
    """A hand-written torch train/evaluate/test/predict runnable, task hooks left open.

    Args (on top of the family base's):
        batch_size: Rows per batch, baked into the three deferred loader slots.
        num_workers: Loader workers (``persistent_workers`` derives from it).
        persistent_workers: Override that derivation. ``None`` derives (``num_workers != 0``);
            an explicit value is for a HOST fact the derivation cannot see — macOS terminates
            persistent workers slowly, so a config there sets ``false`` and keeps its workers.
        device: Where to run — the closed :data:`~matrainer.torch.device.Device` Literal;
            ``"auto"`` prefers CUDA → MPS → CPU and an explicit value is taken at its word.
        checkpoint_dir: Directory for ``model.pt`` (a plain ``state_dict`` — there is no
            framework format). ``None`` writes no weights, which is what a smoke run wants.
        grad_clip: Clip gradients to this global norm before each ``step()``. ``None`` disables.
        collate_fn: The batch-shape choice baked into the loaders — ``collate_records`` stacks,
            ``collate_list`` does not (detection's ``List[Tensor]`` contract). A machinery-level
            knob for the task base, not a config key.

    Optional OVERRIDE slots (each already defaulted; a same-named top-level YAML key replaces
    one via broadcasting): ``val_set``, ``optimizer``, ``train/val/test_loader``, ``loss_fn``,
    ``predictions_sink``, ``metrics_sidecar_path``.
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
        device: Device = "auto",
        checkpoint_dir: Optional[str] = None,
        grad_clip: Optional[float] = None,
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
        self.batch_size = batch_size
        self.device_choice: Device = device
        self.checkpoint_dir = checkpoint_dir
        self.grad_clip = grad_clip
        slots = loader_slots(batch_size, num_workers, collate_fn=collate_fn, persistent_workers=persistent_workers)
        self.train_loader: Partial[DataLoader[Any]] = slots.train
        self.val_loader: Partial[DataLoader[Any]] = slots.val
        self.test_loader: Partial[DataLoader[Any]] = slots.test
        self.optimizer: Partial[Optimizer] = PartialClass(torch.optim.AdamW, lr=1e-4, weight_decay=0.05)
        self._history: Dict[str, List[float]] = {}
        self._computed_metrics: Dict[str, float] = {}
        self._device: torch.device = torch.device("cpu")

    @property
    def device(self) -> torch.device:  # type: ignore[override]  # narrows the family declaration
        """The resolved run device — the same name Lightning's property gives the task hooks."""
        return self._device

    # ---- run dispatch: the device brackets around the task's own initialize --------------- #
    def run(self) -> None:
        self._device = resolve_device(self.device_choice)
        logger.info(f"{self.log_label} torch backend on {self._device}.")
        self._require_model()
        self.initialize()
        self._finalize_device()
        # The `@entrypoint` markers (inherited from the API base) ARE the dispatch table.
        run_entrypoint(self, self.task)

    def _finalize_device(self) -> None:
        """Place the task-built model and loss on the run device, after `initialize()` ran.

        The loss moves too, and it is not decoration: a balanced/weighted loss holds its class
        weights in a BUFFER (and a learned perceptual loss a whole network), and a CPU buffer
        against CUDA/MPS logits raises mid-batch rather than at setup.
        """
        if not isinstance(self.model, nn.Module):
            raise TypeError(
                f"{type(self).__name__}.model must be a torch.nn.Module, got {type(self.model).__name__}. "
                "Wire a `!lazy:` model (or any nn.Module) into the model slot."
            )
        self.model.to(self._device)
        loss = flow(self.loss_fn)
        if isinstance(loss, nn.Module):
            self.loss_fn = loss.to(self._device)

    # ---- the batch hooks (what a task genuinely decides about the loop) -------------------- #
    def _train_batch_loss(self, batch: Record) -> Tensor:
        """ONE training batch → the loss to backpropagate, updating the train metrics.

        The default is the read-forward-loss shape three of the four tasks share; a task whose
        model computes its loss inside ``forward`` (detection's loss dict) overrides this.
        """
        x = batch_tensor(batch, self.input_key, self._device)
        y = self._targets(batch)
        logits = cast(nn.Module, self.model)(x)
        loss: Tensor = cast(nn.Module, self.loss_fn)(logits, y)
        if isinstance(self.train_metrics, (Metric, MetricCollection)):
            self.train_metrics.update(logits.detach(), y)
        return loss

    def _eval_batch(self, batch: Record, collection: MetricsSpec) -> Optional[float]:
        """ONE eval batch → its loss (or ``None`` for a task whose eval pass has no loss).

        The default mirrors :meth:`_train_batch_loss`; detection overrides it (predictions
        against targets, no eval loss — the native contract returns predictions in eval mode).
        """
        x = batch_tensor(batch, self.input_key, self._device)
        y = self._targets(batch)
        logits = cast(nn.Module, self.model)(x)
        if isinstance(collection, (Metric, MetricCollection)):
            collection.update(logits, y)
        return float(cast(nn.Module, self.loss_fn)(logits, y))

    def _predict_batch(self, batch: Record) -> Any:
        """ONE predict batch → the task's prediction CONTRACT (via ``_prediction_output``)."""
        return self._prediction_output(cast(nn.Module, self.model)(batch_tensor(batch, self.input_key, self._device)))

    # ---- the loop --------------------------------------------------------------------------- #
    def _train_epoch(self, loader: DataLoader[Any], optimizer: Optimizer) -> float:
        """One training pass, returning the mean loss. The whole reason this base exists."""
        model = cast(nn.Module, self.model)
        model.train()
        total, batches = 0.0, 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = self._train_batch_loss(batch)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        return total / batches if batches else float("nan")

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader[Any], collection: MetricsSpec) -> Optional[float]:
        """One evaluation pass, returning the mean loss (or ``None`` when the task has none).

        ``no_grad`` because nothing here trains.
        """
        cast(nn.Module, self.model).eval()
        losses: List[float] = []
        for batch in loader:
            value = self._eval_batch(batch, collection)
            if value is not None:
                losses.append(value)
        return sum(losses) / len(losses) if losses else None

    def _epoch_metrics(self, collection: MetricsSpec, step: Optional[int] = None) -> Dict[str, float]:
        """Compute and RESET a split's metrics — the bookkeeping a ``Trainer`` hides.

        Only the SCALARS are RETURNED (``Tracker.log_metrics`` is a ``{name: float}`` channel);
        the non-scalars go out through the task's ``_report_nonscalars`` hook (matrices /
        per-class vectors → the trackers' text channel), after the task's ``_reportable``
        filter — identical reporting to the Lightning base, which matters because the two
        machinery bases exist to be compared.
        """
        if not isinstance(collection, (Metric, MetricCollection)):
            return {}
        computed = self._reportable(collection.compute())
        self._report_nonscalars(computed, step, self._live_trackers())
        values = scalar_metrics(computed)
        collection.reset()
        return values

    def _loaders(self) -> Dict[str, Optional[DataLoader[Any]]]:
        """The deferred loader slots, flowed with the datasets this run resolved."""
        return {
            "train": flow(self.train_loader, dataset=self.train_set) if self.train_set is not None else None,
            "val": flow(self.val_loader, dataset=self.val_set) if self.val_set is not None else None,
            "test": flow(self.test_loader, dataset=self.test_set) if self.test_set is not None else None,
        }

    # ---- the four entry points (markers inherited from the API base) ------------------------ #
    def fit(self) -> None:
        """Epochs, in the open: forward, backward, step, validate, log, checkpoint."""
        logger.info(f"{self.log_label} torch training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError(f"{type(self).__name__}.fit() requires a 'train_set'.")
        loaders = self._loaders()
        train_loader = cast(DataLoader[Any], loaders["train"])
        optimizer = cast(Optimizer, flow(self.optimizer, params=cast(nn.Module, self.model).parameters()))
        trackers = self._live_trackers()
        manifest = self._log_run_context(trackers)
        self._move_metrics()

        for epoch in range(self.max_epochs):
            logs: Dict[str, float] = {"train/loss": self._train_epoch(train_loader, optimizer)}
            logs.update(self._epoch_metrics(self.train_metrics, step=epoch))
            if loaders["val"] is not None:
                val_loss = self._eval_epoch(cast(DataLoader[Any], loaders["val"]), self.val_metrics)
                if val_loss is not None:
                    logs["val/loss"] = val_loss
                logs.update(self._epoch_metrics(self.val_metrics, step=epoch))
            logs = as_metric_names(self.metric_names)(logs)
            for tracker in trackers:
                tracker.log_metrics(logs, step=epoch)
            for key, value in logs.items():
                self._history.setdefault(key, []).append(value)
            logger.info(f"epoch {epoch + 1}/{self.max_epochs}: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items()))
            self._computed_metrics = dict(logs)

        logger.info("Training complete.")
        self._save_checkpoint()
        self._after_fit(self, self.train_set, self.val_set, self.test_set)
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        if loaders["test"] is not None:
            self._computed_metrics.update(self._score(cast(DataLoader[Any], loaders["test"])))
        # Uploads the log file, stamps the manifest's terminal status AND finalizes every
        # tracker. Not optional once a manifest exists: left unfinished it stays `running`, and
        # matrainer's atexit hook then calls a successful run `interrupted`.
        from matrainer.tracking.provenance import capture_log_file

        capture_log_file(trackers, manifest=manifest, status="success")

    def evaluate(self) -> None:
        """Score the VALIDATION split — the `evaluate` / `test` split is the only difference."""
        if self.val_set is None:
            raise ValueError(f"{type(self).__name__}.evaluate() requires a 'val_set'.")
        self._move_metrics()
        loader = cast(DataLoader[Any], self._loaders()["val"])
        self._computed_metrics = self._score(loader, split="val", metrics=self.val_metrics)
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    def test(self) -> None:
        if self.test_set is None:
            raise ValueError(f"{type(self).__name__}.test() requires a 'test_set'.")
        self._move_metrics()
        self._computed_metrics = self._score(cast(DataLoader[Any], self._loaders()["test"]))
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    def _score(self, loader: DataLoader[Any], split: str = "test", metrics: MetricsSpec = None) -> Dict[str, float]:
        """Evaluate over ``loader`` and report — shared by ``test``, ``evaluate`` and the
        post-fit pass."""
        trackers = self._live_trackers()
        collection = metrics if metrics is not None else self.test_metrics
        loss = self._eval_epoch(loader, collection)
        results: Dict[str, float] = {} if loss is None else {f"{split}/loss": loss}
        results.update(self._epoch_metrics(collection))
        results = as_metric_names(self.metric_names)(results)
        for tracker in trackers:
            tracker.log_metrics(results)
        logger.info(f"Evaluation complete: {results}")
        return results

    @torch.no_grad()
    def predict(self) -> None:
        """Stream per-record predictions to the sink, pairing each batch with its metadata."""
        logger.info(f"{self.log_label} torch predicting: {self.experiment_name} ({self.run_name})")
        if self.test_set is None:
            raise ValueError(f"{type(self).__name__}.predict() requires a 'test_set'.")
        cast(nn.Module, self.model).eval()
        self.predictions_sink = flow(self.predictions_sink)
        loader = cast(DataLoader[Any], self._loaders()["test"])
        for batch in loader:
            out = self._predict_batch(batch)
            if self.predictions_sink is not None:
                self._write_predictions(out, batch)
        if self.predictions_sink is not None and hasattr(self.predictions_sink, "close"):
            self.predictions_sink.close()
        logger.info("Prediction complete.")

    # ---- artifacts + tracking --------------------------------------------------------------- #
    def _save_checkpoint(self) -> None:
        """A plain ``state_dict`` — no framework format, because there is no framework."""
        if not self.checkpoint_dir:
            return
        directory = Path(self.checkpoint_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "model.pt"
        torch.save(cast(nn.Module, self.model).state_dict(), path)
        logger.info(f"Saved weights to {path}")

    def _log_run_context(self, trackers: List[Tracker]) -> Any:
        """Provenance through matrainer's plain FUNCTIONS — hookup style (1).

        There is no callback protocol to hang a ``TrackingCallback`` on, which is exactly the
        case matrainer's function-level API exists for. ``runnable=self`` rather than a
        hand-written params dict: matrainer owns the ONE hyperparameter derivation, so no two
        machinery bases can answer "what were the hyperparameters?" differently. Returns the
        run manifest so ``fit()`` can finish it.
        """
        if not trackers:
            return None
        from matrainer.tracking.provenance import log_run_context

        return log_run_context(trackers, {"device": str(self._device)}, runnable=self)

    def _move_metrics(self) -> None:
        """Metric state lives on the same device as the values it consumes."""
        for collection in (self.train_metrics, self.val_metrics, self.test_metrics):
            if isinstance(collection, (Metric, MetricCollection)):
                collection.to(self._device)

    # ---- @outputs --------------------------------------------------------------------------- #
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
        """Per-epoch metric history — what a ``Trainer``'s logger would have accumulated."""
        return {key: list(values) for key, values in self._history.items()}

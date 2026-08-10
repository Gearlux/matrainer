"""``HuggingFaceRunnable`` — the ``transformers.Trainer`` machinery base of the trainer API.

The member that writes least while swapping most (``docs/architecture.md`` record 9): the
engine owns somebody else's *whole training program* — ``Trainer`` runs the epoch, the
optimizer, the scheduler, gradient accumulation, mixed precision and distribution (via
Accelerate) — so what this base writes is the assembly around it: the deferred
``TrainingArguments`` / ``Trainer`` slots, the run-time argument FLOOR, the entry points with
the plain-function tracking hookup, and the ``log_history`` replay. Every method here was
measured near-identical across the four task projects' huggingface backends before the
consolidation; what differed per task is exactly the hooks below (``_to_hf_batch``, the
metrics hook, the loss seam, detection's own decode+torchmetrics scoring path).

Four engine facts the whole family encodes:

* **The transformers objects are DEFERRED SLOTS with STRING dotted targets** —
  ``LazyClass("transformers.TrainingArguments")`` / ``("transformers.Trainer")`` — so this
  module has no top-level ``import transformers`` (the methods that need it import it in
  their bodies) and a config reaches every knob they take (~100 on ``TrainingArguments``
  alone) without one of them becoming a matrainer param.
* **The derived knobs are a run-time FLOOR, never pinned on the slot** — the slot is a
  whole-value replacement, so pinning would mean ``training_arguments: {use_cpu: true}``
  silently dropped the epoch count AND ``remove_unused_columns=False``, whose absence
  surfaces as a ``KeyError`` three frames inside Trainer's own collator. A key the config
  pinned WINS; only the ones it did not mention are injected.
* **``report_to`` defaults to ``"none"`` ON PURPOSE** — Trainer auto-enables every tracker
  library it can import, so on a workspace with wandb AND mlflow installed an innocent run
  would silently start reporting to both. Tracking goes through the ``trackers`` slot, whose
  curves come from replaying ``Trainer``'s own ``state.log_history`` after the fit.
* **``evaluate()`` scores the VAL split** (``test()`` the held-out one, with
  ``metric_key_prefix="test"`` — left at HF's ``eval`` default the held-out scores would land
  in the columns a validation pass fills) — the same harmonization every other machinery
  base made.

The metric BUILDERS are deliberate no-ops here: three of the four tasks resolve metrics by
NAME (plain numpy callables — there is no class for a registry to tag), so there is no
torchmetrics collection to build. A task that scores with the fleet's torchmetrics anyway
(detection's decode + ``MeanAveragePrecision`` pass) restores the family builders — the same
arrangement the fastai base uses.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional, Sequence, Union, cast

import torch
from confluid import Lazy, LazyClass, configurable, flow, output
from loggair import get_logger
from recordstream import PredictionsSink, RecordSource, batch_tensor, collate_records
from recordstream.runnable import RunnableTask
from torch.utils.data import DataLoader

from matrainer.artifacts import write_metrics_sidecar
from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.torch.runnable import TorchTrainerRunnable
from matrainer.tracking.base import Tracker
from matrainer.tracking.provenance import capture_log_file, log_run_context

if TYPE_CHECKING:  # the deferred slots' real types, without the module-level import
    from transformers import Trainer, TrainingArguments

logger = get_logger(__name__)

#: What the ``callbacks`` slot holds. This is the ONE family whose element type stays open,
#: and the reason is measured rather than stylistic: the honest type is
#: ``transformers.TrainerCallback``, naming it would force the module-level
#: ``import transformers`` this module defers, and unlike the model slot there is no lighter
#: stand-in — a ``TrainerCallback`` is a plain object with ``on_*`` hooks that inherits
#: nothing importable from elsewhere. The type lives in the docstring instead.
HuggingFaceCallback = Any

__all__ = ["HuggingFaceCallback", "HuggingFaceRunnable"]


# `@configurable(validate=False)` for the same reason as the other bases: the mark makes
# confluid's body-slot scans see THIS layer's slots (`training_arguments`, `trainer`,
# `test_loader`, `callbacks`) in a concrete subclass's MRO. The concrete class's flat
# signature stays the validated config surface.
@configurable(validate=False)
class HuggingFaceRunnable(TorchTrainerRunnable):
    """A ``transformers.Trainer`` train/evaluate/test/predict runnable, task hooks left open.

    Args (on top of the family base's):
        batch_size: Rows per batch (``per_device_train_batch_size`` / ``_eval_batch_size``,
            and the record loader :meth:`predict` walks).
        num_workers: DataLoader workers (``dataloader_num_workers``).
        learning_rate: Peak LR. The default is transformers' own fine-tuning default — two
            orders below a from-scratch LR, because this family's whole point is starting
            from pretrained weights.
        weight_decay: ``TrainingArguments.weight_decay``.
        report_to: Trainer's own tracking integrations — ``"none"`` on purpose (module
            docstring).
        output_dir: Where Trainer writes its state. Defaults beside the metrics sidecar when
            one is configured, else ``./runs/hf_runs/<run_name>``.
        callbacks: ``transformers.TrainerCallback`` instances, handed to the ``Trainer``.
            Stored VERBATIM (entries may be deferred markers) and flowed where used.

    There is deliberately no ``train_metrics`` param anywhere in this family:
    ``compute_metrics`` is a ``Trainer``'s EVALUATION hook, so a train-split slot could only
    be honoured with a second forward pass over the whole train split at every epoch end — a
    cost every user would carry, where offering the slot and quietly never calling it is the
    exact failure per-split slots exist to remove.

    Optional OVERRIDE slots (each already defaulted; a same-named top-level YAML key replaces
    one via broadcasting): ``val_set``, ``training_arguments``, ``trainer``, ``test_loader``,
    ``predictions_sink``, ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        *,
        model: Optional[Lazy[Any]] = None,
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        task: RunnableTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 8,
        num_workers: int = 0,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.0,
        report_to: str = "none",
        output_dir: Optional[str] = None,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        callbacks: Optional[List[HuggingFaceCallback]] = None,
        experiment_name: str = "matrainer",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
        # `Any`: the metric VOCABULARY differs per task — name/callable lists in three tasks,
        # torchmetrics templates in detection — and the CONCRETE ctor declares the task's real
        # type; this keyword-only machinery signature is not a config surface.
        metrics: Any = None,
        val_metrics: Any = None,
        test_metrics: Any = None,
        metric_names: Optional[Lazy[MetricNames]] = None,
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
            train_metrics=None,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            metric_names=metric_names,
        )
        self.batch_size = batch_size
        # Stored rather than closed over: `_argument_floor()` reads them at RUN time, so a
        # post-construction `runnable.learning_rate = …` still reaches the Trainer.
        self.num_workers = int(num_workers)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.report_to = report_to
        self.output_dir = output_dir
        self.callbacks: Optional[List[HuggingFaceCallback]] = callbacks
        # The transformers objects are DEFERRED SLOTS with STRING dotted targets (module
        # docstring). `training_arguments` is EMPTY on purpose — every derived knob is a
        # run-time FLOOR in `_training_arguments()`, so a config override ADDS to them.
        self.training_arguments: Lazy["TrainingArguments"] = LazyClass("transformers.TrainingArguments")
        self.trainer: Lazy["Trainer"] = LazyClass("transformers.Trainer")
        # The record-dict loader `predict` walks: prediction pairs each output row with the
        # metadata that travelled beside the image, which `_to_hf_batch` has already dropped.
        self.test_loader: Lazy[DataLoader[Any]] = LazyClass(
            DataLoader,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_records,
        )
        self._computed_metrics: Dict[str, float] = {}

    # ---- the task hooks --------------------------------------------------------------------- #
    def _network(self) -> Any:
        """The live ``transformers`` model — the wrapper's, or a bare one wired directly."""
        return getattr(self.model, "model", self.model)

    def _to_hf_batch(self, records: Sequence[Any]) -> Dict[str, Any]:
        """One list of records → the mapping ``Trainer`` feeds the model — THE batch boundary.

        Handed to ``Trainer`` as its ``data_collator``, the same seat a ``collate_fn``
        occupies in the other torch machinery bases. The KEYS are transformers' contract, not
        ours: ``pixel_values`` is what an image model's ``forward`` takes and ``labels`` is
        what makes it return a loss. The default is the image-to-image pair; a task with a
        dtype rule (int64 ids, float multi-hot) or its own shaping (a checkpoint's image
        processor) overrides.
        """
        batch = collate_records(list(records))
        return {
            "pixel_values": batch_tensor(batch, self.input_key),
            "labels": batch_tensor(batch, self.target_key),
        }

    def _default_hf_metrics(self) -> Optional[List[Any]]:
        """The task's default metric NAMES, for a task whose hook resolves names.

        ``None`` (the default) leaves the sentinel unresolved. A task scoring with
        torchmetrics instead implements the family's ``_default_metrics``, which
        :meth:`_resolve_task_defaults` consults first.
        """
        return None

    def _metrics_hook(self, split: str) -> Optional[Any]:
        """Trainer's ``compute_metrics``, bound to ONE split's metric list — or ``None``.

        A ``Trainer`` holds a single hook, so the split is decided where the Trainer is
        BUILT — ``fit`` evaluates the validation split, ``test`` the held-out one — rather
        than inside the hook, which cannot tell which dataset it was called on. ``None``
        (the default) wires no hook: the Trainer then reports its loss only, which is what a
        task that scores OUTSIDE the Trainer (detection's decode + torchmetrics pass) wants.
        """
        return None

    def _trainer_factory(self) -> Optional[Any]:
        """A ``Trainer`` SUBCLASS to build when the ``trainer`` slot was not overridden.

        ``None`` (the default) keeps the plain ``transformers.Trainer``. A task whose model
        will not compute its own loss (Swin2SR raises on ``labels=``) returns a subclass
        overriding ``compute_loss`` — the seam, because the slot must stay replaceable by a
        config while the DEFAULT must be the subclass that actually trains.
        """
        return None

    # ---- sentinel resolution + the deliberate no-op builders -------------------------------- #
    def _resolve_task_defaults(self) -> None:
        """The family resolution first (loss + torchmetrics tasks), then the name-list hook."""
        super()._resolve_task_defaults()
        if self.metrics is None:
            self.metrics = self._default_hf_metrics()

    def _build_fit_metrics(self, include_train: bool = True) -> None:
        """Deliberate no-op: three of the four tasks resolve metric NAMES in their hook, so
        there is no torchmetrics collection to build (the fastai-base arrangement). A
        torchmetrics-scoring task restores ``TorchTrainerRunnable``'s builders."""

    def _build_eval_metrics(self) -> None:
        """Deliberate no-op — see :meth:`_build_fit_metrics`."""

    # ---- the Trainer assembly --------------------------------------------------------------- #
    def _argument_floor(self) -> Dict[str, Any]:
        """The derived ``TrainingArguments`` knobs — the flat config, in Trainer's words.

        A task extends via ``super()`` (detection adds its ``seed``); the floor semantics —
        a config-pinned key wins — are :meth:`_training_arguments`'s.
        """
        return {
            "output_dir": self._output_dir(),
            "num_train_epochs": self.max_epochs,
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dataloader_num_workers": self.num_workers,
            # A generic map-style source is not a `datasets.Dataset`, so Trainer must not
            # drop "unused" columns by inspecting the model signature — the collate already
            # produces exactly the keys the model takes.
            "remove_unused_columns": False,
            "report_to": self.report_to,
            "run_name": self.run_name,
            "disable_tqdm": True,
            "save_strategy": "no",
            "eval_strategy": "epoch" if self.val_set is not None else "no",
        }

    def _training_arguments(self) -> Any:
        """Flow the ``training_arguments`` slot with the FLOOR — a config-pinned key WINS."""
        floor = self._argument_floor()
        pinned = getattr(self.training_arguments, "kwargs", None) or {}
        return flow(self.training_arguments, **{k: v for k, v in floor.items() if k not in pinned})

    def _build_trainer(self, train: Any = None, evaluate: Any = None, split: str = "val") -> Any:
        """Flow the ``Trainer`` with everything only the RUN knows.

        The model, the split datasets, the batch boundary and this split's metric hook cannot
        be config values — they are derived from the data or built per entry point — so they
        are flow kwargs, and every other ``Trainer`` knob is whatever the config put on the
        slot. When the slot was NOT overridden and the task supplies a subclass
        (:meth:`_trainer_factory`), the subclass is what gets built.
        """
        kwargs: Dict[str, Any] = {
            "model": self._network(),
            "args": self._training_arguments(),
            "train_dataset": train,
            "eval_dataset": evaluate,
            "data_collator": self._to_hf_batch,
            "compute_metrics": self._metrics_hook(split),
            "callbacks": [flow(cb) for cb in (self.callbacks or [])] or None,
        }
        trainer = flow(self.trainer, **kwargs)
        factory = self._trainer_factory()
        if factory is not None:
            from transformers import Trainer as HFTrainer

            if type(trainer) is HFTrainer:
                trainer = factory(**kwargs)
        return trainer

    def _output_dir(self) -> str:
        """Where Trainer writes its state — beside the sidecar when one is configured.

        The last resort roots at ``runs/``, the ONE gitignored output directory every
        project carries (workspace ``AGENTS.md``, "Run Output Lands In A Gitignored
        ``runs/``"), so an unconfigured run does not scatter ``Trainer`` state into
        whatever directory it was launched from.
        """
        if self.output_dir:
            return str(Path(self.output_dir).expanduser())
        if self.metrics_sidecar_path:
            return str(Path(self.metrics_sidecar_path).expanduser().parent / self.run_name)
        return str(Path("./runs/hf_runs") / self.run_name)

    def _record_loader(self, source: Any) -> DataLoader[Any]:
        """A loader of COLLATED RECORD BATCHES — the shape :meth:`predict` needs."""
        return cast(DataLoader[Any], flow(self.test_loader, dataset=self._prepare(source)))

    # ---- the four entry points --------------------------------------------------------------- #
    def fit(self) -> None:
        logger.info(f"{self.log_label} huggingface training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError(f"{type(self).__name__}.fit() requires a 'train_set'.")
        # Warm every wired split IN THIS PROCESS: the Trainer's own DataLoader forks workers,
        # and a lazy source built in a forked child SIGSEGVs through _scproxy on macOS.
        # Idempotent over a split the task's initialize already prepared.
        self.train_set = self._prepare(self.train_set)
        self.val_set = self._prepare(self.val_set)
        self.test_set = self._prepare(self.test_set)
        # Provenance through matrainer's plain FUNCTIONS — `runnable=self` so the
        # hyperparameters and the resolved config come from the ONE derivation.
        trackers = self._live_trackers()
        manifest = log_run_context(trackers, runnable=self)
        trainer = self._build_trainer(train=self.train_set, evaluate=self.val_set)
        result = trainer.train()
        names = as_metric_names(self.metric_names)
        scored = {k: float(v) for k, v in result.metrics.items() if _is_number(v)}
        if self.val_set is not None:
            scored.update(self._fit_val_scores(trainer))
        # Idempotent over a task hook that already produced canonical `<split>/<name>` keys —
        # the vocabulary recognises the canonical shape whatever the source.
        self._computed_metrics = names(scored, source="huggingface")
        _log_trainer_history(trainer, trackers, names)
        for tracker in trackers:
            tracker.log_metrics(self._computed_metrics)
        logger.info(f"Training complete: {self._computed_metrics}")
        self._after_fit(trainer, self.train_set, self.val_set, self.test_set)
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        # Uploads the log file AND stamps the manifest's terminal status. Not optional once a
        # manifest exists: left unfinished it stays `running`, and matrainer's atexit hook
        # then calls a successful run `interrupted`.
        capture_log_file(trackers, manifest=manifest, status="success")

    def _fit_val_scores(self, trainer: Any) -> Dict[str, float]:
        """The post-fit validation scores merged into the fit's computed metrics.

        The engine's own ``evaluate()`` by default (raw HF names — translated with the rest);
        a task that scores outside the Trainer overrides with its own pass.
        """
        return {k: float(v) for k, v in trainer.evaluate().items() if _is_number(v)}

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
        """``Trainer.evaluate`` over one split, translated into the canonical vocabulary.

        ``metric_key_prefix="test"`` on the held-out split because HF's ``eval`` default
        would file its scores in the columns a validation pass fills; the val split keeps
        the ``eval`` prefix the vocabulary already translates to ``val/``.
        """
        prefix = "test" if split == "test" else "eval"
        results = self._build_trainer(evaluate=source, split=split).evaluate(metric_key_prefix=prefix)
        return as_metric_names(self.metric_names)(
            {k: float(v) for k, v in results.items() if _is_number(v)}, source="huggingface"
        )

    def predict(self) -> None:
        """Stream per-record predictions to the sink, pairing each row with its metadata.

        A plain ``torch.no_grad()`` forward rather than ``Trainer.predict``: that returns
        only the stacked logits, and the sink contract needs each row paired with the
        metadata from the record it came from. One walk over the record loader keeps the two
        in step by construction.
        """
        if self.test_set is None:
            raise ValueError(f"{type(self).__name__}.predict() requires a 'test_set'.")
        self.predictions_sink = flow(self.predictions_sink)
        network = self._network().eval()
        device = next(network.parameters()).device
        with torch.no_grad():
            for batch in self._record_loader(self.test_set):
                preds = self._prediction_output(self._predict_batch(network, batch, device))
                if self.predictions_sink is not None:
                    self._write_predictions(preds, batch)
        if self.predictions_sink is not None and hasattr(self.predictions_sink, "close"):
            cast(PredictionsSink, self.predictions_sink).close()
        logger.info("Prediction complete.")

    def _predict_batch(self, network: Any, batch: Any, device: Any) -> Any:
        """ONE record batch → the raw model output the prediction contract wraps.

        The default is the transformers image contract (``pixel_values=`` in, ``.logits``
        out); a task whose output is not logits (a reduced-resolution map to upsample, a
        restored image on a wrapper's own forward) overrides.
        """
        return network(pixel_values=batch_tensor(batch, self.input_key, device)).logits

    # ---- @outputs ----------------------------------------------------------------------------- #
    @property
    @output
    def computed_metrics(self) -> Dict[str, float]:
        """Final SCALAR metrics (post-run ``@output``). Raises before ``run()``."""
        if not self._computed_metrics:
            raise RuntimeError("computed_metrics is only available after run().")
        return dict(self._computed_metrics)


def _is_number(value: Any) -> bool:
    """Whether a ``Trainer`` metrics entry is a scalar (it also reports strings and counts)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _log_trainer_history(trainer: Any, trackers: Sequence[Tracker], names: MetricNames) -> None:
    """Replay ``Trainer``'s own log history onto the trackers, translated.

    ``transformers.Trainer`` owns the loop, so there is no epoch boundary to hook — but it
    records every log entry in ``state.log_history``, which is exactly what its own MLflow
    and W&B integrations consume. Replaying it after the fit gives the same curves without
    ``report_to``, which would silently start reporting to every tracker library that
    happens to be importable. Without the replay a tracked run shows one point where every
    other backend shows a curve (one project's backend shipped exactly that — a completely
    blank sweep row beside a log printing its scores).

    ``step`` is the ENTRY's global optimizer step, not the list index: HF logs training and
    evaluation entries at different cadences, so indexing would interleave two series onto
    one misleading axis. ``epoch``/``step`` are dropped from the VALUES — they are axes, not
    metrics, and a series of the x-value adds a column no reader wants.
    """
    if not trackers:
        return
    for entry in getattr(getattr(trainer, "state", None), "log_history", []) or []:
        values = names(
            {k: float(v) for k, v in entry.items() if _is_number(v) and k not in ("epoch", "step")},
            source="huggingface",
        )
        if not values:
            continue
        step = entry.get("step")
        for tracker in trackers:
            tracker.log_metrics(values, step=int(step) if _is_number(step) else None)

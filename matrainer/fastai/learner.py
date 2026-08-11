"""``FastaiRunnable`` — the fastai machinery base of the trainer API.

The ``Learner`` member of the family (``docs/architecture.md`` record 9): fastai runs on torch,
so the task hooks a consumer implements for the torch family serve here unchanged — what this
base swaps is WHO OWNS THE EPOCH (``Learner.fit_one_cycle`` instead of a ``pl.Trainer`` or the
open-coded loop), and with it the three things measured near-identical across the task projects'
hand-written fastai backends: the ``Learner``/``DataLoaders`` assembly, the ``Recorder``
read-back (fastai reports through its Recorder and nothing else sees those rows), and the
positional batch adaptation (``_to_tuple`` — fastai unpacks a batch positionally, so the batch
must BE a tuple rather than the record dict the other machinery bases read by key).

Three fastai facts this base encodes so no consumer re-learns them:

* ``from fastai.callback import schedule`` is load-bearing — ``Learner.fit_one_cycle`` is a
  ``@patch`` installed as that module's IMPORT SIDE EFFECT, and fastai's ``@patch`` returns
  ``None``, so importing the function by name binds None and ``learn.fit_one_cycle`` then falls
  through fastcore's attribute delegation to the MODEL. Imported ONCE, here.
* metrics and optimizers are NAMES, not classes — fastai ships plain functions and factories,
  so there is nothing for a ``!class:`` tag to resolve; :meth:`_resolve_metric` /
  :func:`as_opt_func` resolve strings, with a per-task alias table
  (:meth:`_metric_aliases`) tried FIRST because an alias exists precisely where the upstream
  name would fail or resolve to the wrong quantity.
* a ``Learner`` ALWAYS validates (no "skip the val loop" mode) and, with
  ``Recorder.train_metrics`` on, reports ONE metric list on both passes — so the fit path
  combines ``train_metrics``/``val_metrics`` by UNION where the test path narrows by merge
  (:meth:`_learner_metrics`), and a run with no ``val_set`` reuses the training data with a
  warning that the metrics then measure fit, not generalization.

What a Learner implies, and this base deliberately does NOT have: no framework loggers
(tracking goes through ``trackers:`` via matrainer's plain functions — the same hookup as the
owned loop), and the batches are built on CPU (:attr:`device` is the CPU device the task hooks
read; fastai owns placement, exactly as a ``pl.Trainer`` does).
"""

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
from confluid import Lazy, LazyClass, configurable, flow
from fastai import metrics as fastai_metrics
from fastai import optimizer as fastai_optimizer

# `Learner.fit_one_cycle` is a `@patch` installed as this module's import SIDE EFFECT — see the
# module docstring. Do not "tidy" it away.
from fastai.callback import schedule  # noqa: F401
from fastai.callback.core import Callback as FastaiCallback
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from loggair import get_logger
from recordstream import RecordSource, batch_tensor, collate_records
from recordstream.items import Record
from recordstream.loaders import loader_slots
from recordstream.runnable import RunnableTask, run_entrypoint
from torch.utils.data import DataLoader

from matrainer.artifacts import write_metrics_sidecar
from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.metric_slots import merge_metric_specs, union_metric_specs
from matrainer.torch.runnable import TorchTrainerRunnable
from matrainer.tracking.base import Tracker

logger = get_logger(__name__)

#: A fastai metric: a live callable/metric object, or the NAME of one in ``fastai.metrics``
#: (or a task's configured alias). A NAME because fastai's metrics are plain functions and
#: factories — there is no class for a ``!class:`` tag to resolve or for confluid's registry
#: to tag.
FastaiMetric = Union[str, Callable[..., Any]]

#: A fastai optimizer: a live ``OptimWrapper`` factory, or the NAME of one in
#: ``fastai.optimizer`` (``"Adam"``, ``"SGD"``, ``"ranger"``, …).
FastaiOptimizer = Union[str, Callable[..., Any]]

#: What the ``loss`` slot admits: any callable over ``(predictions, targets)``. Deliberately
#: NOT ``torch.nn.Module`` — fastai's ``*Flat`` family are ``fastai.losses.BaseLoss`` OBJECTS
#: holding the real loss in ``.func``, not modules — while a plain ``torch.nn`` loss satisfies
#: the same call and a ``Learner`` takes either.
FastaiLoss = Callable[..., Any]

__all__ = [
    "FastaiLoss",
    "FastaiMetric",
    "FastaiOptimizer",
    "FastaiRunnable",
    "as_opt_func",
    "log_recorded_epochs",
    "metric_columns",
    "named_values",
    "recorded_metrics",
]


# `@configurable(validate=False)` for the same reason as the other bases: the mark makes
# confluid's body-slot scans see THIS layer's slots (the loader triple, `dataloaders`,
# `learner`) in a concrete subclass's MRO. The concrete class's flat signature stays the
# validated config surface.
@configurable(validate=False)
class FastaiRunnable(TorchTrainerRunnable):
    """A fastai ``Learner`` train/evaluate/test/predict runnable, task hooks left open.

    Args (on top of the family base's):
        batch_size: Rows per batch, baked into the three deferred loader slots.
        num_workers: Loader workers (``persistent_workers`` derives from it).
        learning_rate: Peak LR for ``fit_one_cycle``, or the flat LR for ``fit``. Coerced
            EXPLICITLY to float: YAML 1.1 parses ``1e-4`` (no decimal point) as a STRING, and
            fastai's scheduler then dies in a numpy divide three frames from the config line
            (hit for real).
        weight_decay: fastai's ``wd``; ``None`` is the ``Learner`` default.
        one_cycle: Use ``fit_one_cycle`` (fastai's headline schedule) instead of a flat ``fit``.
        opt_func: fastai optimizer — a callable or a name in ``fastai.optimizer``.
        loss: The loss callable — stored on the family's ``loss_fn`` slot (the :attr:`loss`
            property maps the fastai spelling onto it). ``None`` resolves to the task's
            :meth:`_default_fastai_loss`.
        callbacks: fastai callbacks, stored verbatim (entries may be deferred markers).
        collate_fn: The loaders' batch-shape choice. ``None`` bakes :meth:`_to_tuple` (the
            positional ``(x, y)`` tuple a Learner unpacks); a loss-inside-forward task passes
            its own packing instead.

    Optional OVERRIDE slots (each already defaulted; a same-named top-level YAML key replaces
    one via broadcasting): ``val_set``, ``train/val/test_loader``, ``dataloaders`` (fastai's
    ``DataLoaders`` around them — its ``path``), ``learner`` (every ``Learner`` knob this class
    does not restate: ``splitter``, ``model_dir``, ``moms``, ``wd_bn_bias``, ``train_bn``,
    ``default_cbs``), ``predictions_sink``, ``metrics_sidecar_path``.
    """

    def __init__(
        self,
        *,
        model: Optional[Lazy[nn.Module]] = None,
        train_set: Optional[Lazy[RecordSource]] = None,
        test_set: Optional[Lazy[RecordSource]] = None,
        task: RunnableTask = "fit",
        max_epochs: int = 10,
        batch_size: int = 16,
        num_workers: int = 0,
        trackers: Optional[Union[Lazy[Tracker], Iterable[Lazy[Tracker]]]] = None,
        experiment_name: str = "matrainer",
        run_name: Optional[str] = None,
        input_key: str = "image",
        target_key: str = "target",
        metrics: Any = None,
        train_metrics: Any = None,
        val_metrics: Any = None,
        test_metrics: Any = None,
        metric_names: Optional[Lazy[MetricNames]] = None,
        learning_rate: float = 1e-3,
        weight_decay: Optional[float] = None,
        one_cycle: bool = True,
        opt_func: FastaiOptimizer = "Adam",
        loss: Optional[Lazy[FastaiLoss]] = None,
        callbacks: Optional[List[Lazy[FastaiCallback]]] = None,
        collate_fn: Optional[Callable[[List[Record]], Any]] = None,
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
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay) if weight_decay is not None else None
        self.one_cycle = one_cycle
        self.opt_func: FastaiOptimizer = opt_func
        if loss is not None:
            self.loss = loss
        # Stored VERBATIM (lazy-init): entries may be deferred `Fluid` markers, flowed where used.
        self.callbacks: Optional[List[Lazy[FastaiCallback]]] = callbacks
        # The cast: `_to_tuple` returns fastai's positional tuple rather than a Record —
        # deliberate, the whole point of this collate — while `loader_slots` types the
        # record-shaped default.
        slots = loader_slots(
            batch_size, num_workers, collate_fn=cast(Callable[[List[Record]], Any], collate_fn or self._to_tuple)
        )
        self.train_loader: Lazy[DataLoader[Any]] = slots.train
        self.val_loader: Lazy[DataLoader[Any]] = slots.val
        self.test_loader: Lazy[DataLoader[Any]] = slots.test
        # fastai's own two objects, deferred for the same reason the loaders are: everything this
        # base CONSTRUCTS should be reachable from config, or every knob they take and this class
        # does not restate — a `DataLoaders` path, a `Learner`'s `moms` / `splitter` /
        # `model_dir` — is unreachable from YAML. `DataLoaders(*loaders, …)` takes its loaders
        # POSITIONALLY, so `_dataloaders()` supplies them as `flow()`'s positional runtime args.
        self.dataloaders: Lazy[DataLoaders] = LazyClass(DataLoaders)
        self.learner: Lazy[Learner] = LazyClass(Learner, lr=self.learning_rate, wd=self.weight_decay)
        self._computed_metrics: Dict[str, float] = {}
        self._learner: Optional[Learner] = None
        self._device: torch.device = torch.device("cpu")

    # ---- the fastai spelling of the family's loss slot ------------------------------------- #
    @property
    def loss(self) -> Optional[Lazy[FastaiLoss]]:
        """The fastai spelling of the family's ``loss_fn`` slot — ONE slot, two names.

        fastai configs and the fleet's tests say ``loss:`` (a ``Learner``'s own vocabulary,
        and the slot admits ``BaseLoss`` callables that are not modules); the task hooks the
        family base declares read ``loss_fn``. A property pair keeps them the same value, so
        a hook that inspects ``self.loss_fn`` (a weight-capability probe) sees what a config
        wired as ``loss:``.
        """
        return cast(Optional[Lazy[FastaiLoss]], self.loss_fn)

    @loss.setter
    def loss(self, value: Optional[Lazy[FastaiLoss]]) -> None:
        self.loss_fn = cast(Any, value)

    @property
    def device(self) -> torch.device:  # type: ignore[override]  # narrows the family declaration
        """The device the task hooks build batches on — CPU, because fastai owns placement.

        A collate runs in a DataLoader worker before any engine placement happens, so the
        shared ``_targets`` hook must produce CPU tensors here; fastai's ``TrainEvalCallback``
        moves the model (and its batch handling the batches) exactly as a ``pl.Trainer`` does.
        """
        return self._device

    # ---- run dispatch ----------------------------------------------------------------------- #
    def run(self) -> None:
        logger.info(f"{self.log_label} fastai backend.")
        self._require_model()
        self.initialize()
        # The `@entrypoint` markers (inherited from the API base) ARE the dispatch table.
        run_entrypoint(self, self.task)

    # ---- the fastai shape of the family's sentinels + builders ------------------------------ #
    def _default_fastai_loss(self) -> Any:
        """The loss ``loss: null`` resolves to — **fastai hook**, defaulting to the task's own.

        The task hooks' ``_default_loss`` is a plain ``torch.nn`` loss, which a ``Learner``
        consumes as-is; a task whose fastai spelling differs (a ``*Flat`` wrapper with a
        corrected ``axis``) overrides THIS hook, leaving the family one to the other engines.
        """
        return self._default_loss()

    def _default_fastai_metrics(self) -> Any:
        """The metric list ``metrics: null`` resolves to — **fastai hook**.

        Defaults to the family hook, which suits a task whose fastai backend scores through
        the SAME vocabulary (a torchmetrics collection fed from a callback). A task whose
        fastai metrics are NAMES overrides this with its name list — the family default is
        torchmetrics TEMPLATES, which a ``Learner`` cannot consume.
        """
        return self._default_metrics()

    def _metric_aliases(self) -> Dict[str, Callable[[], Any]]:
        """Names the TASK configures rather than resolves from ``fastai.metrics`` — **hook**.

        Zero-arg thunks (close over the task's own knobs) so nothing is built until a config
        asks for it. An alias exists precisely where the upstream lookup would fail or return
        the wrong quantity, which is why :meth:`_resolve_metric` tries these FIRST.
        """
        return {}

    def _metric_vocabulary_hint(self) -> str:
        """One sentence naming the task's usable ``fastai.metrics`` names — **hook**, for the
        unknown-metric error."""
        return ""

    def _resolve_task_defaults(self) -> None:
        """Fill the ``loss`` / ``metrics`` sentinels with the FASTAI hooks. Idempotent."""
        if self.loss_fn is None:
            self.loss_fn = self._default_fastai_loss()
            if self.loss_fn is not None:
                logger.debug(f"loss defaulted to {type(self.loss_fn).__name__}.")
        if self.metrics is None:
            resolved = self._default_fastai_metrics()
            self.metrics = list(resolved) if isinstance(resolved, (list, tuple)) else resolved

    def _build_fit_metrics(self, include_train: bool = True) -> None:
        """Deliberately a NO-OP: fastai metrics are resolved at Learner assembly, per split.

        The family builders turn specs into prefixed torchmetrics collections, which a
        ``Learner`` cannot consume — the slots here stay the NAME lists a config wired, and
        :meth:`_learner_metrics` combines them when a Learner is built. A task that scores
        through torchmetrics anyway (a loss-inside-forward contract feeding a collection from
        a callback) restores the family builders in its concrete class.
        """

    def _build_eval_metrics(self) -> None:
        """Deliberately a NO-OP — see :meth:`_build_fit_metrics`."""

    # ---- the batch boundary ----------------------------------------------------------------- #
    def _to_tuple(self, records: Sequence[Any], with_targets: bool = True) -> Tuple[torch.Tensor, ...]:
        """One list of records → the ``(x, y)`` tuple fastai consumes.

        The fastai twin of the other bases' by-key batch reads: fastai unpacks a batch
        POSITIONALLY, so the batch must BE a tuple. The target half is the task's own
        ``_targets`` hook — the same implementation the Lightning and loop bases call, built
        on CPU (see :attr:`device`).
        """
        batch = collate_records(list(records))
        x = batch_tensor(batch, self.input_key)
        if not with_targets:
            return (x,)
        return x, self._targets(batch)

    def _dataloaders(self) -> DataLoaders:
        """fastai's ``DataLoaders`` — a train loader and a validation loader, always both.

        ``Learner`` validates at the end of every epoch and its ``Recorder`` reports from that
        pass, so unlike the other engines it has no "no validation" mode. When no ``val_set``
        is wired the training data is reused, which keeps a run working but makes the reported
        metrics training metrics — worth saying out loud, hence the warning.

        Every split is WARMED here (`_prepare`, idempotent), and for this base that is not
        optional: ``import fastai`` sets the multiprocessing start method to ``fork``, so a
        ``DataLoader`` worker that is the FIRST to touch a lazy source builds it inside a
        forked child — reaching ``_scproxy`` / CoreFoundation on macOS, a SIGSEGV with no
        Python traceback. Reading one record in the PARENT leaves the child nothing to build.
        """
        self.train_set = self._prepare(self.train_set)
        train = flow(self.train_loader, dataset=self.train_set)
        if self.val_set is not None:
            self.val_set = self._prepare(self.val_set)
            valid = flow(self.val_loader, dataset=self.val_set)
            return self._place(cast(DataLoaders, flow(self.dataloaders, train, valid)))
        logger.warning(
            "No 'val_set' wired: fastai's Learner always validates, so the TRAINING data is being "
            "used for validation and the reported metrics measure fit, not generalization. Wire a "
            "'val_set' to get a real held-out score."
        )
        return self._place(
            cast(DataLoaders, flow(self.dataloaders, train, flow(self.val_loader, dataset=self.train_set)))
        )

    def _place(self, dataloaders: DataLoaders) -> DataLoaders:
        """Post-assembly hook on the built ``DataLoaders`` — the default touches nothing.

        Exists for a task that must carry a device choice onto the loaders (fastai's
        ``TrainEvalCallback`` falls back to ``default_device()`` over plain torch loaders,
        which a CPU-pinned detection run must override).
        """
        return dataloaders

    # ---- the learner ------------------------------------------------------------------------- #
    def _learner_model(self) -> nn.Module:
        """The module handed to the ``Learner`` — **hook**; the default is the model itself.

        A task whose model does not meet ``loss_func(model(*xb), *yb)`` (a loss-inside-forward
        contract) overrides this with its adapter.
        """
        if not isinstance(self.model, nn.Module):
            raise TypeError(
                f"{type(self).__name__}.model must be a torch.nn.Module, got {type(self.model).__name__}. "
                "Wire a `!lazy:` model (or any nn.Module) into the model slot."
            )
        return self.model

    def _learner_loss(self) -> Any:
        """The ``loss_func`` handed to the ``Learner`` — **hook**; default: the flowed slot."""
        return flow(self.loss_fn)

    def _learner_metrics(self, split: str) -> Sequence[FastaiMetric]:
        """The ONE metric list this Learner gets — and why the two paths combine differently.

        A ``Learner`` holds a single ``metrics`` list and, with ``train_metrics`` on, reports
        it on BOTH passes: ``Recorder._train_mets`` and ``_valid_mets`` are the same
        ``self.metrics`` with different prefixes. So during a fit, ``train_metrics:`` and
        ``val_metrics:`` cannot be narrowed apart and the honest combination is the UNION —
        a metric computed on a split that did not ask for it is noise a reader skips, where
        dropping it would be a wired metric that silently reports nothing. An eval path has no
        such constraint — it builds its own Learner and runs one pass — so it keeps the
        narrowing merge and that split's entry shadows the shared one.
        """
        if split == "test" or (split == "val" and self.task != "fit"):
            return merge_metric_specs(self.metrics, getattr(self, f"{split}_metrics")) or []
        return union_metric_specs(self.metrics, self.train_metrics, self.val_metrics) or []

    def _learner_metric_objects(self, split: str) -> List[Any]:
        """The resolved metric OBJECTS for this Learner — **hook** past the resolution."""
        return [self._resolve_metric(m) for m in self._learner_metrics(split)]

    def _learner_callbacks(self, split: str) -> List[Any]:
        """The callbacks handed to the ``Learner`` — **hook**; default: the wired ones."""
        return self._callbacks()

    def _build_learner(self, dataloaders: DataLoaders, split: str = "val") -> Learner:
        """Assemble the ``Learner``: model, loss, metrics, callbacks, artifact path.

        Everything passed here is something only the RUN knows — the data, the flowed loss,
        the resolved optimizer function, this split's metrics, the callbacks, the artifact
        path. The ergonomic knobs (``lr`` / ``wd``) are already baked into the deferred
        ``learner`` slot, and every knob neither names is whatever the config put there.
        """
        path = self._artifact_dir()
        learner = cast(
            Learner,
            flow(
                self.learner,
                dls=dataloaders,
                model=self._learner_model(),
                loss_func=self._learner_loss(),
                opt_func=as_opt_func(self.opt_func),
                metrics=self._learner_metric_objects(split),
                cbs=self._learner_callbacks(split),
                path=Path(path) if path else None,
            ),
        )
        # A `Recorder` computes its metrics on the VALIDATION pass unless told otherwise, and
        # `train_metrics=True` makes it accumulate the SAME list over the TRAINING batches as
        # they go by — no second pass over the data, and the numbers describe the network as it
        # TRAINED (a second eval-mode pass would cost a pass AND measure a different network:
        # dropout, batch norm). The columns become `train_loss, train_<m>…, valid_loss,
        # valid_<m>…`, exactly the shape `matrainer.metric_names`' fastai vocabulary reads.
        #
        # Set on the FIT path only: `Learner.validate()` runs no training pass, so asking for
        # train columns there would declare names no row ever fills.
        if split == "val" and self.task == "fit":
            learner.recorder.train_metrics = True
        self._learner = learner
        return learner

    def _resolve_metric(self, value: FastaiMetric) -> Any:
        """A fastai metric from a callable, a CONFIGURED ALIAS, or a NAME in ``fastai.metrics``.

        The string form is not sugar: fastai's metrics are plain FUNCTIONS and factories, so
        none is uniform enough for a ``!class:`` tag to name or for confluid's registry to tag.
        A name is the only thing a YAML config can write, so this resolves it and says which
        names exist when it cannot. The task's aliases come FIRST — an alias exists precisely
        where the upstream lookup would fail or resolve to the wrong quantity.
        """
        built = flow(value)
        if not isinstance(built, str):
            return built
        aliases = self._metric_aliases()
        if built in aliases:
            return aliases[built]()
        metric = getattr(fastai_metrics, built, None)
        if metric is None:
            hint = self._metric_vocabulary_hint()
            alias_note = f" Plus this task's configured aliases: {', '.join(sorted(aliases))}." if aliases else ""
            raise ValueError(f"Unknown fastai metric {built!r}. Names come from `fastai.metrics`. {hint}{alias_note}")
        return metric

    def _callbacks(self) -> List[Any]:
        """The wired fastai callbacks, with any file destination's parent directory created.

        fastai's ``CSVLogger`` opens ``learn.path/fname`` without creating it, so a perfectly
        reasonable ``./fastai_logs/run.csv`` would die *after* the data walk and the model
        build.
        """
        callbacks = [flow(cb) for cb in (self.callbacks or [])]
        base = Path(self._artifact_dir() or ".")
        for callback in callbacks:
            fname = getattr(callback, "fname", None)
            if isinstance(fname, (str, Path)):
                (base / fname).expanduser().parent.mkdir(parents=True, exist_ok=True)
        return callbacks

    def _artifact_dir(self) -> Optional[str]:
        """Where run artifacts land. fastai has no checkpoint-callback convention to read, so
        this follows the sidecar's directory when one is configured."""
        if self.metrics_sidecar_path:
            return str(Path(self.metrics_sidecar_path).expanduser().parent)
        return None

    # ---- the four entry points (markers inherited from the API base) ------------------------ #
    def fit(self) -> None:
        """One ``Learner`` fit: provenance in, one-cycle (or flat) epochs, Recorder read-back."""
        logger.info(f"{self.log_label} fastai training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError(f"{type(self).__name__}.fit() requires a 'train_set'.")
        # Provenance through matrainer's plain FUNCTIONS — `runnable=self` so the
        # hyperparameters and the resolved config come from matrainer's ONE derivation.
        from matrainer.tracking.provenance import capture_log_file, log_run_context

        trackers = self._live_trackers()
        manifest = log_run_context(trackers, runnable=self)
        learner = self._build_learner(self._dataloaders(), "val")
        if self.one_cycle:
            learner.fit_one_cycle(self.max_epochs, lr_max=self.learning_rate)
        else:
            learner.fit(self.max_epochs, lr=self.learning_rate)
        names = as_metric_names(self.metric_names)
        self._computed_metrics = self._fit_results(learner, names)
        self._log_epoch_history(learner, trackers, names)
        logger.info(f"Training complete: {self._computed_metrics}")
        self._after_fit(learner, self.train_set, self.val_set, self.test_set)
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)
        # Uploads the log file, stamps the manifest's terminal status AND finalizes every
        # tracker — left unfinished a manifest stays `running`, and matrainer's atexit hook
        # then calls a successful run `interrupted`.
        capture_log_file(trackers, manifest=manifest, status="success")

    def _fit_results(self, learner: Learner, names: MetricNames) -> Dict[str, float]:
        """What a finished fit RECORDS — **hook**; default: the Recorder's last epoch."""
        return recorded_metrics(learner, names)

    def _log_epoch_history(self, learner: Learner, trackers: Sequence[Tracker], names: MetricNames) -> None:
        """Forward the run's per-epoch history to the trackers — **hook**; default: every
        Recorder row (fastai reports through its Recorder and nothing else sees those rows,
        so without this a tracked run shows one point where every other engine shows a curve).
        """
        log_recorded_epochs(learner, trackers, names)

    def evaluate(self) -> None:
        """Score the VALIDATION split — the ``evaluate`` / ``test`` split is the only
        difference, matching every other machinery base."""
        self._validate_split("val")

    def test(self) -> None:
        self._validate_split("test")

    def _validate_split(self, split: str) -> None:
        """One ``Learner.validate()`` pass over the named split, read back and recorded."""
        dataset = self.val_set if split == "val" else self.test_set
        if dataset is None:
            verb = "evaluate" if split == "val" else "test"
            raise ValueError(f"{type(self).__name__}.{verb}() requires a '{split}_set'.")
        dataset = self._prepare(dataset)  # the fork-safety warm — see `_dataloaders`
        setattr(self, f"{split}_set", dataset)
        loader = flow(self.val_loader if split == "val" else self.test_loader, dataset=dataset)
        learner = self._build_learner(self._place(cast(DataLoaders, flow(self.dataloaders, loader, loader))), split)
        values = learner.validate()
        self._computed_metrics = self._validate_results(learner, values, split)
        logger.info(f"Evaluation complete: {self._computed_metrics}")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    def _validate_results(self, learner: Learner, values: Sequence[Any], split: str) -> Dict[str, float]:
        """What a validate pass RECORDS — **hook**; default: the Recorder row, translated.

        ``validate()`` fills the VALIDATION columns whichever split it walked, so the row is
        read as fastai spells it and lands under ``val/`` — the same names a fit's last epoch
        reports.
        """
        return named_values(learner, values, as_metric_names(self.metric_names))

    @torch.no_grad()
    def predict(self) -> None:
        """Stream per-record predictions to the sink, pairing each batch with its metadata.

        A plain forward over a RECORD-dict loader rather than ``Learner.get_preds``:
        ``get_preds`` returns only the stacked predictions, and the sink contract needs each
        row paired with the metadata the positional collate has already dropped. Same
        ``test_loader`` slot as :meth:`test`, flowed with a runtime ``collate_fn`` override
        back to the record shape — a runtime kwarg wins over the stored one, so every other
        knob the operator set still applies.
        """
        logger.info(f"{self.log_label} fastai predicting: {self.experiment_name} ({self.run_name})")
        if self.test_set is None:
            raise ValueError(f"{type(self).__name__}.predict() requires a 'test_set'.")
        model = self._learner_model().eval()
        device = next(model.parameters()).device
        self.predictions_sink = flow(self.predictions_sink)
        self.test_set = self._prepare(self.test_set)  # the fork-safety warm — see `_dataloaders`
        records = flow(self.test_loader, dataset=self.test_set, collate_fn=self._record_collate())
        for batch in records:
            out = self._predict_batch(batch, device)
            if self.predictions_sink is not None:
                self._write_predictions(out, batch)
        if self.predictions_sink is not None and hasattr(self.predictions_sink, "close"):
            self.predictions_sink.close()
        logger.info("Prediction complete.")

    def _record_collate(self) -> Callable[[List[Record]], Any]:
        """The record-shaped collate ``predict`` swaps in — **hook** for a non-stacking task."""
        return collate_records

    def _predict_batch(self, batch: Record, device: torch.device) -> Any:
        """ONE predict batch → the task's prediction CONTRACT (via ``_prediction_output``)."""
        return self._prediction_output(cast(nn.Module, self.model)(batch_tensor(batch, self.input_key, device)))

    # ---- @outputs --------------------------------------------------------------------------- #
    @property
    def computed_metrics(self) -> Dict[str, float]:
        """Final SCALAR metrics (post-run ``@output``). Raises before ``run()``."""
        if not self._computed_metrics:
            raise RuntimeError("computed_metrics is only available after run().")
        return dict(self._computed_metrics)


def as_opt_func(value: FastaiOptimizer) -> Any:
    """A fastai optimizer function from a callable or its name (``"Adam"``, ``"SGD"``, …)."""
    built = flow(value)
    if not isinstance(built, str):
        return built
    opt = getattr(fastai_optimizer, built, None)
    if opt is None:
        raise ValueError(
            f"Unknown fastai optimizer {built!r}. Names come from `fastai.optimizer` — e.g. Adam, "
            f"SGD, RMSProp, RAdam, QHAdam, Lamb, ranger."
        )
    return opt


def metric_columns(learner: Learner) -> List[str]:
    """The Recorder's column names, minus its two bookkeeping columns.

    fastai names the columns ``['epoch', 'train_loss', 'valid_loss', <metrics…>, 'time']``
    while ``recorder.values`` holds only the numeric middle, so the two ends have to come off
    before the names line up with the values.
    """
    names = list(getattr(learner.recorder, "metric_names", []) or [])
    return [name for name in names if name not in ("epoch", "time")]


def named_values(learner: Learner, values: Sequence[Any], names: MetricNames) -> Dict[str, float]:
    """Pair a Recorder row with its column names, tolerating a shorter row.

    ``Learner.validate`` returns one fewer value than a training epoch does (there is no
    ``train_loss``), so the columns are matched from the END — where ``valid_loss`` and the
    metrics line up — rather than from the front.

    The row is then translated out of fastai's vocabulary (``names``), which is what makes
    this run's record comparable with another engine's. Two things only a fastai reader knows
    and the translation encodes: ``valid_loss`` is the validation loss, and an UNPREFIXED
    column is a VALIDATION metric — the Recorder computes its metrics on the validation pass
    and prefixes none of them, where every other engine's bare key is a training one. (An
    unprefixed column only appears with ``Recorder.train_metrics`` off — the ``validate()``
    path; a fit turns it on and every column arrives ``train_``/``valid_``-prefixed, which the
    same vocabulary also reads.)
    """
    columns = metric_columns(learner)
    if len(columns) > len(values):
        columns = columns[len(columns) - len(values) :]
    return names({name: float(value) for name, value in zip(columns, values)}, source="fastai")


def recorded_metrics(learner: Learner, names: MetricNames) -> Dict[str, float]:
    """The last epoch's metrics as ``{name: value}``, or ``{}`` if nothing was recorded."""
    values = list(getattr(learner.recorder, "values", []) or [])
    return named_values(learner, values[-1], names) if values else {}


def log_recorded_epochs(learner: Learner, trackers: Sequence[Tracker], names: MetricNames) -> None:
    """Forward EVERY recorded epoch to the trackers, not just the final one.

    fastai's ``Recorder`` already holds the whole history (``recorder.values`` is one row per
    epoch), so replaying it after the fit gives the same curve a per-epoch callback would,
    without inserting a matrainer callback into fastai's callback chain — one fewer thing that
    can reorder or swallow. :func:`recorded_metrics` reads the last row of this same list, so
    the two cannot disagree about what the run scored.
    """
    if not trackers:
        return
    for epoch, row in enumerate(list(getattr(learner.recorder, "values", []) or [])):
        metrics = named_values(learner, row, names)
        for tracker in trackers:
            tracker.log_metrics(metrics, step=epoch)

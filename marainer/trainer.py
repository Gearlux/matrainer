import json
import shutil
from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Fluid, LazyClass, configurable, flow
from logflow import get_logger
from pytorch_lightning.loggers import Logger
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MetricCollection

from marainer.run_names import generate_run_name
from marainer.runner import TorchRunner

logger = get_logger(__name__)


def build_metric_collection(metrics: Any, **flow_dims: Any) -> MetricCollection:
    """Build a ``MetricCollection`` from a dict of single-metric templates OR a ready-made collection.

    Task subclasses configure their ``metrics`` slot as a dict ``{name: LazyClass(Metric, ...)}`` of
    single-metric templates (so a dimension like ``num_classes`` can be injected late) OR as a
    fully-built ``MetricCollection``. This normalizes both into one ``MetricCollection`` the caller
    then clones per split (``train/`` · ``val/`` · ``test/``):

    * **dict** — each value is flowed with ``flow_dims`` (a confluid ``LazyClass`` / ``Fluid`` template
      builds with the dims, e.g. ``num_classes``; an already-live ``Metric`` passes through ``flow``
      unchanged) and the results are wrapped in a ``MetricCollection``.
    * **MetricCollection** — returned as-is (re-wrapping its live metrics, or "flowing" already-built
      ones, would be wrong).

    Shared by the classification trainer and evaluator (and available to any task trainer) so the
    dict-vs-collection handling lives in ONE place.
    """
    if isinstance(metrics, MetricCollection):
        return metrics
    flowed = {k: flow(v, **flow_dims) for k, v in metrics.items()}
    return MetricCollection(flowed)


def ensure_sample_dataset(source: Any) -> Dataset[Any]:
    """Normalize any wired source into a map-style ``Dataset`` that is GUARANTEED to yield ``Sample``s.

    A wired ``train_set`` / ``val_set`` / ``test_set`` may be a `dataflux.Flux`, another `torch`
    `Dataset`, a `dataflux` source like `HuggingFaceSource` (only `__getitem__`/`__len__`/`__iter__`),
    or a plain list — and its items may be `Sample`s or raw rows (dicts, tuples, …). A `dataflux.Flux`
    already coerces every item to a `Sample` (`Sample.from_any` in `__getitem__`/`__iter__`), so:

    * a `Flux` is returned as-is (already a `Dataset[Sample]`; this preserves a subclass's own wrap —
      e.g. sonair's label-encoding Flux with its `label_names`), and
    * anything else is wrapped in a `Flux`, which makes it both a map-style `Dataset` AND a
      `Sample`-yielding one.

    Calling this once up front lets the rest of the pipeline (target detection, label fitting/encoding,
    collate, metrics) assume `Sample` items unconditionally — no per-call "is this a Sample?" checks.
    """
    from dataflux import Flux

    if isinstance(source, Flux):
        return source
    return Flux(source=source)


def _resolve_label_names(dataset: Any) -> Optional[List[str]]:
    """Return the dataset's ``label_names`` as a plain list, or None.

    Informal protocol: any dataset MAY expose ``label_names: Sequence[str]``.
    Classification datasets may expose this; bespoke datasets too.
    Returns None when the attribute is missing or empty so callers
    can skip the class-name plumbing entirely.
    """
    names = getattr(dataset, "label_names", None)
    if not names:
        return None
    try:
        return [str(n) for n in names]
    except TypeError:
        return None


def _resolve_checkpoint_dir(lightning_trainer: L.Trainer) -> Optional[Path]:
    """Locate the directory where ModelCheckpoint is saving .ckpt files.

    Falls back to ``trainer.default_root_dir`` so we always have a
    deterministic location to drop ``class_names.json`` next to the run.
    """
    cb = getattr(lightning_trainer, "checkpoint_callback", None)
    dirpath = getattr(cb, "dirpath", None) if cb is not None else None
    if dirpath:
        return Path(dirpath)
    root = getattr(lightning_trainer, "default_root_dir", None)
    return Path(root) if root else None


def log_config_artifacts(lightning_trainer: L.Trainer) -> None:
    """Log the full YAML config tree to every wired Lightning logger.

    Resolves the file list from
    :attr:`liquifai.context.LiquifyContext.included_paths` (entrypoint +
    every transitively ``include:``-d YAML, populated by the CLI bootstrap)
    and falls back to just ``ctx.config_path`` when an older liquifai is
    installed. Each wired logger receives the YAML files via its
    backend-appropriate channel — see :func:`_log_paths_to_logger`.

    Shared module-level helper so both :class:`LightningTrainer.run` and
    :class:`marainer.evaluator.LightningEvaluator.run` capture the same
    artifacts without duplicating the dispatch logic.
    """
    from liquifai.context import get_context

    ctx = get_context()
    if ctx is None:
        return

    paths: List[Path] = list(getattr(ctx, "included_paths", None) or [])
    if not paths and ctx.config_path is not None:
        paths = [ctx.config_path]
    paths = [p for p in paths if p.exists()]
    if not paths:
        return

    for lg in getattr(lightning_trainer, "loggers", []) or []:
        _log_paths_to_logger(lg, paths)


def _log_paths_to_logger(lg: Any, paths: List[Path]) -> None:
    """Forward a list of YAML config files to a single Lightning logger.

    Dispatch order:
        * MLflow-style loggers (``experiment.log_artifact(run_id, path, dir)``
          with a non-None ``run_id``) — files are uploaded under ``config/``.
        * Everything else with a writable ``log_dir`` / ``save_dir`` — files
          are copied into ``<log_dir>/config/<name>`` so they sit alongside
          events files and ``hparams.yaml``.

    Failures are swallowed at ``debug`` level — config-artifact logging is
    a best-effort convenience that must never break a real training run.
    """
    exp = getattr(lg, "experiment", None)
    run_id = getattr(lg, "run_id", None)
    log_artifact = getattr(exp, "log_artifact", None) if exp is not None else None
    if callable(log_artifact) and run_id:
        for p in paths:
            try:
                log_artifact(run_id, str(p), "config")
                logger.debug(f"Logged {p.name} as MLflow artifact for run {run_id}")
            except Exception as exc:  # pragma: no cover - logger backends are external
                logger.debug(f"MLflow log_artifact failed on {type(lg).__name__} for {p}: {exc}")
        return

    log_dir = getattr(lg, "log_dir", None) or getattr(lg, "save_dir", None)
    if log_dir is None:
        logger.debug(f"{type(lg).__name__} has no log_dir/save_dir; skipping YAML copy")
        return
    target_dir = Path(log_dir) / "config"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover - filesystem permissions
        logger.debug(f"Could not create {target_dir}: {exc}")
        return
    for p in paths:
        try:
            shutil.copy2(p, target_dir / p.name)
            logger.debug(f"Copied {p.name} to {target_dir}")
        except OSError as exc:  # pragma: no cover - filesystem permissions
            logger.debug(f"Could not copy {p} to {target_dir}: {exc}")


def _write_class_names(lightning_trainer: L.Trainer, label_names: List[str]) -> None:
    """Persist ``{"class_names": [...], "num_classes": N}`` to disk + MLflow.

    Filesystem: ``<checkpoint_dir>/class_names.json`` so the file travels
    with the .ckpt artifacts the model loads from.

    MLflow: for every wired :class:`pytorch_lightning.loggers.MLFlowLogger`,
    log the same payload as a run artifact. Silently skipped when no MLflow
    logger is wired.
    """
    payload = {"class_names": label_names, "num_classes": len(label_names)}

    ckpt_dir = _resolve_checkpoint_dir(lightning_trainer)
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        out_path = ckpt_dir / "class_names.json"
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        logger.info(f"Wrote class_names.json to {out_path}")
    else:
        logger.debug("No checkpoint dir or default_root_dir; skipping filesystem class_names.json")

    for lg in getattr(lightning_trainer, "loggers", []) or []:
        run_id = getattr(lg, "run_id", None)
        exp = getattr(lg, "experiment", None)
        log_dict = getattr(exp, "log_dict", None) if exp is not None else None
        if callable(log_dict) and run_id:
            try:
                log_dict(run_id, payload, "class_names.json")
                logger.info(f"Logged class_names.json as MLflow artifact for run {run_id}")
            except Exception as exc:  # pragma: no cover - logger backends are external
                logger.debug(f"MLflow log_dict failed on {type(lg).__name__}: {exc}")


@configurable
class LightningTrainer(TorchRunner, L.LightningModule):
    """Marainer LightningTrainer: configurable Lightning module with integrated training loop.

    Supports YAML-configurable metrics via torchmetrics. Train and validation
    ``MetricCollection`` instances are configured directly in YAML using
    ``!clone:`` for independent copies (torchmetrics requires separate state).

    Follows confluid's minimal-ctor / post-construction discipline ("Lazy
    Initialization & Zero-Arg Construction", rule 4): the constructor takes only
    ``model`` / ``train_set`` (+ identity ``experiment_name`` / ``run_name``) and
    does **no functional work**. Every other slot is a defaulted **body
    attribute** configured post-construction (by YAML, Confluid broadcasting, or
    a subclass). The deferred slots — ``optimizer`` / ``train_loader`` /
    ``val_loader`` / ``test_loader`` / ``lightning`` — default to ``LazyClass(...)``
    and are flowed by :meth:`run` (loaders with ``dataset=``, the optimizer with
    ``params=`` via :meth:`configure_optimizers`, ``lightning`` after the
    auto-``ModelCheckpoint`` / logger wiring). ``to_pydantic`` introspects these
    body slots, so navigaitor / MCP / FluxStudio still enumerate them.

    YAML contract: a deferred slot MUST be wired with ``!lazy:`` (NOT ``!class:``)
    — a bare ``!class:`` post-init attr is eagerly materialized on assignment by
    confluid, which breaks runtime injection (``optimizer`` / loaders) and skips
    lightning's pre-flow wiring (:meth:`run` warns when a slot arrives
    already-materialized, see ``_marainer_warn_eager_runtime_slots``).

    Args:
        model: The neural network model (deferred via Confluid).
        train_set: Training dataset or Flux pipeline.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name. ``None`` (default) auto-generates a random, human-friendly slug
            like ``train_funny_rabbit`` (see :func:`marainer.run_names.generate_run_name`).

    Post-construction body-attribute slots (set via YAML / broadcasting):
        val_set: Validation dataset or Flux pipeline (used during ``fit``).
        test_set: Held-out test dataset; when set, ``lightning.test(...)`` runs
            after ``fit``.
        optimizer: Deferred optimizer (default: ``LazyClass(Adam, lr=1e-3)``).
        train_loader / val_loader / test_loader: Deferred ``DataLoader`` configs.
        lightning: Deferred PyTorch Lightning Trainer. Lightning loggers
            (TensorBoard / CSV / MLflow) belong under ``lightning.logger:`` in
            YAML — there is NO top-level ``loggers:`` argument here.
        train_metrics / val_metrics / test_metrics: ``MetricCollection`` per split.
        loss_fn: Loss function (default: ``CrossEntropyLoss``).
        log_system_metrics: Toggle MLflow system-metric capture (default ``True``).

    Example YAML (note ``!lazy:`` for the deferred lightning slot)::

        trainer: !class:marainer.trainer.LightningTrainer
          model: !lazy:my.Model
          train_set: !ref:train_set
        optimizer: !lazy:torch.optim.Adam
          lr: 0.001
        lightning: !lazy:pytorch_lightning.Trainer
          max_epochs: 10
          logger:
            - !class:pytorch_lightning.loggers.TensorBoardLogger()
              save_dir: ./logs
    """

    def __init__(
        self,
        model: Union[nn.Module, Fluid],
        train_set: Union[Dataset[Any], Fluid],
        experiment_name: str = "marainer_run",
        run_name: Optional[str] = None,
        max_epochs: int = 10,
        logger: Optional[Union[Logger, Iterable[Logger], bool]] = None,
    ) -> None:
        super().__init__()
        # Post-construction-configurable slots: each is reconfigured (by YAML /
        # Confluid broadcasting, or a subclass) to a Fluid / live object, so they
        # are typed Union[..., Fluid] rather than pinned to Any.
        self.model: Union[nn.Module, Fluid] = model
        self.train_set: Union[Dataset[Any], Fluid] = train_set
        self.val_set: Union[Dataset[Any], Fluid, None] = None
        self.test_set: Union[Dataset[Any], Fluid, None] = None
        self.optimizer: Any = LazyClass(torch.optim.Adam, lr=1e-3)
        self.train_loader: Any = LazyClass(DataLoader)
        self.val_loader: Any = LazyClass(DataLoader)
        self.test_loader: Any = LazyClass(DataLoader)
        self.lightning: Any = LazyClass(L.Trainer, logger=logger, max_epochs=max_epochs)
        self.experiment_name = experiment_name
        # A None run_name gets a random, human-friendly slug (e.g. ``train_funny_rabbit``) so every
        # run is distinguishable in logs / MLflow without the caller having to invent one.
        self.run_name = run_name if run_name is not None else generate_run_name(prefix="train")
        self.log_system_metrics: bool = True
        self.loss_fn: nn.Module = nn.CrossEntropyLoss()
        self.train_metrics: Any = None
        self.val_metrics: Any = None
        self.test_metrics: Any = None

        # The trainer flows these deferred slots itself at runtime, so they must be
        # declared here as non-deferred defaults (``LazyClass(...)``) to satisfy
        # the type system and navigaitor's form-spec / FluxStudio pickers.
        # Overwite this list with the slot names if you add more deferred slots that require runtime or
        # change the flow with pre-flow wiring (e.g. the default ModelCheckpoint).
        self.runtime_slots = ["optimizer", "train_loader", "val_loader", "test_loader", "lightning"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from typing import cast

        model = cast(nn.Module, self.model)
        return cast(torch.Tensor, model(x))

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds = self(batch.input)
        loss: torch.Tensor = self.loss_fn(preds, batch.target)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.input.shape[0],
        )
        if self.train_metrics is not None:
            self.train_metrics(preds, batch.target)
            if isinstance(self.train_metrics, MetricCollection):
                self.log_dict(self.train_metrics, on_step=True, on_epoch=False)
            else:
                self.log("train/metric", self.train_metrics, on_step=True, on_epoch=False)
        return loss

    def on_train_epoch_end(self) -> None:
        if self.train_metrics is not None:
            self.train_metrics.reset()

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds = self(batch.input)
        loss: torch.Tensor = self.loss_fn(preds, batch.target)
        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.input.shape[0],
        )
        if self.val_metrics is not None:
            self.val_metrics.update(preds, batch.target)
        return loss

    def on_val_epoch_end(self) -> None:
        if self.val_metrics is not None:
            if isinstance(self.val_metrics, MetricCollection):
                self.log_dict(self.val_metrics, on_step=True, on_epoch=False)
            else:
                self.log("val/metric", self.val_metrics, on_step=True, on_epoch=False)
            self.val_metrics.reset()

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds = self(batch.input)
        loss: torch.Tensor = self.loss_fn(preds, batch.target)
        self.log(
            "test/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.input.shape[0],
        )
        if self.test_metrics is not None:
            self.test_metrics.update(preds, batch.target)
        return loss

    def on_test_epoch_end(self) -> None:
        if self.test_metrics is not None:
            if isinstance(self.test_metrics, MetricCollection):
                self.log_dict(self.test_metrics, on_step=True, on_epoch=False)
            else:
                self.log("test/metric", self.test_metrics, on_step=True, on_epoch=False)
            self.test_metrics.reset()

    def configure_optimizers(self) -> Any:
        return flow(self.optimizer, params=self.parameters())

    def _marainer_log_hyperparams(self, lightning_trainer: L.Trainer) -> None:
        """Log hyperparameters, git hash, and config path to all loggers."""
        from liquifai.context import get_context
        from liquifai.discovery import get_configurable_paths

        params = get_configurable_paths(self)
        clean_params = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}

        ctx = get_context()
        if ctx and ctx.config_path:
            clean_params["config_script"] = str(ctx.config_path)

        for lg in lightning_trainer.loggers:
            lg.log_hyperparams(clean_params)

    def _marainer_log_config_artifacts(self, lightning_trainer: L.Trainer) -> None:
        """Thin instance-side hook over :func:`log_config_artifacts`."""
        log_config_artifacts(lightning_trainer)

    def _warn_eager_runtime_slots(self) -> None:
        """Warn when a deferred slot arrived already-materialized (misconfigured YAML).

        ``optimizer`` / ``train_loader`` / ``val_loader`` / ``test_loader`` /
        ``lightning`` are deferred body slots that :meth:`run` flows itself — the
        optimizer/loaders with runtime kwargs (``params=`` / ``dataset=``),
        ``lightning`` after the auto-``ModelCheckpoint`` / logger wiring. A YAML
        that wires one with ``!class:`` (a bare ``Class``) instead of ``!lazy:``
        is eagerly materialized by confluid on attribute assignment, so it
        reaches here already-built: the runtime kwargs can no longer be injected
        and lightning's pre-flow wiring is silently skipped. The optimizer/loader
        cases usually fail loudly at construction; ``lightning`` (no required
        args) fails silently — this surfaces it as an actionable warning.
        """
        from confluid.fluid import Fluid

        for name in self.runtime_slots:
            value = getattr(self, name, None)
            if value is not None and not isinstance(value, Fluid):
                logger.warning(
                    f"Trainer slot '{name}' arrived already-materialized "
                    f"({type(value).__name__}); it is a deferred slot the trainer flows itself. "
                    f"Declare it '!lazy:' (not '!class:') in YAML so runtime injection / pre-flow "
                    f"wiring (e.g. the default ModelCheckpoint) works."
                )

    def run(self) -> None:
        """Execute the training loop."""
        logger.info(f"Marainer navigating to: {self.experiment_name} ({self.run_name})")
        self._warn_eager_runtime_slots()

        # Materialize datasets ONCE — reused for dimension derivation, the
        # DataLoaders, and label-name persistence. (Flowing them here also fixes
        # an older latent issue where the label-name lookup ran against the
        # still-deferred Fluid and silently skipped class_names.json.)
        train_ds = ensure_sample_dataset(flow(self.train_set))
        val_ds = ensure_sample_dataset(flow(self.val_set)) if self.val_set is not None else None
        test_ds = ensure_sample_dataset(flow(self.test_set)) if self.test_set is not None else None

        # Flow the model + metrics into live instances. The base stays
        # task-agnostic: it materializes whatever is wired (an explicit config, or a
        # node a task subclass already flowed with its dimensions). Task-specific
        # dimension injection — e.g. a classification trainer flowing the model +
        # metrics with num_classes — is owned by the subclass's initialize() at the
        # top of run(); flow() here is idempotent on the already-live instance.
        self.model = flow(self.model)
        if self.train_metrics is not None:
            self.train_metrics = flow(self.train_metrics)
        if self.val_metrics is not None:
            self.val_metrics = flow(self.val_metrics)
        if self.test_metrics is not None:
            self.test_metrics = flow(self.test_metrics)
        # Materialize the loss if a config supplied it as a deferred Fluid
        # (the loss is a configurable slot now); idempotent on the live default.
        self.loss_fn = flow(self.loss_fn)

        t_loader = flow(self.train_loader, dataset=train_ds)
        v_loader = flow(self.val_loader, dataset=val_ds) if val_ds is not None else None

        lightning_trainer = flow(self.lightning)
        if self.log_system_metrics:
            from marainer.system_metrics import activate_for_lightning

            activate_for_lightning(lightning_trainer)
        self._marainer_log_config_artifacts(lightning_trainer)
        self._marainer_log_hyperparams(lightning_trainer)
        lightning_trainer.fit(model=self, train_dataloaders=t_loader, val_dataloaders=v_loader)
        logger.info("Training complete.")

        label_names = _resolve_label_names(train_ds)
        if label_names:
            _write_class_names(lightning_trainer, label_names)
        else:
            logger.debug("train_set exposes no label_names; skipping class_names.json")

        if test_ds is not None:
            logger.info("Evaluating on held-out test set...")
            test_loader = flow(self.test_loader, dataset=test_ds)
            lightning_trainer.test(model=self, dataloaders=test_loader)
            logger.info("Test evaluation complete.")

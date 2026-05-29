import json
import shutil
from pathlib import Path
from typing import Any, List, Optional, Tuple

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class, Instance, Lazy, configurable, flow
from logflow import get_logger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from marainer.dimensions import collect_derived_dimensions, flow_node_with_dims, needs_derivation

logger = get_logger(__name__)


def _is_model_checkpoint_target(cb: Any) -> bool:
    """True when ``cb`` is, or defers to, a ``ModelCheckpoint``.

    Accepts: a live ``ModelCheckpoint`` instance, a confluid ``Class`` /
    ``Instance`` whose target resolves to ``ModelCheckpoint`` (subclass-aware
    when target is a real class object; string targets are matched by name
    so YAML that uses the dotted form ``pytorch_lightning.callbacks.ModelCheckpoint``
    is still recognised).
    """
    if isinstance(cb, ModelCheckpoint):
        return True
    if isinstance(cb, (Class, Instance)):
        target = cb.target
        if isinstance(target, type) and issubclass(target, ModelCheckpoint):
            return True
        if isinstance(target, str):
            return target.endswith("ModelCheckpoint")
    return False


def _logger_wants_all_checkpoints(lg: Any) -> bool:
    """True when a logger (Fluid or instantiated) opts into archiving every checkpoint.

    The flag lives on the instance for already-flowed loggers and inside
    ``Fluid.kwargs`` for deferred ones. See :mod:`marainer.loggers` for the
    canonical opt-in subclasses (``MarainerMLFlowLogger`` and friends).
    """
    if isinstance(lg, (Class, Instance)):
        return bool(lg.kwargs.get("log_all_checkpoints", False))
    return bool(getattr(lg, "log_all_checkpoints", False))


def _resolve_label_names(dataset: Any) -> Optional[List[str]]:
    """Return the dataset's ``label_names`` as a plain list, or None.

    Informal protocol: any dataset MAY expose ``label_names: Sequence[str]``.
    HuggingFaceClassificationDataset in sonair does; bespoke datasets may
    too. Returns None when the attribute is missing or empty so callers
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
class LightningTrainer(L.LightningModule):
    """Marainer LightningTrainer: configurable Lightning module with integrated training loop.

    Supports YAML-configurable metrics via torchmetrics. Train and validation
    ``MetricCollection`` instances are configured directly in YAML using
    ``!clone:`` for independent copies (torchmetrics requires separate state).

    Args:
        model: The neural network model (deferred via Confluid).
        train_set: Training dataset or Flux pipeline.
        val_set: Validation dataset or Flux pipeline (used during ``fit``).
        test_set: Held-out test dataset or Flux pipeline. When set,
            ``lightning_trainer.test(...)`` runs after ``fit``.
        optimizer: Optimizer class/config (deferred, default: Adam lr=1e-3).
        train_loader: DataLoader config for training data.
        val_loader: DataLoader config for validation data.
        test_loader: DataLoader config for test data.
        lightning: PyTorch Lightning Trainer config. Lightning loggers
            (TensorBoard / CSV / MLflow) belong under ``lightning.logger:``
            in YAML — that's the kwarg Lightning's own ``Trainer.__init__``
            picks up. There is NO top-level ``loggers:`` argument here.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.

    Post-construction attributes (set via ``@configurable``):
        train_metrics: Train ``MetricCollection`` (from YAML).
        val_metrics: Validation ``MetricCollection`` (from YAML).
        test_metrics: Test ``MetricCollection`` (from YAML), used after ``fit``.
        loss_fn: Loss function (default: ``CrossEntropyLoss``).

    Example YAML::

        metrics:
          - !class:torchmetrics.classification.MulticlassAccuracy()
          - !class:torchmetrics.classification.MulticlassF1Score()

        train_metrics: !class:torchmetrics.MetricCollection()
          metrics: !clone:metrics
          prefix: "train/"

        val_metrics: !class:torchmetrics.MetricCollection()
          metrics: !clone:metrics
          prefix: "val/"
    """

    def __init__(
        self,
        model: Any,
        train_set: Any,
        val_set: Any = None,
        test_set: Any = None,
        optimizer: Lazy[Any] = Class(torch.optim.Adam, lr=1e-3),
        train_loader: Lazy[Any] = Class(DataLoader),
        val_loader: Lazy[Any] = Class(DataLoader),
        test_loader: Lazy[Any] = Class(DataLoader),
        lightning: Any = Class(L.Trainer),
        experiment_name: str = "marainer_run",
        run_name: str = "train",
        log_system_metrics: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.train_set = train_set
        self.val_set = val_set
        self.test_set: Any = test_set
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lightning = lightning
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.log_system_metrics = bool(log_system_metrics)
        self.loss_fn: nn.Module = nn.CrossEntropyLoss()
        self.train_metrics: MetricCollection | None = None
        self.val_metrics: MetricCollection | None = None
        self.test_metrics: MetricCollection | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from typing import cast

        return cast(torch.Tensor, self.model(x))

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
            scalars = {k: v for k, v in self.train_metrics.items() if isinstance(v, torch.Tensor) and v.numel() == 1}
            if scalars:
                self.log_dict(
                    scalars,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch.input.shape[0],
                )
        return loss

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
            self.val_metrics(preds, batch.target)
            scalars = {k: v for k, v in self.val_metrics.items() if isinstance(v, torch.Tensor) and v.numel() == 1}
            if scalars:
                self.log_dict(
                    scalars,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch.input.shape[0],
                )
        return loss

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
            self.test_metrics(preds, batch.target)
            scalars = {k: v for k, v in self.test_metrics.items() if isinstance(v, torch.Tensor) and v.numel() == 1}
            if scalars:
                self.log_dict(
                    scalars,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch.input.shape[0],
                )
        return loss

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

    def _resolve_default_checkpoint_monitor(self) -> Tuple[str, str]:
        """Pick ``(monitor, mode)`` for the auto-injected ``ModelCheckpoint``.

        Resolution order:

        1. First key in ``self.val_metrics`` when configured AND a validation
           set is wired — keys produced by
           :class:`torchmetrics.MetricCollection.items` already carry the
           collection's prefix (``"val/top1"``), matching what
           :meth:`validation_step` logs. ``mode`` comes from the metric's
           :attr:`higher_is_better` class attribute (``True`` → max,
           ``False`` → min, missing → max).
        2. ``("val/loss", "min")`` when a val set is wired but no metrics.
        3. ``("train/loss", "min")`` when no val set is wired —
           :meth:`training_step` always logs that scalar, so the monitor
           cannot dangle.
        """
        if self.val_set is not None:
            if self.val_metrics is not None:
                try:
                    items = list(self.val_metrics.items())
                except Exception:  # pragma: no cover - torchmetrics raised; treat as no metrics
                    items = []
                if items:
                    raw_key, first_metric = items[0]
                    prefix = getattr(self.val_metrics, "prefix", "") or ""
                    monitor = raw_key if raw_key.startswith(prefix) else f"{prefix}{raw_key}"
                    higher = getattr(first_metric, "higher_is_better", None)
                    mode = "min" if higher is False else "max"
                    return monitor, mode
            return "val/loss", "min"
        return "train/loss", "min"

    def _marainer_ensure_default_checkpoint(self) -> None:
        """Inject a default ``ModelCheckpoint`` into ``self.lightning.kwargs`` when missing.

        Runs BEFORE ``flow(self.lightning)`` so the callback lands in the
        Lightning Trainer's own ``callbacks`` argument at construction time —
        Lightning then knows checkpoint handling is already wired and does
        not silently auto-inject its own ``monitor=None`` default. When the
        user has already declared any ``ModelCheckpoint`` (Fluid or
        instance) under ``lightning.callbacks``, this is a no-op.

        Defaults:
            * ``save_last=True``  — always keep ``last.ckpt``
            * ``save_top_k=1``    — keep the single best checkpoint by
              the first val metric (escalates to ``-1`` when any wired
              logger has ``log_all_checkpoints=True``)
            * ``monitor`` / ``mode`` — see
              :meth:`_resolve_default_checkpoint_monitor`
        """
        if not hasattr(self.lightning, "kwargs"):
            return  # Already flowed or exotic type — leave alone.

        # ``enable_checkpointing=False`` is the user's explicit opt-out — Lightning
        # would also reject a ModelCheckpoint in callbacks alongside the flag
        # (``MisconfigurationException``), so skip injection rather than fight it.
        if self.lightning.kwargs.get("enable_checkpointing") is False:
            logger.debug("enable_checkpointing=False; skipping default ModelCheckpoint injection")
            return

        existing_callbacks = self.lightning.kwargs.get("callbacks") or []
        if not isinstance(existing_callbacks, list):
            existing_callbacks = [existing_callbacks]
        if any(_is_model_checkpoint_target(cb) for cb in existing_callbacks):
            logger.debug("User-supplied ModelCheckpoint detected; skipping default injection")
            return

        existing_loggers = self.lightning.kwargs.get("logger") or []
        if not isinstance(existing_loggers, list):
            existing_loggers = [existing_loggers]
        save_top_k = -1 if any(_logger_wants_all_checkpoints(lg) for lg in existing_loggers) else 1

        monitor, mode = self._resolve_default_checkpoint_monitor()
        default_cb = ModelCheckpoint(
            monitor=monitor,
            mode=mode,
            save_top_k=save_top_k,
            save_last=True,
            filename="best-{epoch:02d}-{step}",
        )
        self.lightning.kwargs["callbacks"] = [*existing_callbacks, default_cb]
        logger.info(
            f"Auto-injected default ModelCheckpoint: monitor={monitor!r}, "
            f"mode={mode!r}, save_top_k={save_top_k}, save_last=True"
        )

    def _marainer_log_config_artifacts(self, lightning_trainer: L.Trainer) -> None:
        """Thin instance-side hook over :func:`log_config_artifacts`."""
        log_config_artifacts(lightning_trainer)

    def run(self) -> None:
        """Execute the training loop."""
        logger.info(f"Marainer navigating to: {self.experiment_name} ({self.run_name})")

        # Materialize datasets ONCE — reused for dimension derivation, the
        # DataLoaders, and label-name persistence. (Flowing them here also fixes
        # an older latent issue where the label-name lookup ran against the
        # still-deferred Fluid and silently skipped class_names.json.)
        train_ds = flow(self.train_set)
        val_ds = flow(self.val_set) if self.val_set is not None else None
        test_ds = flow(self.test_set) if self.test_set is not None else None

        # Generic, task-agnostic dimension injection: a config may declare the
        # model / metrics as `!lazy:` and omit a dimension (e.g. num_classes);
        # the dataset derives it (sonair walks its targets) and we inject it when
        # constructing those deferred nodes. We only walk the dataset(s) when a
        # node is actually deferred-lazy, so the explicit path pays nothing.
        dims = (
            collect_derived_dimensions([train_ds, val_ds, test_ds])
            if needs_derivation([self.model, self.train_metrics, self.val_metrics, self.test_metrics])
            else {}
        )
        if dims:
            logger.info(f"Derived model dimensions from dataset(s): {dims}")

        self.model = flow_node_with_dims(self.model, dims)
        if self.train_metrics is not None:
            self.train_metrics = flow_node_with_dims(self.train_metrics, dims)
        if self.val_metrics is not None:
            self.val_metrics = flow_node_with_dims(self.val_metrics, dims)
        if self.test_metrics is not None:
            self.test_metrics = flow_node_with_dims(self.test_metrics, dims)
        # Materialize the loss if a config supplied it as a deferred Fluid
        # (the loss is a configurable slot now); idempotent on the live default.
        self.loss_fn = flow(self.loss_fn)

        t_loader = flow(self.train_loader, dataset=train_ds)
        v_loader = flow(self.val_loader, dataset=val_ds) if val_ds is not None else None

        # Must run before `flow(self.lightning)` so the default callback lands in
        # the L.Trainer's own callbacks list at construction time — otherwise
        # Lightning silently auto-injects a monitor=None default that we'd have
        # to surgically remove afterwards.
        self._marainer_ensure_default_checkpoint()

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

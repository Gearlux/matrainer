import json
from pathlib import Path
from typing import Any, List, Optional

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class, Lazy, configurable, flow
from logflow import get_logger
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

logger = get_logger(__name__)


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
        lightning: PyTorch Lightning Trainer config.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.
        loggers: List of Lightning loggers (deferred via Confluid).

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
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch.input.shape[0])
        if self.train_metrics is not None:
            self.train_metrics(preds, batch.target)
            scalars = {k: v for k, v in self.train_metrics.items() if isinstance(v, torch.Tensor) and v.numel() == 1}
            if scalars:
                self.log_dict(scalars, on_step=False, on_epoch=True, batch_size=batch.input.shape[0])
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds = self(batch.input)
        loss: torch.Tensor = self.loss_fn(preds, batch.target)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=batch.input.shape[0])
        if self.val_metrics is not None:
            self.val_metrics(preds, batch.target)
            scalars = {k: v for k, v in self.val_metrics.items() if isinstance(v, torch.Tensor) and v.numel() == 1}
            if scalars:
                self.log_dict(scalars, on_step=False, on_epoch=True, batch_size=batch.input.shape[0])
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds = self(batch.input)
        loss: torch.Tensor = self.loss_fn(preds, batch.target)
        self.log("test/loss", loss, on_epoch=True, prog_bar=True, batch_size=batch.input.shape[0])
        if self.test_metrics is not None:
            self.test_metrics(preds, batch.target)
            scalars = {k: v for k, v in self.test_metrics.items() if isinstance(v, torch.Tensor) and v.numel() == 1}
            if scalars:
                self.log_dict(scalars, on_step=False, on_epoch=True, batch_size=batch.input.shape[0])
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

    def run(self) -> None:
        """Execute the training loop."""
        logger.info(f"Marainer navigating to: {self.experiment_name} ({self.run_name})")

        self.model = flow(self.model)
        if self.train_metrics is not None:
            self.train_metrics = flow(self.train_metrics)
        if self.val_metrics is not None:
            self.val_metrics = flow(self.val_metrics)
        if self.test_metrics is not None:
            self.test_metrics = flow(self.test_metrics)

        t_loader = flow(self.train_loader, dataset=flow(self.train_set))
        v_loader = flow(self.val_loader, dataset=flow(self.val_set)) if self.val_set else None

        lightning_trainer = flow(self.lightning)
        self._marainer_log_hyperparams(lightning_trainer)
        lightning_trainer.fit(model=self, train_dataloaders=t_loader, val_dataloaders=v_loader)
        logger.info("Training complete.")

        label_names = _resolve_label_names(self.train_set)
        if label_names:
            _write_class_names(lightning_trainer, label_names)
        else:
            logger.debug("train_set exposes no label_names; skipping class_names.json")

        if self.test_set is not None:
            logger.info("Evaluating on held-out test set...")
            test_loader = flow(self.test_loader, dataset=flow(self.test_set))
            lightning_trainer.test(model=self, dataloaders=test_loader)
            logger.info("Test evaluation complete.")

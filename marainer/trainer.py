import subprocess
from typing import Any

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class, configurable, flow
from logflow import get_logger
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

logger = get_logger(__name__)


def _get_git_hash() -> str | None:
    """Return current git commit hash, or None if unavailable."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


@configurable
class Trainer(L.LightningModule):
    """Marainer Trainer: configurable Lightning module with integrated training loop.

    Supports YAML-configurable metrics via torchmetrics. Train and validation
    ``MetricCollection`` instances are configured directly in YAML using
    ``!clone:`` for independent copies (torchmetrics requires separate state).

    Args:
        model: The neural network model (deferred via Confluid).
        train_set: Training dataset or Flux pipeline.
        val_set: Validation dataset or Flux pipeline.
        optimizer: Optimizer class/config (deferred, default: Adam lr=1e-3).
        train_loader: DataLoader config for training data.
        val_loader: DataLoader config for validation data.
        lightning: PyTorch Lightning Trainer config.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.
        loggers: List of Lightning loggers (deferred via Confluid).

    Post-construction attributes (set via ``@configurable``):
        train_metrics: Train ``MetricCollection`` (from YAML).
        val_metrics: Validation ``MetricCollection`` (from YAML).
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
        optimizer: Any = Class(torch.optim.Adam, lr=1e-3),
        train_loader: Any = Class(DataLoader),
        val_loader: Any = Class(DataLoader),
        lightning: Any = Class(L.Trainer),
        experiment_name: str = "marainer_run",
        run_name: str = "train",
        loggers: Any = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.train_set = train_set
        self.val_set = val_set
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lightning = lightning
        self.experiment_name = experiment_name
        self.run_name = run_name
        self._loggers_config = loggers
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_metrics: MetricCollection | None = None
        self.val_metrics: MetricCollection | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from typing import cast

        return cast(torch.Tensor, self.model(x))

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds = self(batch.input)
        loss: torch.Tensor = self.loss_fn(preds, batch.target)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch.input.shape[0])
        if self.train_metrics is not None:
            self.train_metrics(preds, batch.target)
            self.log_dict(self.train_metrics, on_step=False, on_epoch=True, batch_size=batch.input.shape[0])
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        preds = self(batch.input)
        loss: torch.Tensor = self.loss_fn(preds, batch.target)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, batch_size=batch.input.shape[0])
        if self.val_metrics is not None:
            self.val_metrics(preds, batch.target)
            self.log_dict(self.val_metrics, on_step=False, on_epoch=True, batch_size=batch.input.shape[0])
        return loss

    def configure_optimizers(self) -> Any:
        return flow(self.optimizer, params=self.parameters())

    def _marainer_log_hyperparams(self, lightning_trainer: L.Trainer) -> None:
        """Log hyperparameters, git hash, and config path to all loggers."""
        from liquify.context import get_context
        from liquify.discovery import get_configurable_paths

        params = get_configurable_paths(self)
        clean_params = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}

        git_hash = _get_git_hash()
        if git_hash:
            clean_params["git_hash"] = git_hash

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

        t_loader = flow(self.train_loader, dataset=flow(self.train_set))
        v_loader = flow(self.val_loader, dataset=flow(self.val_set)) if self.val_set else None

        # Pass loggers to Lightning Trainer — Confluid auto-flows Instance objects in the list
        logger_kwarg: dict[str, Any] = {}
        if self._loggers_config is not None:
            logger_kwarg["logger"] = self._loggers_config

        lightning_trainer = flow(self.lightning, **logger_kwarg)
        self._marainer_log_hyperparams(lightning_trainer)
        lightning_trainer.fit(model=self, train_dataloaders=t_loader, val_dataloaders=v_loader)
        logger.info("Training complete.")

from typing import Any

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class, configurable, flow
from logflow import get_logger
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

logger = get_logger(__name__)


@configurable
class Trainer(L.LightningModule):
    """Marainer Trainer: configurable Lightning module with integrated training loop.

    Supports YAML-configurable metrics via torchmetrics. Metrics can be declared
    individually using ``!class:`` tags or loaded in bulk via
    :class:`~marainer.metrics.MetricPreset`.

    Args:
        model: The neural network model (deferred via Confluid).
        train_set: Training dataset or Flux pipeline.
        val_set: Validation dataset or Flux pipeline.
        optimizer: Optimizer class/config (deferred, default: Adam lr=1e-3).
        train_loader: DataLoader config for training data.
        val_loader: DataLoader config for validation data.
        lightning: PyTorch Lightning Trainer config.
        metrics: List of torchmetrics ``Metric`` instances and/or
            :class:`~marainer.metrics.MetricPreset` objects. When provided,
            separate train and validation MetricCollections are created
            automatically. Shared parameters like ``num_classes`` are
            broadcast from the YAML root by Confluid.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.

    Example YAML with metrics::

        num_classes: 10
        metrics:
          - !class:torchmetrics.classification.MulticlassAccuracy()
          - !class:torchmetrics.classification.MulticlassF1Score()
    """

    def __init__(
        self,
        model: Any = None,
        train_set: Any = None,
        val_set: Any = None,
        optimizer: Any = Class(torch.optim.Adam, lr=1e-3),
        train_loader: Any = Class(DataLoader),
        val_loader: Any = Class(DataLoader),
        lightning: Any = Class(L.Trainer),
        metrics: list[Any] | None = None,
        experiment_name: str = "marainer_run",
        run_name: str = "train",
    ) -> None:
        super().__init__()
        self.model = model
        self.train_set = train_set
        self.val_set = val_set
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lightning = lightning
        self.metrics = metrics
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_metrics: MetricCollection | None = None
        self.val_metrics: MetricCollection | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # type: ignore[no-any-return]

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

    def _setup_metrics(self) -> None:
        """Materialize and split metrics into train/val MetricCollections."""
        if self.metrics is None:
            return

        from marainer.metrics import build_metric_collection

        raw = self.metrics if isinstance(self.metrics, list) else [self.metrics]
        materialized = [flow(m) for m in raw]
        base = build_metric_collection(materialized)
        self.train_metrics = base.clone(prefix="train/")
        self.val_metrics = base.clone(prefix="val/")
        logger.info(f"Metrics configured: {list(base.keys())}")

    def run(self) -> None:
        """Execute the training loop."""
        logger.info(f"Marainer navigating to: {self.experiment_name} ({self.run_name})")

        self.model = flow(self.model)
        self._setup_metrics()

        t_loader = flow(self.train_loader, dataset=flow(self.train_set))
        v_loader = flow(self.val_loader, dataset=flow(self.val_set)) if self.val_set else None

        lightning_trainer = flow(self.lightning)
        lightning_trainer.fit(model=self, train_dataloaders=t_loader, val_dataloaders=v_loader)
        logger.info("Training complete.")

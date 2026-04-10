from typing import Any

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class, configurable, flow
from logflow import get_logger
from torch.utils.data import DataLoader

logger = get_logger(__name__)


@configurable
class Trainer(L.LightningModule):
    """Marainer Trainer: configurable Lightning module with integrated training loop."""

    def __init__(
        self,
        model: Any = None,
        train_set: Any = None,
        val_set: Any = None,
        optimizer: Any = Class(torch.optim.Adam, lr=1e-3),
        train_loader: Any = Class(DataLoader),
        val_loader: Any = Class(DataLoader),
        lightning: Any = Class(L.Trainer),
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
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # type: ignore[no-any-return]

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss: torch.Tensor = self.loss_fn(self(batch.input), batch.target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch.input.shape[0])
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss: torch.Tensor = self.loss_fn(self(batch.input), batch.target)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch.input.shape[0])
        return loss

    def configure_optimizers(self) -> Any:
        return flow(self.optimizer, params=self.parameters())

    def run(self) -> None:
        """Execute the training loop."""
        logger.info(f"Marainer navigating to: {self.experiment_name} ({self.run_name})")

        self.model = flow(self.model)

        t_loader = flow(self.train_loader, dataset=flow(self.train_set))
        v_loader = flow(self.val_loader, dataset=flow(self.val_set)) if self.val_set else None

        lightning_trainer = flow(self.lightning)
        lightning_trainer.fit(model=self, train_dataloaders=t_loader, val_dataloaders=v_loader)
        logger.info("Training complete.")

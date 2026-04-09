from typing import Any, Union

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class, configurable, flow
from logflow import get_logger
from torch.utils.data import DataLoader, IterableDataset

logger = get_logger(__name__)


class MarainerLightningModule(L.LightningModule):
    """
    Generic Lightning wrapper for Marainer models.
    """

    def __init__(self, model: nn.Module, optimizer: Any):
        super().__init__()
        self.model = model
        self.optimizer_factory = optimizer
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # type: ignore[no-any-return]

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        data, target = self._unpack_batch(batch)
        output = self(data)
        loss: torch.Tensor = self.loss_fn(output, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        data, target = self._unpack_batch(batch)
        output = self(data)
        loss: torch.Tensor = self.loss_fn(output, target)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Any:
        # Use confluid.flow to instantiate the optimizer with model parameters
        return flow(self.optimizer_factory, params=self.parameters())

    def _unpack_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Standardizes batch unpacking for DataFlux Samples or tuples.

        Handles three formats:
        1. List of Samples (from Flux chunk_size batching)
        2. Collated Sample (from DataLoader batch_size collation)
        3. Tuple/list of (data, target) tensors (standard PyTorch)
        """
        if isinstance(batch, list) and len(batch) > 0 and hasattr(batch[0], "input"):
            # List of individual Samples → stack into batch
            data = torch.stack([s.input for s in batch])
            target = torch.tensor([s.target for s in batch])
        elif hasattr(batch, "input") and hasattr(batch, "target"):
            # Collated Sample (DataLoader auto-batched) — already batched tensors
            data = batch.input
            target = batch.target if isinstance(batch.target, torch.Tensor) else torch.tensor(batch.target)
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            data, target = batch[0], batch[1]
        else:
            raise ValueError(f"Unknown batch format: {type(batch)}")
        return data, target


class FluxIterableDataset(IterableDataset):
    """Bridges DataFlux Flux to PyTorch IterableDataset."""

    def __init__(self, flux: Any):
        self.flux = flux

    def __iter__(self) -> Any:
        yield from self.flux


@configurable
class Trainer:
    """
    Marainer Trainer powered by PyTorch Lightning.
    Orchestrates the lifecycle using explicitly defined datasets and deferred loaders.
    """

    def __init__(
        self,
        model: Any = None,
        optimizer: Any = None,
        train_set: Any = None,
        val_set: Any = None,
        train_loader: Any = Class(DataLoader),  # Default Class citizen
        val_loader: Any = Class(DataLoader),  # Default Class citizen
        epochs: int = 10,
        accelerator: str = "auto",
        devices: Union[str, int] = "auto",
        experiment_name: str = "marainer_run",
        run_name: str = "train",
        **lightning_kwargs: Any,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_set = train_set
        self.val_set = val_set
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.accelerator = accelerator
        self.devices = devices
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.lightning_kwargs = lightning_kwargs

    def run(self) -> None:
        """Executes the training loop using PyTorch Lightning."""
        logger.info(f"Marainer navigating to: {self.experiment_name} ({self.run_name})")

        # 1. Flow Core Components
        model_instance = flow(self.model)
        lightning_model = MarainerLightningModule(model_instance, self.optimizer)

        # 2. Flow Data Components (Deferred materialization)
        # We flow the set first, then use it as the 'dataset' argument for the loader
        train_data = flow(self.train_set)
        if hasattr(train_data, "__iter__") and not hasattr(train_data, "__getitem__"):
            train_data = FluxIterableDataset(train_data)

        # Materialize the loader with the live dataset
        # This will also pick up any CLI overrides like --num_workers
        t_loader = flow(self.train_loader, dataset=train_data)

        v_loader = None
        if self.val_set:
            val_data = flow(self.val_set)
            if hasattr(val_data, "__iter__") and not hasattr(val_data, "__getitem__"):
                val_data = FluxIterableDataset(val_data)
            v_loader = flow(self.val_loader, dataset=val_data)

        # 3. Initialize Lightning Trainer
        trainer = L.Trainer(
            accelerator=self.accelerator,
            devices=self.devices,
            max_epochs=self.epochs,
            **self.lightning_kwargs,
        )

        # 4. Fit
        logger.info(f"Starting Lightning Fit. Train Loader: {t_loader.__class__.__name__}")
        trainer.fit(model=lightning_model, train_dataloaders=t_loader, val_dataloaders=v_loader)

        logger.info("Training complete. Mariner has safely docked.")

"""Integration tests: verify training works with both batching strategies.

Scenario A: chunk_size on Flux, batch_size=null on DataLoader
Scenario B: chunk_size=null on Flux, batch_size on DataLoader
"""

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class, configurable
from dataflux.core import Flux
from dataflux.sample import Sample
from torch.utils.data import DataLoader

from marainer.trainer import FluxIterableDataset, MarainerLightningModule


@configurable
class TinyModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self, input_size: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # type: ignore[no-any-return]


def _make_samples(n: int = 32, input_size: int = 4) -> list[Sample]:
    """Create n synthetic samples."""
    return [
        Sample(
            input=torch.randn(input_size),
            target=i % 2,
            metadata={"idx": i},
        )
        for i in range(n)
    ]


class TestScenarioA:
    """Flux does batching (chunk_size), DataLoader passes through (batch_size=None)."""

    def test_flux_chunked_dataloader_passthrough(self) -> None:
        """Verify Flux chunk_size produces list-of-Samples batches."""
        flux = Flux(source=_make_samples(32), chunk_size=8)
        ds = FluxIterableDataset(flux)
        loader = DataLoader(ds, batch_size=None)

        batches = list(loader)
        assert len(batches) == 4  # 32 / 8
        assert isinstance(batches[0], list)
        assert len(batches[0]) == 8
        assert hasattr(batches[0][0], "input")

    def test_training_step_with_chunked_flux(self) -> None:
        """Verify the training step works with Flux-chunked batches."""
        model = TinyModel()
        module = MarainerLightningModule(model=model, optimizer=Class(torch.optim.SGD, lr=0.01))

        flux = Flux(source=_make_samples(16), chunk_size=8)
        ds = FluxIterableDataset(flux)
        loader = DataLoader(ds, batch_size=None)

        batch = next(iter(loader))
        loss = module.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar

    def test_full_epoch_with_chunked_flux(self) -> None:
        """Run a full training epoch with Flux chunking."""
        model = TinyModel()
        module = MarainerLightningModule(model=model, optimizer=Class(torch.optim.SGD, lr=0.01))

        flux = Flux(source=_make_samples(32), chunk_size=16)
        ds = FluxIterableDataset(flux)
        loader = DataLoader(ds, batch_size=None)

        trainer = L.Trainer(max_epochs=1, accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)
        trainer.fit(model=module, train_dataloaders=loader)


class TestScenarioB:
    """Flux yields individual Samples (no chunking), DataLoader batches them."""

    def test_flux_individual_dataloader_batches(self) -> None:
        """Verify DataLoader collates individual Samples into batched Sample."""
        flux = Flux(source=_make_samples(32))  # chunk_size=0 (no chunking)
        ds = FluxIterableDataset(flux)
        loader = DataLoader(ds, batch_size=8)

        batches = list(loader)
        assert len(batches) == 4  # 32 / 8
        # DataLoader default_collate on NamedTuples produces a collated NamedTuple
        batch = batches[0]
        assert hasattr(batch, "input")
        assert batch.input.shape == (8, 4)

    def test_training_step_with_dataloader_batching(self) -> None:
        """Verify the training step works with DataLoader-collated batches."""
        model = TinyModel()
        module = MarainerLightningModule(model=model, optimizer=Class(torch.optim.SGD, lr=0.01))

        flux = Flux(source=_make_samples(16))
        ds = FluxIterableDataset(flux)
        loader = DataLoader(ds, batch_size=8)

        batch = next(iter(loader))
        loss = module.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_full_epoch_with_dataloader_batching(self) -> None:
        """Run a full training epoch with DataLoader batching."""
        model = TinyModel()
        module = MarainerLightningModule(model=model, optimizer=Class(torch.optim.SGD, lr=0.01))

        flux = Flux(source=_make_samples(32))
        ds = FluxIterableDataset(flux)
        loader = DataLoader(ds, batch_size=16)

        trainer = L.Trainer(max_epochs=1, accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)
        trainer.fit(model=module, train_dataloaders=loader)


class TestScenarioBWithImages:
    """Same as B but with image-shaped data (CNN-like)."""

    def test_image_dataloader_batching(self) -> None:
        """Verify DataLoader batching works with image tensors."""
        samples = [Sample(input=torch.randn(1, 28, 28), target=i % 10, metadata={}) for i in range(32)]
        flux = Flux(source=samples)
        ds = FluxIterableDataset(flux)
        loader = DataLoader(ds, batch_size=16)

        batch = next(iter(loader))
        assert batch.input.shape == (16, 1, 28, 28)

        module = MarainerLightningModule(
            model=nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10)),
            optimizer=Class(torch.optim.SGD, lr=0.01),
        )
        data, target = module._unpack_batch(batch)
        assert data.shape == (16, 1, 28, 28)
        assert target.shape == (16,)


class TestFluxChunkSizeNone:
    """Verify chunk_size=None (from YAML null) works correctly."""

    def test_none_chunk_size(self) -> None:
        flux = Flux(source=_make_samples(4), chunk_size=None)
        results = list(flux)
        assert len(results) == 4
        assert hasattr(results[0], "input")  # Individual Samples, not lists

    def test_zero_chunk_size(self) -> None:
        flux = Flux(source=_make_samples(4), chunk_size=0)
        results = list(flux)
        assert len(results) == 4
        assert hasattr(results[0], "input")

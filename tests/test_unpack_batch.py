"""Tests for Trainer with DataLoader-collated Samples."""

import torch
from dataflux.core import Flux
from dataflux.sample import Sample
from torch.utils.data import DataLoader

from marainer.trainer import LightningTrainer


def test_training_step_with_collated_sample() -> None:
    trainer = LightningTrainer(model=torch.nn.Linear(4, 2), train_set=[])
    batch = Sample(input=torch.randn(8, 4), target=torch.randint(0, 2, (8,)), metadata={})
    loss = trainer.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_flux_as_map_dataset() -> None:
    samples = [Sample(input=torch.randn(4), target=i % 2, metadata={}) for i in range(16)]
    flux = Flux(source=samples)
    loader: DataLoader = DataLoader(flux, batch_size=8)
    batch = next(iter(loader))
    assert batch.input.shape == (8, 4)
    assert batch.target.shape == (8,)


def _double(x: torch.Tensor) -> torch.Tensor:
    return x * 2


def test_flux_with_ops_as_map_dataset() -> None:
    samples = [Sample(input=torch.ones(4), target=0, metadata={}) for _ in range(8)]
    flux = Flux(source=samples).map(_double)
    loader: DataLoader = DataLoader(flux, batch_size=4)
    batch = next(iter(loader))
    assert torch.allclose(batch.input, torch.full((4, 4), 2.0))

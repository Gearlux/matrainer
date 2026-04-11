"""Integration tests for Trainer with map-style Flux."""

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class, configurable
from dataflux.core import Flux
from dataflux.sample import Sample
from torch.utils.data import DataLoader

from marainer.trainer import Trainer


@configurable
class TinyModel(nn.Module):
    def __init__(self, input_size: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # type: ignore[no-any-return]


def _make_samples(n: int = 32, input_size: int = 4) -> list[Sample]:
    return [Sample(input=torch.randn(input_size), target=i % 2, metadata={}) for i in range(n)]


def _negate(x: torch.Tensor) -> torch.Tensor:
    return -x


class TestMapStyleFlux:
    def test_dataloader_batches_flux(self) -> None:
        flux = Flux(source=_make_samples(32))
        loader: DataLoader = DataLoader(flux, batch_size=8)  # type: ignore[arg-type]
        batches = list(loader)
        assert len(batches) == 4
        assert batches[0].input.shape == (8, 4)

    def test_training_step(self) -> None:
        trainer = Trainer(model=TinyModel(), train_set=_make_samples(16))
        flux = Flux(source=_make_samples(16))
        loader: DataLoader = DataLoader(flux, batch_size=8)  # type: ignore[arg-type]
        loss = trainer.training_step(next(iter(loader)), 0)
        assert loss.dim() == 0

    def test_full_epoch(self) -> None:
        trainer = Trainer(
            model=TinyModel(),
            train_set=_make_samples(32),
            optimizer=Class(torch.optim.SGD, lr=0.01),
        )
        flux = Flux(source=_make_samples(32))
        loader: DataLoader = DataLoader(flux, batch_size=16)  # type: ignore[arg-type]
        lt = L.Trainer(max_epochs=1, accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)
        lt.fit(model=trainer, train_dataloaders=loader)

    def test_with_num_workers(self) -> None:
        flux = Flux(source=_make_samples(32))
        loader: DataLoader = DataLoader(flux, batch_size=16, num_workers=2)  # type: ignore[arg-type]
        assert len(list(loader)) == 2


class TestFluxGetitem:
    def test_returns_sample(self) -> None:
        assert isinstance(Flux(source=_make_samples(4))[0], Sample)

    def test_with_ops(self) -> None:
        samples = [Sample(input=torch.ones(2), target=0, metadata={}) for _ in range(4)]
        assert torch.allclose(Flux(source=samples).map(_negate)[0].input, -torch.ones(2))

    def test_len(self) -> None:
        assert len(Flux(source=_make_samples(10))) == 10

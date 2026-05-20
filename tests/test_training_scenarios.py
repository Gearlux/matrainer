"""Integration tests for Trainer with map-style Flux."""

from typing import Any, cast

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class, configurable
from dataflux.core import Flux
from dataflux.sample import Sample
from torch.utils.data import DataLoader

from marainer.trainer import LightningTrainer


@configurable
class TinyModel(nn.Module):
    def __init__(self, input_size: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.fc(x))


def _make_samples(n: int = 32, input_size: int = 4) -> list[Sample]:
    return [Sample(input=torch.randn(input_size), target=i % 2, metadata={}) for i in range(n)]


def _negate(x: torch.Tensor) -> torch.Tensor:
    return -x


class TestMapStyleFlux:
    def test_dataloader_batches_flux(self) -> None:
        flux = Flux(source=_make_samples(32))
        loader: DataLoader = DataLoader(flux, batch_size=8)
        batches = list(loader)
        assert len(batches) == 4
        assert batches[0].input.shape == (8, 4)

    def test_training_step(self) -> None:
        trainer = LightningTrainer(model=TinyModel(), train_set=_make_samples(16))
        flux = Flux(source=_make_samples(16))
        loader: DataLoader = DataLoader(flux, batch_size=8)
        loss = trainer.training_step(next(iter(loader)), 0)
        assert loss.dim() == 0

    def test_full_epoch(self) -> None:
        trainer = LightningTrainer(
            model=TinyModel(),
            train_set=_make_samples(32),
            optimizer=Class(torch.optim.SGD, lr=0.01),
        )
        flux = Flux(source=_make_samples(32))
        loader: DataLoader = DataLoader(flux, batch_size=16)
        lt = L.Trainer(max_epochs=1, accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)
        lt.fit(model=trainer, train_dataloaders=loader)

    def test_with_num_workers(self) -> None:
        flux = Flux(source=_make_samples(32))
        loader: DataLoader = DataLoader(flux, batch_size=16, num_workers=2)
        assert len(list(loader)) == 2


class TestFluxGetitem:
    def test_returns_sample(self) -> None:
        assert isinstance(Flux(source=_make_samples(4))[0], Sample)

    def test_with_ops(self) -> None:
        samples = [Sample(input=torch.ones(2), target=0, metadata={}) for _ in range(4)]
        assert torch.allclose(Flux(source=samples).map(_negate)[0].input, -torch.ones(2))

    def test_len(self) -> None:
        assert len(Flux(source=_make_samples(10))) == 10


class TestTestSet:
    """Verify that an optional test_set triggers lightning_trainer.test() after fit()."""

    def test_run_invokes_test_when_test_set_provided(self) -> None:
        """When test_set is set, LightningTrainer.run() should call lightning_trainer.test()."""
        trainer = LightningTrainer(
            model=TinyModel(),
            train_set=Flux(source=_make_samples(16)),
        )
        trainer.val_set = Flux(source=_make_samples(8))
        trainer.test_set = Flux(source=_make_samples(8))
        trainer.lightning = Class(
            L.Trainer,
            max_epochs=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        trainer.train_loader = Class(DataLoader, batch_size=8)
        trainer.val_loader = Class(DataLoader, batch_size=8)
        trainer.test_loader = Class(DataLoader, batch_size=8)

        # Spy on Lightning's test() method.
        original_test = L.Trainer.test
        call_counter = {"count": 0}

        def _spy(self: L.Trainer, *args: Any, **kwargs: Any) -> Any:
            call_counter["count"] += 1
            return original_test(self, *args, **kwargs)

        L.Trainer.test = _spy  # type: ignore[method-assign]
        try:
            trainer.run()
        finally:
            L.Trainer.test = original_test  # type: ignore[method-assign]

        assert call_counter["count"] == 1
        logged = trainer.trainer.callback_metrics
        assert any(k.startswith("test/") for k in logged), f"expected test/* metrics, got: {list(logged)}"

    def test_run_skips_test_when_test_set_absent(self) -> None:
        trainer = LightningTrainer(
            model=TinyModel(),
            train_set=Flux(source=_make_samples(16)),
        )
        trainer.lightning = Class(
            L.Trainer,
            max_epochs=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        trainer.train_loader = Class(DataLoader, batch_size=8)

        original_test = L.Trainer.test
        call_counter = {"count": 0}

        def _spy(self: L.Trainer, *args: Any, **kwargs: Any) -> Any:
            call_counter["count"] += 1
            return original_test(self, *args, **kwargs)

        L.Trainer.test = _spy  # type: ignore[method-assign]
        try:
            trainer.run()
        finally:
            L.Trainer.test = original_test  # type: ignore[method-assign]

        assert call_counter["count"] == 0

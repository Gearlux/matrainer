"""Integration tests for Trainer with map-style Flux."""

from typing import Any, cast

import pytest
import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class, configurable, output_specs
from sampleflux.core import Flux
from sampleflux.sample import Sample
from torch.utils.data import DataLoader

from marainer.trainer import LightningTrainer


@configurable
class TinyModel(nn.Module):
    def __init__(self, input_size: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.fc(x))


def _make_samples(n: int = 32, input_size: int = 4) -> Flux:
    return Flux(source=[Sample(input=torch.randn(input_size), target=i % 2, metadata={}) for i in range(n)])


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
        trainer = LightningTrainer(model=TinyModel(), train_set=_make_samples(32))
        trainer.optimizer = Class(torch.optim.SGD, lr=0.01)
        flux = Flux(source=_make_samples(32))
        loader: DataLoader = DataLoader(flux, batch_size=16)
        lt = L.Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        lt.fit(model=trainer, train_dataloaders=loader)

    def test_with_num_workers(self) -> None:
        flux = Flux(source=_make_samples(32))
        loader: DataLoader = DataLoader(flux, batch_size=16, num_workers=2)
        assert len(list(loader)) == 2


class TestProgressCallback:
    """The comfy-free progress callback fires through a REAL Lightning fit (not a faked trainer).

    Verifies the whole marainer-side wiring end-to-end: ``run()`` attaches the
    ``LightningProgressCallback`` after ``flow(self.lightning)``, Lightning's real per-batch hook
    drives it, and the executor's sink receives a monotonic ``global_step`` against the constant
    ``estimated_stepping_batches`` total — the whole-run bar mapping.
    """

    def test_progress_callback_fires_during_real_fit(self) -> None:
        from confluid import LazyClass

        reports: list = []
        trainer = LightningTrainer(model=TinyModel(), train_set=_make_samples(16))
        trainer.optimizer = Class(torch.optim.SGD, lr=0.01)
        trainer.train_loader = LazyClass(DataLoader, batch_size=8)  # 16 / 8 = 2 batches/epoch
        trainer.lightning = LazyClass(
            L.Trainer,
            max_epochs=2,  # 2 batches/epoch * 2 epochs = 4 optimizer steps total
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
        )
        trainer.set_progress_callback(lambda value, total, desc: reports.append((value, total, desc)))
        trainer.run()

        assert reports, "no progress was reported during a real fit"
        assert {desc for *_, desc in reports} == {"Training"}
        # estimated_stepping_batches is constant across the whole run (4 steps); global_step is the
        # monotonic value, reaching the total on the final batch.
        assert {total for _, total, _ in reports} == {4.0}
        values = [value for value, _, _ in reports]
        assert values == sorted(values)
        assert max(values) == 4.0


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


class _LabeledFlux(Flux):
    """A Flux exposing the informal ``label_names`` protocol the trainer surfaces."""

    label_names = ("neg", "pos")


def _make_runnable_trainer(train_set: Flux) -> LightningTrainer:
    trainer = LightningTrainer(model=TinyModel(), train_set=train_set)
    trainer.lightning = Class(
        L.Trainer,
        max_epochs=1,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.train_loader = Class(DataLoader, batch_size=8)
    return trainer


class TestRunOutputs:
    """The post-run ``@output`` I/O contract surfaced by FluxStudio / navigaitor."""

    def test_declares_computed_metrics_and_label_names_outputs(self) -> None:
        names = {s["name"] for s in output_specs(LightningTrainer)}
        assert {"computed_metrics", "label_names"} <= names
        # @output marks the underlying getter, not the property object.
        for name in ("computed_metrics", "label_names"):
            fget = getattr(LightningTrainer, name).fget
            assert getattr(fget, "__confluid_output__", False) is True

    def test_computed_metrics_raises_before_run(self) -> None:
        trainer = LightningTrainer(model=TinyModel(), train_set=Flux(source=_make_samples(8)))
        with pytest.raises(RuntimeError, match="only available after run"):
            _ = trainer.computed_metrics

    def test_label_names_raises_before_run(self) -> None:
        trainer = LightningTrainer(model=TinyModel(), train_set=Flux(source=_make_samples(8)))
        with pytest.raises(RuntimeError, match="only available after run"):
            _ = trainer.label_names

    def test_computed_metrics_after_run_returns_scalar_floats(self) -> None:
        trainer = _make_runnable_trainer(Flux(source=_make_samples(16)))
        trainer.run()
        metrics = trainer.computed_metrics
        assert isinstance(metrics, dict) and metrics, "expected non-empty scalar metrics"
        assert "train/loss" in metrics  # the base logs train/loss every run
        assert all(isinstance(v, float) for v in metrics.values()), metrics

    def test_label_names_after_run_from_dataset(self) -> None:
        trainer = _make_runnable_trainer(_LabeledFlux(source=_make_samples(16)))
        trainer.run()
        assert trainer.label_names == ["neg", "pos"]

    def test_label_names_after_run_empty_when_absent(self) -> None:
        trainer = _make_runnable_trainer(Flux(source=_make_samples(16)))  # no label_names
        trainer.run()
        assert trainer.label_names == []

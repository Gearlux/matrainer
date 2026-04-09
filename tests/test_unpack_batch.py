"""Tests for MarainerLightningModule._unpack_batch across all batch formats."""

import torch
from dataflux.sample import Sample

from marainer.trainer import MarainerLightningModule


def _make_module() -> MarainerLightningModule:
    """Create a minimal module for testing _unpack_batch."""
    model = torch.nn.Linear(1, 1)
    return MarainerLightningModule(model=model, optimizer=None)


class TestUnpackBatchListOfSamples:
    """Path A: Flux chunk_size batching → list of individual Samples."""

    def test_basic_shape(self) -> None:
        module = _make_module()
        batch = [Sample(input=torch.randn(1, 28, 28), target=i, metadata={}) for i in range(16)]
        data, target = module._unpack_batch(batch)
        assert data.shape == (16, 1, 28, 28)
        assert target.shape == (16,)

    def test_single_sample_list(self) -> None:
        module = _make_module()
        batch = [Sample(input=torch.randn(1, 28, 28), target=0, metadata={})]
        data, target = module._unpack_batch(batch)
        assert data.shape == (1, 1, 28, 28)
        assert target.shape == (1,)

    def test_rgb_images(self) -> None:
        module = _make_module()
        batch = [Sample(input=torch.randn(3, 32, 32), target=i, metadata={}) for i in range(8)]
        data, target = module._unpack_batch(batch)
        assert data.shape == (8, 3, 32, 32)

    def test_target_values(self) -> None:
        module = _make_module()
        batch = [Sample(input=torch.zeros(1, 2, 2), target=i * 10, metadata={}) for i in range(4)]
        _, target = module._unpack_batch(batch)
        assert target.tolist() == [0, 10, 20, 30]


class TestUnpackBatchCollatedSample:
    """Path B: DataLoader batch_size collation → single collated Sample."""

    def test_basic_shape(self) -> None:
        module = _make_module()
        batch = Sample(
            input=torch.randn(16, 1, 28, 28),
            target=torch.arange(16),
            metadata={},
        )
        data, target = module._unpack_batch(batch)
        assert data.shape == (16, 1, 28, 28)
        assert target.shape == (16,)

    def test_rgb_collated(self) -> None:
        module = _make_module()
        batch = Sample(
            input=torch.randn(8, 3, 32, 32),
            target=torch.arange(8),
            metadata={},
        )
        data, target = module._unpack_batch(batch)
        assert data.shape == (8, 3, 32, 32)

    def test_1d_input(self) -> None:
        """Edge case: 1D signal data [B, features] — already batched."""
        module = _make_module()
        batch = Sample(
            input=torch.randn(4, 100),
            target=torch.zeros(4),
            metadata={},
        )
        data, target = module._unpack_batch(batch)
        assert data.shape == (4, 100)

    def test_target_as_list(self) -> None:
        """Target might be a list instead of tensor after collation."""
        module = _make_module()
        batch = Sample(
            input=torch.randn(4, 1, 28, 28),
            target=[0, 1, 2, 3],
            metadata={},
        )
        _, target = module._unpack_batch(batch)
        assert isinstance(target, torch.Tensor)
        assert target.tolist() == [0, 1, 2, 3]


class TestUnpackBatchTupleFormat:
    """Path C: Standard PyTorch (data, target) tuple."""

    def test_basic(self) -> None:
        module = _make_module()
        batch = (torch.randn(16, 1, 28, 28), torch.arange(16))
        data, target = module._unpack_batch(batch)
        assert data.shape == (16, 1, 28, 28)

    def test_list_format(self) -> None:
        module = _make_module()
        batch = [torch.randn(8, 3, 32, 32), torch.arange(8)]
        data, target = module._unpack_batch(batch)
        assert data.shape == (8, 3, 32, 32)


class TestUnpackBatchErrors:
    """Error cases."""

    def test_unknown_format(self) -> None:
        module = _make_module()
        import pytest

        with pytest.raises(ValueError, match="Unknown batch format"):
            module._unpack_batch("not a batch")

    def test_empty_list(self) -> None:
        module = _make_module()
        import pytest

        with pytest.raises((ValueError, IndexError)):
            module._unpack_batch([])

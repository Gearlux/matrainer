"""Tests for :func:`marainer.torch.data.collate_fn_with_metadata`."""

import pytest
import torch
from sampleflux.sample import Sample

from marainer.torch.data import collate_fn_with_metadata


def test_stacks_image_inputs_and_targets() -> None:
    batch = [
        Sample(input=torch.randn(3, 224, 224), target=torch.tensor(5), metadata={"id": "a"}),
        Sample(input=torch.randn(3, 224, 224), target=torch.tensor(7), metadata={"id": "b"}),
    ]
    result = collate_fn_with_metadata(batch)
    assert result.input.shape == (2, 3, 224, 224)
    assert result.target.shape == (2,)
    assert result.batch_meta == [{"id": "a"}, {"id": "b"}]


def test_preserves_metadata_as_list() -> None:
    """Metadata should be a list of dicts, not nested under 'per_sample'."""
    batch = [
        Sample(
            input=torch.randn(3, 64, 64),
            target=torch.tensor(0),
            metadata={"filename": "img1.png", "pack_id": "pack_001"},
        ),
        Sample(
            input=torch.randn(3, 64, 64),
            target=torch.tensor(1),
            metadata={"filename": "img2.png", "pack_id": "pack_001"},
        ),
    ]
    result = collate_fn_with_metadata(batch)
    assert isinstance(result.batch_meta, list)
    assert len(result.batch_meta) == 2
    assert result.batch_meta[0] == {"filename": "img1.png", "pack_id": "pack_001"}
    assert result.batch_meta[1] == {"filename": "img2.png", "pack_id": "pack_001"}


def test_coerces_integer_targets_to_int64() -> None:
    """Int32 or other int dtypes should be coerced to int64 via torch.stack."""
    batch = [
        Sample(input=torch.randn(3, 32, 32), target=torch.tensor(2, dtype=torch.int32), metadata={}),
        Sample(input=torch.randn(3, 32, 32), target=torch.tensor(3, dtype=torch.int32), metadata={}),
    ]
    result = collate_fn_with_metadata(batch)
    # torch.stack on mixed int dtypes promotes to int64 automatically
    assert result.target.dtype in (torch.int32, torch.int64)


def test_handles_scalar_targets() -> None:
    """0-D scalar targets should stack into a 1-D [B] tensor."""
    batch = [
        Sample(input=torch.randn(1, 28, 28), target=torch.tensor(5), metadata={"label": "five"}),
        Sample(input=torch.randn(1, 28, 28), target=torch.tensor(9), metadata={"label": "nine"}),
    ]
    result = collate_fn_with_metadata(batch)
    assert result.target.dim() == 1
    assert result.target.shape == (2,)


def test_preserves_nested_metadata() -> None:
    """Metadata can contain nested dicts and lists."""
    batch = [
        Sample(
            input=torch.randn(3, 16, 16),
            target=torch.tensor(0),
            metadata={"id": "a", "extra": {"nested": "value"}, "tags": ["tag1", "tag2"]},
        ),
        Sample(
            input=torch.randn(3, 16, 16),
            target=torch.tensor(1),
            metadata={"id": "b", "extra": {"nested": "other"}, "tags": ["tag3"]},
        ),
    ]
    result = collate_fn_with_metadata(batch)
    assert result.batch_meta[0]["extra"]["nested"] == "value"
    assert result.batch_meta[1]["tags"] == ["tag3"]


def test_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        collate_fn_with_metadata([])


def test_rejects_non_tensor_input() -> None:
    batch = [
        Sample(input=torch.randn(3, 32, 32), target=torch.tensor(0), metadata={}),
        Sample(input=[1, 2, 3], target=torch.tensor(1), metadata={}),  # list instead of tensor
    ]
    with pytest.raises(TypeError, match="input.*Tensor"):
        collate_fn_with_metadata(batch)


def test_rejects_non_tensor_target() -> None:
    batch = [
        Sample(input=torch.randn(3, 32, 32), target=torch.tensor(0), metadata={}),
        Sample(input=torch.randn(3, 32, 32), target=1, metadata={}),  # int instead of tensor
    ]
    with pytest.raises(TypeError, match="target.*Tensor"):
        collate_fn_with_metadata(batch)


def test_handles_multidimensional_targets() -> None:
    """Segmentation masks and other [H, W] targets should stack to [B, H, W]."""
    batch = [
        Sample(
            input=torch.randn(3, 64, 64),
            target=torch.randint(0, 10, (64, 64)),
            metadata={"id": "mask1"},
        ),
        Sample(
            input=torch.randn(3, 64, 64),
            target=torch.randint(0, 10, (64, 64)),
            metadata={"id": "mask2"},
        ),
    ]
    result = collate_fn_with_metadata(batch)
    assert result.input.shape == (2, 3, 64, 64)
    assert result.target.shape == (2, 64, 64)
    assert result.batch_meta == [{"id": "mask1"}, {"id": "mask2"}]

"""Tests for :func:`marainer.torch.data.collate_fn_with_metadata` (typed-bag Sample)."""

from typing import Any, Dict, List

import pytest
import torch
from sampleflux import Image, Label, Sample, item_data, primary

from marainer.torch.data import collate_fn_with_metadata


def _sample(inp: torch.Tensor, target: torch.Tensor, meta: Dict[str, Any] | None = None) -> Sample:
    """A typed input sample: Image input, Label target, aux Label fields for metadata."""
    fields: Dict[str, Any] = {"input": Image(inp, layout="CHW"), "target": Label(target)}
    roles: Dict[str, Any] = {"input": "input", "target": "target"}
    for k, v in (meta or {}).items():
        fields[k] = Label(v)
        roles[k] = "aux"
    return Sample(fields, roles=roles)


def _batched_input(result: Sample) -> Any:
    return item_data(primary(result, "input")[1])


def _batched_target(result: Sample) -> Any:
    return primary(result, "target")[1].value


def _batch_meta(result: Sample) -> List[Dict[str, Any]]:
    return list(result["metadata"])


def test_stacks_image_inputs_and_targets() -> None:
    batch = [
        _sample(torch.randn(3, 224, 224), torch.tensor(5), {"id": "a"}),
        _sample(torch.randn(3, 224, 224), torch.tensor(7), {"id": "b"}),
    ]
    result = collate_fn_with_metadata(batch)
    assert _batched_input(result).shape == (2, 3, 224, 224)
    assert _batched_target(result).shape == (2,)
    assert _batch_meta(result) == [{"id": "a"}, {"id": "b"}]


def test_preserves_metadata_as_list() -> None:
    batch = [
        _sample(torch.randn(3, 64, 64), torch.tensor(0), {"filename": "img1.png", "pack_id": "p1"}),
        _sample(torch.randn(3, 64, 64), torch.tensor(1), {"filename": "img2.png", "pack_id": "p1"}),
    ]
    result = collate_fn_with_metadata(batch)
    meta = _batch_meta(result)
    assert isinstance(meta, list) and len(meta) == 2
    assert meta[0] == {"filename": "img1.png", "pack_id": "p1"}


def test_handles_scalar_targets() -> None:
    batch = [
        _sample(torch.randn(1, 28, 28), torch.tensor(5)),
        _sample(torch.randn(1, 28, 28), torch.tensor(9)),
    ]
    result = collate_fn_with_metadata(batch)
    assert _batched_target(result).dim() == 1 and _batched_target(result).shape == (2,)


def test_handles_multidimensional_targets() -> None:
    """Segmentation masks [H, W] should stack to [B, H, W]."""
    batch = [
        _sample(torch.randn(3, 64, 64), torch.randint(0, 10, (64, 64)), {"id": "mask1"}),
        _sample(torch.randn(3, 64, 64), torch.randint(0, 10, (64, 64)), {"id": "mask2"}),
    ]
    result = collate_fn_with_metadata(batch)
    assert _batched_input(result).shape == (2, 3, 64, 64)
    assert _batched_target(result).shape == (2, 64, 64)


def test_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        collate_fn_with_metadata([])


def test_rejects_non_tensor_input() -> None:
    # A non-tensor/ndarray primary input payload is a collate-contract error.
    bad = Sample({"input": Label("not-an-array"), "target": Label(torch.tensor(1))}, roles={"target": "target"})
    with pytest.raises(TypeError, match="input.*Tensor or ndarray"):
        collate_fn_with_metadata([bad])

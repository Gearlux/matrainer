"""Tests for :func:`marainer.torch.data.collate_fn_with_metadata` (thin wrapper over typed_collate)."""

from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from sampleflux import Image, Label, Mask, Sample, item_data, primary

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
    """Transpose the aux (non-primary) fields typed_collate leaves as per-sample lists into dicts."""
    input_key = primary(result, "input")[0]
    target_key = primary(result, "target")[0]
    columns = {
        k: (result[k].value if isinstance(result[k], Label) else item_data(result[k]))
        for k in result.keys()
        if k not in (input_key, target_key)
    }
    n = min((len(v) for v in columns.values()), default=0)
    return [{k: columns[k][i] for k in columns} for i in range(n)]


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


def test_stacks_scalar_label_targets() -> None:
    """A scalar Label target (which typed_collate leaves as a list) is stacked to an [N] tensor."""
    batch = [
        _sample(torch.randn(1, 28, 28), torch.tensor(5)),
        _sample(torch.randn(1, 28, 28), torch.tensor(9)),
    ]
    result = collate_fn_with_metadata(batch)
    tgt = _batched_target(result)
    assert isinstance(tgt, torch.Tensor) and tgt.dim() == 1 and tgt.shape == (2,)


def test_stacks_multidimensional_label_targets() -> None:
    """Per-sample [H, W] Label targets stack to [B, H, W]."""
    batch = [
        _sample(torch.randn(3, 64, 64), torch.randint(0, 10, (64, 64)), {"id": "mask1"}),
        _sample(torch.randn(3, 64, 64), torch.randint(0, 10, (64, 64)), {"id": "mask2"}),
    ]
    result = collate_fn_with_metadata(batch)
    assert _batched_input(result).shape == (2, 3, 64, 64)
    assert _batched_target(result).shape == (2, 64, 64)


def test_array_item_target_stacked_by_typed_collate() -> None:
    """An array-item target (Mask) is stacked by typed_collate itself — the wrapper leaves it be."""
    batch = [
        Sample(
            {
                "input": Image(np.zeros((3, 8, 8), dtype="float32"), layout="CHW"),
                "mask": Mask(np.zeros((8, 8), "int64")),
            },
            roles={"mask": "target"},
        )
        for _ in range(3)
    ]
    result = collate_fn_with_metadata(batch)
    _, tgt_item = primary(result, "target")
    assert isinstance(tgt_item, Mask)
    assert item_data(tgt_item).shape == (3, 8, 8)


def test_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        collate_fn_with_metadata([])

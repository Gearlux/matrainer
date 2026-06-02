"""Generic collate functions for DataFlux Samples preserving metadata.

The default PyTorch DataLoader collate fails on heterogeneous per-sample metadata
(strings, variable-length lists, nested dicts). These helpers stack tensors and
preserve metadata as a list of dicts so sinks and metrics can correlate
predictions back to source samples.
"""

from typing import Any, Dict, List

import torch
from dataflux.sample import Sample
from torch import Tensor


def collate_fn_with_metadata(batch: List[Sample]) -> Sample:
    """Collate per-sample Samples, stacking tensors and preserving metadata as a list.

    Each input :class:`~dataflux.sample.Sample` is expected to carry:

    * ``sample.input`` — A tensor or scalar (typically ``[C, H, W]`` for images,
      ``[T]`` or ``[F, T]`` for signals, or a scalar).
    * ``sample.target`` — A tensor or scalar (typically ``[H, W]`` for segmentation
      masks, a scalar for classification/regression, or ``[N, 4]`` for detection boxes).

    The returned :class:`~dataflux.sample.Sample` has:

    * ``.input`` — Stacked input tensors (``[B, C, H, W]`` for images, etc.).
    * ``.target`` — Stacked target tensors (``[B, ...]`` matching input batch dim).
    * ``.metadata`` — A list of per-sample metadata dicts (NOT nested under a
      ``"per_sample"`` key), so string-valued fields (``pack_id``, ``iq_file``,
      ``drone``, ``filename``) survive collation intact.

    All per-sample input/target tensors must already share compatible shapes for
    stacking (same spatial dims for images, same time/freq dims for signals, etc.);
    mixed sizes must be resized upstream.

    Args:
        batch: List of :class:`~dataflux.sample.Sample` objects.

    Returns:
        A single batched Sample with stacked tensors and metadata preserved as a list.

    Raises:
        ValueError: If batch is empty or input/target shapes are incompatible.
        TypeError: If input or target is not a Tensor.
    """
    if not batch:
        raise ValueError("collate_fn_with_metadata received an empty batch.")

    inputs: List[Tensor] = []
    targets: List[Tensor] = []
    metas: List[Dict[str, Any]] = []

    for s in batch:
        if not isinstance(s.input, Tensor):
            raise TypeError(
                f"collate_fn_with_metadata expects sample.input to be a torch.Tensor; " f"got {type(s.input).__name__}."
            )
        if not isinstance(s.target, Tensor):
            raise TypeError(
                f"collate_fn_with_metadata expects sample.target to be a torch.Tensor; "
                f"got {type(s.target).__name__}."
            )
        inputs.append(s.input)
        targets.append(s.target)
        metas.append(s.metadata)

    return Sample(
        input=torch.stack(inputs),
        target=torch.stack(targets),
        metadata=metas,
    )


__all__ = ["collate_fn_with_metadata"]

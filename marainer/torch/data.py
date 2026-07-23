"""Dataset normalization + metadata-preserving collate for torch training.

* :func:`ensure_sample_dataset` — normalize any wired source into a map-style
  ``Dataset`` guaranteed to yield ``Sample``s, so the rest of a training pipeline
  can assume ``Sample`` items unconditionally.
* :func:`collate_fn_with_metadata` — stack per-sample tensors and preserve
  metadata as a list of dicts (the default DataLoader collate fails on
  heterogeneous per-sample metadata).

Both import the sampleflux surface from the package top level.
"""

from typing import Any, List, cast

import torch
from sampleflux import Flux, Label, Sample, primary, typed_collate
from torch import Tensor
from torch.utils.data import Dataset


def ensure_sample_dataset(source: Any) -> Dataset[Any]:
    """Normalize any wired source into a map-style ``Dataset`` that yields ``Sample``s.

    A wired ``train_set`` / ``val_set`` / ``test_set`` may be a ``sampleflux.Flux``,
    another ``torch`` ``Dataset``, a sampleflux source (``HuggingFaceSource``), or a
    plain list — and its items may be ``Sample``s or raw rows. A ``Flux`` already
    coerces every item to a ``Sample``, so:

    * a ``Flux`` is returned as-is (already a ``Dataset[Sample]``; this preserves a
      subclass's own wrap, e.g. a label-encoding Flux with its ``label_names``), and
    * anything else is wrapped in a ``Flux``, which makes it both a map-style
      ``Dataset`` AND a ``Sample``-yielding one.

    Calling this once up front lets the rest of the pipeline (target detection,
    label fitting/encoding, collate, metrics) assume ``Sample`` items — no per-call
    "is this a Sample?" checks.
    """
    if isinstance(source, Flux):
        return source
    return Flux(source=source)


def collate_fn_with_metadata(batch: List[Sample]) -> Sample:
    """Collate typed-bag ``Sample``\\ s via :func:`sampleflux.typed_collate`, stacking the target tensor.

    Thin wrapper over the workspace's generic ``typed_collate`` — which stacks each field's array
    payload and gathers wrapper-item attrs / aux (non-primary) fields as per-sample lists. The one thing
    it does NOT do for a training batch is stack a wrapper-item :class:`~sampleflux.Label` **target**'s
    per-sample values into an ``[N, …]`` tensor (an array-item target, e.g. a :class:`~sampleflux.Mask`,
    IS stacked by ``typed_collate``); this wrapper adds exactly that so a torch trainer sees a batched
    target tensor. Aux fields survive as their own per-sample list-fields (``typed_collate``'s metadata
    shape) — recover them at the model boundary.

    Args:
        batch: List of typed-bag :class:`~sampleflux.Sample` objects.

    Returns:
        A single batched ``Sample`` (stacked input + stacked target + per-sample aux list-fields).

    Raises:
        ValueError: If ``batch`` is empty.
    """
    if not batch:
        raise ValueError("collate_fn_with_metadata received an empty batch.")
    collated = cast(Sample, typed_collate(batch))
    key, item = primary(collated, "target")
    if isinstance(item, Label) and isinstance(item.value, list):
        stacked = torch.stack([_as_tensor(v) for v in item.value])
        collated = collated.replace_field(key, Label(stacked))
    return collated


def _as_tensor(value: Any) -> Tensor:
    """Normalize a per-sample target value (tensor OR ndarray/scalar) to a ``torch.Tensor``."""
    return value if isinstance(value, Tensor) else torch.as_tensor(value)


__all__ = ["collate_fn_with_metadata", "ensure_sample_dataset"]

"""Dataset normalization + metadata-preserving collate for torch training.

* :func:`ensure_record_dataset` — normalize any wired source into a map-style
  ``Dataset`` guaranteed to yield record dicts, so the rest of a training
  pipeline can assume record items unconditionally.
* :func:`collate_fn_with_metadata` — stack per-record tensors and preserve
  metadata as a list of dicts (the default DataLoader collate fails on
  heterogeneous per-record metadata).

Both import the recordstream surface from the package top level.
"""

from typing import Any, List, cast

import torch
from recordstream import Label, Record, Stream, collate_records
from torch import Tensor
from torch.utils.data import Dataset


def ensure_record_dataset(source: Any) -> Dataset[Any]:
    """Normalize any wired source into a map-style ``Dataset`` that yields record dicts.

    A wired ``train_set`` / ``val_set`` / ``test_set`` may be a ``recordstream.Stream``,
    another ``torch`` ``Dataset``, a recordstream source (``HuggingFaceSource``), or a
    plain list — and its items may be record dicts or raw rows. A ``Stream`` already
    coerces every item to a record, so:

    * a ``Stream`` is returned as-is (already a ``Dataset`` of records; this preserves a
      subclass's own wrap, e.g. a label-encoding Stream with its ``label_names``), and
    * anything else is wrapped in a ``Stream``, which makes it both a map-style
      ``Dataset`` AND a record-yielding one.

    Calling this once up front lets the rest of the pipeline (target detection,
    label fitting/encoding, collate, metrics) assume record items — no per-call
    "is this a record?" checks.
    """
    if isinstance(source, Stream):
        return source
    return Stream(source=source)


def collate_fn_with_metadata(batch: List[Record], target_key: str = "class") -> Record:
    """Collate record dicts via :func:`recordstream.collate_records`, stacking the target tensor.

    Thin wrapper over the workspace's generic ``collate_records`` — which stacks each key's array
    payload and gathers wrapper-item attrs / plain values as per-record lists. The one thing it
    does NOT do for a training batch is stack a wrapper-item :class:`~recordstream.Label` **target**'s
    per-record values into an ``[N, …]`` tensor (an array-item target, e.g. a :class:`~recordstream.Mask`,
    IS stacked by ``collate_records``); this wrapper adds exactly that so a torch trainer sees a batched
    target tensor. The remaining keys survive as their own per-record list values (``collate_records``'s
    metadata shape) — recover them at the model boundary.

    Args:
        batch: List of record dicts.
        target_key: Record key carrying the classification target (default ``"class"``).

    Returns:
        A single batched record (stacked input + stacked target + per-record list values).

    Raises:
        ValueError: If ``batch`` is empty.
    """
    if not batch:
        raise ValueError("collate_fn_with_metadata received an empty batch.")
    collated = cast(Record, collate_records(batch))
    item = collated[target_key]
    if isinstance(item, Label) and isinstance(item.value, list):
        stacked = torch.stack([_as_tensor(v) for v in item.value])
        collated = {**collated, target_key: Label(stacked)}
    return collated


def _as_tensor(value: Any) -> Tensor:
    """Normalize a per-record target value (tensor OR ndarray/scalar) to a ``torch.Tensor``."""
    return value if isinstance(value, Tensor) else torch.as_tensor(value)


__all__ = ["collate_fn_with_metadata", "ensure_record_dataset"]

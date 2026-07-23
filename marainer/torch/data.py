"""Dataset normalization + metadata-preserving collate for torch training.

* :func:`ensure_sample_dataset` — normalize any wired source into a map-style
  ``Dataset`` guaranteed to yield ``Sample``s, so the rest of a training pipeline
  can assume ``Sample`` items unconditionally.
* :func:`collate_fn_with_metadata` — stack per-sample tensors and preserve
  metadata as a list of dicts (the default DataLoader collate fails on
  heterogeneous per-sample metadata).

Both import the sampleflux surface from the package top level.
"""

from typing import Any, Dict, List

import torch
from sampleflux import Flux, Sample
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
    """Collate per-sample Samples, stacking tensors and preserving metadata as a list.

    Each input :class:`~sampleflux.Sample` carries ``sample.input`` (a tensor) and
    ``sample.target`` (a tensor). The returned batched Sample has stacked
    ``.input`` / ``.target`` and ``.metadata`` as a list of per-sample dicts (NOT
    nested under a ``"per_sample"`` key), so string-valued fields survive collation.

    Args:
        batch: List of :class:`~sampleflux.Sample` objects.

    Returns:
        A single batched Sample with stacked tensors and metadata preserved as a list.

    Raises:
        ValueError: If batch is empty.
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
                f"collate_fn_with_metadata expects sample.input to be a torch.Tensor; got {type(s.input).__name__}."
            )
        if not isinstance(s.target, Tensor):
            raise TypeError(
                f"collate_fn_with_metadata expects sample.target to be a torch.Tensor; got {type(s.target).__name__}."
            )
        inputs.append(s.input)
        targets.append(s.target)
        metas.append(s.meta)  # per-sample dict (the batched output below carries the list)

    return Sample(input=torch.stack(inputs), target=torch.stack(targets), metadata=metas)


__all__ = ["collate_fn_with_metadata", "ensure_sample_dataset"]

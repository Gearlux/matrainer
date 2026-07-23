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
from sampleflux import Flux, Image, Label, Sample, item_data, primary
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


def _as_tensor(value: Any, what: str) -> Tensor:
    """Normalize a primary field payload (tensor OR ndarray) to a ``torch.Tensor``."""
    if isinstance(value, Tensor):
        return value
    try:
        return torch.as_tensor(value)
    except Exception as exc:  # noqa: BLE001 — surface a clear collate-contract error
        raise TypeError(
            f"collate_fn_with_metadata: the primary {what} payload must be a torch.Tensor or ndarray; "
            f"got {type(value).__name__}."
        ) from exc


def _sample_metadata(s: Sample) -> Dict[str, Any]:
    """Per-sample metadata dict = the non-primary (aux) fields' payloads, so string-valued
    fields (``pack_id``/``iq_file``/…) survive collation."""
    meta: Dict[str, Any] = {}
    _, input_item = primary(s, "input")
    _, target_item = primary(s, "target")
    for key in s.keys():
        item = s[key]
        if item is input_item or item is target_item:
            continue
        meta[key] = item.value if isinstance(item, Label) else item_data(item)
    return meta


def collate_fn_with_metadata(batch: List[Sample]) -> Sample:
    """Collate per-sample typed-bag ``Sample``\\ s, stacking tensors and preserving metadata as a list.

    Each ``Sample`` carries a primary ``input`` item (tensor payload) and a primary ``target``
    item (tensor payload / :class:`~sampleflux.Label`). The returned batched ``Sample`` carries
    field ``"input"`` — an :class:`~sampleflux.Image` (``layout="CHW"``) over the stacked ``[N, …]``
    payload (an ``Image`` cannot hold a live ``torch.Tensor`` — ``NDArrayItem.__new__`` runs
    ``np.asarray`` — so the consumer converts back at the model boundary), field ``"target"``
    (role ``target``) — a :class:`~sampleflux.Label` whose ``.value`` is the stacked target tensor,
    and field ``"metadata"`` (role ``aux``) — a ``list`` of per-sample metadata dicts.

    Args:
        batch: List of typed-bag :class:`~sampleflux.Sample` objects.

    Returns:
        A single batched ``Sample`` (stacked input/target + per-sample metadata list).

    Raises:
        ValueError: If batch is empty.
        TypeError: If a sample's primary input or target payload is not a Tensor.
    """
    if not batch:
        raise ValueError("collate_fn_with_metadata received an empty batch.")

    inputs: List[Tensor] = []
    targets: List[Tensor] = []
    metas: List[Dict[str, Any]] = []

    for s in batch:
        _, input_item = primary(s, "input")
        _, target_item = primary(s, "target")
        raw_inp = item_data(input_item)
        raw_tgt = target_item.value if isinstance(target_item, Label) else item_data(target_item)
        # An Image item can't hold a live tensor (NDArrayItem coerces to ndarray), so accept
        # tensor OR ndarray and normalize to a tensor. Anything else is a real error.
        inp = _as_tensor(raw_inp, "input")
        tgt = _as_tensor(raw_tgt, "target")
        inputs.append(inp)
        targets.append(tgt)
        metas.append(_sample_metadata(s))

    return Sample(
        {
            "input": Image(torch.stack(inputs), layout="CHW"),
            "target": Label(torch.stack(targets)),
            "metadata": metas,
        },
        roles={"input": "input", "target": "target", "metadata": "aux"},
    )


__all__ = ["collate_fn_with_metadata", "ensure_sample_dataset"]

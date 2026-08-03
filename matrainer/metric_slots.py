"""How a per-split metrics slot combines with the shared one — the ONE rule, engine- and task-free.

A training runnable typically takes ``metrics`` (what EVERY split measures) plus
``train_metrics`` / ``val_metrics`` / ``test_metrics`` (what THAT split measures on top), so an
expensive or split-specific metric — a confusion matrix, a per-class IoU — can be added to
validation alone without restating the shared set beside it.

What a "metric" *is* differs per engine: a mapping of torchmetrics templates in the torch
backends, a list of ``keras.metrics`` objects in the Keras ones, a list of function NAMES in
fastai / JAX / MLX. So the COMBINATION rule lives here and is applied to whatever vocabulary the
engine uses, rather than being rewritten per backend against its own value type.

**Why matrainer and not a training project.** It is the metrics vocabulary, which is what this
package already owns — ``build_metric_collection`` / ``scalar_metrics`` (in
:mod:`matrainer.torch`) and ``write_metrics_sidecar`` (in :mod:`matrainer.artifacts`) — and a
second task, on the same eight engines, would otherwise copy it verbatim. It sits at the TOP
LEVEL rather than under :mod:`matrainer.torch` deliberately: that subpackage requires torch,
and the whole point of this module is that it imports **nothing but the standard library**, so
the deliberately framework-free backends (JAX, MLX, Keras, and any hand-written loop) can use
it. Nothing here touches a tracker, so it is not re-exported from the package root, which is
the *tracking* surface.

Two functions, because engines fall into two groups:

* :func:`merge_metric_specs` — for an engine whose loop the CALLER owns (bare torch, Lightning,
  TensorFlow, JAX, MLX): each split gets its own set.
* :func:`union_metric_specs` — for an engine that computes ONE set for several phases and cannot
  be told to narrow one (``keras.Model.compile``, a ``transformers.Trainer``'s
  ``compute_metrics``). Widening is the honest failure: the metric is computed on a split that did
  not ask for it, which is noise, where dropping it would be a wired metric that silently reports
  nothing.
"""

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["merge_metric_specs", "union_metric_specs"]


def _is_list(value: Any) -> bool:
    """True for an ordered metric SEQUENCE — a string is a single named metric, not a sequence."""
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def merge_metric_specs(shared: Any, split: Any) -> Any:
    """The metric set for ONE split: ``split`` combined with ``shared``.

    Merging rather than replacing is what makes the per-split slots usable: a replace-only slot
    forces ``val_metrics`` to restate every shared metric beside the one being added, and a
    config with the same list written twice drifts.

    Two mappings merge by KEY, so a same-named entry in ``split`` shadows the shared one (that is
    how a split overrides a template rather than adding to it). Two sequences CONCATENATE, since
    an engine-native metric list has no keys to merge on. Anything else — a ready-made
    ``MetricCollection``, a mapping against a list — means ``split`` REPLACES: there is no
    key-wise way to combine them, and silently ignoring the split's own spec would be worse.

    ``None`` on either side means "not chosen", so the other one stands.
    """
    if split is None:
        return shared
    if shared is None:
        return split
    if isinstance(shared, Mapping) and isinstance(split, Mapping):
        return {**shared, **split}
    if _is_list(shared) and _is_list(split):
        return [*shared, *split]
    return split


def union_metric_specs(shared: Any, *splits: Any) -> Any:
    """The ONE set an engine that cannot vary metrics per split has to compute.

    ``keras.Model.compile`` takes a single metric list and reports it on train AND validation
    (prefixed ``val_``); a ``transformers.Trainer`` takes a single ``compute_metrics``. Given
    per-split slots such an engine can only compute their union, so a metric asked for on one
    split appears on the others too.

    Duplicates are dropped so the union of overlapping slots does not report a metric twice —
    which for a Keras metric OBJECT would additionally mean two entries with the same ``name``.
    """
    out = shared
    for split in splits:
        if split is None:
            continue
        if isinstance(out, Mapping) and isinstance(split, Mapping):
            out = {**out, **split}
        elif _is_list(out) and _is_list(split):
            out = [*out, *(m for m in split if m not in out)]
        else:
            out = merge_metric_specs(out, split)
    return out

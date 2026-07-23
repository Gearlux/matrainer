"""Modality-neutral predictions sink for a Lightning predict/test loop.

A consumer's predict loop calls ``predictions_sink.write(prediction, metadata)``
once per sample, then ``predictions_sink.close()`` at end-of-run. This module
hosts the sink that bridges that contract to SampleFlux ``Sample``-ops *without*
any domain coupling:

* :class:`ClassificationPredictionsSink` — reads each per-sample
  ``ClassificationOutput`` (``probs`` / ``class_idx``, see
  ``marainer.torch.outputs``), resolves the int class id to a human-readable
  label, builds a top-k list, and threads a fresh ``Sample`` carrying
  ``predicted_class_id`` / ``predicted_class_label`` / ``predicted_confidence``
  / ``predicted_top_k`` (as a single ``aux`` field) through a configurable op
  chain (typically just a ``SampleSinkOp`` wrapping a per-window JSON sink).

It lives in marainer — not in a domain package — because marainer both defines
the ``predictions_sink`` contract AND owns the ``ClassificationOutput`` type it
reads, and because the sink is fully modality-neutral (no signal/image
specifics). Any classification consumer reuses it with no cross-domain
dependency. The signal-domain counterpart that back-projects pixel boxes to
time/frequency regions stays in the signal package because it genuinely depends
on that package's signal geometry.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from confluid import configurable
from loggair import get_logger
from sampleflux import Sample

logger = get_logger(__name__)


@configurable
class ClassificationPredictionsSink:
    """Adapter sink that lifts classification predictions into a Sample and runs ops.

    Bridges a Lightning predict/test loop's ``predictions_sink`` contract —
    ``write(prediction, metadata)`` per sample — to the Sample-based ops in
    sampleflux, giving the train→eval→predict triad a uniform shape across
    detection, segmentation, and classification.

    For each call:

    1. Read the model's ``ClassificationOutput`` (``marainer.torch.outputs``) —
       ``logits`` ``[C]``, ``probs`` ``[C]``, ``class_idx`` scalar (per
       sample).
    2. Resolve the int class id to a human-readable label via
       ``label_names``. Build a top-k list (``top_k`` highest-probability
       classes with their probabilities + labels).
    3. Build a fresh ``Sample`` carrying the original metadata plus the new
       prediction columns as a single ``aux`` field ``"metadata"``:

       * ``predicted_class_id`` (int)
       * ``predicted_class_label`` (str)
       * ``predicted_confidence`` (float — top-1 probability)
       * ``predicted_top_k`` (list of ``{class_id, label, probability}``
         dicts, sorted by descending probability)

       carried on the typed bag as a single ``aux`` field ``"metadata"``
       (``Sample({"metadata": ...}, roles={"metadata": "aux"})``).
    4. Thread the Sample through ``ops`` — typically just
       ``SampleSinkOp(sink=JsonPerWindowSink(...))`` to dump per-window JSON.

    Args:
        ops: Non-empty list of ops to run on each per-prediction Sample.
        label_names: Optional ``{int_class_id: str}`` map for converting
            the model's int64 class ids back to human-readable strings.
            Without it, labels become ``str(class_id)``. YAML int keys
            land here as strings (confluid stringifies all mapping keys
            at load time) — both forms are accepted; keys are normalized
            to ``str`` internally.
        top_k: Number of top-probability predictions to record per sample.
            Defaults to ``1``. ``top_k > num_classes`` is silently clamped.
        confidence_threshold: Predictions whose top-1 probability is below
            this value are skipped entirely (no Sample produced, no ops
            run). Default ``0.0`` keeps every prediction.
    """

    def __init__(
        self,
        ops: List[Any],
        label_names: Optional[Dict[Any, str]] = None,
        top_k: int = 1,
        confidence_threshold: float = 0.0,
    ) -> None:
        if not ops:
            raise ValueError("ClassificationPredictionsSink requires a non-empty 'ops' list.")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")
        self.ops: List[Any] = list(ops)
        self.top_k = int(top_k)
        self.confidence_threshold = float(confidence_threshold)
        # Normalize keys to ``str`` so YAML-loaded maps (which always come
        # through with stringified keys) and Python-constructed maps (which
        # may use int keys) are both addressable by the same lookup.
        self.label_names: Dict[str, str] = {str(k): str(v) for k, v in (label_names or {}).items()}

    def _label_for(self, class_id: int) -> str:
        return self.label_names.get(str(int(class_id)), str(int(class_id)))

    def write(self, prediction: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        from confluid import flow
        from confluid.fluid import Fluid

        probs = prediction.get("probs")
        class_idx = prediction.get("class_idx")

        sample_id = (
            f"{metadata.get('pack_id', '?')}/{metadata.get('iq_file', '?')}"
            f"@win_s{metadata.get('window_start_sample', 0)}"
        )
        if probs is None or class_idx is None:
            logger.warning(
                f"[diag][cls_pred_sink] {sample_id}: prediction missing 'probs' or 'class_idx' "
                f"(got keys: {sorted(prediction.keys())}); skipping."
            )
            return

        probs_arr = probs.detach().cpu().numpy() if hasattr(probs, "detach") else np.asarray(probs)
        # Tolerate both [C] (single-sample) and [1, C] (batched-with-1) shapes.
        if probs_arr.ndim == 2 and probs_arr.shape[0] == 1:
            probs_arr = probs_arr[0]
        if probs_arr.ndim != 1:
            logger.warning(
                f"[diag][cls_pred_sink] {sample_id}: probs has shape {probs_arr.shape}, "
                f"expected [C] or [1,C]; skipping."
            )
            return

        n_classes = int(probs_arr.shape[0])
        top1_id = int(class_idx.item()) if hasattr(class_idx, "item") else int(class_idx)
        top1_prob = float(probs_arr[top1_id])
        if top1_prob < self.confidence_threshold:
            logger.debug(
                f"[diag][cls_pred_sink] {sample_id}: top1_prob={top1_prob:.3f} "
                f"< threshold={self.confidence_threshold:.3f}; skipping."
            )
            return

        k = min(self.top_k, n_classes)
        # argsort descending, take first k.
        top_k_ids = np.argsort(-probs_arr)[:k]
        top_k_list = [
            {
                "class_id": int(cid),
                "label": self._label_for(int(cid)),
                "probability": float(probs_arr[cid]),
            }
            for cid in top_k_ids
        ]

        new_metadata = dict(metadata)
        new_metadata["predicted_class_id"] = top1_id
        new_metadata["predicted_class_label"] = self._label_for(top1_id)
        new_metadata["predicted_confidence"] = top1_prob
        new_metadata["predicted_top_k"] = top_k_list

        logger.debug(
            f"[diag][cls_pred_sink] {sample_id}: top1={new_metadata['predicted_class_label']!r} "
            f"({top1_prob:.3f}), top_k={k}"
        )

        # Carry the prediction metadata as a single ``aux`` field on a typed bag ``Sample``.
        # Downstream ops (typically ``SampleSinkOp`` wrapping a per-window JSON sink) read it
        # via ``sample["metadata"]``. There is no input/target on a prediction-only sample.
        sample = Sample({"metadata": new_metadata}, roles={"metadata": "aux"})
        for i, op in enumerate(self.ops):
            if isinstance(op, Fluid):
                op = flow(op)
                self.ops[i] = op
            sample = op(sample)

    def close(self) -> None:
        """Propagate ``close()`` to ops that own resources (e.g. wrapped sinks).

        The sink itself buffers nothing, but the wrapped ops typically do
        (e.g. ``SampleSinkOp`` wrapping a buffered/file-handle sink).
        """
        for op in self.ops:
            close_fn = getattr(op, "close", None)
            if callable(close_fn):
                close_fn()


__all__ = ["ClassificationPredictionsSink"]

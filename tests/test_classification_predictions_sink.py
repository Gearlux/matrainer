"""Tests for :class:`marainer.sinks.ClassificationPredictionsSink`."""

from typing import Any, Dict, List

import pytest
import torch
from sampleflux.sample import Sample

from marainer.sinks import ClassificationPredictionsSink


class _CapturingOp:
    """Tiny op that records every Sample it sees, for assertion purposes."""

    def __init__(self) -> None:
        self.calls: List[Sample] = []
        self.closed = False

    def __call__(self, sample: Sample) -> Sample:
        self.calls.append(sample)
        return sample

    def close(self) -> None:
        self.closed = True


def _make_prediction(
    probs: List[float],
    class_idx: int,
) -> Dict[str, Any]:
    """Build a ClassificationOutput-shaped dict from a flat prob list."""
    p = torch.tensor(probs, dtype=torch.float32)
    return {
        "logits": torch.log(p),  # not used by sink, but realistic
        "probs": p,
        "class_idx": torch.tensor(class_idx, dtype=torch.int64),
    }


def _make_metadata() -> Dict[str, Any]:
    return {
        "pack_id": "p1",
        "iq_file": "f1.iq",
        "window_start_sample": 0,
        "drone": "DJI AVATA2",
    }


# --- Happy path ------------------------------------------------------------


def test_top1_path_writes_predicted_columns() -> None:
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(
        ops=[op],
        label_names={0: "DJI MINI3", 1: "DJI AVATA2", 2: "Other"},
        top_k=1,
    )
    sink.write(_make_prediction([0.1, 0.7, 0.2], class_idx=1), _make_metadata())

    assert len(op.calls) == 1
    meta = op.calls[0].meta
    assert meta["predicted_class_id"] == 1
    assert meta["predicted_class_label"] == "DJI AVATA2"
    assert meta["predicted_confidence"] == pytest.approx(0.7, abs=1e-6)
    assert len(meta["predicted_top_k"]) == 1
    assert meta["predicted_top_k"][0]["label"] == "DJI AVATA2"


def test_top_k_path_returns_descending_probabilities() -> None:
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(
        ops=[op],
        label_names={0: "a", 1: "b", 2: "c", 3: "d"},
        top_k=3,
    )
    sink.write(_make_prediction([0.1, 0.4, 0.45, 0.05], class_idx=2), _make_metadata())

    top_k = op.calls[0].meta["predicted_top_k"]
    assert len(top_k) == 3
    # Sorted descending by probability.
    probs = [entry["probability"] for entry in top_k]
    assert probs == sorted(probs, reverse=True)
    assert top_k[0]["label"] == "c"
    assert top_k[1]["label"] == "b"
    assert top_k[2]["label"] == "a"


def test_top_k_clamps_when_exceeds_num_classes() -> None:
    """top_k=10 with 3 classes → returns 3, not crash."""
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(ops=[op], top_k=10)
    sink.write(_make_prediction([0.5, 0.3, 0.2], class_idx=0), _make_metadata())
    assert len(op.calls[0].meta["predicted_top_k"]) == 3


def test_label_names_string_keys_work() -> None:
    """YAML loads int keys as strings; both forms must work."""
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(
        ops=[op],
        label_names={"0": "a", "1": "b"},  # str keys
        top_k=1,
    )
    sink.write(_make_prediction([0.2, 0.8], class_idx=1), _make_metadata())
    assert op.calls[0].meta["predicted_class_label"] == "b"


def test_label_names_missing_key_falls_back_to_str_class_id() -> None:
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(
        ops=[op],
        label_names={0: "zero"},  # only class 0 mapped
        top_k=1,
    )
    sink.write(_make_prediction([0.1, 0.9], class_idx=1), _make_metadata())
    # Missing class 1 → falls back to "1".
    assert op.calls[0].meta["predicted_class_label"] == "1"


def test_no_label_names_uses_str_class_id() -> None:
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(ops=[op], top_k=2)
    sink.write(_make_prediction([0.7, 0.3], class_idx=0), _make_metadata())
    meta = op.calls[0].meta
    assert meta["predicted_class_label"] == "0"
    assert {entry["label"] for entry in meta["predicted_top_k"]} == {"0", "1"}


def test_threaded_sample_carries_input_none() -> None:
    """The contract mirrors DetectionPredictionsSink: input/target = None."""
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(ops=[op], top_k=1)
    sink.write(_make_prediction([0.6, 0.4], class_idx=0), _make_metadata())
    s = op.calls[0]
    assert s.input is None
    assert s.target is None


def test_close_propagates_to_ops() -> None:
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(ops=[op], top_k=1)
    sink.close()
    assert op.closed is True


# --- Confidence threshold --------------------------------------------------


def test_confidence_threshold_drops_low_predictions() -> None:
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(
        ops=[op],
        top_k=1,
        confidence_threshold=0.5,
    )
    # top1 prob = 0.4, below threshold → no Sample produced.
    sink.write(_make_prediction([0.4, 0.3, 0.3], class_idx=0), _make_metadata())
    assert op.calls == []


def test_confidence_threshold_keeps_high_predictions() -> None:
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(
        ops=[op],
        top_k=1,
        confidence_threshold=0.5,
    )
    sink.write(_make_prediction([0.7, 0.3], class_idx=0), _make_metadata())
    assert len(op.calls) == 1


# --- Edge cases ------------------------------------------------------------


def test_empty_ops_list_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ClassificationPredictionsSink(ops=[])


def test_zero_top_k_raises() -> None:
    with pytest.raises(ValueError, match="top_k"):
        ClassificationPredictionsSink(ops=[_CapturingOp()], top_k=0)


def test_missing_probs_skips_silently_with_warning() -> None:
    """If the model emits a malformed prediction, the sink skips rather than crashing."""
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(ops=[op], top_k=1)
    sink.write({"logits": torch.zeros(3)}, _make_metadata())  # no probs/class_idx
    assert op.calls == []


def test_handles_2d_probs_with_leading_singleton() -> None:
    """Some pipelines emit [1, C] instead of [C]; sink must accept both."""
    op = _CapturingOp()
    sink = ClassificationPredictionsSink(ops=[op], top_k=1)
    sink.write(
        {
            "logits": torch.zeros(1, 3),
            "probs": torch.tensor([[0.1, 0.6, 0.3]], dtype=torch.float32),
            "class_idx": torch.tensor(1, dtype=torch.int64),
        },
        _make_metadata(),
    )
    assert len(op.calls) == 1
    assert op.calls[0].meta["predicted_class_id"] == 1
    assert op.calls[0].meta["predicted_confidence"] == pytest.approx(0.6, abs=1e-6)

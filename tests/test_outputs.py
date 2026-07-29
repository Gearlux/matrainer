"""Tests for typed prediction-output contracts."""

import torch

from marainer.torch.outputs import (
    ClassificationOutput,
    DetectionOutput,
    SegmentationOutput,
    classification_output,
    segmentation_output,
)


def test_classification_output_shapes_and_dtypes() -> None:
    logits = torch.randn(4, 3)
    out = classification_output(logits)

    # TypedDict values are plain dict entries at runtime.
    assert set(out.keys()) == {"logits", "probs", "class_idx"}
    assert out["logits"] is logits
    assert out["probs"].shape == (4, 3)
    assert torch.allclose(out["probs"].sum(dim=-1), torch.ones(4), atol=1e-5)
    assert out["class_idx"].dtype == torch.int64
    assert out["class_idx"].shape == (4,)
    assert torch.equal(out["class_idx"], logits.argmax(dim=-1).to(torch.int64))


def test_segmentation_output_per_pixel_argmax() -> None:
    logits = torch.randn(2, 5, 8, 8)
    out = segmentation_output(logits)

    assert set(out.keys()) == {"logits", "probs", "mask"}
    assert out["probs"].shape == (2, 5, 8, 8)
    # Softmax over channel dim.
    assert torch.allclose(out["probs"].sum(dim=1), torch.ones(2, 8, 8), atol=1e-5)
    assert out["mask"].dtype == torch.int64
    assert out["mask"].shape == (2, 8, 8)
    assert torch.equal(out["mask"], logits.argmax(dim=1).to(torch.int64))


def test_detection_output_typed_dict_construction() -> None:
    # DetectionOutput is constructed directly at call sites; verify the
    # TypedDict's runtime behavior matches a plain dict.
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
    scores = torch.tensor([0.9], dtype=torch.float32)
    labels = torch.tensor([1], dtype=torch.int64)
    out = DetectionOutput(boxes=boxes, scores=scores, labels=labels)
    assert out["boxes"].dtype == torch.float32
    assert out["scores"].dtype == torch.float32
    assert out["labels"].dtype == torch.int64
    assert set(out.keys()) == {"boxes", "scores", "labels"}


def test_classification_output_is_typed_dict_instance() -> None:
    out: ClassificationOutput = classification_output(torch.zeros(1, 2))
    # TypedDicts are plain dicts at runtime.
    assert isinstance(out, dict)
    assert "logits" in out and "probs" in out and "class_idx" in out


def test_segmentation_output_is_typed_dict_instance() -> None:
    out: SegmentationOutput = segmentation_output(torch.zeros(1, 2, 3, 3))
    assert isinstance(out, dict)
    assert set(out.keys()) == {"logits", "probs", "mask"}


# --------------------------------------------------------------------------- #
# The contracts are framework-neutral; only the builders are torch.
# --------------------------------------------------------------------------- #


def test_the_contracts_import_without_torch() -> None:
    """A Keras/JAX/TF consumer must be able to declare its output without `marainer[torch]`.

    Run in a SUBPROCESS: this test session has already imported torch, so an in-process
    ``sys.modules`` check would prove nothing.
    """
    import subprocess
    import sys

    probe = "import sys, marainer.outputs; assert 'torch' not in sys.modules; print('clean')"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_a_contract_parameterizes_over_the_array_type() -> None:
    """The same contract describes a torch run and a numpy/TF/JAX one."""
    import numpy as np

    from marainer.outputs import ClassificationOutput as NeutralClassificationOutput

    numpy_out: NeutralClassificationOutput[np.ndarray] = {
        "logits": np.zeros(2),
        "probs": np.zeros(2),
        "class_idx": np.zeros(2),
    }
    torch_out: NeutralClassificationOutput[torch.Tensor] = classification_output(torch.zeros(1, 2))

    assert isinstance(numpy_out["probs"], np.ndarray)
    assert isinstance(torch_out["probs"], torch.Tensor)


def test_the_torch_module_reexports_and_does_not_redefine() -> None:
    """One definition per contract — `marainer.torch.outputs` re-exports, never redefines."""
    import marainer.outputs as neutral
    import marainer.torch.outputs as torch_outputs

    for name in ("ClassificationOutput", "DetectionOutput", "SegmentationOutput"):
        assert getattr(torch_outputs, name) is getattr(neutral, name), f"{name} was redefined"

"""The torch-only helpers: loading a checkpoint, choosing a device, sizing and reducing metrics.

Asserts what `docs/torch.md` claims, with weights it writes and reads back rather than a model
anyone has to download:

* `load_lightning_checkpoint` takes three on-disk shapes through ONE call — a plain state dict, a
  Lightning `.ckpt` whose keys carry the `model.` prefix, and a checkpoint dropped into a WRAPPER
  whose named child matches it — and the loaded weights are the saved ones in each case;
* `resolve_device` takes an explicit choice at its word, which is the point: `cpu` means CPU even
  on a machine with a GPU, so a comparison run is not silently moved;
* `build_metric_collection` injects a sizing kwarg PER TEMPLATE — only where the constructor
  declares it, and never over a value the template already pinned;
* `scalar_metrics` keeps the one-number metrics and drops a confusion matrix and a per-class
  vector, because a metric channel holds one number per name.

Not covered here: the Ultralytics `.pt` shape (its checkpoint holds a live `DetectionModel`, so it
needs that package) and `apply_mps_low_watermark`, whose effect is Apple-GPU-specific.

Run:  python examples/torch_helpers.py   (needs `matrainer[torch]`)
"""

import tempfile
from pathlib import Path
from typing import Any, cast, get_args

import torch
from confluid import PartialClass
from torch import nn

from matrainer.torch import (
    Device,
    TorchModelFile,
    build_metric_collection,
    load_lightning_checkpoint,
    resolve_device,
    scalar_metrics,
)

FEATURES, CLASSES = 4, 3


def a_model() -> nn.Linear:
    return nn.Linear(FEATURES, CLASSES)


class Wrapper(nn.Module):
    """A model held as a named child — what `Model(checkpoint_path=…)` looks like inside."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = a_model()


def the_three_checkpoint_shapes(root: Path) -> None:
    trained = a_model()
    with torch.no_grad():  # a recognisable weight, so "did it load?" is a value comparison
        trained.weight.fill_(0.5)
        trained.bias.fill_(-0.25)

    # 1. A plain state dict — `torch.save(model.state_dict(), path)`.
    plain = root / "plain.pth"
    torch.save(trained.state_dict(), plain)
    fresh = a_model()
    load_lightning_checkpoint(fresh, str(plain))
    assert torch.equal(fresh.weight, trained.weight) and torch.equal(fresh.bias, trained.bias)

    # 2. A Lightning `.ckpt`: a `state_dict` key whose entries carry the `model.` prefix the
    #    LightningModule wrapper adds. The helper strips it, so the model loads its own weights
    #    without the wrapper being in the picture.
    lightning = root / "epoch=0.ckpt"
    torch.save({"state_dict": {f"model.{k}": v for k, v in trained.state_dict().items()}}, lightning)
    fresh = a_model()
    load_lightning_checkpoint(fresh, str(lightning))
    assert torch.equal(fresh.weight, trained.weight), "the `model.` prefix was not stripped"

    # 3. The same plain checkpoint dropped into a WRAPPER: its keys match the `backbone` child, so
    #    the load is retargeted there without anyone hand-prefixing the names.
    wrapper = Wrapper()
    load_lightning_checkpoint(wrapper, str(plain))
    assert torch.equal(wrapper.backbone.weight, trained.weight), "the load did not find the child"

    print("one call loaded a plain state dict, a Lightning .ckpt and a wrapper's named child")


def a_checkpoint_is_not_a_whole_model(root: Path) -> None:
    """`TorchModelFile` loads a model saved whole; handed a dict it names the fix instead."""
    state_dict_only = root / "weights.pth"
    torch.save(a_model().state_dict(), state_dict_only)

    try:
        # `solidify()` is where the file is read — the constructor stores the path and nothing
        # else, which is confluid's lazy convention, so the refusal arrives on first use.
        TorchModelFile(file=str(state_dict_only)).solidify()
    except ValueError as exc:  # the refusal is the behaviour under test
        assert "checkpoint_path" in str(exc), exc  # it names where a state dict DOES load
        assert "weights.pth" in str(exc), exc  # and which file it was handed
        print("a state dict handed to TorchModelFile is refused, naming checkpoint_path")
    else:  # pragma: no cover - only if the refusal is lost
        raise AssertionError("expected a refusal naming checkpoint_path")


def choosing_a_device() -> None:
    # The closed literal IS the set of answers — one source of truth, no restated tuple.
    assert get_args(Device) == ("auto", "cpu", "cuda", "mps")

    # Explicit means explicit: this machine may well have a GPU, and `cpu` still gets CPU.
    assert resolve_device("cpu").type == "cpu"

    automatic = resolve_device("auto")  # cuda -> mps -> cpu, whichever is present
    assert automatic.type in {"cuda", "mps", "cpu"}
    print(f"resolve_device('auto') chose {automatic.type!r} here; 'cpu' still gives 'cpu'")


def sizing_and_reducing_metrics() -> None:
    from torchmetrics.classification import MulticlassAccuracy
    from torchmetrics.image import PeakSignalNoiseRatio, SpectralAngleMapper

    # `num_classes` reaches the metric that declares it...
    collection = build_metric_collection({"accuracy": PartialClass(MulticlassAccuracy)}, num_classes=CLASSES)
    assert isinstance(collection["accuracy"], MulticlassAccuracy)

    # ...a metric that cannot take the dimension is left alone, so it can sit beside ones that can
    # (three of torchmetrics' seven full-reference image metrics take `data_range`; four do not).
    beside = build_metric_collection(
        {"sam": PartialClass(SpectralAngleMapper), "psnr": PartialClass(PeakSignalNoiseRatio)}, data_range=255.0
    )
    assert isinstance(beside["sam"], SpectralAngleMapper)
    # A MetricCollection holds different metric types, so reading one's own setting needs a cast —
    # the same spelling this project's own tests use.
    assert float(cast(Any, beside["psnr"]).data_range) == 255.0

    # ...and the injection is a FLOOR: a template that pinned the value keeps its own.
    pinned = build_metric_collection({"psnr": PartialClass(PeakSignalNoiseRatio, data_range=2.0)}, data_range=1.0)
    assert float(cast(Any, pinned["psnr"]).data_range) == 2.0

    # A `compute()` result holds more than numbers; a metric channel holds one number per name.
    computed = {
        "val/accuracy": torch.tensor(0.94),
        "val/confusion": torch.zeros(CLASSES, CLASSES),  # an N×N matrix
        "val/iou_per_class": torch.zeros(CLASSES),  # a [C] vector
    }
    reduced = scalar_metrics(computed)
    assert set(reduced) == {"val/accuracy"}, reduced  # the matrix and the vector are dropped
    # What survives is a plain float a tracker can record — a float32 tensor, so 0.94 reads back
    # as 0.9399999976158142 and a comparison needs a tolerance, not an equality.
    assert abs(reduced["val/accuracy"] - 0.94) < 1e-6
    assert isinstance(reduced["val/accuracy"], float)
    print("a sizing kwarg reaches only the templates that take it; scalar_metrics keeps the numbers")


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="matrainer_torch_"))
    the_three_checkpoint_shapes(root)
    a_checkpoint_is_not_a_whole_model(root)
    choosing_a_device()
    sizing_and_reducing_metrics()


if __name__ == "__main__":
    main()

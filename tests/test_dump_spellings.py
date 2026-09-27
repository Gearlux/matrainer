"""The torch-family dump spellings — a run document that carries its values faithfully.

`confluid.dump()` degrades a value it cannot respell to a bare
``{_target_: <module.qualname>}`` placeholder that reloads DEFAULT-constructed —
which for a run manifest's resolved-config artifact meant a class-weights vector
reloading as an EMPTY tensor and a metric collection crashing the reload outright
(``missing 1 required positional argument: 'metrics'``). ``matrainer.torch``
registers the two spellings this platform's run documents actually need:

* ``torch.Tensor`` → a marker on :func:`tensor_from_values` (values + dtype),
  DECLINED above :data:`TENSOR_DUMP_CAP` — a weights VECTOR belongs in the run
  document, a parameter matrix belongs in a checkpoint;
* ``torchmetrics.MetricCollection`` → a marker rebuilding it from its metrics
  dict + prefix/postfix, each metric rendered by the dumper's ordinary
  registered-class branch.
"""

from typing import Any

import torch
import torch.nn as nn
from confluid import configurable, dump, load, register

import matrainer.torch  # noqa: F401 — the import IS the registration
from matrainer.torch.spellings import TENSOR_DUMP_CAP


def _make_host() -> Any:
    @configurable(name="SpellingTorchHost")
    class Host:
        def __init__(self) -> None:
            self.payload: Any = None

    return Host


def _dump_payload(value: Any) -> str:
    host = _make_host()()
    host.payload = value
    return dump(host)


def test_a_small_tensor_round_trips_with_its_dtype() -> None:
    for tensor in (torch.tensor([0.5, 1.5, 3.0]), torch.tensor([1.0, 2.0], dtype=torch.float64)):
        text = _dump_payload(tensor)
        assert "_target_: torch.Tensor" not in text, "the placeholder is gone"
        reloaded = load(text).payload
        assert isinstance(reloaded, torch.Tensor)
        assert reloaded.dtype == tensor.dtype
        assert torch.equal(reloaded, tensor)


def test_an_int_tensor_round_trips() -> None:
    tensor = torch.tensor([1, 2, 3])
    reloaded = load(_dump_payload(tensor)).payload
    assert reloaded.dtype == torch.int64
    assert torch.equal(reloaded, tensor)


def test_a_parameter_hits_the_tensor_spelling_via_the_mro() -> None:
    """An ``nn.Parameter`` IS a Tensor — it reloads as its VALUE (a plain tensor)."""
    param = nn.Parameter(torch.tensor([0.25, 0.75]))
    text = _dump_payload(param)
    assert "Parameter" not in text
    reloaded = load(text).payload
    assert torch.equal(reloaded, param.detach())


def test_a_large_tensor_keeps_the_placeholder() -> None:
    """Model-scale weights belong to checkpoints — the recipe DECLINES above the cap."""
    big = torch.zeros(TENSOR_DUMP_CAP + 1)
    text = _dump_payload(big)
    assert "_target_: torch.Tensor" in text


def test_a_metric_collection_round_trips_and_still_computes() -> None:
    from torchmetrics import MetricCollection
    from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

    register(MulticlassAccuracy, framework="torch")
    register(MulticlassF1Score, framework="torch")
    collection = MetricCollection(
        {"acc": MulticlassAccuracy(num_classes=3), "f1": MulticlassF1Score(num_classes=3, average="macro")},
        prefix="train/",
    )
    text = _dump_payload(collection)
    assert "metrics:" in text, "the marker carries its metrics, not a bare placeholder"

    reloaded = load(text).payload
    assert isinstance(reloaded, MetricCollection)
    assert sorted(str(k) for k in reloaded.keys()) == ["train/acc", "train/f1"]
    preds, target = torch.tensor([0, 1, 2, 2]), torch.tensor([0, 1, 2, 1])
    reloaded.update(preds, target)
    computed = {k: round(float(v), 4) for k, v in reloaded.compute().items()}
    assert computed == {"train/acc": 0.8333, "train/f1": 0.7778}


def test_the_reloaded_collection_preserves_each_metrics_knobs() -> None:
    from torchmetrics import MetricCollection
    from torchmetrics.classification import MulticlassF1Score

    register(MulticlassF1Score, framework="torch")
    collection = MetricCollection({"f1": MulticlassF1Score(num_classes=7, average="micro", top_k=2)})
    reloaded = load(_dump_payload(collection)).payload
    metric = dict(reloaded.items(keep_base=True, copy_state=False))["f1"]
    assert (metric.num_classes, metric.average, metric.top_k) == (7, "micro", 2)


def test_a_directly_slotted_registered_metric_round_trips() -> None:
    """The metric spelling wins over the generic slot walk even for a REGISTERED
    metric at an ordinary slot — the generic walk emits body slots the ctor chain
    refuses back (``update``/``compute``/``beta``)."""
    from torchmetrics.classification import MulticlassF1Score

    register(MulticlassF1Score, framework="torch")
    text = _dump_payload(MulticlassF1Score(num_classes=5, average="micro"))
    assert "beta" not in text and "update" not in text
    reloaded = load(text).payload
    assert isinstance(reloaded, MulticlassF1Score)
    assert (reloaded.num_classes, reloaded.average) == (5, "micro")


def test_an_unrelated_opaque_still_degrades_to_the_placeholder() -> None:
    """The con case: the registrations are per-type, not a general 'dump anything'."""

    class _Opaque:
        pass

    text = _dump_payload(_Opaque())
    assert "_Opaque" in text

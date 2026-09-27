"""Metric-name translation between engine vocabularies — ``MetricNames`` / parse / render.

The key sets below are not invented: they are the header lines of a real eight-backend sweep
(one CSV per run), which is the state this module exists to remove. Each engine names the same
three quantities differently, so a sweep landed one quantity under four names and no tracker
could plot them together.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pytest

from matrainer.metric_names import (
    FRAMEWORKS,
    SPLITS,
    Framework,
    MetricNames,
    parse_metric_name,
    render_metric_name,
    vocabulary,
)

#: What each backend actually recorded, measured from `sweep_logs/csv/*/<run>/metrics.csv`.
NATIVE: Dict[Framework, List[str]] = {
    "lightning": ["train/loss", "train/accuracy", "val/loss", "val/accuracy", "epoch"],
    "torch": ["train/loss", "train/accuracy", "val/loss", "val/accuracy"],
    "keras": ["loss", "accuracy", "val_loss", "val_accuracy"],
    "tensorflow": ["loss", "sparse_categorical_accuracy", "val_loss", "val_sparse_categorical_accuracy"],
    "fastai": ["train_loss", "valid_loss", "accuracy", "top_k_accuracy"],
    "huggingface": ["train_loss", "eval_loss", "eval_accuracy", "train_runtime", "epoch"],
    "jax": ["loss", "val_loss", "val_accuracy"],
    "mlx": ["loss", "val_loss", "val_accuracy"],
}


# ---------------------------------------------------------------------------
# The canonical direction: every engine's spelling -> <split>/<name>
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "source,native,canonical",
    [
        ("keras", "val_loss", "val/loss"),
        ("keras", "loss", "train/loss"),
        ("keras", "val_sparse_categorical_accuracy", "val/accuracy"),
        ("keras", "binary_accuracy", "train/accuracy"),
        ("keras", "acc", "train/accuracy"),
        ("tensorflow", "val_sparse_categorical_accuracy", "val/accuracy"),
        ("fastai", "train_loss", "train/loss"),
        ("fastai", "valid_loss", "val/loss"),
        ("huggingface", "eval_loss", "val/loss"),
        ("huggingface", "eval_accuracy", "val/accuracy"),
        ("huggingface", "train_runtime", "train/runtime"),
        ("huggingface", "train_samples_per_second", "train/samples_per_second"),
        ("jax", "val_accuracy", "val/accuracy"),
        ("torch", "val/accuracy", "val/accuracy"),
    ],
)
def test_a_native_name_becomes_the_canonical_one(source: Framework, native: str, canonical: str) -> None:
    assert MetricNames().name(native, source=source) == canonical


def test_the_comparable_quantities_line_up_across_every_backend() -> None:
    """The whole point: one sweep, one vocabulary — a column per quantity, not per engine."""
    translated = {
        backend: set(MetricNames()(dict.fromkeys(keys, 0.0), source=backend)) for backend, keys in NATIVE.items()
    }
    assert {"train/loss", "val/loss"} <= set.intersection(*translated.values()), translated
    # Every backend that measures validation accuracy now spells it one way. fastai's extra
    # `top_k_accuracy` is a DIFFERENT measurement (top-5) and keeps its own column, as it should.
    assert {backend for backend, keys in translated.items() if "val/accuracy" in keys} == set(NATIVE)
    assert {key for keys in translated.values() for key in keys if "accur" in key} == {
        "train/accuracy",
        "val/accuracy",
        "val/top_k_accuracy",
    }


def test_an_unprefixed_fastai_metric_is_a_VALIDATION_metric() -> None:
    """fastai's Recorder computes its metrics on the validation split and prefixes none of them.

    Read as a training metric — the default everywhere else — a sweep would report the val
    accuracy of five backends beside the train accuracy of one, in the same column.
    """
    assert MetricNames()({"accuracy": 0.9}, source="fastai") == {"val/accuracy": 0.9}
    assert MetricNames()({"accuracy": 0.9}, source="keras") == {"train/accuracy": 0.9}


def test_the_caller_can_override_the_split_for_an_unprefixed_key() -> None:
    """A test pass reports through the same code that reports training."""
    assert MetricNames()({"loss": 0.2}, source="keras", split="test") == {"test/loss": 0.2}


def test_a_prefixed_key_ignores_the_split_override() -> None:
    """The override answers "which split are the BARE names from?", never overrules a real prefix."""
    assert MetricNames()({"val_loss": 0.2}, source="keras", split="test") == {"val/loss": 0.2}


# ---------------------------------------------------------------------------
# The other directions: canonical -> an engine, and engine -> engine
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "target,canonical,rendered",
    [
        ("keras", "val/loss", "val_loss"),
        ("keras", "train/loss", "loss"),  # Keras writes its training metrics unprefixed
        # The BASE name stays canonical: `sparse_categorical_accuracy` is what `compile()`
        # expanded `"accuracy"` into, not a name anyone writes. A vocabulary spells the SPLIT.
        ("keras", "val/accuracy", "val_accuracy"),
        ("fastai", "val/loss", "valid_loss"),
        ("huggingface", "val/accuracy", "eval_accuracy"),
        ("matrainer", "val/loss", "val/loss"),
    ],
)
def test_a_canonical_name_renders_into_any_vocabulary(target: Framework, canonical: str, rendered: str) -> None:
    assert MetricNames(target=target).name(canonical) == rendered


def test_framework_to_framework_needs_no_canonical_detour_from_the_caller() -> None:
    """`source=` and `target=` are independent, so torch -> keras is one call."""
    assert MetricNames(target="keras")({"val/loss": 0.3}, source="torch") == {"val_loss": 0.3}
    assert MetricNames(target="huggingface")({"val_loss": 0.3}, source="keras") == {"eval_loss": 0.3}


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_a_name_survives_the_trip_out_to_a_vocabulary_and_back(framework: Framework) -> None:
    """canonical -> engine -> canonical is the identity, for every engine.

    Not "render(parse(x)) == x": an engine that omits a prefix (fastai's bare `accuracy`) or
    auto-expands a base name (Keras' `sparse_categorical_accuracy`) has more than one spelling
    for one quantity, and translating INTO it must pick one. What must hold is that no meaning
    is lost on the way out — the trip back lands on the name it started from.
    """
    canonical = {"train/loss": 1.0, "val/loss": 2.0, "val/accuracy": 0.9, "test/loss": 3.0, "epoch": 4.0}
    out = MetricNames(target=framework)(canonical)
    assert MetricNames()(out, source=framework) == canonical


def test_translation_is_idempotent() -> None:
    """A backend that already names things canonically may still declare its real source.

    Without this, a second pass would grow a second prefix (`train/train/loss`) — which is what
    a shared helper applied at two seams would otherwise do.
    """
    once = MetricNames()({"train/loss": 1.0, "val/accuracy": 0.9}, source="keras")
    assert once == {"train/loss": 1.0, "val/accuracy": 0.9}
    assert MetricNames()(once, source="keras") == once


# ---------------------------------------------------------------------------
# The knobs
# ---------------------------------------------------------------------------
def test_renames_apply_to_the_canonical_base_name() -> None:
    """So one entry works whatever the two vocabularies spell it — here fastai in, keras out."""
    names = MetricNames(target="keras", renames={"top_k_accuracy": "accuracy_top5"})
    assert names({"top_k_accuracy": 0.99}, source="fastai") == {"val_accuracy_top5": 0.99}


def test_a_rename_reaches_a_name_the_module_has_never_heard_of() -> None:
    names = MetricNames(renames={"miou": "iou_mean"})
    assert names({"val_miou": 0.7}, source="keras") == {"val/iou_mean": 0.7}


@pytest.mark.parametrize("axis", ["epoch", "step"])
def test_an_axis_key_belongs_to_no_split(axis: str) -> None:
    """`epoch` is the x-value of every other series; prefixing it invents a second one."""
    assert MetricNames()({axis: 3.0}, source="huggingface") == {axis: 3.0}
    assert parse_metric_name(axis) == (None, axis)
    assert render_metric_name(None, axis) == axis


def test_zero_arg_construction_and_no_work_in_the_constructor() -> None:
    """The workspace construction rule: a pure config carrier, translated only on call."""
    names = MetricNames()
    assert (names.target, names.renames) == ("matrainer", {})


def test_an_unknown_framework_says_so_instead_of_renaming_nothing() -> None:
    """A silent fall-back to the canonical vocabulary is indistinguishable from a config typo."""
    with pytest.raises(ValueError, match="unknown framework 'kears'"):
        MetricNames()({"loss": 1.0}, source="kears")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown framework"):
        vocabulary("nope")


def test_the_runtime_tuples_come_from_the_annotations() -> None:
    """One source of truth: `get_args`, never a restated parallel list."""
    assert SPLITS == ("train", "val", "test")
    assert set(FRAMEWORKS) == set(NATIVE) | {"matrainer"}


@pytest.mark.parametrize("split", SPLITS)
def test_a_vocabulary_without_a_word_for_a_split_keeps_the_canonical_one(split: str) -> None:
    """fastai never reports a test split; rendering into it must not drop the split silently."""
    rendered = render_metric_name(split, "loss", target="fastai")
    assert rendered.endswith("loss") and rendered != "loss"


def test_parse_and_render_are_available_without_the_configurable() -> None:
    """The two halves are plain functions, so a caller with no config wiring can use them."""
    parsed: Dict[str, Optional[str]] = dict(zip(("split", "base"), parse_metric_name("eval_loss", "huggingface")))
    assert parsed == {"split": "val", "base": "loss"}
    assert render_metric_name("val", "loss", "keras") == "val_loss"
    assert vocabulary("fastai").default_split == "val"

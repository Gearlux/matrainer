"""The three framework-free metric modules, and the one mistake they exist to prevent.

Asserts what `docs/metrics.md` claims:

* `metric_slots` — how a shared metric spec and a split's own spec combine (add, shadow,
  concatenate, replace), for every shape a vocabulary can take.
* `metric_functions` — that batch-averaging a macro-F1 is WRONG, measured, which is the whole
  reason a hand-written loop should accumulate a confusion matrix instead.
* `metric_names` — translating one engine's metric names to another's, in both directions.

Nothing here imports torch: these three modules are stdlib + numpy on purpose, so a
deliberately framework-free training loop can use them.

Run:  python examples/metrics.py
"""

import numpy as np

from matrainer import MetricNames
from matrainer.metric_functions import accumulate_confusion, macro_f1, zero_confusion
from matrainer.metric_slots import merge_metric_specs, union_metric_specs


def combining_metric_specs() -> None:
    """A runnable takes `metrics` (every split) plus `val_metrics` (validation only)."""
    # Two mappings merge by key: the split ADDS what the shared set does not have...
    assert merge_metric_specs({"acc": "A"}, {"confusion": "C"}) == {"acc": "A", "confusion": "C"}
    # ...and SHADOWS a name it restates, so a split overrides a template instead of doubling it.
    assert merge_metric_specs({"acc": "A"}, {"acc": "A2"}) == {"acc": "A2"}
    # Two sequences concatenate.
    assert merge_metric_specs(["accuracy"], ["top_k"]) == ["accuracy", "top_k"]
    # `None` means "this split chose nothing", NOT "this split chose empty".
    assert merge_metric_specs({"acc": "A"}, None) == {"acc": "A"}
    # Shapes that cannot combine key-wise: the split REPLACES, rather than being ignored.
    assert merge_metric_specs({"acc": "A"}, ["top_k"]) == ["top_k"]

    # An engine that compiles ONE list for every phase (keras, transformers) cannot be narrowed,
    # so everything asked for anywhere is computed everywhere.
    assert union_metric_specs({"acc": "A"}, {"confusion": "C"}, {"iou": "I"}) == {
        "acc": "A",
        "confusion": "C",
        "iou": "I",
    }
    print("metric_slots: add, shadow, concatenate, replace, and the union — all as documented")


def why_a_loop_accumulates_a_confusion_matrix() -> None:
    """Score each batch and average, and a macro-F1 comes out low. Measured, not asserted on faith."""
    classes, batch, batches = 10, 32, 40
    rng = np.random.default_rng(0)

    matrix = zero_confusion(classes)
    per_batch = []
    for _ in range(batches):
        targets = rng.integers(0, classes, size=batch)
        predictions = targets.copy()
        wrong = rng.random(batch) > 0.9  # ≈ 90 % accuracy
        predictions[wrong] = (targets[wrong] + 1) % classes
        # What a model hands you is one score per class; `accumulate_confusion` argmaxes it.
        logits = np.zeros((batch, classes))
        logits[np.arange(batch), predictions] = 1.0

        accumulate_confusion(matrix, logits, targets)  # the C×C array, size independent of N
        per_batch.append(macro_f1(accumulate_confusion(zero_confusion(classes), logits, targets)))

    global_f1 = macro_f1(matrix)
    averaged = float(np.mean(per_batch))

    # The claim: batch-averaging is always LOWER, because a class absent from a batch scores 0
    # for that batch and drags the mean down.
    assert averaged < global_f1, (averaged, global_f1)
    assert 0.01 < global_f1 - averaged < 0.10, global_f1 - averaged
    print(
        f"macro-F1 over the whole run {global_f1:.4f}, batch-averaged {averaged:.4f} "
        f"— {100 * (global_f1 - averaged):.1f} points lost by averaging"
    )


def translating_metric_names() -> None:
    """Every tracker records the key it is handed, so the same run lands in different columns."""
    names = MetricNames()  # default target: matrainer's canonical `<split>/<name>`

    assert names({"loss": 0.5, "val_loss": 0.4}, source="keras") == {"train/loss": 0.5, "val/loss": 0.4}
    # fastai computes its metrics on the validation pass and prefixes none of them.
    assert names({"accuracy": 0.9}, source="fastai") == {"val/accuracy": 0.9}
    assert names({"eval_accuracy": 0.9}, source="huggingface") == {"val/accuracy": 0.9}

    # ...and back out again, for a backend that wants Keras spelling.
    assert MetricNames(target="keras")({"val/loss": 0.4}) == {"val_loss": 0.4}

    # `epoch` and `step` are the x-axis, not a measurement: they pass through untouched.
    assert names({"epoch": 3, "step": 120}, source="keras") == {"epoch": 3, "step": 120}

    # Idempotent, so a loop already naming things canonically can declare its real source.
    once = names({"loss": 0.5}, source="keras")
    assert names(once, source="matrainer") == once
    print("metric_names: keras/fastai/huggingface in, canonical out, keras back out, epoch untouched")


def main() -> None:
    combining_metric_specs()
    why_a_loop_accumulates_a_confusion_matrix()
    translating_metric_names()


if __name__ == "__main__":
    main()

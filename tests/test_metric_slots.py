"""The per-split metric-slot combination rule — ``merge_metric_specs`` / ``union_metric_specs``.

A training runnable takes ``metrics`` (what every split measures) plus ``<split>_metrics`` (what
that split measures on top). How the two combine must not vary between engines or tasks, even
though what a "metric" *is* does — a torchmetrics template mapping, a list of ``keras.metrics``
objects, a list of fastai function names. Hence one module of pure functions here, applied to
whichever vocabulary the caller's engine uses.
"""

from __future__ import annotations

import pytest

from matrainer.metric_slots import merge_metric_specs, union_metric_specs


# ---------------------------------------------------------------------------
# merge — for an engine whose loop the caller owns: each split gets its own set
# ---------------------------------------------------------------------------
def test_a_split_slot_merges_over_the_shared_mapping() -> None:
    """Merging is what makes the slots usable — a replace-only slot forces restatement."""
    assert merge_metric_specs({"acc": 1}, {"confusion": 2}) == {"acc": 1, "confusion": 2}


def test_a_same_named_entry_shadows_the_shared_one() -> None:
    """So a split can OVERRIDE a template rather than only add to it."""
    assert merge_metric_specs({"acc": 1}, {"acc": 2}) == {"acc": 2}


def test_two_sequences_concatenate() -> None:
    """An engine-native metric LIST has no keys to merge on, so the split's entries append."""
    assert merge_metric_specs(["accuracy"], ["top_k_accuracy"]) == ["accuracy", "top_k_accuracy"]


def test_a_string_is_one_metric_not_a_sequence_of_characters() -> None:
    """`Sequence` would happily concatenate "abc" — a named metric must survive intact."""
    assert merge_metric_specs("accuracy", "top_k") == "top_k"


@pytest.mark.parametrize(
    "shared,split,expected",
    [
        ({"acc": 1}, None, {"acc": 1}),  # nothing chosen for this split -> the shared set
        (None, {"acc": 1}, {"acc": 1}),  # nothing shared -> the split's own
        (None, None, None),  # nothing at all stays "not chosen", never an empty set
    ],
)
def test_none_means_not_chosen_on_either_side(shared: object, split: object, expected: object) -> None:
    assert merge_metric_specs(shared, split) == expected


def test_incompatible_shapes_mean_the_split_replaces() -> None:
    """There is no key-wise way to combine a list with a mapping.

    Replacing is the lesser evil: ignoring the split's spec would be a wired metric set that
    silently does nothing, which is the failure this whole surface exists to avoid.
    """
    assert merge_metric_specs({"acc": 1}, ["accuracy"]) == ["accuracy"]
    assert merge_metric_specs(["accuracy"], {"acc": 1}) == {"acc": 1}


# ---------------------------------------------------------------------------
# union — for an engine that computes ONE set for several phases
# ---------------------------------------------------------------------------
def test_the_union_is_what_a_single_set_engine_computes() -> None:
    """`keras.Model.compile` takes ONE list; the honest answer is every split's metrics at once."""
    assert union_metric_specs(["accuracy"], None, ["auc"], ["precision"]) == ["accuracy", "auc", "precision"]


def test_the_union_does_not_report_a_metric_twice() -> None:
    """Overlapping slots would otherwise compile two entries with the same metric name."""
    assert union_metric_specs(["accuracy"], ["accuracy"], ["auc"]) == ["accuracy", "auc"]


def test_the_union_of_mappings_merges_by_key() -> None:
    assert union_metric_specs({"a": 1}, {"b": 2}, {"a": 3}) == {"a": 3, "b": 2}


# ---------------------------------------------------------------------------
# the property that lets the framework-free backends use it
# ---------------------------------------------------------------------------
def test_the_module_imports_nothing_but_the_standard_library() -> None:
    """The reason it sits at the top level rather than under ``matrainer.torch``.

    A JAX / MLX / Keras backend must be able to apply the rule without a torch import appearing
    in its dependency graph, and those backends' own suites AST-scan for exactly that. Checked
    by SOURCE rather than by ``sys.modules``, because an earlier test in the same process may
    already have imported anything.
    """
    import ast
    import pathlib
    import sys

    import matrainer.metric_slots

    source = pathlib.Path(str(matrainer.metric_slots.__file__)).read_text()
    imported = {
        (node.module or "").split(".")[0] for node in ast.walk(ast.parse(source)) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imported <= set(sys.stdlib_module_names), f"non-stdlib imports: {imported - set(sys.stdlib_module_names)}"

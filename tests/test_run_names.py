"""Tests for matrainer.run_names — the random slug plus the task→prefix mapping."""

import pytest

from matrainer.run_names import generate_run_name, run_prefix


def test_bare_slug_is_two_words() -> None:
    name = generate_run_name()
    parts = name.split("_")
    assert len(parts) == 2 and all(parts)


def test_prefix_is_prepended() -> None:
    name = generate_run_name(prefix="train")
    assert name.startswith("train_")
    # prefix + adjective + noun
    assert len(name.split("_")) == 3


def test_postfix_is_appended() -> None:
    name = generate_run_name(postfix="fold0")
    assert name.endswith("_fold0")
    assert len(name.split("_")) == 3


def test_prefix_and_postfix() -> None:
    name = generate_run_name(prefix="train", postfix="fold0")
    assert name.startswith("train_") and name.endswith("_fold0")
    assert len(name.split("_")) == 4


def test_custom_separator() -> None:
    name = generate_run_name(prefix="train", separator="-")
    assert name.startswith("train-") and "_" not in name
    assert len(name.split("-")) == 3


def test_names_are_random() -> None:
    # Extremely unlikely to collide across coolname's large word lists.
    names = {generate_run_name() for _ in range(20)}
    assert len(names) > 1


def test_empty_parts_skipped() -> None:
    # Empty prefix/postfix must not introduce leading/trailing/double separators.
    name = generate_run_name(prefix="", postfix="")
    assert not name.startswith("_") and not name.endswith("_") and "__" not in name


# ---------------------------------------------------------------------------
# run_prefix — the task verb a run NAME is prefixed with
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("fit", "train"),
        ("predict", "predict"),
        ("evaluate", "evaluate"),
        # `test` and `evaluate` are two entry points but ONE kind of run — splitting them
        # across two prefixes would split one concept in every tracking backend.
        ("test", "evaluate"),
    ],
)
def test_run_prefix_maps_the_task_verbs(task: str, expected: str) -> None:
    assert run_prefix(task) == expected


def test_an_unknown_verb_is_an_evaluation_rather_than_an_error() -> None:
    """A run name is a LABEL — refusing to produce one is never the right answer."""
    assert run_prefix("some_future_verb") == "evaluate"
    assert run_prefix("") == "evaluate"


def test_it_composes_into_the_one_line_every_backend_writes() -> None:
    assert generate_run_name(prefix=run_prefix("fit")).startswith("train_")

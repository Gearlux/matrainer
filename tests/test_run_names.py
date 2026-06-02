"""Tests for marainer.run_names.generate_run_name."""

from marainer.run_names import generate_run_name


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

"""The PEP-562 lazy-name machinery (`matrainer._exports.lazy_exports`).

The `_BACKEND_NAMES` table stays project data; what is pinned here is the resolver the task
projects' package roots share — resolution on first access, the ImportError that names the
exact extra, the untranslated pass-through for a core name, and the dir() advertisement.
"""

import pytest

from matrainer._exports import lazy_exports


def _pair(names: dict) -> tuple:  # helper: build against THIS package
    return lazy_exports("matrainer", names)


def test_a_name_resolves_to_the_attribute_on_first_access() -> None:
    getattr_, _ = _pair({"Tracker": ("matrainer.tracking.base", "")})
    from matrainer.tracking.base import Tracker

    assert getattr_("Tracker") is Tracker


def test_an_unknown_name_is_an_attribute_error() -> None:
    getattr_, _ = _pair({})
    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        getattr_("Nope")


def test_a_missing_backend_names_the_exact_extra() -> None:
    getattr_, _ = _pair({"Thing": ("matrainer._no_such_module", "lightning")})
    with pytest.raises(ImportError) as excinfo:
        getattr_("Thing")
    message = str(excinfo.value)
    assert "pip install 'matrainer[lightning]'" in message
    assert excinfo.value.__cause__ is not None, "the underlying error must stay visible (chained)"


def test_an_empty_extra_passes_the_import_error_through_untranslated() -> None:
    """A core name's failure is real — advertising an extra that does not exist would be a
    fix that fails."""
    getattr_, _ = _pair({"Thing": ("matrainer._no_such_module", "")})
    with pytest.raises(ImportError) as excinfo:
        getattr_("Thing")
    assert "pip install" not in str(excinfo.value)


def test_dir_advertises_the_lazy_names() -> None:
    _, dir_ = _pair({"SomethingLazy": ("matrainer.tracking.base", "")})
    names = dir_()
    assert "SomethingLazy" in names
    assert "run_names" in names or "tracking" in names  # the module's own globals stay listed

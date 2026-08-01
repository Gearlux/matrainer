"""Backends are DISCOVERED, not listed.

The claim these pin: adding a backend is a single new file — no import line, no ``__all__``
entry, no registry. The old shape needed both, and an out-of-tree backend could appear in
neither.
"""

import importlib
from pathlib import Path
from typing import Any, Iterator

import pytest

import marainer
from marainer import tracking
from marainer.tracking.base import Tracker

EXPECTED = {
    "AimTracker",
    "ClearMLTracker",
    "CsvTracker",
    "MlflowTracker",
    "TensorBoardTracker",
    "WandbTracker",
}


@pytest.fixture
def fresh_scan() -> Iterator[None]:
    """Drop the discovery cache around a test, and restore it afterwards."""
    before = tracking._backends
    tracking._backends = None
    try:
        yield
    finally:
        tracking._backends = before


def test_every_in_tree_backend_is_discovered() -> None:
    assert EXPECTED <= set(tracking.discover_backends())


def test_discovery_matches_the_backend_modules_on_disk(fresh_scan: None) -> None:
    """The set is derived from the package directory, not from a list someone maintains."""
    package_dir = Path(tracking.__file__).parent
    modules = {p.stem for p in package_dir.glob(f"*{tracking.BACKEND_SUFFIX}.py")}
    discovered = {cls.__module__.rsplit(".", 1)[-1] for cls in tracking.discover_backends().values()}
    assert discovered == modules


def test_discovered_values_are_tracker_subclasses() -> None:
    for name, cls in tracking.discover_backends().items():
        assert issubclass(cls, Tracker) and cls is not Tracker, name


def test_the_base_classes_are_not_mistaken_for_backends() -> None:
    found = tracking.discover_backends()
    assert "Tracker" not in found and "TrackerGroup" not in found


@pytest.mark.parametrize("module", [tracking, marainer])
def test_backends_resolve_as_attributes(module: Any) -> None:
    assert module.CsvTracker is tracking.discover_backends()["CsvTracker"]
    assert module.AimTracker.__module__ == "marainer.tracking.aim_backend"


@pytest.mark.parametrize("module", [tracking, marainer])
def test_all_and_dir_include_the_backends(module: Any) -> None:
    assert EXPECTED <= set(module.__all__)
    assert EXPECTED <= set(dir(module))


@pytest.mark.parametrize("module", [tracking, marainer])
def test_an_unknown_attribute_still_raises(module: Any) -> None:
    with pytest.raises(AttributeError, match="has no attribute 'NopeTracker'"):
        getattr(module, "NopeTracker")


def test_discovery_returns_a_copy_so_a_caller_cannot_poison_it() -> None:
    tracking.discover_backends()["CsvTracker"] = object  # type: ignore[assignment]
    assert tracking.discover_backends()["CsvTracker"] is not object


def test_a_new_backend_file_needs_no_edit_to_an_existing_file(fresh_scan: None) -> None:
    """The whole point: drop a module in, and it is importable by name.

    Writes a real module into the package directory rather than mocking the scan — mocking
    it would test the mock, and what is being claimed is a property of the file system
    layout.
    """
    package_dir = Path(tracking.__file__).parent
    new_module = package_dir / f"zzztemp{tracking.BACKEND_SUFFIX}.py"
    new_module.write_text(
        "from typing import Any, Mapping, Optional\n"
        "from marainer.tracking.base import Tracker\n"
        "\n"
        "class ZzzTempTracker(Tracker):\n"
        "    @property\n"
        "    def experiment(self) -> Any:\n"
        "        return 'handle'\n"
        "    def log_hyperparams(self, params: Mapping[str, Any]) -> None:\n"
        "        pass\n"
        "    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:\n"
        "        pass\n"
    )
    try:
        tracking._backends = None
        assert "ZzzTempTracker" in tracking.discover_backends()
        assert tracking.ZzzTempTracker.__name__ == "ZzzTempTracker"  # resolved via __getattr__
        assert "ZzzTempTracker" in dir(tracking)
    finally:
        new_module.unlink()
        importlib.invalidate_caches()
        tracking._backends = None

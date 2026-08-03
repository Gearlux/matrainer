"""Model identity — ``marainer.uri``: the ``ModelRef`` vocabulary and the ``model_refs`` walk."""

import ast
from pathlib import Path
from typing import Any, List, Optional, Sequence, Set

import pytest

from marainer.uri import MAX_WRAPPER_DEPTH, ModelRef, SupportsModelIdentity, file_model_ref, hf_model_ref, model_refs


class _Wrapper:
    """A model wrapper standing in for a real one — it only has to report and to wrap.

    ``refs=None`` leaves ``model_refs`` UNSET rather than empty, which is the distinction the walk
    turns on: a runnable holding a model slot does not implement the protocol at all (follow it),
    while a wrapper reporting an empty list has answered (believe it).
    """

    def __init__(self, refs: Optional[Sequence[Any]] = None, model: Any = None) -> None:
        if refs is not None:
            self.model_refs = refs
        self.model = model


# --- the vocabulary ----------------------------------------------------------------------------


def test_a_ref_defaults_to_a_pretrained_artifact_with_no_link() -> None:
    ref = ModelRef(uri="zoo://thing")
    assert ref.url is None
    assert ref.role == "pretrained"


def test_hf_model_ref_names_the_repo_and_links_to_it() -> None:
    ref = hf_model_ref("timm/resnet50.a1_in1k")
    assert ref.uri == "hf://models/timm/resnet50.a1_in1k"
    assert ref.url == "https://huggingface.co/timm/resnet50.a1_in1k"
    assert ref.role == "pretrained"


def test_hf_model_ref_pins_a_revision_when_given_one() -> None:
    assert hf_model_ref("org/net", revision="abc123").uri == "hf://models/org/net?revision=abc123"


def test_hf_model_ref_uses_the_models_authority_not_the_datasets_one() -> None:
    """The two halves of a run's lineage must read as one vocabulary, not two spellings."""
    assert hf_model_ref("org/net").uri.startswith("hf://models/")


def test_file_model_ref_is_absolute_and_has_no_url(tmp_path: Path) -> None:
    ref = file_model_ref(tmp_path / "epoch=3.ckpt")
    assert ref.uri == (tmp_path / "epoch=3.ckpt").resolve().as_uri()
    assert ref.url is None  # a local artifact has no web page; inventing one would be a lie
    assert ref.role == "checkpoint"


def test_file_model_ref_can_name_an_artifact_brought_in_from_outside(tmp_path: Path) -> None:
    """A SavedModel wired into a config is `pretrained`: this run did not produce it."""
    assert file_model_ref(tmp_path, role="pretrained").role == "pretrained"


def test_a_relative_path_is_resolved_so_the_reference_travels(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    assert file_model_ref("runs/last.ckpt").uri == (tmp_path / "runs" / "last.ckpt").resolve().as_uri()


def test_a_reporting_wrapper_satisfies_the_protocol() -> None:
    assert isinstance(_Wrapper([ModelRef("zoo://x")]), SupportsModelIdentity)


# --- the walk ----------------------------------------------------------------------------------


def test_model_refs_reads_a_wrappers_own_references() -> None:
    refs = [hf_model_ref("org/net"), file_model_ref("/tmp/last.ckpt")]
    assert model_refs(_Wrapper(refs)) == refs


def test_model_refs_answers_empty_for_things_that_report_nothing() -> None:
    assert model_refs(None) == []
    assert model_refs(object()) == []
    assert model_refs(_Wrapper([])) == []


def test_model_refs_follows_a_model_attribute_to_the_wrapper_underneath() -> None:
    """A runnable holds its model slot; asking the runnable must reach the wrapper."""
    inner = _Wrapper([hf_model_ref("org/net")])
    runnable = _Wrapper(model=_Wrapper(model=inner))
    assert model_refs(runnable) == [hf_model_ref("org/net")]


def test_a_wrapper_reporting_nothing_is_believed_rather_than_walked_past() -> None:
    """A wrapper that implements the protocol is authoritative, empty answer included.

    Falling through on an empty list would follow `.model` — which on real wrappers is a property
    that BUILDS the network, so a from-scratch model's provenance read would download weights (or
    raise, when the runnable has not injected `num_classes` yet). Caught by a consumer test.
    """

    class BuildingWrapper:
        model_refs: List[ModelRef] = []

        @property
        def model(self) -> Any:
            raise AssertionError("the `.model` hop must not be taken past a reporting wrapper")

    assert model_refs(BuildingWrapper()) == []


def test_a_wrapper_with_its_own_references_wins_over_the_one_it_wraps() -> None:
    inner = _Wrapper([hf_model_ref("org/inner")])
    outer = _Wrapper([hf_model_ref("org/outer")], model=inner)
    assert model_refs(outer) == [hf_model_ref("org/outer")]


def test_duplicate_uris_collapse_keeping_the_first() -> None:
    ref = hf_model_ref("org/net")
    assert model_refs(_Wrapper([ref, ModelRef(ref.uri, None, "checkpoint"), file_model_ref("/tmp/a")])) == [
        ref,
        file_model_ref("/tmp/a"),
    ]


def test_a_plain_tuple_is_accepted_because_the_protocol_is_structural() -> None:
    assert model_refs(_Wrapper([("zoo://x", None, "checkpoint")])) == [ModelRef("zoo://x", None, "checkpoint")]


def test_an_empty_uri_is_dropped_rather_than_recorded() -> None:
    assert model_refs(_Wrapper([ModelRef(""), hf_model_ref("org/net")])) == [hf_model_ref("org/net")]


def test_a_cyclic_wrapper_chain_terminates() -> None:
    node = _Wrapper()
    node.model = node
    assert model_refs(node) == []


def test_a_chain_deeper_than_the_cap_gives_up_rather_than_walking_forever() -> None:
    chain: Any = _Wrapper([hf_model_ref("org/net")])
    for _ in range(MAX_WRAPPER_DEPTH + 2):
        chain = _Wrapper(model=chain)
    assert model_refs(chain) == []


# --- the "asking never builds" property --------------------------------------------------------


def test_a_deferred_marker_is_materialized_before_being_asked() -> None:
    """A `!lazy:` model straight out of a config must answer without a flow() at the call site."""
    from confluid import LazyClass, configurable

    @configurable
    class MarkerToy:
        def __init__(self, name: str = "org/net") -> None:
            self.name = name

        @property
        def model_refs(self) -> List[ModelRef]:
            return [hf_model_ref(self.name)]

    assert model_refs(LazyClass(MarkerToy)) == [hf_model_ref("org/net")]


def test_asking_a_live_model_does_not_run_its_solidify_hook() -> None:
    """The whole property: `flow()` on a LIVE object runs `solidify()`, which BUILDS the model.

    A model wrapper's build creates the network and downloads its weights — and raises outright
    when the runnable has not injected `num_classes` yet — so reading a run's provenance must
    never take that path.
    """
    from confluid import configurable

    built: List[int] = []

    @configurable
    class SolidifyToy:
        def __init__(self, name: str = "org/net") -> None:
            self.name = name

        def solidify(self) -> Any:
            built.append(1)
            raise AssertionError("solidify must not run while reading model identity")

        @property
        def model_refs(self) -> List[ModelRef]:
            return [hf_model_ref(self.name)]

    assert model_refs(SolidifyToy()) == [hf_model_ref("org/net")]
    assert built == []


def test_the_module_imports_nothing_but_the_standard_library() -> None:
    """A framework-free model wrapper must name its weights without a tracking backend or a
    framework entering its dependency graph — the same rule `marainer.metric_slots` obeys.

    Read from the SOURCE with `ast` so the check fails wherever the mistake is made, and so a
    body-local import (the confluid one inside `_materialize`) is judged by the same rule.
    """
    import marainer.uri as module

    source = ast.parse(Path(module.__file__).read_text())
    roots: Set[str] = set()
    for node in ast.walk(source):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    # confluid is the one exception and it is BODY-LOCAL on purpose: materializing a deferred
    # marker needs it, and nothing else in the module does.
    assert roots <= {"pathlib", "typing", "urllib", "confluid"}, roots
    assert "confluid" not in {
        node.module.split(".")[0]
        for node in ast.iter_child_nodes(source)
        if isinstance(node, ast.ImportFrom) and node.module
    }, "confluid must stay a body-local import"


@pytest.mark.parametrize("forbidden", ["torch", "pytorch_lightning", "keras", "tensorflow"])
def test_importing_model_identity_pulls_in_no_framework(forbidden: str) -> None:
    """A model wrapper naming its weights must not drag a framework in to do it.

    Only the FRAMEWORKS are checked, not `marainer.tracking`: importing any submodule runs the
    package `__init__`, which is the tracking surface, so that one is not avoidable from here and
    claiming otherwise would be a test that lies. What this module genuinely owes — and what the
    source check above enforces — is adding no dependency of its own.
    """
    import subprocess
    import sys

    code = f"import marainer.uri, sys; assert {forbidden!r} not in sys.modules, {forbidden!r}"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

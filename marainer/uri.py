"""Model identity — the artifacts a model wrapper loads, named so a run record can point at them.

The model counterpart of the dataset identity a data engine's sources carry. A run's weights come
from somewhere — a hub repository, a zoo variant, a checkpoint a previous run wrote — and a record
that cannot name that source leaves a reader guessing which registry resolves a bare architecture
name. This module is the opt-in protocol for saying so, plus the helpers that keep one spelling
across the wrappers that resolve to the same place.

**It reports a LIST, and that is the deliberate difference from the dataset side.** A dataset
source reads one dataset; a model wrapper routinely *composes* artifacts into one network — a
pretrained backbone with a fine-tuned checkpoint loaded over it used two, and both belong in the
record. A pipeline that runs several models is the same shape one level up. So the protocol is
plural-first and every reference carries the :data:`ModelRole` it played, where the dataset
protocol is two plain strings. The shapes differ because the things differ.

Top level rather than under :mod:`marainer.tracking`, and importing nothing but the standard
library, for the reason :mod:`marainer.metric_slots` is: a model wrapper in a deliberately
framework-free project must be able to name its weights without a framework entering its
dependency graph, and every consumer of this module is a wrapper rather than a tracker. It is not
re-exported from the package root, which is the tracking surface and which nothing here touches.
"""

from pathlib import Path
from typing import Any, List, Literal, NamedTuple, Optional, Protocol, Sequence, runtime_checkable
from urllib.parse import quote, urlencode

#: What a referenced artifact contributed to the running model. ``pretrained`` is weights the run
#: did NOT produce — a hub checkpoint, a zoo variant, a SavedModel brought in from elsewhere;
#: ``checkpoint`` is weights a previous run of this project wrote. Two values rather than a
#: taxonomy: the distinction a reader needs is "did this come from outside, or from us?".
ModelRole = Literal["pretrained", "checkpoint"]

#: How deep :func:`model_refs` follows ``.model`` wrappers. A classifier holding a wrapper holding
#: a network is two hops; the cap is what keeps a self-referential graph from hanging a tracking
#: call, exactly as on the dataset side.
MAX_WRAPPER_DEPTH = 16

#: Where a model repository lives on the Hugging Face Hub, and where it is browsable. The
#: ``models`` authority mirrors the ``datasets`` one a data engine's sources use, so the two halves
#: of a run's lineage read as one vocabulary.
HF_URI_PREFIX = "hf://models/"
HF_BROWSE_PREFIX = "https://huggingface.co/"


class ModelRef(NamedTuple):
    """One artifact a model loaded.

    Args:
        uri: The canonical, machine-parseable identifier — ``hf://models/timm/resnet50.a1_in1k``,
            ``file:///runs/epoch%3D3.ckpt``. This is the string two runs are compared on.
        url: A link a person can open, or ``None`` when the artifact has no web page (a local
            checkpoint has none, and inventing one would be a lie).
        role: What it contributed — see :data:`ModelRole`.
    """

    uri: str
    url: Optional[str] = None
    role: ModelRole = "pretrained"


@runtime_checkable
class SupportsModelIdentity(Protocol):
    """A model wrapper that can name the artifacts it loads.

    One member, and it is a property returning a possibly-empty sequence. Implementations MUST be
    cheap and side-effect free: this is asked of a wrapper that may never run, so it reads stored
    configuration and never builds, downloads or opens anything.

    **An empty list is the right answer when nothing external was loaded.** A model built from an
    architecture with random weights (``pretrained=False``, no checkpoint) referenced no artifact;
    the architecture it used is configuration, recorded as such, and inventing a reference for it
    would name something nobody loaded.
    """

    @property
    def model_refs(self) -> Sequence[ModelRef]:
        """The artifacts this model loads, in the order they are applied."""
        ...


def hf_model_ref(repo_id: str, revision: Optional[str] = None, role: ModelRole = "pretrained") -> ModelRef:
    """A reference to a Hugging Face Hub model repository.

    The one spelling shared by every wrapper that resolves to the Hub — several do, through
    different libraries — so they cannot drift into three near-identical URI formats that no longer
    compare equal for the same weights.

    Args:
        repo_id: The Hub repository, ``"google/vit-base-patch16-224"`` or ``"timm/resnet50.a1_in1k"``.
        revision: Optional git revision (branch, tag or commit) pinning the weights.
        role: What the artifact contributes; see :data:`ModelRole`.
    """
    path = quote(repo_id.strip("/"))
    uri = HF_URI_PREFIX + path
    if revision:
        uri = f"{uri}?{urlencode({'revision': revision})}"
    return ModelRef(uri=uri, url=HF_BROWSE_PREFIX + path, role=role)


def file_model_ref(path: Any, role: ModelRole = "checkpoint") -> ModelRef:
    """A reference to a weights file or directory on disk.

    Absolute, so the reference means the same thing read from anywhere; ``url`` is ``None`` because
    a local artifact has no web page.

    Args:
        path: The checkpoint / SavedModel / state-dict location.
        role: What the artifact contributes. Defaults to ``checkpoint`` — the common case — but a
            model you brought in from elsewhere is ``pretrained`` even when it lives on disk.
    """
    return ModelRef(uri=Path(path).resolve().as_uri(), url=None, role=role)


def model_refs(model: Any) -> List[ModelRef]:
    """Every model artifact reachable from ``model``, in order, deduplicated by URI.

    A DEFERRED model (a ``!class:`` / ``!lazy:`` marker straight out of a config) is materialized
    first, so a caller never has to remember which entry point flows and which does not.

    A wrapper that holds no references of its own but wraps another via ``.model`` — a runnable
    holding its model slot, a classifier holding its backbone — is followed, so asking the outer
    object works. Depth-capped and cycle-safe.

    Args:
        model: Any model, wrapper, runnable, or deferred config marker. ``None`` is accepted and
            answers ``[]``, so a caller needs no guard for an unwired slot.

    Example::

        model_refs(TimmModel(model_name="resnet50", checkpoint_path="runs/last.ckpt"))
        # [ModelRef('hf://models/timm/resnet50.a1_in1k', 'https://…', 'pretrained'),
        #  ModelRef('file:///w/runs/last.ckpt', None, 'checkpoint')]
    """
    current = _materialize(model)
    for _ in range(MAX_WRAPPER_DEPTH):
        if current is None:
            return []
        refs = getattr(current, "model_refs", None)
        # `is not None`, NOT truthiness: a wrapper that IMPLEMENTS the protocol is authoritative
        # even when its answer is "nothing loaded", and falling through on an empty list would
        # walk into `.model` — which on several real wrappers is a property that BUILDS the
        # network. Reading provenance would then download weights for a from-scratch model, and
        # raise outright when the runnable has not injected `num_classes` yet (measured against
        # two wrappers before this line said `is not None`).
        if refs is not None:
            return _deduplicate(refs)
        wrapped = getattr(current, "model", None)
        if wrapped is None or wrapped is current:
            return []
        current = _materialize(wrapped)
    return []


def _materialize(node: Any) -> Any:
    """Build ``node`` only if it is a DEFERRED config marker; hand a live object back untouched.

    Narrower than a bare ``flow()`` on purpose, and the difference is the whole "asking never
    builds" property. ``flow()`` on a LIVE object still runs confluid's post-construction
    ``solidify()`` hook — which for a model wrapper is exactly the expensive thing: it creates the
    network and downloads its weights. Reading a run's provenance must never do that, and it would
    also fail outright on a wrapper whose ``num_classes`` the runnable has not injected yet
    (measured: ``flow(TimmModel(model_name="resnet50"))`` raises before it can be asked anything).

    A marker, by contrast, has nothing to read until it is built, and building one is cheap by the
    workspace's lazy-construction rule.
    """
    from confluid import Fluid, flow

    return flow(node) if isinstance(node, Fluid) else node


def _deduplicate(refs: Any) -> List[ModelRef]:
    """Coerce a reported sequence into a clean ``ModelRef`` list, first occurrence per URI winning.

    Tolerant of a wrapper returning plain tuples rather than the NamedTuple, because the protocol
    is structural and a consumer may legitimately build one from data.
    """
    found: List[ModelRef] = []
    seen: set = set()
    for ref in refs:
        entry = ref if isinstance(ref, ModelRef) else ModelRef(*ref)
        if not entry.uri or entry.uri in seen:
            continue
        seen.add(entry.uri)
        found.append(entry)
    return found


__all__ = [
    "HF_BROWSE_PREFIX",
    "HF_URI_PREFIX",
    "MAX_WRAPPER_DEPTH",
    "ModelRef",
    "ModelRole",
    "SupportsModelIdentity",
    "file_model_ref",
    "hf_model_ref",
    "model_refs",
]

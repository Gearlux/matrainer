"""Generic, task-agnostic injection of dataset-derived model dimensions.

A training / evaluation config may declare the model or metrics as ``!lazy:``
and **omit** a dimension the dataset can supply (canonically ``num_classes``).
The dataset exposes those dimensions via a duck-typed
``derived_dimensions() -> dict`` hook — sonair's classification dataset returns
``{"num_classes": …}`` by walking its targets. The trainer / evaluator collects
them and injects only the **unset** ones when constructing the deferred nodes.

marainer never names a specific dimension here — the keys come entirely from the
dataset — so :class:`marainer.trainer.LightningTrainer` stays generic across
classification / segmentation / detection. Datasets that don't implement the
hook (e.g. detection / segmentation datasets) contribute nothing and are
untouched.

Contract: a node declared ``!lazy:`` with a dimension omitted is the signal
"derive this from the dataset". A node that sets the dimension explicitly (or a
plain ``!class:`` node) always wins — derived values never override configured
ones.
"""

from typing import Any, Dict, Iterable

from confluid import flow
from confluid.fluid import Fluid, Lazy  # the Fluid class produced by ``!lazy:`` (not the Lazy[T] annotation)
from logflow import get_logger

logger = get_logger(__name__)


def needs_derivation(nodes: Iterable[Any]) -> bool:
    """True when any node is a deferred ``Lazy`` — the "derive my dims" signal.

    Used to gate the (potentially expensive) dataset walk: when nothing is
    lazily deferred, the explicit path pays nothing.
    """
    return any(isinstance(n, Lazy) for n in nodes)


def collect_derived_dimensions(datasets: Iterable[Any]) -> Dict[str, Any]:
    """Merge the ``derived_dimensions()`` a dataset can supply about itself.

    Duck-typed: a dataset opts in by exposing ``derived_dimensions() -> dict``.
    When the same integer dimension appears across splits (e.g. ``num_classes``
    from train + val), the **max** wins — robust to a class id that appears in
    one split only. Non-integer collisions take the last writer.
    """
    merged: Dict[str, Any] = {}
    for ds in datasets:
        if ds is None:
            continue
        derive = getattr(ds, "derived_dimensions", None)
        if not callable(derive):
            continue
        for key, value in derive().items():
            if key in merged and isinstance(value, int) and isinstance(merged[key], int):
                merged[key] = max(merged[key], value)
            else:
                merged[key] = value
    return merged


def flow_node_with_dims(node: Any, dims: Dict[str, Any]) -> Any:
    """Flow a (possibly deferred) model / metrics node, injecting only *unset* dims.

    Derived dimensions are passed as runtime kwargs only for keys the node
    doesn't already set, so an explicitly-configured value always wins. For a
    deferred ``Lazy`` node — the shape emitted when a config omits a dimension —
    that injected kwarg is also what forces construction. For a
    ``MetricCollection`` the injected ``num_classes`` broadcasts into the nested
    metrics (which accept it) and is filtered out of the collection's own
    constructor.

    If a lazy node is left with nothing to inject it would stay deferred — that
    means a lazy slot expected a dimension no wired dataset supplied, so we raise
    a clear error instead of letting it fail cryptically downstream.
    """
    if not isinstance(node, Fluid):
        return flow(node)  # already live — flow() is idempotent
    inject = {k: v for k, v in dims.items() if k not in node.kwargs}
    result = flow(node, **inject)
    if isinstance(result, Fluid):
        target = getattr(node, "target", "?")
        raise RuntimeError(
            f"{type(node).__name__}(target={target!r}) is declared lazy but no wired dataset supplied "
            f"a value to construct it (derived dimensions available: {sorted(dims)}). Provide the value "
            "explicitly in the config, or wire a dataset that exposes derived_dimensions()."
        )
    return result

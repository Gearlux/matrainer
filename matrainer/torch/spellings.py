"""Dump spellings for the torch-family VALUES a run document must carry faithfully.

``confluid.dump()`` degrades a value it cannot respell to a bare
``{_target_: <module.qualname>}`` placeholder that reloads DEFAULT-constructed.
For the resolved-config artifact a tracked run archives, two torch-family values
used to land there and both lied on reload: a small ``torch.Tensor`` (a
class-weights vector, a loss's ``weight``) came back as an EMPTY tensor, and a
``torchmetrics.MetricCollection`` crashed the reload outright (its required
``metrics`` argument is not reconstructible from the live object's attributes).

The spellings registered here emit ordinary markers, so the reload needs no
machinery beyond confluid's own build pass. Registration happens at import —
``matrainer.torch``'s package ``__init__`` imports this module, the same
import-side-effect contract as a class registration — so any process training on
the torch stack has them, and a torch-free process keeps confluid's placeholder
behaviour, which is correct there.
"""

from typing import Any, Optional

import torch
from confluid import Target, register_dump_spelling
from confluid.introspect import slots
from torchmetrics import Metric, MetricCollection

#: Above this many elements a tensor DECLINES its spelling and keeps the
#: placeholder + warning: a weights VECTOR belongs in the run document, a
#: parameter matrix belongs in a checkpoint. Generous on purpose — the guard is
#: against megabyte YAML artifacts, not a statement about what is configuration.
TENSOR_DUMP_CAP = 65_536


def tensor_from_values(values: Any, dtype: str = "float32") -> torch.Tensor:
    """Rebuild a dumped tensor — the reload TARGET of the tensor spelling.

    The keyword twin of ``torch.tensor``, which cannot be a marker target itself:
    its ``data`` parameter is positional-only (a C extension), and a marker
    delivers kwargs. Deliberately unregistered — the emitted target is the
    importable dotted path, so the artifact reloads with no registrations at all.

    Args:
        values: The tensor's values as (nested) Python lists.
        dtype: The torch dtype NAME (``"float32"``, ``"int64"``, …).
    """
    return torch.tensor(values, dtype=getattr(torch, dtype))


def _spell_tensor(value: torch.Tensor) -> Optional[Target]:
    if value.numel() > TENSOR_DUMP_CAP:
        return None  # declined — confluid keeps the placeholder and says so
    return Target(
        tensor_from_values,
        values=value.detach().cpu().tolist(),
        dtype=str(value.dtype).removeprefix("torch."),
    )


def _spell_metric(value: Metric) -> Target:
    """A metric's document is its CONFIG surface, never the generic slot walk.

    Every torchmetrics constructor forwards ``**kwargs`` up to ``Metric``, whose
    ``__init__`` refuses leftovers — and the generic reconstruction emits body
    slots the chain cannot re-accept (``beta``, bound positionally by the F1
    subclasses, crashes the reload with "multiple values for 'beta'"). So: the
    ctor kwargs CAPTURED at flow when the metric came from a config (they carry
    exactly what the author wrote, ``compute_on_cpu`` included), else the
    signature params read off the live same-named attributes.
    """
    captured = getattr(value, "__confluid_kwargs__", None)
    if captured:
        return Target(type(value), **dict(captured))
    params = {
        s.name: getattr(value, s.name) for s in slots(type(value)) if s.kind == "keyword" and hasattr(value, s.name)
    }
    return Target(type(value), **params)


def _spell_metric_collection(value: MetricCollection) -> Target:
    # The base (un-prefixed) keys, each metric a LIVE object — the dumper's
    # ordinary registered-class branch renders each as a marker with its real
    # kwargs, so the collection's document is exactly what a config would write.
    kwargs: dict = {"metrics": dict(value.items(keep_base=True, copy_state=False))}
    if value.prefix is not None:
        kwargs["prefix"] = value.prefix
    if value.postfix is not None:
        kwargs["postfix"] = value.postfix
    return Target(MetricCollection, **kwargs)


register_dump_spelling(torch.Tensor, _spell_tensor)
register_dump_spelling(Metric, _spell_metric)
register_dump_spelling(MetricCollection, _spell_metric_collection)

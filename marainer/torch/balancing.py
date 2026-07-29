"""Class-imbalance weighting for torch losses.

Training on a skewed label distribution biases a model toward the majority
class. The standard remedy is inverse-frequency weighting: pass a per-class
``weight`` tensor to the loss so rare classes contribute proportionally more.

The policy is three steps, and they belong together — a consumer that only
copies the capability check still has to reimplement the arithmetic and the
injection:

1. :func:`loss_accepts_weight` — can this loss take weights at all?
2. :func:`inverse_frequency_weights` — derive them from the observed class ids.
3. :func:`apply_class_weights` — do both, and inject.

**Torch-specific by design.** ``weight=`` as the balancing knob is a
``torch.nn`` *constructor* convention; Keras takes ``class_weight`` on
``fit()`` instead. That is why this lives under ``marainer.torch`` rather than
at the package root — the subpackage split is the framework boundary.

Targets arrive as **already-encoded integer class ids**. Mapping strings to ids
is the caller's job (it owns the label map), which keeps this module free of
any dataset or encoding dependency.
"""

import inspect
from typing import Any, Iterable, Optional, Tuple

import torch
from confluid.fluid import Fluid
from confluid.registry import resolve_class
from loggair import get_logger
from torch import Tensor

logger = get_logger(__name__)

__all__ = ["apply_class_weights", "inverse_frequency_weights", "loss_accepts_weight"]


def loss_accepts_weight(loss: Any) -> bool:
    """True when ``loss``'s constructor declares a ``weight`` parameter.

    Works on a live loss instance OR a deferred :class:`confluid.fluid.Fluid`
    (a ``!class:``/``!lazy:`` marker), so the check can run before the loss is
    built — which is what lets a trainer decide whether walking the targets for
    weights is even worthwhile.

    Returns ``False`` for anything unresolvable or without an introspectable
    signature (a C-extension constructor), so the caller degrades to unweighted
    training rather than raising.

    Example::

        loss_accepts_weight(nn.CrossEntropyLoss())      # True
        loss_accepts_weight(nn.MSELoss())               # False
    """
    loss_cls = resolve_class(loss.target) if isinstance(loss, Fluid) else type(loss)
    if loss_cls is None:
        return False
    try:
        return "weight" in inspect.signature(loss_cls).parameters
    except (ValueError, TypeError):  # pragma: no cover - C-extension ctor without a signature
        return False


def inverse_frequency_weights(class_ids: Iterable[int], num_classes: int) -> Optional[Tensor]:
    """Per-class weights inversely proportional to observed frequency.

    ``w[c] = total / (num_classes * count[c])`` — so a class at exactly the mean
    frequency gets ``1.0``, rarer classes more, commoner classes less.

    Args:
        class_ids: Observed integer class ids (already encoded). Ids outside
            ``[0, num_classes)`` are ignored rather than raising — a stray label
            should not abort a training run.
        num_classes: Width of the returned tensor.

    Returns:
        A ``float32`` tensor of length ``num_classes``, or ``None`` when nothing
        was counted (an empty or fully out-of-range set), so the caller can tell
        "no weights" from "all-zero weights". A class observed **zero** times
        gets weight ``0.0``, not infinity.

    Example::

        inverse_frequency_weights([0, 0, 0, 1], num_classes=2)  # tensor([0.6667, 2.0000])
    """
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for class_id in class_ids:
        if 0 <= class_id < num_classes:
            counts[class_id] += 1
    total = float(counts.sum())
    if total <= 0:
        return None
    weights = torch.where(counts > 0, total / (num_classes * counts), torch.zeros_like(counts))
    return weights.to(torch.float32)


def apply_class_weights(
    loss_fn: Any,
    class_ids: Iterable[int],
    num_classes: int,
) -> Tuple[Any, Optional[Tensor]]:
    """Derive inverse-frequency weights and inject them into ``loss_fn``.

    The whole policy in one call: capability check, weight derivation, and
    injection — including the diagnostic when a loss cannot be balanced.

    Args:
        loss_fn: A live loss module OR a deferred :class:`~confluid.fluid.Fluid`.
        class_ids: Observed integer class ids (already encoded).
        num_classes: Number of classes.

    Returns:
        ``(loss_fn, weights)``. The loss is returned rather than mutated in
        place because a ``Fluid`` cannot be: it is FLOWED with ``weight=`` and
        the caller must rebind. ``weights`` is ``None`` when balancing did not
        happen (the loss takes no ``weight``, or nothing was counted), in which
        case ``loss_fn`` comes back untouched.

    Example::

        self.loss_fn, self.class_weights = apply_class_weights(
            self.loss_fn, encoded_targets, num_classes
        )
    """
    if not loss_accepts_weight(loss_fn):
        name: Any = getattr(loss_fn, "target", type(loss_fn).__name__)
        logger.warning(f"Class balancing requested but the loss ({name}) has no 'weight' arg; training unweighted.")
        return loss_fn, None

    weights = inverse_frequency_weights(class_ids, num_classes)
    if weights is None:
        logger.warning("Class balancing requested but no class ids were observed; training unweighted.")
        return loss_fn, None

    if isinstance(loss_fn, Fluid):
        from confluid import flow

        loss_fn = flow(loss_fn, weight=weights)
    else:
        loss_fn.weight = weights
    logger.info(f"Balanced loss: injected inverse-frequency class weights for {num_classes} classes.")
    return loss_fn, weights

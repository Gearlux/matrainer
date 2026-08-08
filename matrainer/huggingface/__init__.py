"""HuggingFace trainer machinery (``pip install matrainer[huggingface]``).

One member: :class:`~matrainer.huggingface.trainer.HuggingFaceRunnable`, the
``transformers.Trainer`` machinery base of the trainer API — the member that writes least
while swapping most, because the engine owns the whole training program (epoch, optimizer,
scheduler, gradient accumulation, mixed precision, distribution via Accelerate). Resolved
LAZILY (PEP 562) so importing this package name costs nothing until the class is touched;
the ``trainer`` module itself imports torch (the family base) at module level and
``transformers`` only inside the methods that need it.
"""

from typing import Any

_LAZY_NAMES = {
    "HuggingFaceRunnable": "matrainer.huggingface.trainer",
    "HuggingFaceCallback": "matrainer.huggingface.trainer",
}

__all__ = ["HuggingFaceCallback", "HuggingFaceRunnable"]


def __getattr__(name: str) -> Any:
    module_path = _LAZY_NAMES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_path), name)

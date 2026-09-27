"""Native-TensorFlow trainer machinery (``pip install matrainer[tensorflow]``).

One member: :class:`~matrainer.tensorflow.loop.TensorFlowRunnable`, the hand-written
``tf.data`` + ``GradientTape`` machinery base of the trainer API — the sibling of the
compile/fit member in :mod:`matrainer.keras`. Resolved LAZILY (PEP 562) so importing this
package name costs nothing until the class is touched; the ``loop`` module imports TensorFlow
at module level (and ``setdefault``s ``KERAS_BACKEND=tensorflow`` first — it is the lowest
layer that imports the engine).

Historical note: ``matrainer.tensorflow`` was the NAME of the keras tracking subpackage until
2026-07-30, when that one was corrected to :mod:`matrainer.keras` (an optional helper's extra
names the API it imports). This package is new (P5, 2026-08-07) and genuinely needs the
TensorFlow distribution, which is what the ``[tensorflow]`` extra installs.
"""

from typing import Any

_LAZY_NAMES = {
    "TensorFlowRunnable": "matrainer.tensorflow.loop",
    "TensorFlowNetwork": "matrainer.tensorflow.loop",
}

__all__ = ["TensorFlowNetwork", "TensorFlowRunnable"]


def __getattr__(name: str) -> Any:
    module_path = _LAZY_NAMES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_path), name)

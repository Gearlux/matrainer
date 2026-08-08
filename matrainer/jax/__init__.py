"""JAX trainer machinery (``pip install matrainer[jax]``).

One member: :class:`~matrainer.jax.loop.JaxLoopRunnable`, the jitted ``optax`` machinery base
of the trainer API — the owned-loop member for the functional engine: JAX brings function
transformations and expects you to write the loop yourself. Resolved LAZILY (PEP 562) so
importing this package name costs nothing until the class is touched; the ``loop`` module
imports jax/flax/optax at module level.
"""

from typing import Any

_LAZY_NAMES = {
    "JaxLoopRunnable": "matrainer.jax.loop",
    "EpochCallback": "matrainer.jax.loop",
    "JaxLoss": "matrainer.jax.loop",
    "JaxMetric": "matrainer.jax.loop",
    "JaxOptimizer": "matrainer.jax.loop",
    "as_optax_loss": "matrainer.jax.loop",
    "as_optax_optimizer": "matrainer.jax.loop",
}

__all__ = [
    "EpochCallback",
    "JaxLoopRunnable",
    "JaxLoss",
    "JaxMetric",
    "JaxOptimizer",
    "as_optax_loss",
    "as_optax_optimizer",
]


def __getattr__(name: str) -> Any:
    module_path = _LAZY_NAMES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_path), name)

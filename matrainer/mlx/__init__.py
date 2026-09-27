"""MLX trainer machinery (``pip install matrainer[mlx]`` — Apple silicon only).

One member: :class:`~matrainer.mlx.loop.MlxLoopRunnable`, the owned-loop machinery base for
Apple's MLX — the jax sibling in hook surface, with the engine's three facts (unified memory,
lazy arrays forced by ``mx.eval``, train/eval as module STATE) in the loop. Resolved LAZILY
(PEP 562) so importing this package name costs nothing until the class is touched.
"""

from typing import Any

_LAZY_NAMES = {
    "MlxLoopRunnable": "matrainer.mlx.loop",
    "EpochCallback": "matrainer.mlx.loop",
    "MlxLoss": "matrainer.mlx.loop",
    "MlxMetric": "matrainer.mlx.loop",
    "MlxOptimizer": "matrainer.mlx.loop",
    "as_mlx_loss": "matrainer.mlx.loop",
    "as_mlx_optimizer": "matrainer.mlx.loop",
}

__all__ = [
    "EpochCallback",
    "MlxLoopRunnable",
    "MlxLoss",
    "MlxMetric",
    "MlxOptimizer",
    "as_mlx_loss",
    "as_mlx_optimizer",
]


def __getattr__(name: str) -> Any:
    module_path = _LAZY_NAMES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_path), name)

"""Framework-neutral checkpoint policies.

A checkpoint policy declares *what* to save, once, independent of the training
framework. Each framework subpackage translates the declared policies into its
native mechanism — ``matrainer.lightning`` maps them to
``pytorch_lightning.callbacks.ModelCheckpoint``\\ s, a TensorFlow subpackage to
Keras ``ModelCheckpoint`` callbacks.

The policies:

* :class:`SaveLast` — keep the most recent checkpoint (overwrites).
* :class:`SaveEveryEpoch` — snapshot every ``every_n_epochs`` epochs.
* :class:`SaveBest` — track a monitored metric and keep the best ``top_k``.

All are Confluid ``@configurable`` (zero-arg constructible, no work in
``__init__`` — pure config carriers) so a single YAML wires them, and a run may
declare several (a list). Checkpointing is **opt-in**: no policy ⇒ the framework's
own default (or none) applies.

Example YAML::

    checkpoint:
      - !class:matrainer.checkpoints.SaveBest
          monitor: val/loss
          mode: min
          top_k: 3
      - !class:matrainer.checkpoints.SaveLast()
"""

from __future__ import annotations

from typing import Literal, Optional

from confluid import configurable

CheckpointMode = Literal["min", "max"]


@configurable
class CheckpointPolicy:
    """Base checkpoint policy — a framework-neutral declaration of what to save.

    Args:
        dirpath: Directory to write checkpoints into. ``None`` (default) lets the
            framework pick its run-relative default location.
        filename: Optional filename pattern (framework-interpreted); ``None`` = default.
    """

    def __init__(self, dirpath: Optional[str] = None, filename: Optional[str] = None) -> None:
        self.dirpath = dirpath
        self.filename = filename


@configurable
class SaveLast(CheckpointPolicy):
    """Keep the single most recent checkpoint (overwritten each save).

    Args:
        dirpath: Directory to write into (``None`` = framework default).
        filename: Optional filename pattern (``None`` = framework default).
    """


@configurable
class SaveEveryEpoch(CheckpointPolicy):
    """Snapshot a checkpoint every ``every_n_epochs`` epochs.

    Args:
        every_n_epochs: Save cadence in epochs (default ``1`` = every epoch).
        dirpath: Directory to write into (``None`` = framework default).
        filename: Optional filename pattern (``None`` = framework default).
    """

    def __init__(
        self,
        every_n_epochs: int = 1,
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> None:
        super().__init__(dirpath, filename)
        self.every_n_epochs = every_n_epochs


@configurable
class SaveBest(CheckpointPolicy):
    """Track a monitored metric and keep the best ``top_k`` checkpoints.

    Args:
        monitor: The logged metric key to monitor (e.g. ``"val/loss"``).
        mode: ``"min"`` keeps the lowest values, ``"max"`` the highest.
        top_k: How many best checkpoints to keep (default ``1``).
        dirpath: Directory to write into (``None`` = framework default).
        filename: Optional filename pattern (``None`` = framework default).
    """

    def __init__(
        self,
        monitor: str = "val/loss",
        mode: CheckpointMode = "min",
        top_k: int = 1,
        dirpath: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> None:
        super().__init__(dirpath, filename)
        self.monitor = monitor
        self.mode = mode
        self.top_k = top_k


__all__ = [
    "CheckpointMode",
    "CheckpointPolicy",
    "SaveBest",
    "SaveEveryEpoch",
    "SaveLast",
]

"""Marainer Lightning logger subclasses with a uniform ``log_all_checkpoints`` knob.

Each subclass wraps the corresponding ``pytorch_lightning.loggers`` class
and adds a single per-instance flag — ``log_all_checkpoints: bool``. The
flag is read by :class:`marainer.trainer.LightningTrainer` at run time to
decide how the auto-injected :class:`pytorch_lightning.callbacks.ModelCheckpoint`
should be configured:

* Default (any logger has the flag set to ``False`` or omitted): the default
  callback uses ``save_top_k=1`` with ``save_last=True`` — i.e. it keeps
  the best checkpoint by the first val metric plus the most recent one.
* When **any** wired logger has ``log_all_checkpoints=True``: the default
  callback escalates to ``save_top_k=-1`` so every checkpoint is written
  to disk and remains available for that logger's artifact store.

For :class:`MarainerMLFlowLogger` the flag also propagates into Lightning's
own ``log_model`` parameter (``True`` → log the best model; ``"all"`` →
log every checkpoint). The other backends (TensorBoard, CSV) have no
equivalent server-side concept — the checkpoint files already live next
to the logger's ``log_dir``, so the flag's only effect is governing the
shared ``save_top_k`` decision above.

User-facing YAML::

    lightning: !class:pytorch_lightning.Trainer
      logger:
        - !class:marainer.loggers.MarainerMLFlowLogger
          tracking_uri: file:./mlruns
          experiment_name: rfuav
          log_all_checkpoints: true   # archive every checkpoint in MLflow
        - !class:marainer.loggers.MarainerTensorBoardLogger
          save_dir: ./lightning_logs
          # log_all_checkpoints omitted → defaults to False

Sticking with the stock ``!class:pytorch_lightning.loggers.TensorBoardLogger``
form still works; ``getattr(lg, "log_all_checkpoints", False)`` returns
``False`` cleanly and the trainer falls back to the conservative default.
"""

from typing import Any

from pytorch_lightning.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger


class MarainerTensorBoardLogger(TensorBoardLogger):
    """``TensorBoardLogger`` + ``log_all_checkpoints`` marker.

    TensorBoard has no server-side notion of "model artifacts", so the
    flag is purely a marker for the trainer's checkpoint-callback wiring.
    """

    def __init__(self, *args: Any, log_all_checkpoints: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.log_all_checkpoints = log_all_checkpoints


class MarainerCSVLogger(CSVLogger):
    """``CSVLogger`` + ``log_all_checkpoints`` marker.

    CSV is purely a per-step metric sink; like TensorBoard it has no
    server-side artifact API, so the flag only nudges the shared
    ``save_top_k`` decision in the trainer.
    """

    def __init__(self, *args: Any, log_all_checkpoints: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.log_all_checkpoints = log_all_checkpoints


class MarainerMLFlowLogger(MLFlowLogger):
    """``MLFlowLogger`` + ``log_all_checkpoints`` marker.

    Also forwards into Lightning's own ``log_model`` parameter so MLflow
    picks the matching server-side behaviour: ``True`` → log the best
    checkpoint as an MLflow model when training ends; ``"all"`` → log
    every checkpoint as the ``ModelCheckpoint`` callback fires. If the
    caller already supplies an explicit ``log_model`` we respect it
    (``setdefault`` semantics).
    """

    def __init__(self, *args: Any, log_all_checkpoints: bool = False, **kwargs: Any) -> None:
        kwargs.setdefault("log_model", "all" if log_all_checkpoints else True)
        super().__init__(*args, **kwargs)
        self.log_all_checkpoints = log_all_checkpoints


__all__ = [
    "MarainerTensorBoardLogger",
    "MarainerCSVLogger",
    "MarainerMLFlowLogger",
]

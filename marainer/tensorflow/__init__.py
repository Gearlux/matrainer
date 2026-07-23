"""TensorFlow/Keras tracking utilities (``pip install marainer[tensorflow]``).

marainer does NOT own a TensorFlow training loop — a consumer's model is a plain
compiled ``keras.Model``. This subpackage provides the optional Keras tracking
helper: :func:`keras_tracking_callback`, a ``keras.callbacks.Callback`` that forwards
epoch logs (and, on train begin, run provenance) to marainer trackers — the
TensorFlow analogue of :class:`~marainer.lightning.callbacks.TrackingCallback`.

``tensorflow`` is imported lazily (raised with a clear ``pip install
marainer[tensorflow]`` message when the callback is built).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from marainer._imports import require
from marainer.tracking import provenance
from marainer.tracking.base import Tracker


def keras_tracking_callback(
    trackers: Sequence[Tracker],
    params: Optional[Mapping[str, Any]] = None,
    capture_log: bool = True,
) -> Any:
    """Build a ``keras.callbacks.Callback`` that forwards run tracking to marainer trackers.

    On train begin it logs run provenance (git / config / params); on each epoch end it
    forwards the epoch ``logs`` as metrics; on train end it captures the log file. Attach it
    via ``model.fit(..., callbacks=[keras_tracking_callback([tracker])])``.

    Args:
        trackers: The marainer trackers to log to.
        params: Optional hyperparameters (only JSON-scalar values forwarded).
        capture_log: Upload the loggair log file as an artifact on train end.
    """
    keras = require("tensorflow.keras", "tensorflow", feature="keras_tracking_callback")
    members = list(trackers)

    class _KerasTrackingCallback(keras.callbacks.Callback):  # type: ignore[misc, name-defined]
        def on_train_begin(self, logs: Optional[Dict[str, float]] = None) -> None:
            provenance.log_run_context(members, params)

        def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
            for tracker in members:
                tracker.log_metrics(dict(logs or {}), step=epoch)

        def on_train_end(self, logs: Optional[Dict[str, float]] = None) -> None:
            if capture_log:
                provenance.capture_log_file(members)

    return _KerasTrackingCallback()


__all__ = ["keras_tracking_callback"]

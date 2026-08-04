"""Keras tracking utilities (``pip install matrainer[keras]``).

matrainer does NOT own a TensorFlow training loop — a consumer's model is a plain
compiled ``keras.Model``. This subpackage provides the optional Keras tracking
helper: :func:`keras_tracking_callback`, a ``keras.callbacks.Callback`` that forwards
epoch logs (and, on train begin, run provenance) to matrainer trackers — the
Keras analogue of :class:`~matrainer.lightning.callbacks.TrackingCallback`.

``keras`` is imported lazily (raised with a clear ``pip install
matrainer[keras]`` message when the callback is built).


**Why ``keras``, not ``tensorflow``.** Keras 3 is a standalone package that runs on PyTorch,
TensorFlow or JAX, chosen by ``KERAS_BACKEND``. Requiring ``tensorflow.keras`` — as this module
did until 2026-07-30 — forced a ~2GB TensorFlow install on a consumer training with Keras 3 on
the torch backend, which is a framework it had deliberately not chosen. ``import keras`` is the
same API (``tf.keras`` IS Keras 3 in TensorFlow >= 2.16) with none of that.

**Why the import goes THROUGH ``recordstream.keras`` (2026-07-30).** That choice of backend is
read by Keras at IMPORT time, and its own default is ``tensorflow`` — which ``matrainer[keras]``
deliberately does not install. So a bare ``import keras`` here died with
``ModuleNotFoundError: No module named 'tensorflow'`` whenever this callback was the FIRST keras
import in a process, and :func:`~matrainer._imports.require` then reported it as *"the 'keras'
package is not installed"* — advice that fixes nothing, because keras WAS installed.
``recordstream.keras`` sets the variable to the first engine actually present before importing
keras, so routing through it removes the failure instead of describing it. Requires no new
dependency: recordstream is already a hard requirement, and that module needs nothing beyond the
``keras`` this extra installs.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from matrainer._imports import require
from matrainer.metric_names import MetricNames, as_metric_names
from matrainer.tracking import provenance
from matrainer.tracking.base import Tracker


def keras_tracking_callback(
    trackers: Sequence[Tracker],
    params: Optional[Mapping[str, Any]] = None,
    runnable: Any = None,
    capture_log: bool = True,
    metric_names: Optional[MetricNames] = None,
) -> Any:
    """Build a ``keras.callbacks.Callback`` that forwards run tracking to matrainer trackers.

    On train begin it captures the run context (git / code state / config / params) into a run
    manifest and projects it onto every tracker; on each epoch end it forwards the epoch
    ``logs`` as metrics; on train end it finishes the manifest and captures the log file.
    Attach it via ``model.fit(..., callbacks=[keras_tracking_callback([tracker])])``.

    Args:
        trackers: The matrainer trackers to log to.
        params: Optional hyperparameters (only JSON-scalar values forwarded). Merged OVER
            anything derived from ``runnable``, so an explicit value wins.
        runnable: The configured object driving the run. Given one, its hyperparameters are
            derived and its resolved configuration dumped — hand it the object rather than
            building a params dict, so that answer comes from one implementation.
        capture_log: Upload the loggair log file as an artifact on train end.
        metric_names: How the epoch logs are named on the way to the trackers. ``None`` (default)
            uses :class:`~matrainer.metric_names.MetricNames`, which renders Keras' own spelling
            (``loss`` / ``val_sparse_categorical_accuracy``) as the canonical ``train/loss`` /
            ``val/accuracy`` — so a run recorded here is comparable with one from any other
            engine. The translation happens HERE and not to the ``logs`` dict itself: Keras hands
            that same dict to the user's own callbacks, where ``EarlyStopping(monitor="val_loss")``
            must keep finding the name it was given.
    """
    # `via=` so KERAS_BACKEND is set before keras loads (see the module docstring), while the
    # error an operator without the extra sees still names `keras` — the thing they must install.
    keras = require("keras", "keras", feature="keras_tracking_callback", via="recordstream.keras").keras
    members = list(trackers)
    names = as_metric_names(metric_names)

    class _KerasTrackingCallback(keras.callbacks.Callback):  # type: ignore[misc, name-defined]
        #: The run manifest, available from train begin onward (``None`` before that).
        manifest: Any = None

        def on_train_begin(self, logs: Optional[Dict[str, float]] = None) -> None:
            self.manifest = provenance.log_run_context(members, params, runnable=runnable)

        def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, float]] = None) -> None:
            # `source="keras"`: the keys arriving here are Keras', whatever engine it runs on —
            # bare training keys, `val_`-prefixed validation ones. Translated for the trackers
            # only; `logs` itself stays untouched for the callbacks after this one.
            reported = names(dict(logs or {}), source="keras")
            for tracker in members:
                tracker.log_metrics(reported, step=epoch)

        def on_train_end(self, logs: Optional[Dict[str, float]] = None) -> None:
            # Finishing is not optional once a manifest exists: left at `running`, matrainer's
            # atexit hook would label a successful run `interrupted`.
            if capture_log or self.manifest is not None:
                provenance.capture_log_file(members, manifest=self.manifest, status="success")

    return _KerasTrackingCallback()


__all__ = ["keras_tracking_callback"]

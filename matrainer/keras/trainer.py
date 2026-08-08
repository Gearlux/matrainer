"""``KerasRunnable`` — the compile/fit machinery base of the trainer API.

The Keras 3 member of the family (``docs/architecture.md`` record 9): the engine owns the loop
(``keras.Model.fit``), so what this base writes is the assembly around it — build + compile,
the ``History`` read-back, the entry points — with everything task-shaped left to the hooks a
consumer's task base implements. Every method here was measured near-identical across the four
task projects' keras backends before the consolidation; what differed per task is exactly the
family hooks (``_to_arrays``, the default loss/metrics) plus the compile seams below
(:meth:`_compiled_loss` / :meth:`_compiled_optimizer` / :meth:`_check_compiled_metrics` — a
detection task compiles the spec's positional loss LIST and scores with numpy AP instead of
compile-slot metrics; a per-pixel task warns when the IoU family meets the torch engine).

Two vocabulary rules the whole family shares:

* **The HISTORY keeps Keras's own names** — it is Keras's object, and a ``CSVLogger`` beside
  it writes those same columns, so a wired ``EarlyStopping(monitor="val_loss")`` must keep
  finding them. What LEAVES the run (the sidecar, ``computed_metrics``, the trackers) is
  translated through ``as_metric_names(..., source="keras")`` so it reads beside another
  engine's.
* **``evaluate()`` scores the VAL split** (``test()`` the held-out one) — the same
  harmonization the torch-loop and fastai bases made: ``evaluate`` used to delegate to
  ``test()`` in every keras backend, which scored the WRONG split under the fleet's shared
  verb vocabulary. Keras's ``Model.evaluate`` returns UNPREFIXED keys whichever split it ran
  on, so the caller states which one it was — Keras cannot.

Tracking is :func:`matrainer.keras.keras_tracking_callback` (appended by the family's
``_callbacks``), which owns the run lifecycle — provenance on train begin, epoch logs, the
manifest + log capture on train end. That is why :meth:`fit` sequences its artifact writes
BEFORE ``model.fit`` where they touch a tracker (the run has ENDED when ``fit`` returns) and
keeps only local file writes after.
"""

from typing import Any, Dict, Literal, cast

import numpy as np
from confluid import configurable, flow
from loggair import get_logger
from recordstream import PredictionsSink
from recordstream.runnable import run_entrypoint

from matrainer.artifacts import write_metrics_sidecar
from matrainer.keras.runnable import KerasTrainerRunnable, keras_backend
from matrainer.metric_names import as_metric_names

logger = get_logger(__name__)

__all__ = ["KerasRunnable"]


@configurable(validate=False)
class KerasRunnable(KerasTrainerRunnable):
    """A Keras 3 train/evaluate/test/predict runnable, task hooks left open.

    Adds no constructor slots of its own — the family base carries them all; what this layer
    contributes is the compile step and the four entry points over ``keras.Model``'s own
    ``fit`` / ``evaluate`` / ``predict``.
    """

    # ---- run dispatch: compile between the task's initialize and the entry point ----------- #
    def run(self) -> None:
        logger.info(f"Keras backend in effect: {keras_backend()}")
        self.initialize()
        self._compile()
        # The `@entrypoint` markers (inherited from the API base) ARE the dispatch table.
        run_entrypoint(self, self.task)

    # ---- compile ---------------------------------------------------------------------------- #
    def _compile(self) -> None:
        """Build the model, then compile it.

        The compile slots are flowed at USE: a nested ``!class:`` in a constructor slot
        arrives as a deferred ``Class`` marker, and Keras takes an unbuilt marker for a custom
        callable and fails deep inside ``compile`` rather than saying what is wrong. A bare
        ``flow()`` is the whole guard — it returns a live object, a string alias and a list
        unchanged.

        The model reaches here LIVE on the standard path (the task base's ``initialize``
        already flowed it with its resolved dimensions) and the plain ``flow()`` is then a
        pass-through; it stays here so a directly-driven instance still builds.
        """
        self._resolve_task_defaults()
        self.model = flow(self.model)
        metrics = [flow(m) for m in self._compiled_metrics()]
        self._check_compiled_metrics(metrics)
        self.model.compile(optimizer=self._compiled_optimizer(), loss=self._compiled_loss(), metrics=metrics)

    def _compiled_loss(self) -> Any:
        """The value ``compile(loss=...)`` receives — a seam, because one task's loss is not a
        slot: the detection spec compiles a positional loss LIST matched to the model's raw
        head maps, and overrides this."""
        return flow(self.loss)

    def _compiled_optimizer(self) -> Any:
        """The value ``compile(optimizer=...)`` receives — a seam for a task that translates
        knobs into a string-alias optimizer (a learning rate, say) before handing it over."""
        return flow(self.optimizer)

    def _check_compiled_metrics(self, metrics: Any) -> None:
        """Inspect the flowed metric list before it compiles — no-op by default.

        The per-pixel task warns here when the IoU family meets the torch engine (a run that
        would look HUNG rather than slow); a check belongs before the fit, not in a postmortem.
        """

    # ---- fit hooks -------------------------------------------------------------------------- #
    def _before_fit(self) -> None:
        """Task prologue, after compile and before ``model.fit`` — no-op by default.

        BEFORE the fit is the only safe place for a tracker-touching artifact write: this
        backend hands the run's lifecycle to the tracking callback, whose ``on_train_end``
        closes the run out — a write sequenced after ``fit()`` returns arrives at an ENDED run
        (W&B drops the handle and would open a NEW one). The classification task persists its
        class names here.
        """

    def _fit_scores(self, final: Dict[str, float]) -> Dict[str, float]:
        """Extend the last epoch's history values with task-computed scores — identity by
        default. The detection task appends its numpy AP over the validation split here (Keras
        metrics cannot score variable-size detections, so the history carries losses only)."""
        return final

    # ---- the four entry points (markers inherited from the API base) ------------------------ #
    def fit(self) -> None:
        logger.info(f"{self.log_label} keras training: {self.experiment_name} ({self.run_name})")
        if self.train_set is None:
            raise ValueError(f"{type(self).__name__}.fit() requires a 'train_set'.")
        train_seq = self._sequence(self.train_set, shuffle=True)
        val_seq = self._sequence(self.val_set) if self.val_set is not None else None
        self._before_fit()
        history = self.model.fit(
            train_seq, validation_data=val_seq, epochs=self.max_epochs, callbacks=self._callbacks()
        )
        self._history = {k: [float(x) for x in v] for k, v in history.history.items()}
        final = self._fit_scores({k: v[-1] for k, v in self._history.items()})
        self._computed_metrics = as_metric_names(self.metric_names)(final, source="keras")
        logger.info(f"Training complete: {self._computed_metrics}")
        # Local file only — no tracker touch, so it is safe after the run has ended.
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    def evaluate(self) -> None:
        """Score the VALIDATION split — the ``evaluate`` / ``test`` split is the only difference."""
        if self.val_set is None:
            raise ValueError(f"{type(self).__name__}.evaluate() requires a 'val_set'.")
        self._computed_metrics = self._evaluate_split(self.val_set, "val")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    def test(self) -> None:
        if self.test_set is None:
            raise ValueError(f"{type(self).__name__}.test() requires a 'test_set'.")
        self._computed_metrics = self._evaluate_split(self.test_set, "test")
        write_metrics_sidecar(self.metrics_sidecar_path, self._computed_metrics, self.experiment_name, self.run_name)

    def _evaluate_split(self, source: Any, split: Literal["val", "test"]) -> Dict[str, float]:
        """``model.evaluate`` over one split, translated into the canonical vocabulary.

        ``split=`` because ``evaluate()`` returns UNPREFIXED keys (``loss``, ``accuracy``)
        whichever split it ran on — the caller states which one this was; Keras cannot.
        """
        results = self.model.evaluate(self._sequence(source), return_dict=True, callbacks=self._callbacks())
        computed = as_metric_names(self.metric_names)(
            {k: float(v) for k, v in results.items()}, source="keras", split=split
        )
        logger.info(f"Evaluation complete: {computed}")
        return computed

    def predict(self) -> None:
        """Stream per-record predictions to the sink, pairing each batch with its metadata."""
        if self.test_set is None:
            raise ValueError(f"{type(self).__name__}.predict() requires a 'test_set'.")
        self.predictions_sink = flow(self.predictions_sink)
        sequence = self._sequence(self.test_set, with_targets=False)
        for index, batch in enumerate(sequence.batches()):
            preds = self._prediction_output(np.asarray(self.model.predict(sequence[index][0], verbose=0)))
            if self.predictions_sink is not None:
                self._write_predictions(preds, batch)
        if self.predictions_sink is not None and hasattr(self.predictions_sink, "close"):
            cast(PredictionsSink, self.predictions_sink).close()
        logger.info("Prediction complete.")

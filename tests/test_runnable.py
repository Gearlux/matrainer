"""The framework-neutral trainer API (`matrainer.runnable.TrainerRunnable`).

What is pinned here is the neutral base's OWN behaviour — slot storage, dispatch, the
neutral hook defaults, the tracker plumbing, the light-core claim — against a minimal stub.
The torch-family layer's typed behaviour is pinned in `test_torch_runnable.py`; a machinery
base's (trainer construction, step functions) in its own suite.
"""

import subprocess
import sys
import textwrap
from typing import Any, List

import pytest
from confluid import PartialClass
from recordstream.runnable import entrypoint_tasks, runnable_entrypoints

from matrainer.runnable import TrainerRunnable


class _Stub(TrainerRunnable):
    """The smallest concrete subclass: every abstract member filled with a recorder."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls: List[str] = []

    def initialize(self) -> None:
        self.calls.append("initialize")

    def _resolve_task_defaults(self) -> None:
        self.calls.append("_resolve_task_defaults")

    def _build_fit_metrics(self, include_train: bool = True) -> None:
        self.calls.append("_build_fit_metrics")

    def _build_eval_metrics(self) -> None:
        self.calls.append("_build_eval_metrics")

    def fit(self) -> None:
        self.calls.append("fit")

    def evaluate(self) -> None:
        self.calls.append("evaluate")

    def test(self) -> None:
        self.calls.append("test")

    def predict(self) -> None:
        self.calls.append("predict")


class _SinkRecorder:
    def __init__(self) -> None:
        self.writes: List[Any] = []

    def write(self, prediction: Any, metadata: Any) -> None:
        self.writes.append((prediction, metadata))

    def close(self) -> None:  # pragma: no cover - protocol completeness
        pass


# ---------------------------------------------------------------------------
# construction + dispatch
# ---------------------------------------------------------------------------
def test_zero_arg_construction_works_and_stores_verbatim() -> None:
    """The lazy-init mandate: every param defaulted, the constructor does no work."""
    runnable = _Stub()
    assert runnable.task == "fit"
    assert runnable.val_set is None and runnable.predictions_sink is None
    assert runnable.input_key == "image" and runnable.target_key == "target"


def test_run_dispatches_through_the_entrypoint_markers() -> None:
    runnable = _Stub(task="test")
    runnable.run()
    assert runnable.calls == ["initialize", "test"]


def test_a_runnable_without_a_model_slot_is_untouched_by_the_guard() -> None:
    """The stub declares no model slot, so ``run()`` must not invent a requirement for one.

    The sentinel in :meth:`TrainerRunnable._require_model` exists for exactly this: a bare
    class annotation creates no attribute, so "no model slot" and "slot present but empty" are
    different states and only the second is an error.
    """
    runnable = _Stub(task="test")
    runnable.run()  # no raise
    assert not hasattr(runnable, "model")


def test_an_empty_model_slot_raises_a_clear_error_at_run() -> None:
    """The other half of the defaulted-`Mandatory` trade.

    Every family defaults its `model` param so the class stays zero-arg constructible (and
    `Mandatory[T]` keeps confluid reporting it required). That moves "you forgot the model"
    out of construction, so ``run()`` is where it has to be caught — before a framework
    reports it as an unhelpful ``NoneType`` complaint several layers down.
    """
    runnable = _Stub(task="test")
    runnable.model = None  # what every family base stores when the slot was never wired

    with pytest.raises(ValueError, match=r"`model` slot is empty"):
        runnable.run()

    assert runnable.calls == [], "the guard must fire BEFORE initialize()"


def test_the_markers_are_declared_once_on_the_base_and_found_on_a_subclass() -> None:
    """`runnable_entrypoints` walks the MRO, so the API base's markers ARE the subclass's
    dispatch table — an override needs no re-decoration."""
    marks = runnable_entrypoints(_Stub)
    assert list(marks) == ["fit", "evaluate", "test", "predict"]
    assert entrypoint_tasks(_Stub, "trainer") == ["fit"]
    assert entrypoint_tasks(_Stub, "evaluator") == ["test", "evaluate"]  # primary first
    assert entrypoint_tasks(_Stub, "predictor") == ["predict"]


def test_a_default_run_name_carries_the_task_prefix() -> None:
    assert _Stub(task="fit").run_name.startswith("train_")
    assert _Stub(task="test").run_name.startswith("evaluate_")
    assert _Stub(run_name="pinned").run_name == "pinned"


def test_the_neutral_hook_defaults_are_safe_no_ops() -> None:
    runnable = _Stub()
    assert runnable._prediction_output("raw") == "raw"
    assert runnable._reportable({"a": 1}) == {"a": 1}
    runnable._report_nonscalars({}, step=None, trackers=[])  # no-op, must not raise


def test_an_incomplete_subclass_cannot_be_instantiated() -> None:
    class _Partial(TrainerRunnable):
        pass

    with pytest.raises(TypeError):
        _Partial()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# tracker plumbing
# ---------------------------------------------------------------------------
def test_live_trackers_flows_once_and_writes_back(tmp_path: Any) -> None:
    from matrainer.tracking.csv_backend import CsvTracker

    runnable = _Stub(trackers=PartialClass(CsvTracker, save_dir=str(tmp_path)))
    live = runnable._live_trackers()
    assert len(live) == 1 and isinstance(live[0], CsvTracker)
    # Written back: the second call returns the SAME object, not a rebuilt one.
    assert runnable._live_trackers()[0] is live[0]


def test_no_trackers_is_an_empty_list_not_an_error() -> None:
    assert _Stub()._live_trackers() == []


# ---------------------------------------------------------------------------
# prediction writing
# ---------------------------------------------------------------------------
def test_write_predictions_slices_the_batch_and_pairs_metadata() -> None:
    import numpy as np

    runnable = _Stub(input_key="image", target_key="class")
    sink = _SinkRecorder()
    runnable.predictions_sink = sink  # type: ignore[assignment]
    batch = {"image": np.zeros((2, 3)), "class": [0, 1], "idx": [10, 11]}
    runnable._write_predictions({"probs": np.eye(2)}, batch)
    assert len(sink.writes) == 2
    prediction, metadata = sink.writes[1]
    assert metadata == {"idx": 11}, "input/target keys are excluded; the rest pairs per record"
    assert list(prediction["probs"]) == [0.0, 1.0]


# ---------------------------------------------------------------------------
# the light-core claim, executed
# ---------------------------------------------------------------------------
def test_the_api_base_imports_no_framework() -> None:
    """`matrainer.runnable` is the framework-free layer — importing it must pull no
    torch/lightning/keras, or the keras/jax/mlx family bases could never build on it."""
    probe = textwrap.dedent(
        """
        import sys
        import matrainer.runnable
        leaked = sorted(
            m for m in sys.modules
            if m.split(".")[0] in {"torch", "pytorch_lightning", "keras", "tensorflow", "jax", "mlx"}
        )
        assert not leaked, f"matrainer.runnable leaked frameworks: {leaked}"
        """
    )
    subprocess.run([sys.executable, "-c", probe], check=True)

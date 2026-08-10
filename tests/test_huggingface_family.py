"""The HuggingFace machinery base — `HuggingFaceRunnable`.

What this pins is the INHERITED wiring the task projects' fleet tests exempt their concretes
from showing: the string-dotted deferred slots (no module-level transformers import), the
run-time argument FLOOR with a pinned key winning, the sentinel resolution through the
name-list hook, the deliberate no-op metric builders, the Trainer-subclass seam, the
harmonized `evaluate`-scores-val rule, the fit sequencing (warm → provenance → train →
scores → history replay → `_after_fit` → terminal capture), and the `log_history` replay's
axis-key rule.

Structure-level where behaviour needs a real `transformers.Trainer` (the four task projects
run the real fits in their own suites); behavioural where torch + recordstream suffice.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pytest

torch = pytest.importorskip("torch")

from matrainer.huggingface.trainer import HuggingFaceRunnable, _log_trainer_history  # noqa: E402
from matrainer.metric_names import MetricNames  # noqa: E402


class _Hooked(HuggingFaceRunnable):
    """A minimal task stand-in over the machinery base."""

    def initialize(self) -> None:  # pragma: no cover - structure only
        pass

    def _default_hf_metrics(self) -> Optional[List[Any]]:
        return ["accuracy"]


# --------------------------------------------------------------------------- #
# The deferred transformers objects
# --------------------------------------------------------------------------- #
def test_transformers_stays_a_deferred_dotted_target() -> None:
    """No module-level `import transformers` — the slots are STRING dotted paths confluid
    resolves at flow time, so importing the machinery stays light."""
    import matrainer.huggingface.trainer as module

    source = inspect.getsource(module)
    top_level = [line for line in source.splitlines() if line.startswith(("import transformers", "from transformers"))]
    assert not top_level, f"module-level transformers import: {top_level}"
    runnable = _Hooked()
    assert cast(Any, runnable.training_arguments).target == "transformers.TrainingArguments"
    assert cast(Any, runnable.trainer).target == "transformers.Trainer"
    assert not cast(Any, runnable.training_arguments).kwargs, "the slot is EMPTY — the floor is derived at run time"


# --------------------------------------------------------------------------- #
# The argument floor
# --------------------------------------------------------------------------- #
def test_the_floor_is_derived_and_a_pinned_key_wins(monkeypatch: Any) -> None:
    """The slot is a whole-value replacement, so pinning the derived knobs on it would mean
    `training_arguments: {use_cpu: true}` silently dropped the epoch count AND
    `remove_unused_columns=False` — which surfaces as a `KeyError` inside Trainer's collator."""
    runnable = _Hooked()
    runnable.max_epochs, runnable.batch_size = 3, 4
    captured: Dict[str, Any] = {}

    def _fake_flow(slot: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return slot

    monkeypatch.setattr("matrainer.huggingface.trainer.flow", _fake_flow)
    cast(Any, runnable.training_arguments).kwargs.update({"num_train_epochs": 99})  # the config pinned a key
    runnable._training_arguments()
    assert "num_train_epochs" not in captured, "a pinned key must WIN — the floor may not restate it"
    assert captured["per_device_train_batch_size"] == 4
    assert captured["remove_unused_columns"] is False
    assert captured["report_to"] == "none"
    assert captured["save_strategy"] == "no"


def test_the_eval_strategy_follows_the_wired_val_split() -> None:
    runnable = _Hooked()
    assert runnable._argument_floor()["eval_strategy"] == "no"
    runnable.val_set = [1]
    assert runnable._argument_floor()["eval_strategy"] == "epoch"


def test_the_unconfigured_output_dir_roots_at_runs(tmp_path: Any) -> None:
    """`runs/` is the ONE gitignored output root every project carries.

    An unconfigured `Trainer` writes checkpoints and optimizer state, and the last-resort
    path is the one nobody chose — so it must not scatter that into the directory the run
    happened to be launched from (workspace `AGENTS.md`, "Run Output Lands In A Gitignored
    `runs/`"). The two configured paths still win, in order.
    """
    runnable = _Hooked(run_name="funny_rabbit")
    assert runnable._output_dir() == str(Path("./runs/hf_runs/funny_rabbit"))

    runnable.metrics_sidecar_path = str(tmp_path / "metrics.json")
    assert runnable._output_dir() == str(tmp_path / "funny_rabbit"), "the sidecar wins over the default"

    runnable.output_dir = str(tmp_path / "explicit")
    assert runnable._output_dir() == str(tmp_path / "explicit"), "an explicit path wins over both"


# --------------------------------------------------------------------------- #
# Sentinel resolution + the deliberate no-op builders
# --------------------------------------------------------------------------- #
def test_the_metrics_sentinel_resolves_via_the_name_list_hook_idempotently() -> None:
    runnable = _Hooked()
    assert runnable.metrics is None
    runnable._resolve_task_defaults()
    assert runnable.metrics == ["accuracy"]

    wired = _Hooked()
    wired.metrics = ["f1_macro"]
    wired._resolve_task_defaults()
    assert wired.metrics == ["f1_macro"], "an explicitly wired metric list must never be replaced"


def test_the_metric_builders_are_deliberate_noops() -> None:
    """Three of the four tasks resolve metric NAMES in their hook — there is no torchmetrics
    collection to build (the fastai-base arrangement); a torchmetrics task restores the family's."""
    runnable = _Hooked()
    runnable.metrics = ["accuracy"]
    runnable._build_fit_metrics()
    runnable._build_eval_metrics()
    assert runnable.metrics == ["accuracy"], "the builders must not transform the specs"


def test_there_is_no_train_metrics_param() -> None:
    """`compute_metrics` is a Trainer's EVALUATION hook — a train slot would accept a wired
    set and silently never call it."""
    assert "train_metrics" not in inspect.signature(HuggingFaceRunnable.__init__).parameters


# --------------------------------------------------------------------------- #
# The Trainer assembly seams
# --------------------------------------------------------------------------- #
def test_the_trainer_factory_seam_replaces_only_the_unoverridden_default() -> None:
    source = inspect.getsource(HuggingFaceRunnable._build_trainer)
    assert "_trainer_factory()" in source
    assert "type(trainer) is HFTrainer" in source, "a config-replaced Trainer must never be substituted"
    assert _Hooked()._trainer_factory() is None


def test_the_default_metrics_hook_is_none() -> None:
    """No hook wired = the Trainer reports its loss only — what a task scoring OUTSIDE the
    Trainer (detection's decode + torchmetrics pass) wants."""
    assert _Hooked()._metrics_hook("val") is None


# --------------------------------------------------------------------------- #
# The entry points
# --------------------------------------------------------------------------- #
def test_fit_sequences_warm_provenance_train_scores_and_terminal_capture() -> None:
    body = inspect.getsource(HuggingFaceRunnable.fit)
    assert body.index("_prepare(") < body.index("log_run_context(")
    assert body.index("log_run_context(") < body.index("trainer.train()")
    assert body.index("trainer.train()") < body.index("_fit_val_scores(")
    assert body.index("_log_trainer_history(") < body.index("_after_fit(")
    assert body.index("_after_fit(") < body.index("capture_log_file(")
    assert "tracker.log_metrics(self._computed_metrics)" in body


def test_evaluate_scores_the_validation_split() -> None:
    """The fleet harmonization: `evaluate` used to delegate to `test()` in three of the four
    huggingface backends."""
    assert '_run_eval_task(self.val_set, "val")' in inspect.getsource(HuggingFaceRunnable.evaluate)
    assert '_run_eval_task(self.test_set, "test")' in inspect.getsource(HuggingFaceRunnable.test)


def test_the_test_split_carries_its_own_metric_key_prefix() -> None:
    """Left at HF's `eval` default, the held-out scores would land in the val columns."""
    source = inspect.getsource(HuggingFaceRunnable._evaluate_split)
    assert '"test" if split == "test" else "eval"' in source


def test_an_eval_task_records_the_run() -> None:
    """The distilaide sweep lesson: an eval run without the tracker + manifest wiring shows a
    blank row and gets relabelled `interrupted` by the atexit hook."""
    body = inspect.getsource(HuggingFaceRunnable._run_eval_task)
    assert "log_run_context(" in body and "capture_log_file(" in body
    assert "tracker.log_metrics(" in body


# --------------------------------------------------------------------------- #
# The batch boundary
# --------------------------------------------------------------------------- #
def test_the_default_batch_is_the_transformers_image_pair() -> None:
    runnable = _Hooked()
    records = [{"image": torch.rand(3, 4, 4), "target": torch.rand(3, 4, 4)} for _ in range(2)]
    batch = runnable._to_hf_batch(records)
    assert set(batch) == {"pixel_values", "labels"}
    assert batch["pixel_values"].shape == (2, 3, 4, 4)


# --------------------------------------------------------------------------- #
# The log-history replay
# --------------------------------------------------------------------------- #
class _FakeTrainer:
    class state:
        log_history = [
            {"loss": 1.5, "epoch": 1.0, "step": 10},
            {"eval_loss": 0.9, "eval_accuracy": 0.5, "epoch": 1.0, "step": 10},
            {"train_runtime": 3.2, "step": 20},
        ]


class _FakeTracker:
    def __init__(self) -> None:
        self.calls: List[Any] = []

    def log_metrics(self, values: Dict[str, float], step: Any = None) -> None:
        self.calls.append((dict(values), step))


def test_the_history_replay_translates_and_drops_the_axis_keys() -> None:
    """`epoch`/`step` are AXES, not metrics — a series of the x-value adds a column no reader
    wants; `step` still drives the axis argument."""
    tracker = _FakeTracker()
    _log_trainer_history(_FakeTrainer(), cast(Any, [tracker]), MetricNames())
    assert len(tracker.calls) == 3
    values, step = tracker.calls[0]
    assert values == {"train/loss": 1.5} and step == 10
    values, step = tracker.calls[1]
    assert values == {"val/loss": 0.9, "val/accuracy": 0.5} and step == 10
    assert all("epoch" not in v and "step" not in v for v, _ in tracker.calls)


def test_canonical_keys_survive_the_fit_translation() -> None:
    """A task hook may hand back already-canonical `<split>/<name>` scores (detection's
    torchmetrics pass) — the vocabulary recognises the shape whatever the source, so the
    fit's closing translation is idempotent over them."""
    names = MetricNames()
    translated = names({"val/map_50": 0.5, "eval_loss": 1.0}, source="huggingface")
    assert translated == {"val/map_50": 0.5, "val/loss": 1.0}


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def test_computed_metrics_raises_before_the_run_and_answers_after() -> None:
    runnable = _Hooked()
    with pytest.raises(RuntimeError, match="after run"):
        _ = runnable.computed_metrics
    runnable._computed_metrics = {"val/loss": 1.0}
    assert runnable.computed_metrics == {"val/loss": 1.0}


def test_the_prediction_walk_uses_the_record_loader(tmp_path: Any) -> None:
    """`Trainer.predict` returns only stacked logits — the sink contract needs each row paired
    with its record's metadata, so predict walks a RECORD loader instead."""
    body = inspect.getsource(HuggingFaceRunnable.predict)
    assert "_record_loader(" in body and "_write_predictions(" in body
    assert "_prediction_output(" in body

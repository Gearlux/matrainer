"""The MLX machinery base — `MlxLoopRunnable` (Apple silicon only).

What this pins is the INHERITED wiring the task projects' fleet tests exempt their concretes
from showing: the three MLX facts in the loop (the per-step ``mx.eval`` over parameters +
optimizer state + loss, the train/eval MODE flips at the loop boundaries, the aux-tuple
objective — MLX differentiates the FIRST element of a returned tuple), the class-not-factory
optimizer resolution, the sentinel hooks, the splat-tolerant driver, the per-epoch reshuffle,
and the harmonized `evaluate`-scores-val rule.
"""

from __future__ import annotations

import inspect
from typing import Any, List, Optional

import pytest

mx = pytest.importorskip("mlx.core")
optim = pytest.importorskip("mlx.optimizers")

from matrainer.mlx.loop import MlxLoopRunnable, MlxLoss, as_mlx_loss, as_mlx_optimizer  # noqa: E402


class _Hooked(MlxLoopRunnable):
    """A minimal task stand-in over the machinery base."""

    def initialize(self) -> None:  # pragma: no cover - structure only
        pass

    def _default_mlx_loss(self) -> Optional[MlxLoss]:
        return "mse_loss"

    def _default_mlx_metrics(self) -> Optional[List[Any]]:
        return ["mse"]


def test_the_sentinels_resolve_via_the_task_hooks_idempotently() -> None:
    runnable = _Hooked()
    assert runnable.loss is None and runnable.metrics is None
    runnable._resolve_task_defaults()
    assert runnable.loss == "mse_loss" and runnable.metrics == ["mse"]

    wired = _Hooked(loss="l1_loss", metrics=["mae"])
    wired._resolve_task_defaults()
    assert wired.loss == "l1_loss" and wired.metrics == ["mae"], "a wired choice is never replaced"


def test_losses_resolve_by_name_and_an_unknown_name_says_where_names_come_from() -> None:
    import mlx.nn as nn

    assert as_mlx_loss("l1_loss") is nn.losses.l1_loss
    with pytest.raises(ValueError, match="mlx.nn.losses"):
        as_mlx_loss("nope")


def test_optimizers_are_CLASSES_here_not_factories() -> None:
    """MLX ships optimizer classes where optax ships factories — a name is INSTANTIATED, and
    a lowercase spelling is capitalised so the two siblings' configs read alike."""
    made = as_mlx_optimizer("adam", 0.5)
    assert isinstance(made, optim.Adam)
    assert as_mlx_optimizer(made, 0.1) is made, "a ready-made optimizer is taken as-is"
    with pytest.raises(ValueError, match="mlx.optimizers"):
        as_mlx_optimizer("NotAnOptimizer", 1e-3)


def test_the_loop_forces_the_lazy_graph_once_per_step() -> None:
    """MLX arrays are LAZY and omitting `mx.eval` does NOT raise: the graph grows every step
    until memory runs out. The eval covers the parameters AND the optimizer state, not only
    the loss — the state is what carries the momentum buffers."""
    source = inspect.getsource(MlxLoopRunnable._train_epoch)
    assert "mx.eval(network.parameters(), optimizer.state, loss)" in source


def test_the_train_mode_is_set_PER_PHASE_not_per_call() -> None:
    """MLX carries train/eval as module STATE — a `training=False` default on `__call__`
    would flip the network to eval on every forward, including the one inside the training
    step, where `nn.value_and_grad` calls `model(x)` with no flag to pass."""
    assert "network.train(True)" in inspect.getsource(MlxLoopRunnable._train_epoch)
    assert "network.train(False)" in inspect.getsource(MlxLoopRunnable._eval_pass)
    assert ".train(False)" in inspect.getsource(MlxLoopRunnable.predict)


def test_the_default_objective_returns_its_outputs_beside_the_loss() -> None:
    """MLX differentiates the FIRST element of a returned tuple — the outputs ride along so
    the training metrics are scored from the very tensor the gradient was taken through."""
    source = inspect.getsource(MlxLoopRunnable._make_step)
    assert "return loss_fn(outputs, y), outputs" in source


def test_the_epoch_driver_splats_and_accepts_a_bare_loss() -> None:
    source = inspect.getsource(MlxLoopRunnable._train_epoch)
    assert "*arrays" in source and "isinstance(result, tuple)" in source


def test_the_reshuffle_and_the_callback_fanout_are_in_the_fit() -> None:
    body = inspect.getsource(MlxLoopRunnable.fit)
    assert "on_epoch_end()" in body, "forgetting the reshuffle is invisible"
    assert "flow(callback)(epoch, logs)" in body


def test_evaluate_scores_the_validation_split() -> None:
    """The fleet harmonization: `evaluate` used to delegate to `test()` in every mlx backend."""
    assert '_run_eval_task(self.val_set, "val")' in inspect.getsource(MlxLoopRunnable.evaluate)
    assert '_run_eval_task(self.test_set, "test")' in inspect.getsource(MlxLoopRunnable.test)


def test_an_eval_task_records_the_run() -> None:
    body = inspect.getsource(MlxLoopRunnable._run_eval_task)
    assert "log_run_context(" in body and "capture_log_file(" in body
    assert "tracker.log_metrics(" in body


def test_computed_metrics_raises_before_the_run() -> None:
    runnable = _Hooked()
    with pytest.raises(RuntimeError, match="after run"):
        _ = runnable.computed_metrics

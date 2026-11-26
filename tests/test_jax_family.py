"""The JAX machinery base — `JaxLoopRunnable`.

What this pins is the INHERITED wiring the task projects' fleet tests exempt their concretes
from showing: the sentinel resolution via the jax task hooks, the deliberate no-op metric
builders, the jit-ONCE step builders, the SPLAT-tolerant epoch driver (bare-loss steps and
non-pair arities), the per-epoch reshuffle, the harmonized `evaluate`-scores-val rule, the
eval-task run recording, and the name-resolution helpers (optax ships functions and
FACTORIES — a named optimizer is resolved and CALLED with the learning rate).

Structure-level where behaviour needs a real fit (the four task projects run those in their
own suites); behavioural where jax-on-CPU suffices.
"""

from __future__ import annotations

import inspect
from typing import Any, List, Optional

import pytest

jax = pytest.importorskip("jax")
optax = pytest.importorskip("optax")

from matrainer.jax.loop import JaxLoopRunnable, JaxLoss, as_optax_loss, as_optax_optimizer  # noqa: E402


class _Hooked(JaxLoopRunnable):
    """A minimal task stand-in over the machinery base."""

    def initialize(self) -> None:  # pragma: no cover - structure only
        pass

    def _default_jax_loss(self) -> Optional[JaxLoss]:
        return "squared_error"

    def _default_jax_metrics(self) -> Optional[List[Any]]:
        return ["mse"]


def test_the_sentinels_resolve_via_the_task_hooks_idempotently() -> None:
    runnable = _Hooked()
    assert runnable.loss is None and runnable.metrics is None
    runnable._resolve_task_defaults()
    assert runnable.loss == "squared_error" and runnable.metrics == ["mse"]

    wired = _Hooked(loss="huber_loss", metrics=["mae"])
    wired._resolve_task_defaults()
    assert wired.loss == "huber_loss" and wired.metrics == ["mae"], "a wired choice is never replaced"


def test_the_metric_builders_are_deliberate_noops() -> None:
    runnable = _Hooked()
    runnable.metrics = ["mse"]
    runnable._build_fit_metrics()
    runnable._build_eval_metrics()
    assert runnable.metrics == ["mse"], "the builders must not transform the specs"


def test_losses_and_optimizers_resolve_by_name() -> None:
    """optax ships FUNCTIONS and FACTORIES — a named optimizer is resolved and CALLED."""
    assert as_optax_loss("squared_error") is optax.losses.squared_error
    made = as_optax_optimizer("adam", 1e-3)
    assert hasattr(made, "update"), "the factory was CALLED, not handed back"
    with pytest.raises(ValueError, match="optax.losses"):
        as_optax_loss("nope")
    with pytest.raises(ValueError, match="optax"):
        as_optax_optimizer("nope", 1e-3)


def test_the_default_loss_fn_asks_the_hook_when_unresolved() -> None:
    """A directly-driven instance works without `initialize()` — resolve-on-use."""
    import jax.numpy as jnp

    runnable = _Hooked()
    compute = runnable._loss_fn()
    value = compute(jnp.ones((2, 3)), jnp.zeros((2, 3)))
    assert float(value) == pytest.approx(1.0)  # mean squared error over ones vs zeros


def test_the_steps_are_jitted_once_and_reused() -> None:
    """A fresh `nnx.jit` per epoch retraces the whole graph — `fit` builds the steps OUTSIDE
    its epoch loop."""
    import ast
    import textwrap

    fit = ast.parse(textwrap.dedent(inspect.getsource(JaxLoopRunnable.fit))).body[0]
    calls_in_loops = [
        ast.unparse(call)
        for loop in ast.walk(fit)
        if isinstance(loop, (ast.For, ast.While))
        for call in ast.walk(loop)
        if isinstance(call, ast.Call) and "make_steps" in ast.unparse(call)
    ]
    assert not calls_in_loops, f"_make_steps must be called once, outside the epoch loop: {calls_in_loops}"
    assert "_make_steps" in inspect.getsource(JaxLoopRunnable.fit)


def test_the_epoch_driver_splats_and_accepts_a_bare_loss() -> None:
    """A 4-arity dense-map task drives the same loop, and its step may return a bare loss."""
    for driver in (JaxLoopRunnable._train_epoch, JaxLoopRunnable._eval_pass):
        source = inspect.getsource(driver)
        assert "*arrays" in source, f"{driver.__name__} must splat the batch"
        assert "isinstance(result, tuple)" in source, f"{driver.__name__} must unpack a bare loss"


def test_the_reshuffle_and_the_callback_fanout_are_in_the_fit() -> None:
    body = inspect.getsource(JaxLoopRunnable.fit)
    assert "on_epoch_end()" in body, "forgetting the reshuffle is invisible"
    assert "flow(callback)(epoch, logs)" in body


def test_evaluate_scores_the_validation_split() -> None:
    """The fleet harmonization: `evaluate` used to delegate to `test()` in every jax backend."""
    assert '_run_eval_task(self.val_set, "val")' in inspect.getsource(JaxLoopRunnable.evaluate)
    assert '_run_eval_task(self.test_set, "test")' in inspect.getsource(JaxLoopRunnable.test)


def test_an_eval_task_records_the_run() -> None:
    body = inspect.getsource(JaxLoopRunnable._run_eval_task)
    assert body.index("log_run_context(") < body.index("with ending_the_run(trackers, manifest):")
    assert body.index("with ending_the_run(trackers, manifest):") < body.index("_evaluate_split(")
    assert "tracker.log_metrics(" in body


def test_fit_warms_the_splits_and_records_the_run() -> None:
    body = inspect.getsource(JaxLoopRunnable.fit)
    assert body.index("_prepare(") < body.index("log_run_context(")
    assert body.index("_fit_scores(") < body.index("_fit_epilogue()")
    # The run is ended by `ending_the_run`, as failed when anything inside raises.
    assert body.index("log_run_context(") < body.index("with ending_the_run(trackers, manifest):")
    assert body.index("with ending_the_run(trackers, manifest):") < body.index("_fit_scores(")
    assert "capture_log_file(" not in body


def test_computed_metrics_raises_before_the_run() -> None:
    runnable = _Hooked()
    with pytest.raises(RuntimeError, match="after run"):
        _ = runnable.computed_metrics

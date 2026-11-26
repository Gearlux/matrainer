"""The trainer API on the Lightning base: the SAME task, a different machinery base.

The companion to `examples/trainer_api_loop.py`, and together they are what
`docs/trainer-api.md` claims: a project writes its task once and composes a concrete class per
engine. Put the two files side by side — the `IdentityTaskHooks` mixin is identical, and only the
base and what that base needs differ:

* the loop file inherits `TorchLoopRunnable` and sets `device`; the epoch is hand-written there;
* this one inherits `LightningRunnable` and fills the deferred `lightning` slot with a
  `pl.Trainer`; the epoch belongs to Lightning.

Everything else — the typed `model` slot, the metrics quartet, `run()` dispatching on `task`, the
per-split prefixes, `computed_metrics` — comes from the two layers above the machinery base and is
written nowhere in this file. That is the abstraction stopping at the API.

The hooks are repeated rather than imported from the other example on purpose: each example has
to run standalone, which is also why a real project puts its task mixin in its own module.

Run:  python examples/trainer_api_lightning.py   (needs `matrainer[lightning]`)
"""

import inspect
import tempfile
from typing import Any, Dict

import numpy as np
import pytorch_lightning as pl
import torch
from confluid import PartialClass, flow
from recordstream import Stream, batch_tensor
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError

from matrainer.lightning import LightningRunnable
from matrainer.torch import TorchLoopRunnable
from matrainer.torch.metrics import MetricsSpec

WIDTH = 4


class IdentityTaskHooks:
    """The TASK, written once: learn the identity map on 4-vectors.

    Character for character the mixin in `examples/trainer_api_loop.py` — it names no engine, so
    it composes onto either machinery base.
    """

    def _targets(self, batch: Any) -> torch.Tensor:
        return batch_tensor(batch, self.target_key, self.device)  # type: ignore[attr-defined]

    def _default_loss(self) -> nn.Module:
        return nn.MSELoss()

    def _default_metrics(self) -> MetricsSpec:
        return {"mae": PartialClass(MeanAbsoluteError)}

    def _prediction_output(self, preds: Any) -> Dict[str, Any]:
        return {"restored": preds}  # the task's prediction contract


class LightningIdentity(IdentityTaskHooks, LightningRunnable):
    """The concrete class: the same hooks in front of the Lightning machinery base."""

    log_label = "identity-lightning"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # The one thing this base needs that the loop base does not: the Trainer, as a DEFERRED
        # marker. A consumer's config replaces this whole slot the same way.
        self.lightning = PartialClass(
            pl.Trainer,
            max_epochs=6,
            accelerator="cpu",
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            log_every_n_steps=1,
        )
        self.log_system_metrics = False

    def initialize(self) -> None:
        self.model = flow(self.model)  # the slot may hold a deferred marker
        self._resolve_task_defaults()
        self._build_fit_metrics() if self.task == "fit" else self._build_eval_metrics()


def records(n: int = 16) -> Stream:
    """A recordstream Stream of records — what every base consumes, not raw tensors."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        x = rng.normal(size=WIDTH).astype("float32")
        rows.append({"image": x, "target": x.copy(), "idx": i})
    return Stream(source=rows)


def main() -> None:
    tempfile.mkdtemp(prefix="matrainer_trainer_api_lightning_")  # nothing of ours is written here
    runnable = LightningIdentity(
        model=nn.Linear(WIDTH, WIDTH),
        train_set=records(),
        batch_size=4,
        task="fit",
    )
    # No `checkpoint_dir` here: that is a LOOP-base slot. Lightning takes its checkpoint directory
    # from the logger's `save_dir` instead (docs/lightning.md), which is the same fact from the
    # other side — a machinery base adds only what its engine cannot work out for itself.
    runnable.val_set = records(8)  # a body slot here exactly as on the loop base

    runnable.run()  # the SAME call as the loop example: the neutral base dispatches on `task`

    # The family base's typed slots and prefixes — identical to the loop run, though a
    # pl.Trainer drove this one.
    assert isinstance(runnable.train_metrics, MetricCollection)
    assert set(runnable.val_metrics.keys()) == {"val/mae"}
    computed = runnable.computed_metrics
    assert "train/loss" in computed and "val/mae" in computed, sorted(computed)

    # ...and the same object evaluates, through Lightning's own test loop.
    scorer = LightningIdentity(model=nn.Linear(WIDTH, WIDTH), batch_size=4, task="evaluate")
    scorer.val_set = records(8)
    scorer.run()
    assert "val/mae" in scorer.computed_metrics, sorted(scorer.computed_metrics)

    # The layering, counted: almost every constructor slot comes from the shared layers, and
    # only the loop base adds any of its own — the three things a hand-written epoch must be told
    # and a `pl.Trainer` owns. This is the four-layer claim as a number.
    def slots(cls: type) -> set:
        # `signature(cls)` is the constructor's, with `self` already dropped.
        return set(inspect.signature(cls).parameters)

    lightning_slots, loop_slots = slots(LightningRunnable), slots(TorchLoopRunnable)
    assert not lightning_slots - loop_slots, "the Lightning base adds no constructor slot"
    assert loop_slots - lightning_slots == {"checkpoint_dir", "device", "grad_clip"}
    shared = lightning_slots & loop_slots

    print(f"{type(runnable).__name__}: {type(runnable).__mro__[1].__name__} + " f"{type(runnable).__mro__[2].__name__}")
    print(
        f"  {len(shared)} constructor slots come from the shared layers; the loop base adds "
        f"{sorted(loop_slots - lightning_slots)} and this one adds none"
    )
    print(
        f"  the same hooks as the loop example, with a {type(flow(runnable.lightning)).__name__} " f"doing the epochs"
    )
    print(
        f"  val metrics {sorted(runnable.val_metrics.keys())}, "
        f"evaluate() scored {sorted(k for k in scorer.computed_metrics if k.startswith('val/'))}"
    )


if __name__ == "__main__":
    main()

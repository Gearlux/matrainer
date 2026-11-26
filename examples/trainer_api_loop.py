"""The trainer API on the bare-torch base: a task's hooks plus an owned training loop.

Asserts what `docs/trainer-api.md` claims, for the machinery base that has no `Trainer` behind it
at all — `TorchLoopRunnable` writes the epoch itself (`zero_grad` / `backward` / `step`):

* a concrete runnable IS the task's hooks plus `super().__init__` — the four layers supply
  everything else, and the abstraction stops at the API (there is no abstract trainer object);
* the typed slots come from the torch FAMILY base (`model`, the metrics spec) while the epoch
  comes from the MACHINERY base, so the same hooks compose onto another engine unchanged;
* `fit()` descends the loss, scores each split under its own prefix, and keeps a per-epoch
  history; `evaluate()` scores the val split through the same object.

`examples/trainer_api_lightning.py` is the same task on `LightningRunnable` — the two files
differ only in the base they inherit and what that base needs, which is the whole point of the
layering. The task hooks below are deliberately repeated there rather than imported: an example
has to run standalone.

Run:  python examples/trainer_api_loop.py   (needs `matrainer[torch]`)
"""

import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from confluid import PartialClass, flow
from recordstream import Stream, batch_tensor
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError

from matrainer.torch import TorchLoopRunnable
from matrainer.torch.metrics import MetricsSpec

WIDTH = 4


class IdentityTaskHooks:
    """The TASK, written once: learn the identity map on 4-vectors.

    Nothing here names an engine — that is what lets the same mixin sit in front of either
    machinery base. A real project's task mixin is this shape, with its own loss, metrics and
    prediction contract.
    """

    def _targets(self, batch: Any) -> torch.Tensor:
        return batch_tensor(batch, self.target_key, self.device)  # type: ignore[attr-defined]

    def _default_loss(self) -> nn.Module:
        return nn.MSELoss()

    def _default_metrics(self) -> MetricsSpec:
        return {"mae": PartialClass(MeanAbsoluteError)}

    def _prediction_output(self, preds: Any) -> Dict[str, Any]:
        return {"restored": preds}  # the task's prediction contract


class LoopIdentity(IdentityTaskHooks, TorchLoopRunnable):
    """The concrete class: the task's hooks in front of the bare-torch machinery base."""

    log_label = "identity-loop"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("device", "cpu")  # deterministic, and fast enough for an example
        super().__init__(**kwargs)  # the shared slots are stored verbatim by the bases

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
    output = Path(tempfile.mkdtemp(prefix="matrainer_trainer_api_loop_"))
    runnable = LoopIdentity(
        model=nn.Linear(WIDTH, WIDTH),
        train_set=records(),
        batch_size=4,
        max_epochs=6,
        checkpoint_dir=str(output),
        task="fit",
    )
    # `val_set` is a BODY slot, not a constructor parameter: the bases store configuration in the
    # constructor and take everything else by assignment, which is the post-construction rule.
    runnable.val_set = records(8)

    runnable.run()  # `run()` dispatches on `task` through the neutral base's entry points

    # The typed slots came from the FAMILY base; the per-split collections are built prefixed.
    assert isinstance(runnable.train_metrics, MetricCollection)
    assert set(runnable.val_metrics.keys()) == {"val/mae"}

    # The MACHINERY base owned the epoch: a per-epoch history, and a loss that went down.
    # `history` is {metric name: one value per epoch}, so each list is as long as the run.
    history = runnable.history
    assert {"train/loss", "train/mae", "val/loss", "val/mae"} <= set(history), sorted(history)
    losses = history["train/loss"]
    assert len(losses) == 6, losses
    first, last = losses[0], losses[-1]
    assert last < first, (first, last)

    computed = runnable.computed_metrics
    assert "val/mae" in computed, computed

    # ...and the same object evaluates, with no Trainer anywhere in the picture.
    scorer = LoopIdentity(model=nn.Linear(WIDTH, WIDTH), batch_size=4, checkpoint_dir=str(output), task="evaluate")
    scorer.val_set = records(8)
    scorer.run()
    assert "val/mae" in scorer.computed_metrics, scorer.computed_metrics

    print(f"{type(runnable).__name__}: {type(runnable).__mro__[1].__name__} + " f"{type(runnable).__mro__[2].__name__}")
    print(f"  train/loss {first:.4f} -> {last:.4f} over {len(losses)} epochs (an owned loop)")
    print(
        f"  val metrics {sorted(runnable.val_metrics.keys())}, "
        f"evaluate() scored {sorted(k for k in scorer.computed_metrics if k.startswith('val/'))}"
    )


if __name__ == "__main__":
    main()

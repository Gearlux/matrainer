"""Tracking a PyTorch Lightning run: the three hookups, on a real one-epoch fit.

Asserts what `docs/lightning.md` claims, by training a four-feature classifier on synthetic
tensors — small enough that the whole example is a couple of seconds, real enough that Lightning
resolves its own paths and writes its own checkpoint:

* **`TrackerLogger`** receives the metrics the `LightningModule` logs, and reports the tracker's
  `save_dir` — so a `ModelCheckpoint` with no `dirpath` lands under
  `<save_dir>/<experiment>/<run>/checkpoints/` and NOT in the directory the run was launched from.
  That is the whole reason the bridge has a `save_dir`.
* **`TrackingCallback`** captures the run context on fit start and finishes the manifest on end,
  carries it as `.manifest`, and lets `params=` win over anything derived from the runnable.
* a run that **raises** is recorded as `failed` rather than left looking successful.

Style 1 — the plain `log_run_context` / `capture_log_file` functions, with no Lightning at all —
is `examples/standalone_tracking.py`.

`matrainer.lightning` is optional helpers, not a platform: the module below is a standard
`LightningModule`, and dropping both the callback and the logger leaves it training exactly as it
does here.

Run:  python examples/lightning_tracking.py   (needs `matrainer[lightning]`)
"""

import tempfile
from pathlib import Path
from typing import List, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from matrainer import CsvTracker
from matrainer.lightning import TrackerLogger, TrackingCallback

FEATURES, CLASSES, SAMPLES, BATCH = 4, 2, 64, 16


class ToyClassifier(pl.LightningModule):
    """A plain LightningModule — nothing here knows matrainer exists."""

    def __init__(self, fail_at: int = -1) -> None:
        super().__init__()
        self.layer = nn.Linear(FEATURES, CLASSES)
        self.loss = nn.CrossEntropyLoss()
        self.fail_at = fail_at  # the batch index that raises, for the failure half

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        if batch_idx == self.fail_at:
            raise RuntimeError("the model diverged")  # what a real run dying looks like
        inputs, targets = batch
        loss: torch.Tensor = self.loss(self.layer(inputs), targets)
        self.log("train/loss", loss)  # reaches the tracker through TrackerLogger
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.1)


def loader() -> DataLoader:
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(SAMPLES, FEATURES, generator=generator)
    targets = torch.randint(0, CLASSES, (SAMPLES,), generator=generator)
    return DataLoader(TensorDataset(inputs, targets), batch_size=BATCH)


def trainer_for(tracker: CsvTracker, launched_from: Path, callbacks: List[pl.Callback]) -> pl.Trainer:
    """A Trainer wired both ways: the tracker as the logger AND as the provenance callback."""
    return pl.Trainer(
        callbacks=callbacks,
        max_epochs=1,
        accelerator="cpu",
        # One epoch of this dataset is four batches, and Lightning's default only flushes metrics
        # to the logger every 50 steps — so without this the tracker would receive nothing.
        log_every_n_steps=1,
        logger=TrackerLogger(tracker),
        # `default_root_dir` is where Lightning would put things WITHOUT a logger's save_dir —
        # naming it explicitly is what lets the assertions below prove nothing landed there.
        default_root_dir=str(launched_from),
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def a_tracked_run(root: Path) -> None:
    tracker = CsvTracker(experiment_name="lightning-demo", run_name="toy", save_dir=str(root / "runs"))
    checkpoints = ModelCheckpoint()  # no dirpath: it asks the logger where to write
    callback = TrackingCallback([tracker], params={"note": "set by hand"})
    launched_from = root / "cwd"

    trainer = trainer_for(tracker, launched_from, [checkpoints, callback])
    trainer.fit(ToyClassifier(), loader())

    run_dir = root / "runs" / "lightning-demo" / "toy"

    # 1. The checkpoint followed the tracker, not the launch directory.
    written = Path(checkpoints.best_model_path)
    assert written.is_file(), checkpoints.best_model_path
    assert written.parent == run_dir / "checkpoints", written
    assert launched_from not in written.parents, "never the launch directory"

    # 2. The metrics the module logged reached the tracker.
    metrics = (run_dir / "metrics.csv").read_text()
    assert "train/loss" in metrics, metrics[:200]

    # 3. The manifest rode along on the callback, and the run is recorded as a success.
    assert callback.manifest is not None, "the callback carries it from fit start onward"
    assert callback.manifest.status == "success"
    # `params=` is for values outside the config, and wins over anything derived.
    assert callback.manifest.section("params")["note"] == "set by hand"
    assert (run_dir / "manifest.yaml").is_file()

    print(f"run tracked under {run_dir}")
    for path in sorted(p.relative_to(run_dir) for p in run_dir.rglob("*") if p.is_file()):
        print(f"  {path}")
    assert not launched_from.exists() or not any(
        launched_from.rglob("*.ckpt")
    ), "the launch directory holds no checkpoint"


def a_run_that_dies(root: Path) -> None:
    """A run that raises must not be left looking successful."""
    tracker = CsvTracker(experiment_name="lightning-demo", run_name="dies", save_dir=str(root / "runs"))
    callback = TrackingCallback([tracker])

    trainer = trainer_for(tracker, root / "cwd", [callback])
    try:
        trainer.fit(ToyClassifier(fail_at=2), loader())
    except RuntimeError as exc:
        assert "diverged" in str(exc)  # the error is the caller's, never swallowed
    else:  # pragma: no cover - the module raises by construction
        raise AssertionError("expected the training step to raise")

    assert callback.manifest is not None
    assert callback.manifest.status == "failed", callback.manifest.status
    print(f"a run that raised is recorded as {callback.manifest.status!r}, not left as successful")


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="matrainer_lightning_"))
    a_tracked_run(root)
    a_run_that_dies(root)


if __name__ == "__main__":
    main()

"""Standalone tracking example — the `Tracker` surface end to end, no framework required.

Runs a fake "training" loop and tracks it with `CsvTracker` (stdlib only), so you can see every
artifact land without torch / lightning / mlflow installed. This is the torch-free proof of the
tracking contract; a real Lightning run wires the same `Tracker` via `TrackingCallback` /
`TrackerLogger` (see the README).

Run:  python examples/standalone_tracking.py
"""

import math
import tempfile
from pathlib import Path

from marainer import CsvTracker, generate_run_name, log_run_context
from marainer.tracking import capture_log_file


def main() -> None:
    # Zero-arg constructible; the native handle (here a run directory) is built on first use.
    tracker = CsvTracker(
        experiment_name="standalone-demo",
        run_name=generate_run_name(prefix="demo"),  # friendly adjective_noun slug
        save_dir=tempfile.mkdtemp(prefix="marainer_demo_"),
    )

    # ONE call at the start of the run: captures how it was invoked, what code it runs, what
    # config it read, and projects all of it onto every tracker. The manifest it returns is
    # re-emitted on every later call, so killing this script mid-loop still leaves a complete
    # record — marked "interrupted" rather than absent.
    manifest = log_run_context([tracker], params={"lr": 1e-3, "batch_size": 32, "model": "toy"})
    assert manifest is not None  # a tracker was given, so a manifest was built
    manifest.record_model(source="demo", name="toy")

    # A fake training loop — scalar metrics over time.
    for step in range(20):
        loss = math.exp(-step / 5.0)
        acc = 1.0 - loss
        tracker.log_metrics({"train/loss": loss, "train/acc": acc}, step=step)

    tracker.set_tags({"stage": "smoke"})
    manifest.record_output("epoch=19.ckpt", monitor="train/acc", value=acc)

    # ONE call at the end: uploads the log file, finishes the manifest, re-projects it.
    capture_log_file([tracker], manifest=manifest, status="success")
    tracker.finalize("FINISHED")

    run_dir = tracker.log_dir
    if run_dir is None:
        print("Tracker exposed no run directory.")
        return
    run_dir = Path(run_dir)
    print(f"Tracked run written to: {run_dir}")
    for artifact in sorted(run_dir.rglob("*")):
        if artifact.is_file():
            print(f"  {artifact.relative_to(run_dir)}")


if __name__ == "__main__":
    main()

"""A sweep driven end to end, with a stub standing in for the training.

Asserts what `docs/sweeps.md` claims about the DRIVER and the REPORT, which is all of it except
what needs a server:

* one cell per matrix row, each its own subprocess, with the row's `env:` scoped to that one cell;
* a cell that dies does not take the sweep with it — the others still run, the failure is named,
  and the exit code says so;
* the experiment is named `<prefix>_<N>` counted from the TRACKER, so a second sweep is `_2` and a
  wiped tracker restarts at 1;
* the table is the union of every metric any run reported, validation first;
* bookkeeping and within-epoch columns are folded away and never silently — the footer says how
  many went and `--all` brings them back.

The spec's `command:` key is what makes this cheap: it replaces the per-cell command prefix, so
the cells here run a 20-line stub that writes a metrics row and exits, and no training framework
is involved. A real sweep leaves `command:` out and each cell is `matrainer train <config>
--framework F --tracker T …`.

One honest limit, worth knowing before you copy this: the stub writes its metrics where the CSV
reader looks (`<save_dir>/csv/<experiment>/<run>/metrics.csv`). A real cell writes wherever its
config's tracker points, and the shipped overlays point at `./runs/<tracker>` while the sweep
reads `./sweep_logs` — a mismatch `TASKS.md` tracks. This example therefore proves the driver and
the report, not that the two halves agree by default.

Run:  python examples/sweep.py
"""

import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

from matrainer.sweep.report import collect, is_folded, next_experiment_name, render
from matrainer.sweep.runner import run_sweep
from matrainer.sweep.spec import load_spec

#: The stand-in for `matrainer train`: it reads the flags the driver appends, writes one metrics
#: row where the CSV tracker would have, and fails for one named backend.
STUB = """\
import csv, sys
from pathlib import Path

# argv: <save_dir> <config> --framework F --tracker T --experiment_name E --run_name R …
# The save_dir is OURS: the spec's `command:` is a free prefix, so a stub takes what it needs
# there rather than reading an environment variable nobody would ever set.
save_dir, rest = Path(sys.argv[1]), sys.argv[3:]
flags = dict(zip(rest, rest[1:]))
framework, run = flags["--framework"], flags["--run_name"]
if framework == "broken":
    print(f"{framework}: the model diverged", file=sys.stderr)
    sys.exit(1)

run_dir = save_dir / "csv" / flags["--experiment_name"] / run
run_dir.mkdir(parents=True, exist_ok=True)
# A real metrics.csv from `CsvTracker`: a result, a bookkeeping column, and Lightning's
# per-batch series — the last two are what the report folds away.
with (run_dir / "metrics.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["step", "val/psnr", "train/loss", "epoch", "train/loss_step"])
    writer.writerow([1, 27.4 if framework == "lightning" else 20.6, 0.05, 5, 0.04])
print(f"{framework}: wrote {run_dir / 'metrics.csv'}")
"""

SPEC = """\
config: cfg.yaml
trackers: [csv]
frameworks:
  - {{name: lightning, requires: [json]}}
  - {{name: torch, requires: [json], env: {{SWEEP_PIN: scoped-to-this-cell}}}}
  - {{name: broken, requires: [json]}}
  - {{name: absent, requires: [a_module_nobody_has]}}
command: [{python}, {stub}, {logs}]
"""


def a_project_to_sweep(root: Path, logs: Path) -> Path:
    (root / "cfg.yaml").write_text("x: 1\n")  # the cells never read it; the stub ignores it
    stub = root / "fake_train.py"
    stub.write_text(STUB)
    spec = root / "sweep.yaml"
    spec.write_text(SPEC.format(python=sys.executable, stub=stub, logs=logs))
    return spec


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="matrainer_sweep_"))
    logs = root / "sweep_logs"
    spec = load_spec(a_project_to_sweep(root, logs))

    # --- what would run, without running it -------------------------------------------------
    printed = io.StringIO()
    with redirect_stdout(printed):
        run_sweep(spec, "backends", dry_run=True, save_dir=logs)
    commands = [line for line in printed.getvalue().splitlines() if "--framework" in line]

    # Three cells, not four: the row whose `requires:` is not importable is skipped, and the
    # skip is reported rather than silent.
    assert len(commands) == 3, commands
    assert all("--tracker csv" in line and "--experiment_name" in line for line in commands)
    assert any("SWEEP_PIN=scoped-to-this-cell" in line for line in commands), "the row env is shown"
    assert not logs.exists(), "a dry run writes nothing"
    print("dry run, one command per cell:")
    print(f"  {commands[0]}")

    # --- the real thing ---------------------------------------------------------------------
    assert next_experiment_name("csv", "demo", logs) == "demo_1", "nothing recorded yet"

    with redirect_stdout(printed := io.StringIO()):
        code = run_sweep(spec, "backends", save_dir=logs)
    output = printed.getvalue()

    # A dead cell is a result, not an abort: the other two ran, the failure is named, exit 1.
    assert code == 1, code
    assert "failed: broken" in output, output
    assert "2 ok, 1 failed" in output, output
    # ...and its log is kept, so the failure can be read afterwards.
    experiment = next(line.split()[-1] for line in output.splitlines() if line.startswith("experiment :"))
    assert (logs / experiment / "broken.log").is_file()
    assert "the model diverged" in (logs / experiment / "broken.log").read_text()

    # The report is rendered BEFORE the non-zero exit — the deliverable, not a casualty.
    assert output.index("failed: broken") < output.index("| run")

    # --- what the table shows ---------------------------------------------------------------
    rows, note = collect("csv", experiment, logs)
    assert note is None, note  # csv is readable from here; clearml/wandb would say where to look
    assert sorted(r["run"] for r in rows) == ["lightning", "torch"]

    table = render(rows, experiment, "csv")
    assert "val/psnr" in table and "train/loss" in table
    # Folded away, and the footer says so and how to get them back.
    assert "epoch" not in table.split("hidden")[0], table
    assert "columns hidden" in table and "--all" in table
    assert "epoch" in render(rows, experiment, "csv", show_all=True)
    assert {k for row in rows for k in row["metrics"] if is_folded(k)} == {"epoch", "train/loss_step"}

    # --- the count comes from the tracker ---------------------------------------------------
    # Not from a local counter file: this reads the experiments the tracker already holds, so a
    # second sweep is _2 and a wiped tracker restarts at 1.
    assert next_experiment_name("csv", experiment.rsplit("_", 1)[0], logs).endswith("_2")

    print(f"ran {len(rows) + 1} cells into '{experiment}' (1 failed, 1 backend skipped as absent)")
    print(table.rstrip())  # the table AND the footer naming what was folded


if __name__ == "__main__":
    main()

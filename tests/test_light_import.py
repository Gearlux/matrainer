"""The light-core import contract: ``import matrainer`` pulls in no heavy optional backend.

Matrainer is a tracking library with a deliberately light core (confluid / liquifai / loggair /
coolname / numpy). Every heavy library (torch, pytorch-lightning, mlflow, tensorboard, wandb,
tensorflow) is behind an extra and imported lazily inside the backend that needs it — see
``docs/architecture.md`` record 2. This test guards that invariant.

The check runs in a FRESH subprocess: the pytest session itself imports torch (other tests need
it), so ``sys.modules`` in-process is already polluted — only a clean interpreter proves the
contract.
"""

import subprocess
import sys

HEAVY_MODULES = ["torch", "pytorch_lightning", "mlflow", "wandb", "tensorboard", "tensorflow"]


def _import_and_report(import_stmt: str) -> set:
    """Run ``import_stmt`` in a clean interpreter; return the heavy modules it pulled in.

    ``import_stmt`` may be multi-line; it is spliced in at column 0 (no shared indentation with a
    surrounding block) so an embedded multi-statement snippet stays syntactically valid.
    """
    script = (
        "import sys\n"
        f"{import_stmt}\n"
        f"heavy = {HEAVY_MODULES!r}\n"
        "present = [m for m in heavy if m in sys.modules]\n"
        'print(",".join(present))\n'
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    return {m for m in line.split(",") if m}


def test_import_matrainer_is_light() -> None:
    """``import matrainer`` imports no torch / lightning / mlflow / wandb / tensorboard / tensorflow."""
    assert _import_and_report("import matrainer") == set()


def test_import_matrainer_tracking_is_light() -> None:
    """``import matrainer.tracking`` (the four backends) stays import-safe — no backend library loaded."""
    assert _import_and_report("import matrainer.tracking") == set()


def test_csv_tracker_needs_no_backend_library() -> None:
    """Constructing + using the stdlib ``CsvTracker`` pulls in no heavy backend library."""
    stmt = (
        "import matrainer\n"
        't = matrainer.CsvTracker(save_dir="/tmp/matrainer_light_test")\n'
        't.log_hyperparams({"lr": 0.1})\n'
        't.log_metrics({"loss": 1.0}, step=0)\n'
        't.finalize("FINISHED")'
    )
    assert _import_and_report(stmt) == set()

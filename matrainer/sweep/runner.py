"""The sweep DRIVER — probe, name, run one subprocess per cell, isolate failures, report.

Lifted from three near-identical per-project bash drivers (2026-08-07); what stayed in each
project is the MATRIX (:mod:`matrainer.sweep.spec`). The four behaviours the bash established
are kept, each for its measured reason:

* **INSTALLED, not declared** — a project's CLI advertises every backend whether or not its
  framework is importable, so the matrix is built by PROBING (``find_spec``) and a machine with
  three extras runs three backends and says so, instead of failing five times.
* **ONE PROCESS PER CELL** — Keras 3 binds its compute engine at import, so ``keras`` and
  ``tensorflow`` cannot share an interpreter; a subprocess per cell is what makes the sweep
  possible at all, and it also contains ``import fastai``'s process-wide switch of the
  multiprocessing start method.
* **A per-row env pin stays scoped to its ONE subprocess** — the row that needs an engine pin
  takes the same code path as the rows that do not.
* **FAILURE IS ISOLATED** — one backend dying must not abort the rest; a missing row is itself
  a result. The report renders BEFORE the non-zero exit, because the comparison is the
  deliverable and it is still worth reading when one backend is broken.

The default per-cell command is the generic ``matrainer train <config> --framework F
--tracker T …`` (invoked as ``python -m matrainer.console`` so it works whether or not the
console script is on PATH); a spec's ``command:`` overrides the prefix.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from matrainer.sweep.report import DEFAULT_MLFLOW_URI, DEFAULT_SWEEP_LOGS, collect, next_experiment_name, render
from matrainer.sweep.spec import FrameworkRow, SweepSpec, load_spec

__all__ = ["Cell", "SWEEP_MODES", "build_matrix", "main", "run_sweep"]

#: The three sweep modes — the dimension(s) each varies.
SWEEP_MODES = ("backends", "trackers", "all")


@dataclass(frozen=True)
class Cell:
    """One run of the sweep: a backend (+ optional model variant) with its env and flags."""

    framework: str
    model: Optional[str]
    env: Mapping[str, str]
    flags: Sequence[str]

    @property
    def run_name(self) -> str:
        return f"{self.framework}_{self.model}" if self.model else self.framework


def installed(row: FrameworkRow) -> bool:
    """Every probe resolves, or the row is skipped. ``find_spec`` so nothing is imported."""
    return all(importlib.util.find_spec(name) is not None for name in row.requires)


def build_matrix(
    spec: SweepSpec,
    mode: str,
    only: Optional[Sequence[str]] = None,
    framework: Optional[str] = None,
) -> Tuple[List[Cell], List[str]]:
    """``(cells, skipped)`` for one tracker's pass — the tracker axis is the caller's loop.

    ``backends`` and ``all`` walk frameworks × their model variants; ``trackers`` runs ONE
    framework (``--framework``, else the first INSTALLED row) at its FIRST variant, since the
    dimension under study is where the record goes, not what trains.
    """
    if mode not in SWEEP_MODES:
        raise ValueError(f"unknown sweep mode {mode!r} — one of {', '.join(SWEEP_MODES)}")

    rows = list(spec.frameworks)
    if only:
        rows = [row for row in rows if row.name in set(only)]
    skipped = [f"{row.name} (no {','.join(row.requires)})" for row in rows if not installed(row)]
    rows = [row for row in rows if installed(row)]

    if mode == "trackers":
        if framework:
            rows = [spec.framework(framework)]
            if not installed(rows[0]):
                raise RuntimeError(f"--framework {framework}: not installed (needs {','.join(rows[0].requires)})")
        elif rows:
            rows = rows[:1]
        return [_cells_for(spec, row, first_variant_only=True)[0] for row in rows] if rows else [], skipped

    cells: List[Cell] = []
    for row in rows:
        cells.extend(_cells_for(spec, row))
    return cells, skipped


def _cells_for(spec: SweepSpec, row: FrameworkRow, first_variant_only: bool = False) -> List[Cell]:
    if not spec.models:
        return [Cell(framework=row.name, model=None, env=row.env, flags=())]
    # A row naming no variants runs the DEFAULT (first-declared) one — the bare-string case.
    names = list(row.models) or [spec.models[0].name]
    if first_variant_only:
        names = names[:1]
    return [Cell(framework=row.name, model=name, env=row.env, flags=tuple(spec.variant(name).flags)) for name in names]


def cell_command(
    spec: SweepSpec,
    cell: Cell,
    tracker: str,
    experiment: str,
    epochs: Optional[int],
) -> List[str]:
    """The full per-cell command line — the generic runner unless the spec overrides it."""
    prefix = list(spec.command) if spec.command else [sys.executable, "-m", "matrainer.console", "train"]
    command = [
        *prefix,
        str(spec.config),
        "--framework",
        cell.framework,
        "--tracker",
        tracker,
        "--experiment_name",
        experiment,
        "--run_name",
        cell.run_name,
        *cell.flags,
    ]
    if epochs is not None:
        command += ["--max_epochs", str(epochs)]
    return command


def run_sweep(
    spec: SweepSpec,
    mode: str,
    only: Optional[Sequence[str]] = None,
    tracker: Optional[str] = None,
    trackers: Optional[Sequence[str]] = None,
    framework: Optional[str] = None,
    epochs: Optional[int] = None,
    dry_run: bool = False,
    save_dir: Path = DEFAULT_SWEEP_LOGS,
    tracking_uri: str = DEFAULT_MLFLOW_URI,
) -> int:
    """Run the sweep and render each tracker's report. Non-zero when any cell failed.

    The tracker axis: ``backends`` runs ONE tracker (``--tracker``, else the spec's first);
    ``trackers`` and ``all`` run ``--trackers`` or the spec's whole list, each tracker getting
    its OWN experiment (the same date-prefix counter) so its table stays self-contained.
    """
    cells, skipped = build_matrix(spec, mode, only=only, framework=framework)
    if not cells:
        print("nothing to run: no selected backend has its framework installed.", file=sys.stderr)
        for entry in skipped:
            print(f"  skipped: {entry}", file=sys.stderr)
        return 1

    tracker_axis = list(trackers) if trackers else (spec.trackers if mode in ("trackers", "all") else [])
    if mode == "backends":
        tracker_axis = [tracker or spec.trackers[0]]

    effective_epochs = epochs if epochs is not None else spec.epochs
    prefix = time.strftime("%a_%b_%-d_%Y")
    failed_total = 0

    for active_tracker in tracker_axis:
        if not dry_run and not _tracker_setup(spec, active_tracker):
            failed_total += len(cells)
            continue
        if dry_run:
            experiment = f"{prefix}_<n>"
        else:
            experiment = next_experiment_name(active_tracker, prefix, save_dir, tracking_uri)
            if not experiment:
                # Checked rather than assumed: an unanswered tracker would flow an empty name
                # into `--experiment_name ''` — every run training into an unnamed experiment.
                print(f"could not name the experiment: the {active_tracker} tracker did not answer.", file=sys.stderr)
                failed_total += len(cells)
                continue

        print()
        print(f"experiment : {experiment}")
        print(f"tracker    : {active_tracker}")
        print(f"config     : {spec.config}")
        print(f"runs       : {len(cells)}")
        if skipped:
            print(f"skipped    : {', '.join(skipped)}")
        print()

        # `save_dir`, not a literal: this is the directory the reader half
        # (`next_experiment_name` / `collect`) was already given, so a hardcoded
        # "sweep_logs" made `--save-dir` move where reports are READ from while the
        # logs kept being WRITTEN to the default — a flag that silently half-worked.
        log_dir = Path(save_dir) / experiment
        passed: List[str] = []
        failed: List[str] = []
        for cell in cells:
            command = cell_command(spec, cell, active_tracker, experiment, effective_epochs)
            if dry_run:
                pin = " ".join(f"{k}={v}" for k, v in cell.env.items())
                print((pin + " " if pin else "") + " ".join(command))
                continue
            log_dir.mkdir(parents=True, exist_ok=True)
            log = log_dir / f"{cell.run_name}.log"
            print(f"==> {cell.run_name:<24} ", end="", flush=True)
            started = time.monotonic()
            with log.open("w") as handle:
                # The row's env overlays the parent's, scoped to this ONE subprocess.
                result = subprocess.run(
                    command, stdout=handle, stderr=subprocess.STDOUT, env={**os.environ, **cell.env}
                )
            seconds = int(time.monotonic() - started)
            if result.returncode == 0:
                print(f"ok   ({seconds}s)")
                passed.append(cell.run_name)
            else:
                print(f"FAIL ({seconds}s) — {log}")
                # The last non-empty line is almost always the exception; enough to triage.
                lines = [line for line in log.read_text().splitlines() if line.strip()]
                if lines:
                    print(f"      {lines[-1]}")
                failed.append(cell.run_name)

        if dry_run:
            continue
        print()
        print("=" * 60)
        print(f" {experiment} — {len(passed)} ok, {len(failed)} failed")
        print("=" * 60)
        if failed:
            print(f"  failed: {', '.join(failed)}\n  logs:   {log_dir}/")
        print()
        rows, note = collect(active_tracker, experiment, save_dir, tracking_uri)
        print(render(rows, experiment, active_tracker, note))
        failed_total += len(failed)

    return 1 if failed_total else 0


def _tracker_setup(spec: SweepSpec, tracker: str) -> bool:
    """The spec's per-tracker setup command, run once — the tracker has to be up BEFORE the
    first cell, not after the first failure. False (and the tracker's pass is skipped) when the
    setup itself fails."""
    command = spec.tracker_setup.get(tracker)
    if not command:
        return True
    print(f"==> tracker setup ({tracker}): {' '.join(command)}")
    result = subprocess.run(list(command))
    if result.returncode != 0:
        print(f"tracker setup for {tracker!r} failed (exit {result.returncode}) — skipping its runs.", file=sys.stderr)
        return False
    return True


def main(argv: Sequence[str]) -> int:
    """``matrainer sweep backends|trackers|all <sweep.yaml> [flags]`` — the runner's dispatch.

    Flags: ``--only a,b`` (filter frameworks), ``--tracker T`` (backends mode),
    ``--trackers a,b`` (trackers/all modes), ``--framework F`` (trackers mode),
    ``--epochs N``, ``--dry-run``, ``--save-dir …``, ``--tracking-uri …``.
    """
    positional = [a for a in argv if not a.startswith("--")]
    if len(positional) < 2:
        print(__doc__)
        return 2
    mode, spec_path = positional[0], positional[1]

    flags: Dict[str, str] = {}
    for flag, following in zip(argv, list(argv[1:]) + [""]):
        if flag.startswith("--") and flag != "--dry-run":
            flags[flag] = following

    try:
        spec = load_spec(Path(spec_path))
        return run_sweep(
            spec,
            mode,
            only=flags["--only"].split(",") if "--only" in flags else None,
            tracker=flags.get("--tracker"),
            trackers=flags["--trackers"].split(",") if "--trackers" in flags else None,
            framework=flags.get("--framework"),
            epochs=int(flags["--epochs"]) if "--epochs" in flags else None,
            dry_run="--dry-run" in argv,
            save_dir=Path(flags.get("--save-dir", DEFAULT_SWEEP_LOGS)),
            tracking_uri=flags.get("--tracking-uri", DEFAULT_MLFLOW_URI),
        )
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        print(f"sweep: {exc}", file=sys.stderr)
        return 2

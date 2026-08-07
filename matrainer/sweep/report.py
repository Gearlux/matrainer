"""Name a sweep's experiment, and read the sweep back out of a tracker.

A *sweep* is one config run under many backends into one tracker, so the runs become comparable.
Driving it is the consuming project's job — which cells the matrix has, which engine each row pins,
what the config overlays — but the two TRACKER-side halves are the same wherever it runs, and they
live here:

* :func:`next_experiment_name` — the experiment name for THIS run of the sweep, ``<prefix>_<N>``
  where N is one past the highest already recorded. Derived from the TRACKER rather than a local
  counter file, so a wiped tracker restarts at 1 and a second checkout does not collide.
* :func:`collect` + :func:`render` — one row per run: what it was called, whether it finished, how
  long it took, and what it scored. Bookkeeping columns are folded away by default and ``--all``
  shows them; see :data:`BOOKKEEPING`.

Only the backends that can actually be READ are implemented (:data:`QUERYABLE`). A tracker is a
write surface first; some have no local query path at all — an offline ClearML task is a directory
waiting to be uploaded, and W&B needs its server — and pretending otherwise would mean a report that
silently shows nothing. Those name the sweep from a local counter and say where to look.

**Why this is matrainer's and not each project's.** It is a query against a tracking backend, and
the vocabulary it folds on (``epoch`` / ``runtime`` / ``total_flos``, Lightning's ``_step`` suffix)
is the one :mod:`matrainer.metric_names` already owns — the same argument that moved
:mod:`matrainer.metric_slots` here. Three task projects had byte-identical copies of it (verified by
an AST comparison with docstrings stripped) differing only in prose.

Usage — it is a module main rather than a console script, because matrainer ships no binary::

    python -m matrainer.sweep next-experiment mlflow Thu_Aug_6_2026
    python -m matrainer.sweep report mlflow Thu_Aug_6_2026_1
    python -m matrainer.sweep report mlflow Thu_Aug_6_2026_1 --all
    python -m matrainer.sweep report csv Thu_Aug_6_2026_1 --save-dir ./sweep_logs
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "BOOKKEEPING",
    "DEFAULT_MLFLOW_URI",
    "DEFAULT_SWEEP_LOGS",
    "QUERYABLE",
    "WITHIN_EPOCH_SUFFIX",
    "collect",
    "is_bookkeeping",
    "is_folded",
    "is_within_epoch_series",
    "next_experiment_name",
    "render",
]

#: Where a sweep's file-backed trackers write, unless a caller says otherwise. Relative on purpose:
#: it is resolved against the project directory the driver runs from.
DEFAULT_SWEEP_LOGS = Path("./sweep_logs")

#: The workspace's MLflow, started by ``aisland mlflow start``.
DEFAULT_MLFLOW_URI = "http://127.0.0.1:5005"

#: Trackers that can be READ back from here. Every other tracker is write-only from this side and
#: :func:`collect` says where to open it instead — see the module docstring.
QUERYABLE = ("mlflow", "csv")

#: Base names that are a run's BOOKKEEPING rather than its result. ``transformers.Trainer`` records
#: its timings, throughput and FLOP count in the same ``log_history`` as the loss, and Lightning and
#: HF both log ``epoch`` — all worth keeping in the tracker, all noise in a table whose job is to
#: compare a dozen runs on the same few numbers. Hidden here, never dropped: ``--all`` shows them and
#: the footer always says how many were folded away.
#:
#: A curated list is the right shape HERE and the wrong shape one layer down: the display decides
#: what a reader wants first, while the name translation (:mod:`matrainer.metric_names`) must not
#: decide that a recorded value is uninteresting. Add a name when a backend starts reporting another.
BOOKKEEPING = (
    "epoch",
    "step",
    "runtime",
    "samples_per_second",
    "steps_per_second",
    "total_flos",
    "grad_norm",
    "learning_rate",
)

#: Lightning's suffix for a series it logs per BATCH beside the same quantity per epoch. It appends
#: ``_step`` / ``_epoch`` only when ONE name carries both, which is why a runnable logs ``train/loss``
#: and ``train/loss_step`` as two explicit names — ``train/loss_epoch`` would be a name no other
#: backend writes, in the column a sweep compares.
WITHIN_EPOCH_SUFFIX = "_step"


def is_bookkeeping(key: str) -> bool:
    """True for a metric key whose BASE name is bookkeeping — ``train/runtime``, ``epoch``, …"""
    return key.rpartition("/")[2] in BOOKKEEPING


def is_within_epoch_series(key: str) -> bool:
    """True for a per-BATCH series of a quantity the table already shows per epoch.

    ``train/loss_step`` is a real result, not bookkeeping — which is why it gets its own rule rather
    than a line in :data:`BOOKKEEPING`, whose footer would then call a metric something it is not. It
    is folded for a different reason: it measures the same quantity as ``train/loss`` on a finer axis
    a one-row-per-run table cannot show, and only the backend whose engine emits the suffix fills it,
    so it is the one column every other row leaves blank.
    """
    base = key.rpartition("/")[2]
    return base.endswith(WITHIN_EPOCH_SUFFIX) and base != WITHIN_EPOCH_SUFFIX


def is_folded(key: str) -> bool:
    """True for a column the comparison table hides by default — either reason above.

    ONE predicate at the call site, so a third reason is a third function here and no change there.
    Hidden is never dropped: ``--all`` shows everything and the footer always says what went and how
    to get it back.
    """
    return is_bookkeeping(key) or is_within_epoch_series(key)


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------
def next_experiment_name(
    tracker: str,
    prefix: str,
    save_dir: Path = DEFAULT_SWEEP_LOGS,
    tracking_uri: str = DEFAULT_MLFLOW_URI,
) -> str:
    """``<prefix>_<N>``, N one past the highest N already recorded for this prefix.

    Args:
        tracker: Which backend to count in — see :data:`QUERYABLE`.
        prefix: The stable part of the name, typically a date (``Thu_Aug_6_2026``).
        save_dir: Where the file-backed trackers write.
        tracking_uri: The MLflow server to ask.
    """
    existing = _existing_experiments(tracker, prefix, save_dir, tracking_uri)
    used = {int(m.group(1)) for name in existing if (m := re.fullmatch(rf"{re.escape(prefix)}_(\d+)", name))}
    return f"{prefix}_{max(used, default=0) + 1}"


def _existing_experiments(tracker: str, prefix: str, save_dir: Path, tracking_uri: str) -> List[str]:
    if tracker == "mlflow":
        client = _mlflow_client(tracking_uri)
        return [e.name for e in client.search_experiments() if e.name.startswith(prefix)]
    if tracker == "csv":
        root = save_dir / "csv"
        return [p.name for p in root.iterdir() if p.is_dir()] if root.is_dir() else []
    # Not queryable: fall back to a local counter, which is per-checkout but never wrong about how
    # many times THIS script has run here.
    counter = save_dir / f".{tracker}_sweeps"
    counter.parent.mkdir(parents=True, exist_ok=True)
    seen = int(counter.read_text().strip() or 0) if counter.exists() else 0
    counter.write_text(str(seen + 1))
    return [f"{prefix}_{i}" for i in range(1, seen + 1)]


def _mlflow_client(tracking_uri: str) -> Any:
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=tracking_uri)


# ---------------------------------------------------------------------------
# Reading a sweep back
# ---------------------------------------------------------------------------
def collect(
    tracker: str,
    experiment: str,
    save_dir: Path = DEFAULT_SWEEP_LOGS,
    tracking_uri: str = DEFAULT_MLFLOW_URI,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """``(rows, note)`` — one row per run, plus where to look when this tracker cannot be read.

    A row is ``{"run": str, "status": str, "seconds": float | None, "metrics": {name: float}}``.
    ``note`` is ``None`` whenever the tracker was actually queried, so an EMPTY list from a queryable
    tracker keeps meaning "this sweep recorded nothing" rather than "this tracker cannot be read".
    """
    if tracker not in QUERYABLE:
        where = {
            "tensorboard": f"tensorboard --logdir {save_dir / 'tensorboard'}",
            "aim": f"aim up --repo {save_dir / 'aim'}",
            "clearml": "the offline task directories printed by each run (clearml-task import)",
            "wandb": "the W&B web UI for this project",
        }.get(tracker, "this tracker's own UI")
        return [], f"{tracker} cannot be queried from here — open it with: {where}"
    if tracker == "mlflow":
        return _collect_mlflow(experiment, tracking_uri), None
    return _collect_csv(experiment, save_dir), None


def _collect_mlflow(experiment: str, tracking_uri: str) -> List[Dict[str, Any]]:
    client = _mlflow_client(tracking_uri)
    found = client.get_experiment_by_name(experiment)
    if found is None:
        return []
    rows: List[Dict[str, Any]] = []
    # A Run object rather than the default DataFrame: the report needs no pandas, and it carries the
    # status and timings a DataFrame flattens into prefixed columns.
    for run in client.search_runs([found.experiment_id], max_results=200):
        info, data = run.info, run.data
        started, ended = info.start_time, info.end_time
        rows.append(
            {
                "run": data.tags.get("mlflow.runName") or info.run_id[:8],
                "status": info.status,
                "seconds": round((ended - started) / 1000.0, 1) if started and ended else None,
                "metrics": {k: v for k, v in data.metrics.items()},
            }
        )
    return sorted(rows, key=lambda r: r["run"])


def _collect_csv(experiment: str, save_dir: Path) -> List[Dict[str, Any]]:
    root = save_dir / "csv" / experiment
    if not root.is_dir():
        return []
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        metrics: Dict[str, float] = {}
        path = run_dir / "metrics.csv"
        if path.exists():
            with path.open() as handle:
                # Last-write-wins: the CSV backend appends a row per `log_metrics` call, and the
                # report wants each metric's final value, exactly like MLflow's `data.metrics`.
                for record in csv.DictReader(handle):
                    for key, value in record.items():
                        if key != "step" and value not in ("", None):
                            metrics[key] = float(value)
        rows.append({"run": run_dir.name, "status": "-", "seconds": None, "metrics": metrics})
    return rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render(
    rows: Sequence[Dict[str, Any]],
    experiment: str,
    tracker: str,
    note: Optional[str] = None,
    show_all: bool = False,
) -> str:
    """The comparison table, as text.

    Args:
        rows: What :func:`collect` returned.
        experiment: The experiment name, for the footer and the ``--all`` hint.
        tracker: Which backend the rows came from, same.
        note: :func:`collect`'s second return value — rendered ALONE when set, because an empty
            table from a write-only backend reads as "the sweep recorded nothing".
        show_all: Put the folded columns back.
    """
    if note:
        return f"{note}\n"
    if not rows:
        return f"no runs recorded under {tracker} experiment '{experiment}'\n"

    # Every metric any run reported, VALIDATION first — that is the column a reader compares
    # backends on, and each backend translates out of its engine's own spelling on the way to the
    # tracker (`matrainer.metric_names`), so one column name means one thing in every row.
    #
    # Still the UNION rather than a hand-picked list, and that is a property of the REPORT rather
    # than of the config it happens to be pointed at: a metric one backend computes and the others do
    # not earns its own column with the rest of the rows blank, which is the honest rendering — the
    # alternative is a table that silently drops whatever does not fit its widest complete row.
    #
    # Non-scalar values never appear here at all, and that is upstream of this module: a `[C]` vector
    # or a `[C, C]` matrix cannot go through a `{name: float}` metric channel, so a runnable sends it
    # to the trackers' TEXT channel instead. Read those in the tracker's own UI (or the run log).
    keys = sorted({k for row in rows for k in row["metrics"]})
    hidden = [] if show_all else [k for k in keys if is_folded(k)]
    keys = [k for k in keys if k not in hidden]
    keys = [k for k in keys if k.startswith("val/")] + [k for k in keys if not k.startswith("val/")]

    header = ["run", "status", "sec", *keys]
    table = [
        [
            row["run"],
            row["status"],
            "" if row["seconds"] is None else f"{row['seconds']:.0f}",
            *[_fmt(row["metrics"].get(k)) for k in keys],
        ]
        for row in rows
    ]
    # What was folded away is SAID, never silently dropped — a reader who wants the timings has to
    # be able to see that they are there and how to get them.
    footer = f"\n{len(rows)} runs in {tracker} experiment '{experiment}'\n"
    if hidden:
        # The REASON is named, not just the count: "bookkeeping" alone would mislabel
        # `train/loss_step`, which is a genuine result folded for a different reason.
        reasons = ", ".join(
            reason
            for reason, holds in (
                ("bookkeeping", any(is_bookkeeping(k) for k in hidden)),
                ("within-epoch series", any(is_within_epoch_series(k) for k in hidden)),
            )
            if holds
        )
        footer += (
            f"{len(hidden)} columns hidden — {reasons} ({', '.join(hidden)}) — see them with:\n"
            f"  python -m matrainer.sweep report {tracker} {experiment} --all\n"
        )
    return _tabulate(header, table) + footer


def _fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.4f}"


def _tabulate(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    try:
        from tabulate import tabulate

        return str(tabulate(rows, headers=header, tablefmt="github"))
    except ImportError:  # pragma: no cover - tabulate rides in with matrainer[lightning]
        widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
        lines = ["  ".join(str(c).ljust(w) for c, w in zip(row, widths)) for row in [header, *rows]]
        return "\n".join(lines)


def main(argv: Sequence[str]) -> int:
    """``python -m matrainer.sweep <command> <tracker> <name> [--all] [--save-dir …] [--tracking-uri …]``."""
    positional = [a for a in argv if not a.startswith("--")]
    if len(positional) < 3:
        print(__doc__)
        return 2
    command, tracker, name = positional[0], positional[1], positional[2]
    save_dir, tracking_uri = DEFAULT_SWEEP_LOGS, DEFAULT_MLFLOW_URI
    for flag, following in zip(argv, list(argv[1:]) + [""]):
        if flag == "--save-dir":
            save_dir = Path(following)
        elif flag == "--tracking-uri":
            tracking_uri = following

    if command == "next-experiment":
        print(next_experiment_name(tracker, name, save_dir, tracking_uri))
        return 0
    if command == "report":
        rows, note = collect(tracker, name, save_dir, tracking_uri)
        print(render(rows, name, tracker, note, show_all="--all" in argv))
        return 0
    if command == "json":  # for a caller that wants the rows rather than the table
        rows, _ = collect(tracker, name, save_dir, tracking_uri)
        print(json.dumps(rows, indent=2))
        return 0
    print(f"unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

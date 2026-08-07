"""The sweep: name it, RUN it, and read it back out of a tracker.

A *sweep* is one config run under many backends and/or many trackers, so the runs become
comparable. The package holds three halves:

* :mod:`matrainer.sweep.report` — the tracker-side halves (naming this run of the sweep,
  collecting and rendering the comparison table). The original module, unchanged.
* :mod:`matrainer.sweep.spec` — the declarative sweep SPEC a consuming project ships
  (``examples/sweep/sweep.yaml``): its matrix, per-row env pins and probes, its trackers.
  The spec is the project's half; everything else here is machinery.
* :mod:`matrainer.sweep.runner` — the driver: probe installed frameworks, name the experiment
  per tracker, run one subprocess per cell through the generic ``matrainer train``, isolate
  failures, render the report. Lifted from three near-identical per-project bash drivers
  (2026-08-07).

Command surface (the ``matrainer`` console script, or ``python -m matrainer.sweep``)::

    matrainer sweep backends examples/sweep/sweep.yaml     # frameworks × models, one tracker
    matrainer sweep trackers examples/sweep/sweep.yaml     # every tracker, one framework
    matrainer sweep all      examples/sweep/sweep.yaml     # frameworks × models × trackers
    matrainer sweep report mlflow Thu_Aug_6_2026_1         # the read-back, as before
"""

from typing import Sequence

from matrainer.sweep.report import (
    BOOKKEEPING,
    DEFAULT_MLFLOW_URI,
    DEFAULT_SWEEP_LOGS,
    QUERYABLE,
    WITHIN_EPOCH_SUFFIX,
    collect,
    is_bookkeeping,
    is_folded,
    is_within_epoch_series,
)
from matrainer.sweep.report import main as _report_main
from matrainer.sweep.report import next_experiment_name, render
from matrainer.sweep.runner import SWEEP_MODES
from matrainer.sweep.runner import main as _runner_main
from matrainer.sweep.spec import SweepSpec, load_spec

__all__ = [
    "BOOKKEEPING",
    "DEFAULT_MLFLOW_URI",
    "DEFAULT_SWEEP_LOGS",
    "QUERYABLE",
    "SWEEP_MODES",
    "SweepSpec",
    "WITHIN_EPOCH_SUFFIX",
    "collect",
    "is_bookkeeping",
    "is_folded",
    "is_within_epoch_series",
    "load_spec",
    "main",
    "next_experiment_name",
    "render",
]


def main(argv: Sequence[str]) -> int:
    """One dispatch for both spellings (``matrainer sweep …`` / ``python -m matrainer.sweep …``).

    The sweep MODES go to the runner; everything else keeps the report module's original
    surface (``report`` / ``next-experiment`` / ``json``) with its pinned exit codes.
    """
    if argv and argv[0] in SWEEP_MODES:
        return _runner_main(argv)
    return _report_main(argv)

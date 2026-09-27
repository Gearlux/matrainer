"""``CsvTracker`` — the always-available, zero-dependency local tracking backend.

Writes everything to plain files under ``<save_dir>/<experiment_name>/<run_name>/``
using only the standard library, so it works in the light ``matrainer`` core install
with no extra:

* ``metrics.csv`` — one row per :meth:`log_metrics` call (``step`` + the metric
  columns, unioned across calls);
* ``hparams.json`` — the hyperparameters from :meth:`log_hyperparams`;
* ``tags.json`` — run tags;
* ``artifacts/<artifact_dir>/<name>`` — copies of files passed to
  :meth:`log_artifact`, and ``<name>.png`` for figures.

It is the sensible default when no remote tracking server is configured, and the
file-copy fallback that the old ``_log_paths_to_logger`` provided for
non-MLflow loggers.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, List, Mapping, Optional, Union

from confluid import configurable
from loggair import get_logger

from matrainer.tracking.base import Tracker

logger = get_logger(__name__)


@configurable
class CsvTracker(Tracker):
    """Local file-based tracker (stdlib only) — the default when no server is wired.

    See the module docstring for the on-disk layout. Zero-arg constructible; the
    run directory is created on first :attr:`experiment` access.

    Args:
        experiment_name: Experiment (sub-directory) name.
        run_name: Run name; ``None`` draws a random slug.
        save_dir: Root directory for all runs.
        tags: Optional tags written to ``tags.json``.
    """

    def __init__(
        self,
        experiment_name: str = "matrainer",
        run_name: Optional[str] = None,
        save_dir: str = "./runs",
        tags: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__(experiment_name, run_name, save_dir, tags)
        self._run_dir: Optional[Path] = None
        self._metric_columns: List[str] = []
        self._metric_rows: List[dict] = []

    @property
    def experiment(self) -> Path:
        """The run directory (created lazily); doubles as this backend's native handle."""
        if self._run_dir is None:
            run_dir = Path(self.save_dir) / self.experiment_name / self.run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            self._run_dir = run_dir
            if self.tags:
                (run_dir / "tags.json").write_text(json.dumps(self.tags, indent=2, sort_keys=True))
        return self._run_dir

    @property
    def log_dir(self) -> Optional[Path]:
        return self.experiment

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        path = self.experiment / "hparams.json"
        existing: dict = {}
        if path.exists():
            try:
                existing = json.loads(path.read_text())
            except json.JSONDecodeError:  # pragma: no cover - corrupted partial write
                existing = {}
        existing.update({str(k): v for k, v in params.items()})
        path.write_text(json.dumps(existing, indent=2, sort_keys=True, default=str))

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        row: dict = {"step": step if step is not None else len(self._metric_rows)}
        row.update({str(k): float(v) for k, v in metrics.items()})
        self._metric_rows.append(row)
        for key in row:
            if key not in self._metric_columns:
                self._metric_columns.append(key)
        self._write_metrics_csv()

    def _write_metrics_csv(self) -> None:
        path = self.experiment / "metrics.csv"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._metric_columns)
            writer.writeheader()
            writer.writerows(self._metric_rows)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        super().set_tags(tags)
        (self.experiment / "tags.json").write_text(json.dumps(self.tags, indent=2, sort_keys=True))

    def log_artifact(self, path: Union[str, Path], artifact_dir: str = "") -> None:
        src = Path(path)
        if not src.exists():
            logger.debug(f"CsvTracker.log_artifact: {src} does not exist; skipping.")
            return
        dest_dir = self.experiment / "artifacts" / artifact_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / src.name)

    def log_figure(self, name: str, figure: Any, step: Optional[int] = None) -> None:
        save = getattr(figure, "savefig", None)
        if not callable(save):
            logger.debug(f"CsvTracker.log_figure: {name} is not a matplotlib Figure; skipping.")
            return
        dest_dir = self.experiment / "artifacts"
        dest_dir.mkdir(parents=True, exist_ok=True)
        safe = name.replace("/", "_")
        try:
            save(dest_dir / f"{safe}.png")
        except Exception as exc:  # pragma: no cover - matplotlib backend specific
            logger.debug(f"CsvTracker.log_figure failed for {name}: {exc}")


__all__ = ["CsvTracker"]

"""``TensorBoardTracker`` — TensorBoard backend (``pip install marainer[tensorboard]``).

Wraps ``torch.utils.tensorboard.SummaryWriter`` (so the ``[tensorboard]`` extra
implies ``torch``). The writer is created lazily on first :attr:`experiment`
access; importing this module needs no torch/tensorboard installed. Artifacts are
file-copied into the run's log directory (TensorBoard has no artifact concept);
figures go through ``SummaryWriter.add_figure``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from confluid import configurable
from loggair import get_logger

from marainer._imports import require
from marainer.tracking.base import Tracker

logger = get_logger(__name__)


@configurable
class TensorBoardTracker(Tracker):
    """Track an experiment run with TensorBoard.

    Writes event files under ``<save_dir>/<experiment_name>/<run_name>/``. Zero-arg
    constructible; the ``SummaryWriter`` is created on first :attr:`experiment` access.

    Args:
        experiment_name: Experiment (sub-directory) name.
        run_name: Run name; ``None`` draws a random slug.
        save_dir: Root directory for event files.
        tags: Tags recorded as text summaries.
    """

    def __init__(
        self,
        experiment_name: str = "marainer",
        run_name: Optional[str] = None,
        save_dir: str = "./runs",
        tags: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__(experiment_name, run_name, save_dir, tags)
        self._writer: Optional[Any] = None

    @property
    def _run_dir(self) -> Path:
        return Path(self.save_dir) / self.experiment_name / self.run_name

    @property
    def experiment(self) -> Any:
        """The ``SummaryWriter``, created lazily under the run directory."""
        if self._writer is None:
            tb = require("torch.utils.tensorboard", "tensorboard", feature="TensorBoardTracker")
            self._run_dir.mkdir(parents=True, exist_ok=True)
            self._writer = tb.SummaryWriter(log_dir=str(self._run_dir))
            for key, value in self.tags.items():
                self._writer.add_text(f"tags/{key}", str(value))
        return self._writer

    @property
    def log_dir(self) -> Optional[Path]:
        return self._run_dir

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        clean = {str(k): v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
        try:
            self.experiment.add_hparams(clean, {})
        except Exception as exc:  # pragma: no cover - tensorboard internal
            logger.debug(f"TensorBoard add_hparams failed: {exc}")

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        writer = self.experiment
        for key, value in metrics.items():
            writer.add_scalar(str(key), float(value), global_step=step)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        super().set_tags(tags)
        writer = self.experiment
        for key, value in tags.items():
            writer.add_text(f"tags/{key}", str(value))

    def log_artifact(self, path: Union[str, Path], artifact_dir: str = "") -> None:
        src = Path(path)
        if not src.exists():
            logger.debug(f"TensorBoardTracker.log_artifact: {src} does not exist; skipping.")
            return
        dest_dir = self._run_dir / (artifact_dir or "artifacts")
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / src.name)

    def log_figure(self, name: str, figure: Any, step: Optional[int] = None) -> None:
        try:
            self.experiment.add_figure(name, figure, global_step=step)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"TensorBoard add_figure failed for {name}: {exc}")

    def save(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def finalize(self, status: str = "success") -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None


__all__ = ["TensorBoardTracker"]

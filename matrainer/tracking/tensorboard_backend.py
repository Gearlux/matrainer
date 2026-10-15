"""``TensorBoardTracker`` — TensorBoard backend (``pip install matrainer[tensorboard]``).

Wraps ``torch.utils.tensorboard.SummaryWriter`` (so the ``[tensorboard]`` extra
implies ``torch``). The writer is created lazily on first :attr:`experiment`
access; importing this module needs no torch/tensorboard installed. Artifacts are
file-copied into the run's log directory (TensorBoard has no artifact concept);
figures go through ``SummaryWriter.add_figure``.
"""

from __future__ import annotations

import shutil
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

from confluid import configurable
from loggair import get_logger

from matrainer._imports import require
from matrainer.tracking.base import Tracker

if TYPE_CHECKING:  # type-only: `manifest` reaches `base`, so a runtime import would cycle.
    from matrainer.tracking.manifest import RunManifest

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
        raise_errors: Raise a failed call's error after its debug line (see :class:`Tracker`).
    """

    def __init__(
        self,
        experiment_name: str = "matrainer",
        run_name: Optional[str] = None,
        save_dir: str = "./runs",
        tags: Optional[Mapping[str, str]] = None,
        raise_errors: bool = False,
    ) -> None:
        super().__init__(experiment_name, run_name, save_dir, tags, raise_errors)
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
        except Exception as exc:  # tensorboard internal
            logger.debug(f"TensorBoard add_hparams failed: {exc}")
            self._raise_failure(exc)

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
        except Exception as exc:
            logger.debug(f"TensorBoard add_figure failed for {name}: {exc}")
            self._raise_failure(exc)

    def log_text(self, name: str, text: str, step: Optional[int] = None) -> None:
        """The Text tab, which is TensorBoard's only non-numeric native slot.

        Durable AND visible: the value goes into the event file (so the base class's
        ``.md`` copy would add nothing) and the UI renders it, with ``step`` driving the
        slider — so a per-epoch value is browsable rather than only the last one.
        """
        try:
            self.experiment.add_text(name, text, global_step=step)
        except Exception as exc:
            logger.debug(f"TensorBoard add_text failed for {name}: {exc}")
            self._raise_failure(exc)

    def log_manifest(self, manifest: "RunManifest") -> None:
        """Artifact + ``add_hparams`` (the default), PLUS the manifest in TensorBoard's Text tab.

        TensorBoard has no artifact store — the default's ``log_artifact`` merely copies
        the files into the run directory, where the UI will never show them. ``add_text``
        is the one native slot it does have, so the manifest is also rendered there as a
        fenced YAML block, which is what makes it *visible* rather than merely present.

        Hyperparameters reach ``add_hparams`` through the inherited default;
        :meth:`log_hyperparams` already drops the non-scalar values TensorBoard rejects
        (it accepts only int / float / str / bool).
        """
        self._run_all(partial(super().log_manifest, manifest), partial(self._render_manifest, manifest))

    def _render_manifest(self, manifest: "RunManifest") -> None:
        """The manifest as a fenced YAML block in the Text tab."""
        try:
            self.experiment.add_text("manifest", f"```yaml\n{manifest.path.read_text()}\n```")
        except Exception as exc:  # tensorboard internal / unreadable file
            logger.debug(f"TensorBoard add_text failed for the manifest: {exc}")
            self._raise_failure(exc)

    def save(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def finalize(self, status: str = "success") -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None


__all__ = ["TensorBoardTracker"]

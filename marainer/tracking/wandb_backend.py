"""``WandbTracker`` — Weights & Biases backend (``pip install marainer[wandb]``).

Wraps a ``wandb`` run. ``wandb`` is imported lazily on first :attr:`experiment`
access; importing this module needs no wandb installed.

``log_text`` is deliberately NOT overridden: W&B has no text primitive
(``wandb.Html`` would mean escaping the caller's text into markup, and a
``wandb.Table`` is a dataframe wearing a different name), while its run directory
IS its upload channel — so the base class's ``.md`` file, written under
:attr:`log_dir`, is synced with the run for free and stays readable as itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Union

from confluid import configurable
from loggair import get_logger

from marainer._imports import require
from marainer.tracking.base import Tracker

logger = get_logger(__name__)


@configurable
class WandbTracker(Tracker):
    """Track an experiment run in Weights & Biases.

    Zero-arg constructible; the ``wandb`` run is created on first :attr:`experiment`
    access (``wandb.init``).

    Args:
        experiment_name: The W&B project name.
        run_name: The W&B run name; ``None`` draws a random slug.
        save_dir: The ``dir=`` passed to ``wandb.init`` (local run files).
        tags: Tags attached to the run.
        entity: Optional W&B entity (team/user); ``None`` uses the default.
    """

    def __init__(
        self,
        experiment_name: str = "marainer",
        run_name: Optional[str] = None,
        save_dir: str = "./runs",
        tags: Optional[Mapping[str, str]] = None,
        entity: Optional[str] = None,
    ) -> None:
        super().__init__(experiment_name, run_name, save_dir, tags)
        self.entity = entity
        self._run: Optional[Any] = None

    @property
    def experiment(self) -> Any:
        """The ``wandb`` run, created lazily via ``wandb.init``."""
        if self._run is None:
            wandb = require("wandb", "wandb", feature="WandbTracker")
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
            self._run = wandb.init(
                project=self.experiment_name,
                name=self.run_name,
                dir=self.save_dir,
                entity=self.entity,
                tags=list(self.tags.keys()) if self.tags else None,
                reinit=True,
            )
            if self.tags:
                self._run.config.update(self.tags, allow_val_change=True)
        return self._run

    @property
    def version(self) -> Optional[str]:
        return getattr(self._run, "id", None) or self.run_name

    @property
    def log_dir(self) -> Optional[Path]:
        d = getattr(self._run, "dir", None)
        return Path(d) if d else None

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        self.experiment.config.update(dict(params), allow_val_change=True)

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        self.experiment.log({str(k): float(v) for k, v in metrics.items()}, step=step)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        super().set_tags(tags)
        self.experiment.config.update(dict(tags), allow_val_change=True)

    def log_artifact(self, path: Union[str, Path], artifact_dir: str = "") -> None:
        src = Path(path)
        if not src.exists():
            logger.debug(f"WandbTracker.log_artifact: {src} does not exist; skipping.")
            return
        try:
            self.experiment.save(str(src), base_path=str(src.parent))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"wandb save failed for {src}: {exc}")

    def log_figure(self, name: str, figure: Any, step: Optional[int] = None) -> None:
        try:
            self.experiment.log({name: figure}, step=step)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"wandb log figure failed for {name}: {exc}")

    def finalize(self, status: str = "success") -> None:
        if self._run is not None:
            exit_code = 0 if status == "success" else 1
            try:
                self._run.finish(exit_code=exit_code)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"wandb finish failed: {exc}")
            self._run = None


__all__ = ["WandbTracker"]

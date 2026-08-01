"""``AimTracker`` — the Aim experiment-tracking backend (``pip install marainer[aim]``).

Aim is Apache-2.0 and entirely self-hosted: runs land in a local ``.aim`` repository and
``aim up`` serves the comparison UI straight from it, with no server to operate and no
account. That is why it is an in-tree backend — see the README's "Choosing a backend".

``aim`` is imported lazily on first :attr:`experiment` access, so importing this module
needs no Aim installed. The mapping onto Aim's own model:

===========================  ==================================================
marainer                     Aim
===========================  ==================================================
``experiment_name``          ``Run(experiment=...)``
``run_name``                 ``run.name``
``save_dir``                 the Aim repository root (Aim creates ``.aim`` under it)
``log_hyperparams``          the ``run["hparams"]`` dict, MERGED across calls
``log_metrics``              ``run.track(value, name=..., step=...)``
``set_tags``                 ``run.add_tag("key=value")`` — Aim tags are flat strings
``log_artifact``             ``run.log_artifact`` (needs an artifacts URI, set lazily)
``log_figure``               ``run.track(aim.Image(figure), ...)``
===========================  ==================================================

**Reading a run back programmatically:** Aim's ``Repo.iter_runs()`` is served from an
*index* that a finished run does not populate by itself — right after a run,
``iter_runs()`` can legitimately return nothing while every value is safely on disk
(``Repo.list_all_runs()`` still lists the hash). ``aim up`` indexes when it starts, so the
UI shows the run; a script that wants it immediately has to reindex the repo first. This
is Aim's model, not a write that failed — measured against aim 3.29, and written down
here because the symptom (a run with no hyperparameters) reads exactly like a bug.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from confluid import configurable
from loggair import get_logger

from marainer._imports import require
from marainer.tracking.base import Tracker

logger = get_logger(__name__)

#: Aim's own default system-metric sampling period, in seconds.
AIM_SYSTEM_INTERVAL = 10.0


@configurable
class AimTracker(Tracker):
    """Track an experiment run in a local Aim repository.

    Zero-arg constructible; the Aim ``Run`` is created on first :attr:`experiment`
    access (which imports ``aim`` lazily).

    Args:
        experiment_name: Aim experiment the run belongs to.
        run_name: Human-readable run name; ``None`` draws a random slug.
        save_dir: Directory holding the Aim repository (Aim creates ``.aim`` inside it).
        tags: Tags stamped on the run at creation, flattened to Aim's ``key=value`` strings.
        log_system_metrics: Let Aim sample CPU/GPU/memory itself. ``False`` disables it —
            useful when :mod:`marainer.tracking.system` is already sampling, so the two
            do not both write the same series.
        capture_terminal_logs: Let Aim mirror stdout/stderr into the repository. Off by
            default, unlike Aim's own default: a tracked run normally has loggair writing
            a log file that :func:`~marainer.tracking.provenance.capture_log_file` uploads,
            and capturing the same stream twice is noise, not provenance.
    """

    def __init__(
        self,
        experiment_name: str = "marainer",
        run_name: Optional[str] = None,
        save_dir: str = "./runs",
        tags: Optional[Mapping[str, str]] = None,
        log_system_metrics: bool = True,
        capture_terminal_logs: bool = False,
    ) -> None:
        super().__init__(experiment_name, run_name, save_dir, tags)
        self.log_system_metrics = log_system_metrics
        self.capture_terminal_logs = capture_terminal_logs
        self._run: Optional[Any] = None
        # Aim replaces `run["hparams"]` wholesale, but `log_hyperparams` must MERGE across
        # calls (a caller logs params in several passes), so the accumulated dict is kept
        # here and re-assigned each time.
        self._hparams: Dict[str, Any] = {}
        self._artifacts_uri: Optional[str] = None

    @property
    def experiment(self) -> Any:
        """The ``aim.Run``, created lazily under :attr:`save_dir`."""
        if self._run is None:
            aim = require("aim", "aim", feature="AimTracker")
            repo = Path(self.save_dir)
            repo.mkdir(parents=True, exist_ok=True)
            run = aim.Run(
                repo=str(repo),
                experiment=self.experiment_name,
                system_tracking_interval=AIM_SYSTEM_INTERVAL if self.log_system_metrics else None,
                capture_terminal_logs=self.capture_terminal_logs,
            )
            run.name = self.run_name
            self._run = run
            logger.info(f"Aim run started: {self.experiment_name}/{self.run_name} ({run.hash})")
            if self.tags:
                self._add_tags(self.tags, run)
        return self._run

    @property
    def version(self) -> Optional[str]:
        """The Aim run hash — its identity in the repository — once the run exists."""
        return getattr(self._run, "hash", None) or self.run_name

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        self._hparams.update(dict(params))
        try:
            self.experiment["hparams"] = self._hparams
        except Exception as exc:  # pragma: no cover - aim rejects unserializable values
            logger.debug(f"Aim hparams assignment failed: {exc}")

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        run = self.experiment
        for key, value in metrics.items():
            try:
                run.track(float(value), name=str(key), step=step)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"Aim track failed for {key}: {exc}")

    def set_tags(self, tags: Mapping[str, str]) -> None:
        # Resolve the run BEFORE merging into self.tags. Creating the run applies whatever
        # self.tags already holds, so merging first would make these tags land twice.
        run = self.experiment
        super().set_tags(tags)
        self._add_tags(tags, run)

    def _add_tags(self, tags: Mapping[str, str], run: Any) -> None:
        """Aim tags are flat strings, so a key/value pair becomes ``key=value``."""
        for key, value in tags.items():
            try:
                run.add_tag(f"{key}={value}")
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"Aim add_tag failed for {key}: {exc}")

    def log_artifact(self, path: Union[str, Path], artifact_dir: str = "") -> None:
        src = Path(path)
        if not src.exists():
            logger.debug(f"AimTracker.log_artifact: {src} does not exist; skipping.")
            return
        run = self.experiment
        try:
            self._ensure_artifacts_uri(run)
            # The name is the key Aim files the artifact under, so it must be unique —
            # `config/train.yaml` and `logs/train.yaml` would otherwise collide.
            name = f"{artifact_dir}/{src.name}" if artifact_dir else src.name
            run.log_artifact(str(src), name=name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"Aim log_artifact failed for {src}: {exc}")

    def _ensure_artifacts_uri(self, run: Any) -> None:
        """Aim raises unless an artifacts URI is set first; point it beside the repository."""
        if self._artifacts_uri is not None:
            return
        target = Path(self.save_dir).resolve() / "artifacts" / self.experiment_name / str(self.run_name)
        target.mkdir(parents=True, exist_ok=True)
        self._artifacts_uri = target.as_uri()
        run.set_artifacts_uri(self._artifacts_uri)

    def log_figure(self, name: str, figure: Any, step: Optional[int] = None) -> None:
        try:
            aim = require("aim", "aim", feature="AimTracker")
            self.experiment.track(aim.Image(figure), name=name, step=step)
        except Exception as exc:  # pragma: no cover - matplotlib/aim conversion
            logger.debug(f"Aim log_figure failed for {name}: {exc}")

    def finalize(self, status: str = "success") -> None:
        if self._run is None:
            return
        try:
            # Aim has no run status of its own; record it where a query can reach it.
            self._run["status"] = status
            self._run.close()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"Aim run close failed: {exc}")


__all__ = ["AIM_SYSTEM_INTERVAL", "AimTracker"]

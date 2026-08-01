"""``ClearMLTracker`` — the ClearML experiment-tracking backend (``pip install marainer[clearml]``).

ClearML is Apache-2.0 with a self-hostable server (docker-compose), which is why it is an
in-tree backend — see the README's "Choosing a backend". ``clearml`` is imported lazily on
first :attr:`experiment` access, so importing this module needs no ClearML installed.

The mapping onto ClearML's own model:

===========================  ==================================================
marainer                     ClearML
===========================  ==================================================
``experiment_name``          ``Task.init(project_name=...)``
``run_name``                 ``Task.init(task_name=...)``
``log_hyperparams``          ``task.connect(dict, name="hparams")``
``log_metrics``              ``logger.report_scalar(title, series, value, iteration)``
``set_tags``                 ``task.add_tags(["key=value"])`` — ClearML tags are strings
``log_artifact``             ``task.upload_artifact(name, path)``
``log_figure``               ``logger.report_matplotlib_figure(...)``
``finalize``                 ``mark_completed`` / ``mark_failed``, then ``close``
===========================  ==================================================

Two constructor choices differ from ClearML's defaults, both deliberately:

* **``auto_connect_frameworks=False``** — ClearML otherwise monkey-patches torch, matplotlib,
  argparse and more to capture them implicitly. A tracking library must not silently change
  how the process it is measuring behaves; marainer logs what it is asked to log.
* **``offline`` is a first-class option** — ``Task.set_offline(True)`` makes ClearML record a
  session to local disk with no server and no credentials, which is what lets a laptop (and
  this project's test suite) use the backend before anyone stands up a server.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Union

from confluid import configurable
from loggair import get_logger

from marainer._imports import require
from marainer.tracking.base import Tracker

if TYPE_CHECKING:  # type-only: `manifest` reaches `base`, so a runtime import would cycle.
    from marainer.tracking.manifest import RunManifest

logger = get_logger(__name__)

#: The ClearML "section" the hyperparameters are connected under.
HPARAMS_SECTION = "hparams"


@configurable
class ClearMLTracker(Tracker):
    """Track an experiment run as a ClearML task.

    Zero-arg constructible; the ``Task`` is created on first :attr:`experiment` access
    (which imports ``clearml`` lazily).

    Args:
        experiment_name: ClearML project the task belongs to.
        run_name: ClearML task name; ``None`` draws a random slug.
        save_dir: Unused for a server-backed backend; kept for interface parity.
        tags: Tags stamped on the task at creation, flattened to ``key=value`` strings.
        offline: Record to local disk with no server and no credentials
            (``Task.set_offline``). Useful on a laptop, and the mode this project's own
            tests run in. ``False`` (default) leaves ClearML's mode untouched and uses the
            ambient configuration — note this flag is **process-wide** and one-way in
            practice: ``Task.set_offline(False)`` validates the ClearML config and raises
            on an unconfigured machine, so a process that went offline stays offline.
        log_system_metrics: Let ClearML sample CPU/GPU/memory itself. ``False`` disables it,
            so it does not duplicate :mod:`marainer.tracking.system` when that is running.
    """

    def __init__(
        self,
        experiment_name: str = "marainer",
        run_name: Optional[str] = None,
        save_dir: str = "./runs",
        tags: Optional[Mapping[str, str]] = None,
        offline: bool = False,
        log_system_metrics: bool = True,
    ) -> None:
        super().__init__(experiment_name, run_name, save_dir, tags)
        self.offline = offline
        self.log_system_metrics = log_system_metrics
        self._task: Optional[Any] = None
        # ClearML replaces a connected dict wholesale, but `log_hyperparams` must MERGE
        # across calls, so the accumulated dict is kept here and re-connected each time.
        self._hparams: Dict[str, Any] = {}

    @property
    def experiment(self) -> Any:
        """The ``clearml.Task``, created lazily."""
        if self._task is None:
            clearml = require("clearml", "clearml", feature="ClearMLTracker")
            if self.offline:
                clearml.Task.set_offline(True)
            self._task = clearml.Task.init(
                project_name=self.experiment_name,
                task_name=self.run_name,
                # See the module docstring: capturing frameworks implicitly would change how
                # the measured process behaves, which a tracking library must not do.
                auto_connect_frameworks=False,
                auto_resource_monitoring=self.log_system_metrics,
                tags=[f"{key}={value}" for key, value in self.tags.items()] or None,
            )
            logger.info(f"ClearML task started: {self.experiment_name}/{self.run_name} ({self._task.id})")
        return self._task

    @property
    def version(self) -> Optional[str]:
        """The ClearML task id — its identity on the server — once the task exists."""
        return getattr(self._task, "id", None) or self.run_name

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        self._hparams.update(dict(params))
        try:
            self.experiment.connect(self._hparams, name=HPARAMS_SECTION)
        except Exception as exc:  # pragma: no cover - clearml rejects unserializable values
            logger.debug(f"ClearML connect(hparams) failed: {exc}")

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        reporter = self.experiment.get_logger()
        for key, value in metrics.items():
            # ClearML plots are (title, series): "train/loss" becomes the `train` graph with a
            # `loss` line, which is how its UI groups related curves.
            title, _, series = str(key).rpartition("/")
            try:
                reporter.report_scalar(
                    title=title or series,
                    series=series,
                    value=float(value),
                    iteration=int(step or 0),
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"ClearML report_scalar failed for {key}: {exc}")

    def set_tags(self, tags: Mapping[str, str]) -> None:
        # Resolve the task BEFORE merging into self.tags: creating it applies whatever
        # self.tags already holds, so merging first would make these tags land twice.
        task = self.experiment
        super().set_tags(tags)
        try:
            task.add_tags([f"{key}={value}" for key, value in tags.items()])
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"ClearML add_tags failed: {exc}")

    def log_artifact(self, path: Union[str, Path], artifact_dir: str = "") -> None:
        src = Path(path)
        if not src.exists():
            logger.debug(f"ClearMLTracker.log_artifact: {src} does not exist; skipping.")
            return
        # Namespaced by artifact_dir: ClearML files artifacts by NAME, so `config/train.yaml`
        # and `logs/train.yaml` would otherwise overwrite each other.
        name = f"{artifact_dir}/{src.name}" if artifact_dir else src.name
        try:
            self.experiment.upload_artifact(name=name, artifact_object=src)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"ClearML upload_artifact failed for {src}: {exc}")

    def log_figure(self, name: str, figure: Any, step: Optional[int] = None) -> None:
        try:
            self.experiment.get_logger().report_matplotlib_figure(
                title=name, series=name, figure=figure, iteration=int(step or 0)
            )
        except Exception as exc:  # pragma: no cover - matplotlib/clearml conversion
            logger.debug(f"ClearML report_matplotlib_figure failed for {name}: {exc}")

    def log_manifest(self, manifest: "RunManifest") -> None:
        """Artifact + hyperparameters (the default), PLUS ClearML's own native slots.

        ClearML has two homes the generic surface cannot reach. ``connect_configuration``
        puts the manifest in the task's **Configuration** tab, where it is readable and
        diffable between tasks rather than being a file to download. ``connect`` under a
        named section puts the datasets and models in the **Hyperparameters** tab, where
        ClearML can compare and filter across runs.

        Both are ADDITIONAL — the manifest artifact still carries everything, so a run
        tracked here stays comparable with the same run tracked to a local directory.
        """
        super().log_manifest(manifest)
        task = self.experiment
        try:
            task.connect_configuration(manifest.path, name="manifest")
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"ClearML connect_configuration failed: {exc}")
        for section in ("datasets", "models"):
            entries = manifest.section(section)
            if not entries:
                continue
            try:
                # Flattened to `0.name` / `1.name`: ClearML sections are flat key/value maps.
                flat = {f"{i}.{k}": v for i, entry in enumerate(entries) for k, v in entry.items()}
                task.connect(flat, name=section)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"ClearML connect({section}) failed: {exc}")

    def finalize(self, status: str = "success") -> None:
        if self._task is None:
            return
        try:
            if status == "success":
                self._task.mark_completed()
            else:
                self._task.mark_failed(status_reason=status)
            self._task.close()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"ClearML task close failed: {exc}")


__all__ = ["HPARAMS_SECTION", "ClearMLTracker"]

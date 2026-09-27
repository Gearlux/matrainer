"""``ClearMLTracker`` — the ClearML experiment-tracking backend (``pip install matrainer[clearml]``).

ClearML is Apache-2.0 with a self-hostable server (docker-compose), which is why it is an
in-tree backend — see the README's "Choosing a backend". ``clearml`` is imported lazily on
first :attr:`experiment` access, so importing this module needs no ClearML installed.

The mapping onto ClearML's own model:

===========================  ==================================================
matrainer                     ClearML
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
  how the process it is measuring behaves; matrainer logs what it is asked to log.
* **``offline`` is a first-class option** — ``Task.set_offline(True)`` makes ClearML record a
  session to local disk with no server and no credentials, which is what lets a laptop (and
  this project's test suite) use the backend before anyone stands up a server.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple, Union

from confluid import configurable
from loggair import get_logger

from matrainer._imports import require
from matrainer.tracking.base import Tracker, text_file_stem

if TYPE_CHECKING:  # type-only: `manifest` reaches `base`, so a runtime import would cycle.
    from matrainer.tracking.manifest import RunManifest

logger = get_logger(__name__)

#: The ClearML "section" the hyperparameters are connected under.
HPARAMS_SECTION = "hparams"

#: Schemes ClearML's storage layer can actually address. Anything else it REWRITES rather than
#: rejecting: ``StorageHelper.conform_url`` treats an unknown scheme as a relative path and
#: resolves it against the CWD, so ``hf://models/timm/x`` is registered as
#: ``file://<cwd>/hf:/models/timm/x`` — a local file that does not exist, pointing wherever the
#: run happened to start. Every zoo scheme this workspace mints (``hf``/``smp``/``keras``/
#: ``ultralytics``/``rfdetr``) hits that path, so they are kept OUT of ``import_model`` and stay
#: in the comparable hyperparameter section instead. Confirmed against a LIVE server, not only
#: read from the source: ``import_model(weights_url="hf://models/timm/resnet50.a1_in1k")`` stored
#: ``file:///<cwd>/hf:/models/timm/resnet50.a1_in1k`` — so the same config records a DIFFERENT
#: wrong path depending on the directory the run was launched from. clearml 2.1; ``azure://``
#: is addressable in principle and raises a configuration error without credentials, which the
#: caller's ``except`` already reports — a loud miss, unlike the silent rewrite above.
ADDRESSABLE_SCHEMES: Tuple[str, ...] = ("file://", "http://", "https://", "s3://", "gs://", "azure://")


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
            so it does not duplicate :mod:`matrainer.tracking.system` when that is running.
    """

    def __init__(
        self,
        experiment_name: str = "matrainer",
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
        # URI -> InputModel id, for models already registered on THIS task. `log_manifest` runs at
        # least twice per run, and a second `import_model` of the same weights is a second entry.
        self._input_models: Dict[str, str] = {}
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

    def log_text(self, name: str, text: str, step: Optional[int] = None) -> None:
        """``upload_artifact``, NOT ``report_text``.

        ClearML's text primitive writes to the task's console log, which is a stream of
        run output — a per-epoch table would be interleaved with everything else the
        process printed and impossible to compare between tasks. An artifact is addressable
        by name, which is what a value logged once an epoch needs; the name carries the
        step for the same reason.
        """
        try:
            self.experiment.upload_artifact(name=f"text/{text_file_stem(name, step)}.md", artifact_object=text)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"ClearML log_text failed for {name}: {exc}")

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
        self._project_input_models(manifest)

    def _project_input_models(self, manifest: "RunManifest") -> None:
        """Register the manifest's models as ClearML ``InputModel``s — a SERVER-ONLY enrichment.

        The hyperparameter section above is comparable across runs but is still just key/value
        text. ``InputModel.import_model`` registers each artifact as a first-class model the task
        links to, which is what puts it in the task's **Models** tab and on ClearML's lineage
        graph. It imports by URL — no weights are uploaded, matching what the manifest holds.

        **It needs a server, and that is why the flat section stays rather than being replaced.**
        Measured against clearml 2.1 in offline mode: ``import_model`` dies with
        ``AttributeError: 'NoneType' object has no attribute 'response'``, because the backend
        call it makes has no backend to answer it. So an offline run — the mode that needs no
        server and no credentials at all — would otherwise log a failure per model per
        :meth:`log_manifest`. Detecting it once up front is cheaper and quieter, and leaves the
        offline run with exactly the projection it had before.

        Models are remembered per URI for the same reason MLflow's are: :meth:`log_manifest` runs
        at least twice per run, and a second registration of the same weights would be a second
        entry rather than the same one.
        """
        entries = manifest.section("models")
        if not entries:
            return
        clearml = require("clearml", "clearml", feature="ClearMLTracker")
        if self.offline or clearml.Task.is_offline():
            logger.debug(
                "ClearML is offline: skipping InputModel registration, which needs a server. "
                "The models are still connected as a comparable hyperparameter section."
            )
            return
        task = self.experiment
        for entry in entries:
            uri = str(entry.get("uri") or "")
            if not uri or uri in self._input_models:
                continue
            weights_url = os.path.expandvars(uri)
            if not weights_url.startswith(ADDRESSABLE_SCHEMES):
                # A zoo scheme — registering it would store a CWD-relative path that resolves
                # nowhere (see ADDRESSABLE_SCHEMES). The model is already in the `models`
                # section, which is the whole reason that section is kept beside this.
                logger.debug(f"ClearML cannot address {uri}; left in the models section only.")
                continue
            try:
                model = clearml.InputModel.import_model(
                    # EXPANDED, unlike the tags beside it: `manifest.section` returns the portable
                    # spelling, so a checkpoint's URI reads `file://$MODEL_ROOT/last.ckpt`, which
                    # is deliberately not a path on any machine. A tag may stay portable — it is
                    # read by a person — but `weights_url` is meant to RESOLVE, so it needs the
                    # bytes on this disk. `os.path.expandvars` is the documented exact inverse.
                    weights_url=weights_url,
                    name=str(entry.get("name") or uri),
                    # `kind` is what `capture_models` writes; `source` is what the manual
                    # `record_model` takes — a section can hold either.
                    framework=str(entry.get("kind") or entry.get("source") or "") or None,
                    comment=str(entry.get("role") or ""),
                    tags=[str(entry["role"])] if entry.get("role") else None,
                )
            except Exception as exc:  # pragma: no cover - server-specific
                logger.debug(f"ClearML InputModel registration failed for {uri}: {exc}")
                continue
            # Memoised the moment the REGISTRATION succeeds, before the link is attempted: it is
            # the registration that would duplicate on a retry, so a failure to link must not send
            # the next `log_manifest` back to `import_model` for weights the server already has.
            self._input_models[uri] = getattr(model, "id", uri)
            try:
                task.connect(model, name=f"model:{entry.get('role') or 'input'}")
            except Exception as exc:  # pragma: no cover - server-specific
                logger.debug(f"ClearML could not link the input model for {uri}: {exc}")

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

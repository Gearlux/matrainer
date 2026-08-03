"""``MlflowTracker`` — MLflow experiment-tracking backend (``pip install matrainer[mlflow]``).

Drives MLflow through :class:`mlflow.tracking.MlflowClient` so the run lifecycle
is explicit and independent of the fluent ``mlflow.start_run`` global state (the
same reason ``pytorch_lightning.loggers.MLFlowLogger`` uses the client directly).
``mlflow`` is imported lazily on first :attr:`experiment` access — importing this
module needs no MLflow installed.

When ``log_system_metrics`` is set, the run also starts MLflow's own
``SystemMetricsMonitor`` (CPU / RAM / disk) plus, on Apple Silicon, the
framework-free :class:`matrainer.tracking.system.MpsMetricsSampler` — so a
non-Lightning training platform gets system metrics for free by using this
tracker.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Union

from confluid import configurable
from loggair import get_logger

from matrainer._imports import require
from matrainer.tracking.base import Tracker, text_file_stem

if TYPE_CHECKING:  # type-only: `manifest` reaches `base`, so a runtime import would cycle.
    from matrainer.tracking.manifest import RunManifest

logger = get_logger(__name__)


#: Characters MLflow rejects in a logged-model NAME (measured: `Invalid model name`). That is
#: every separator a real model identifier uses, so the name is a sanitised LABEL and the exact
#: identifier lives in the entity's `uri` tag.
_UNSAFE_MODEL_NAME = re.compile(r"""[/:.%"']+""")


def _logged_model_name(name: Any) -> str:
    """A display name MLflow will accept for ``name``, never empty."""
    return _UNSAFE_MODEL_NAME.sub("_", str(name or "")).strip("_") or "model"


def _reference_digest(fields: Mapping[str, Any]) -> str:
    """A short digest of a dataset REFERENCE, for when the caller recorded no data digest.

    MLflow requires every dataset input to carry a digest. When the manifest has none,
    hashing the identity (kind / name / revision / split) still gives the run-comparison
    property that matters — two runs on the same reference collide, two runs on different
    references do not. It deliberately fingerprints the reference, NOT the bytes: computing
    a real data digest would mean re-reading a dataset the run has already consumed.
    """
    material = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(material.encode()).hexdigest()[:16]


@configurable
class MlflowTracker(Tracker):
    """Track an experiment run in MLflow.

    Zero-arg constructible; the MLflow experiment + run are created on first
    :attr:`experiment` access (which imports ``mlflow`` lazily).

    Args:
        experiment_name: MLflow experiment name (created if absent).
        run_name: MLflow run name; ``None`` draws a random slug.
        save_dir: Unused for a remote tracking server; kept for interface parity.
        tags: Tags stamped on the run at creation.
        tracking_uri: MLflow tracking URI; ``None`` uses the ambient
            ``MLFLOW_TRACKING_URI`` / local ``mlruns``.
        log_system_metrics: Start CPU/RAM/disk (+ Apple MPS) sampling for the run.
    """

    def __init__(
        self,
        experiment_name: str = "matrainer",
        run_name: Optional[str] = None,
        save_dir: str = "./runs",
        tags: Optional[Mapping[str, str]] = None,
        tracking_uri: Optional[str] = None,
        log_system_metrics: bool = True,
    ) -> None:
        super().__init__(experiment_name, run_name, save_dir, tags)
        self.tracking_uri = tracking_uri
        self.log_system_metrics = log_system_metrics
        self._client: Optional[Any] = None
        self._experiment_id: Optional[str] = None
        self._run_id: Optional[str] = None
        # URI -> logged-model id, for models already projected on THIS run. MLflow dedups a
        # dataset input by digest but does NOT dedup a logged model, and `log_manifest` runs at
        # least twice per run (start and end) — without this every model appears twice (measured).
        self._logged_models: Dict[str, str] = {}
        self._system_monitor: Optional[Any] = None
        self._mps_sampler: Optional[Any] = None

    @property
    def run_id(self) -> Optional[str]:
        """The MLflow run id (available after first :attr:`experiment` access)."""
        return self._run_id

    @property
    def experiment(self) -> Any:
        """The :class:`mlflow.tracking.MlflowClient`, creating the experiment + run lazily."""
        if self._client is None:
            mlflow = require("mlflow", "mlflow", feature="MlflowTracker")
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=self.tracking_uri)
            exp = client.get_experiment_by_name(self.experiment_name)
            exp_id = exp.experiment_id if exp is not None else client.create_experiment(self.experiment_name)
            run = client.create_run(experiment_id=exp_id, run_name=self.run_name, tags=self.tags or None)
            self._client = client
            self._experiment_id = exp_id
            self._run_id = run.info.run_id
            logger.info(f"MLflow run started: {self.experiment_name}/{self.run_name} ({self._run_id})")
            if self.log_system_metrics:
                self._start_system_metrics(mlflow)
        return self._client

    def _start_system_metrics(self, mlflow: Any) -> None:
        from matrainer.tracking.system import MpsMetricsSampler, start_system_metrics_monitor

        self._system_monitor = start_system_metrics_monitor(self._run_id, self.tracking_uri)
        self._mps_sampler = MpsMetricsSampler(self._client, self._run_id)
        self._mps_sampler.start()

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        client = self.experiment
        for key, value in params.items():
            try:
                client.log_param(self._run_id, str(key), value)
            except Exception as exc:  # pragma: no cover - mlflow rejects re-logged params
                logger.debug(f"MLflow log_param failed for {key}: {exc}")

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        client = self.experiment
        for key, value in metrics.items():
            try:
                client.log_metric(self._run_id, str(key), float(value), step=step or 0)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"MLflow log_metric failed for {key}: {exc}")

    def set_tags(self, tags: Mapping[str, str]) -> None:
        super().set_tags(tags)
        client = self.experiment
        for key, value in tags.items():
            try:
                client.set_tag(self._run_id, str(key), str(value))
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"MLflow set_tag failed for {key}: {exc}")

    def log_artifact(self, path: Union[str, Path], artifact_dir: str = "") -> None:
        src = Path(path)
        if not src.exists():
            logger.debug(f"MlflowTracker.log_artifact: {src} does not exist; skipping.")
            return
        client = self.experiment
        try:
            client.log_artifact(self._run_id, str(src), artifact_dir or None)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"MLflow log_artifact failed for {src}: {exc}")

    def log_figure(self, name: str, figure: Any, step: Optional[int] = None) -> None:
        client = self.experiment
        safe = name.replace("/", "_")
        try:
            client.log_figure(self._run_id, figure, f"{safe}.png")
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"MLflow log_figure failed for {name}: {exc}")

    def log_text(self, name: str, text: str, step: Optional[int] = None) -> None:
        """MLflow's own text artifact — the UI previews a ``.md`` file inline, no download."""
        try:
            self.experiment.log_text(self._run_id, text, f"text/{text_file_stem(name, step)}.md")
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"MLflow log_text failed for {name}: {exc}")

    def log_manifest(self, manifest: "RunManifest") -> None:
        """Artifact + hyperparameters (the default), PLUS MLflow's own native slots.

        MLflow has two homes the generic surface cannot reach: the ``mlflow.source.*``
        standard tags, which its UI renders as the run's provenance panel, and
        ``log_inputs``, which makes datasets first-class and links runs that share one.
        Both are ADDITIONAL — the manifest artifact still carries everything, so a run
        tracked here stays comparable with the same run tracked to a local directory.
        """
        super().log_manifest(manifest)
        self._project_source_tags(manifest)
        self._project_datasets(manifest)
        self._project_models(manifest)

    def _project_source_tags(self, manifest: "RunManifest") -> None:
        """Fill MLflow's standard source/git tags from the manifest's command + code sections."""
        try:
            from mlflow.utils.mlflow_tags import (
                MLFLOW_GIT_BRANCH,
                MLFLOW_GIT_COMMIT,
                MLFLOW_GIT_REPO_URL,
                MLFLOW_SOURCE_NAME,
                MLFLOW_SOURCE_TYPE,
            )
        except ImportError as exc:  # pragma: no cover - the constant module has existed since 1.x
            logger.debug(f"MLflow standard tag names unavailable; skipping source tags: {exc}")
            return

        tags: dict = {}
        argv = manifest.section("command").get("argv") or []
        # The config script names the run better than `python` does; fall back to argv[0].
        source = manifest.section("config").get("script") or (argv[0] if argv else None)
        if source:
            tags[MLFLOW_SOURCE_NAME] = str(source)
            tags[MLFLOW_SOURCE_TYPE] = "LOCAL"

        primary = manifest.primary_code()
        if primary:
            tags[MLFLOW_GIT_COMMIT] = str(primary.get("commit"))
            if primary.get("branch"):
                tags[MLFLOW_GIT_BRANCH] = str(primary["branch"])
            if primary.get("remote"):
                tags[MLFLOW_GIT_REPO_URL] = str(primary["remote"])
            if primary.get("dirty") is not None:
                # MLflow's own dirty tag is a lowercase string, matching how it writes it.
                tags["mlflow.source.git.dirty"] = str(bool(primary["dirty"])).lower()
        if tags:
            self.set_tags(tags)

    def _project_datasets(self, manifest: "RunManifest") -> None:
        """Log the manifest's datasets as MLflow dataset INPUTS, not merely as text.

        Uses the entity-level ``Dataset`` rather than ``mlflow.data.from_*``: those load
        the data to fingerprint it, and a manifest records a dataset by REFERENCE — the
        run has already read it, and re-materialising a 12k-image set to log a row would
        be absurd. Re-logging the same dataset is idempotent (MLflow dedups on digest),
        so calling this twice does not duplicate the input.
        """
        datasets = manifest.section("datasets")
        if not datasets:
            return
        try:
            from mlflow.entities import Dataset, DatasetInput, InputTag
        except ImportError as exc:  # pragma: no cover - defensive
            logger.debug(f"MLflow dataset entities unavailable: {exc}")
            return

        inputs = []
        for entry in datasets:
            name = str(entry.get("name") or "dataset")
            fields = {k: v for k, v in entry.items() if v is not None}
            tags = [InputTag(key=k, value=str(entry[k])) for k in ("split", "revision", "count") if entry.get(k)]
            inputs.append(
                DatasetInput(
                    dataset=Dataset(
                        name=name,
                        digest=str(entry.get("digest") or _reference_digest(fields)),
                        source_type=str(entry.get("kind") or "unknown"),
                        source=json.dumps(fields, sort_keys=True, default=str),
                    ),
                    tags=tags or None,
                )
            )
        client = self.experiment
        try:
            client.log_inputs(self._run_id, inputs)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"MLflow log_inputs failed: {exc}")

    def _project_models(self, manifest: "RunManifest") -> None:
        """Log the manifest's model artifacts as MLflow model INPUTS.

        The counterpart of :meth:`_project_datasets`, and it needs one step more: a
        ``LoggedModelInput`` takes a *logged-model id*, which only exists once an entity has been
        created. ``create_logged_model`` makes that entity from metadata alone — no weights are
        uploaded and the Model Registry is not touched — so what lands is a reference, exactly as
        the manifest holds one. Every field rides along as a tag, so the run's UI shows which
        weights it loaded and in which ``role``.

        Two behaviours measured against mlflow 3.14, both of which read as bugs otherwise:

        * **A logged model is NOT deduplicated.** MLflow dedups a dataset input by digest, but
          creating the same logged model twice yields two entities and two inputs — and
          :meth:`log_manifest` runs at least twice per run. Hence :attr:`_logged_models`.
        * **The NAME is restricted** — ``/``, ``:``, ``.``, ``%`` and quotes are rejected outright
          (``Invalid model name``), which is every separator a real model id uses. The name is
          therefore sanitised for display and the exact identifier kept in the ``uri`` tag.

        Reading it back, ``client.get_run(run_id).inputs.model_inputs`` is the query that works.
        ``search_runs`` returns runs whose ``model_inputs`` is EMPTY while their ``dataset_inputs``
        is populated — so verifying lineage that way shows datasets landing and models silently
        missing, which is a convincing-looking bug in code that is doing the right thing.
        """
        entries = manifest.section("models")
        if not entries:
            return
        try:
            from mlflow.entities import LoggedModelInput
        except ImportError as exc:  # pragma: no cover - defensive
            logger.debug(f"MLflow logged-model entities unavailable: {exc}")
            return

        client = self.experiment
        fresh = []
        for entry in entries:
            key = str(entry.get("uri") or json.dumps(entry, sort_keys=True, default=str))
            if key in self._logged_models:
                continue
            try:
                model = client.create_logged_model(
                    experiment_id=self._experiment_id,
                    name=_logged_model_name(entry.get("name")),
                    source_run_id=self._run_id,
                    tags={k: str(v) for k, v in entry.items() if v is not None},
                    # `kind` is what `capture_models` writes and `source` is what the manual
                    # `record_model` takes, so a section can hold either; read both rather than
                    # silently typing a hand-recorded model as unknown.
                    model_type=str(entry.get("kind") or entry.get("source") or "") or None,
                )
            except Exception as exc:  # pragma: no cover - backend-specific
                logger.debug(f"MLflow create_logged_model failed for {key}: {exc}")
                continue
            self._logged_models[key] = model.model_id
            fresh.append(LoggedModelInput(model.model_id))
        if not fresh:
            return
        try:
            client.log_inputs(self._run_id, models=fresh)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"MLflow log_inputs(models=...) failed: {exc}")

    def finalize(self, status: str = "success") -> None:
        if self._mps_sampler is not None:
            self._mps_sampler.stop()
            self._mps_sampler = None
        if self._system_monitor is not None:
            try:
                self._system_monitor.finish()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"system-metrics monitor finish failed: {exc}")
            self._system_monitor = None
        if self._client is not None and self._run_id is not None:
            mlflow_status = "FINISHED" if status == "success" else "FAILED"
            try:
                self._client.set_terminated(self._run_id, status=mlflow_status)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"MLflow set_terminated failed: {exc}")


__all__ = ["MlflowTracker"]

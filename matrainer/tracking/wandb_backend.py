"""``WandbTracker`` — Weights & Biases backend (``pip install matrainer[wandb]``).

Wraps a ``wandb`` run. ``wandb`` is imported lazily on first :attr:`experiment`
access; importing this module needs no wandb installed.

``log_text`` is deliberately NOT overridden: W&B has no text primitive
(``wandb.Html`` would mean escaping the caller's text into markup, and a
``wandb.Table`` is a dataframe wearing a different name), while its run directory
IS its upload channel — so the base class's ``.md`` file, written under
:attr:`log_dir`, is synced with the run for free and stays readable as itself.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Tuple, Union

from confluid import configurable
from loggair import get_logger

from matrainer._imports import require
from matrainer.tracking.base import Tracker

if TYPE_CHECKING:  # type-only: `manifest` reaches `base`, so a runtime import would cycle.
    from matrainer.tracking.manifest import ManifestSection, RunManifest

logger = get_logger(__name__)

#: URI schemes ``wandb.Artifact.add_reference`` can address. A reference records WHERE the bytes
#: live without uploading them, which is what a manifest holds — but only for a location W&B knows
#: how to reach. A zoo's own scheme (``hf://``, ``smp://``, ``ultralytics://``) is not one, and is
#: kept as metadata rather than forced into a reference the library would reject.
REFERENCEABLE_SCHEMES: Tuple[str, ...] = ("file://", "http://", "https://", "s3://", "gs://", "az://")

#: Characters W&B rejects in an artifact NAME — it allows alphanumerics, dashes, underscores and
#: dots only, which excludes the ``/`` in every real dataset or model identifier. The name is
#: therefore a sanitised LABEL, with the exact identifier kept in the artifact's metadata.
_UNSAFE_ARTIFACT_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _artifact_name(name: Any) -> str:
    """A display name W&B will accept for ``name``, never empty."""
    return _UNSAFE_ARTIFACT_NAME.sub("_", str(name or "")).strip("_") or "artifact"


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
        experiment_name: str = "matrainer",
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

    def log_manifest(self, manifest: "RunManifest") -> None:
        """Artifact + hyperparameters (the default), PLUS W&B's artifact-lineage graph.

        W&B's native home for "what went into this run" is an ``Artifact`` marked used, which is
        what puts a run on the lineage graph and links every run that consumed the same data or
        the same weights. Both are ADDITIONAL — the manifest artifact still carries everything,
        per the union contract, so a run tracked here stays comparable with one tracked to a
        local directory.
        """
        super().log_manifest(manifest)
        self._project_inputs(manifest, "datasets", "dataset")
        self._project_inputs(manifest, "models", "model")

    def _project_inputs(self, manifest: "RunManifest", section: "ManifestSection", kind: str) -> None:
        """Mark one manifest section's entries as used W&B artifacts of type ``kind``.

        The artifact carries the entry's fields as metadata and, where the URI names a location
        W&B can address, a REFERENCE to it — ``add_reference`` uploads nothing, it records where
        the bytes live, which is exactly what a manifest holds. A scheme W&B does not know
        (``hf://``, ``smp://``, a zoo's own) is left as metadata alone rather than being forced
        into a reference the library would reject.

        Names are sanitised because W&B's are restricted to alphanumerics, dashes, underscores and
        dots, which excludes the ``/`` in every real dataset or model identifier; the exact
        identifier stays in the metadata. Re-marking the same artifact is what makes a second
        :meth:`log_manifest` harmless — W&B resolves an existing artifact by name rather than
        creating a duplicate.
        """
        entries = manifest.section(section)
        if not entries:
            return
        wandb = require("wandb", "wandb", feature="WandbTracker")
        run = self.experiment
        for entry in entries:
            metadata = {k: v for k, v in entry.items() if v is not None}
            try:
                artifact = wandb.Artifact(_artifact_name(entry.get("name")), type=kind, metadata=metadata)
                self._add_reference(artifact, str(entry.get("uri") or ""))
                run.use_artifact(artifact)
            except Exception as exc:  # pragma: no cover - backend-specific
                logger.debug(f"wandb use_artifact failed for {entry.get('uri') or entry.get('name')}: {exc}")

    @staticmethod
    def _add_reference(artifact: Any, uri: str) -> None:
        """Attach ``uri`` as a reference when W&B can address it — never fatal.

        Two reasons this cannot share the caller's ``try``, both MEASURED against wandb 0.28:
        ``add_reference`` VALIDATES eagerly, raising ``ValueError`` for a ``file://`` path that
        is not on this disk and issuing a live request for ``http(s)://`` (a 404 raises
        ``HTTPError``). Sharing the block meant a reference failure skipped ``use_artifact``
        too, so the entry vanished from the lineage graph entirely rather than degrading to the
        metadata-only form a zoo URI already gets — the artifact is still the record, the
        reference is only an optimisation on top of it.

        The path is EXPANDED first because the manifest is PORTABLE: with ``$MODEL_ROOT`` set,
        a checkpoint's URI is stored as ``file://$MODEL_ROOT/last.ckpt``, which is deliberately
        not a path on any machine. ``os.path.expandvars`` is the documented exact inverse of
        that contraction, so the reference names the bytes on THIS disk while the metadata keeps
        the portable spelling. Without it a checkpoint reference could never resolve on a
        machine that has the roots configured, which is every machine that sets them.
        """
        if not uri.startswith(REFERENCEABLE_SCHEMES):
            return  # a zoo scheme W&B does not know — metadata alone, deliberately
        try:
            artifact.add_reference(os.path.expandvars(uri))
        except Exception as exc:
            logger.debug(f"wandb add_reference skipped for {uri}: {exc}")

    def finalize(self, status: str = "success") -> None:
        if self._run is not None:
            exit_code = 0 if status == "success" else 1
            try:
                self._run.finish(exit_code=exit_code)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(f"wandb finish failed: {exc}")
            self._run = None


__all__ = ["WandbTracker"]

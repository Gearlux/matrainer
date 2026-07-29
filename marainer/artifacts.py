"""Run artifacts written beside a checkpoint — where they live, and that they get logged.

Marainer tracks; it does not train. These three helpers are the tracking side of two things
a consumer produces at the END of a run, and they exist here because each is half a
convention and half a file operation, neither of which belongs in a task package:

* :func:`save_class_names` / :func:`load_class_names` — the label vocabulary, persisted
  beside the checkpoint so an evaluate/predict run picks up the SAME name-to-id ordering the
  training run fitted. The FORMAT is not marainer's — it is
  :meth:`recordstream.LabelMap.save` / :meth:`~recordstream.LabelMap.load`, and these
  delegate to it rather than hand-rolling the JSON (three consumers had written that
  serializer out by hand). What marainer contributes is the checkpoint-adjacency convention
  and the artifact logging.
* :func:`write_metrics_sidecar` — a run's final scalar metrics as a JSON file stamped with
  the run identity, for a discovery service or a human to read without opening a tracking UI.

Framework-neutral by construction: each takes a plain directory or path. Resolving that
directory from a framework's trainer object is the framework subpackage's job
(:func:`marainer.lightning.checkpoint_dir`).
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

from loggair import get_logger
from recordstream import LabelMap

from marainer.tracking.base import Tracker

logger = get_logger(__name__)

#: The filename a run's label vocabulary is persisted under, beside its checkpoints. Fixed
#: rather than configurable so a later evaluate/predict run can FIND it knowing only the
#: checkpoint path — a knob here would have to be repeated identically in both configs.
CLASS_NAMES_FILE = "class_names.json"

__all__ = [
    "CLASS_NAMES_FILE",
    "load_class_names",
    "save_class_names",
    "write_metrics_sidecar",
]


def save_class_names(
    names: Union[LabelMap, Sequence[str], None],
    directory: Union[str, Path, None],
    trackers: Iterable[Tracker] = (),
) -> Optional[Path]:
    """Persist a run's label vocabulary beside its checkpoints, and log it as an artifact.

    Args:
        names: The vocabulary — a :class:`~recordstream.LabelMap`, or the ordered names.
            ``None`` (or empty) writes nothing: a run on integer labels has no vocabulary,
            which is not an error.
        directory: Where the checkpoints live. ``None`` writes nothing — a framework that
            could not tell us where it checkpoints should not make the caller guard.
        trackers: Trackers the written file is logged to as an artifact.

    Returns:
        The path written, or ``None`` when there was nothing to write.

    Example::

        save_class_names(self.label_map, checkpoint_dir(trainer), self._trackers)
    """
    if not names or directory is None:
        return None
    label_map = names if isinstance(names, LabelMap) else LabelMap.from_class_names(list(names))

    path = Path(directory) / CLASS_NAMES_FILE
    label_map.save(path)  # the FORMAT is recordstream's — never re-hand-roll this JSON
    logger.info(f"Wrote {CLASS_NAMES_FILE} to {path}")
    for tracker in trackers:
        tracker.log_artifact(path)
    return path


def load_class_names(checkpoint_path: Union[str, Path, None]) -> Optional[LabelMap]:
    """The :class:`~recordstream.LabelMap` saved beside ``checkpoint_path``, if there is one.

    The read-back half of :func:`save_class_names`, and the reason evaluate/predict share the
    training run's exact name-to-id ordering instead of refitting on a subset — refitting is
    the classic way to silently relabel a model's outputs.

    Best-effort: a missing checkpoint path, a missing file, or an unreadable one returns
    ``None`` (logged), because a caller can still proceed on integer labels.

    Args:
        checkpoint_path: Path to the checkpoint file. The vocabulary is looked for in its
            PARENT directory under :data:`CLASS_NAMES_FILE`.

    Example::

        label_map = self.label_map or load_class_names(getattr(self.model, "checkpoint_path", None))
    """
    if not checkpoint_path:
        return None
    candidate = Path(str(checkpoint_path)).expanduser().parent / CLASS_NAMES_FILE
    if not candidate.is_file():
        return None
    try:
        label_map = LabelMap.load(candidate)
    except Exception as exc:  # pragma: no cover - defensive: a corrupt or foreign file
        logger.warning(f"Could not load a LabelMap from {candidate}: {exc}")
        return None
    logger.info(f"Loaded pinned LabelMap ({label_map.num_classes} classes) from {candidate}.")
    return label_map


def write_metrics_sidecar(
    path: Union[str, Path, None],
    metrics: Mapping[str, Any],
    experiment_name: str = "",
    run_name: str = "",
) -> Optional[Path]:
    """Write a run's final metrics as a JSON sidecar stamped with the run identity.

    A tracking backend already has these numbers; the sidecar is for everything that will not
    open a tracking UI — a discovery service reading a finished run, a CI step asserting a
    threshold, a human with ``cat``.

    Args:
        path: Destination file. ``None`` writes nothing, so a run with no sidecar configured
            needs no guard at the call site. Parent directories are created.
        metrics: Final scalar metrics. Values must already be JSON-safe — computing and
            coercing them is the framework's job (``marainer.lightning.scalar_metrics``).
        experiment_name: Recorded alongside, so a sidecar found on disk identifies its run.
        run_name: Likewise.

    Returns:
        The path written, or ``None``.

    Example::

        write_metrics_sidecar(self.metrics_sidecar_path, scalar_metrics(collection.compute()),
                              self.experiment_name, self.run_name)
    """
    if path is None:
        return None
    payload: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "run_name": run_name,
        "metrics": dict(metrics),
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info(f"Wrote final metrics sidecar to {out_path}")
    return out_path

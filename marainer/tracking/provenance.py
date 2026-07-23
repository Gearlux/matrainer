"""Run-provenance capture: git hash, config artifacts, and the log file.

The one place that answers "what produced this run?" and pushes the answer to the
tracking backends, independent of any training framework:

* :func:`get_git_commit_hash` — ``git rev-parse HEAD`` (+ ``-dirty``).
* :func:`included_config_paths` — the entrypoint YAML + every transitively
  ``include:``-d file (from ``liquifai.context``), so the whole config tree is
  captured as an artifact.
* :func:`resolve_log_file` — the active loggair output file, captured as an
  artifact at the end of a run.
* :func:`log_run_context` — the ONE orchestration entry: stamp the git hash as a
  tag + hyperparam, log the config-script path, and upload the config tree to
  every tracker. :func:`capture_log_file` uploads the log file at finalize time.

Framework-agnostic: it takes a list of :class:`~marainer.tracking.base.Tracker`
objects and calls their neutral surface — a Lightning trainer, a TensorFlow loop,
or an Ultralytics trainer all call it the same way.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from loggair import get_logger

from marainer.tracking.base import Tracker

logger = get_logger(__name__)

GIT_COMMIT_TAG = "git.commit"


def get_git_commit_hash() -> Optional[str]:
    """Return the current git commit hash, with a ``-dirty`` suffix on uncommitted changes.

    Returns ``None`` when not in a git repo or git is unavailable.
    """
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("ascii").strip()
        )
        is_dirty = (
            subprocess.check_output(["git", "status", "--short"], stderr=subprocess.DEVNULL).decode("ascii").strip()
            != ""
        )
        return git_hash + ("-dirty" if is_dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def included_config_paths() -> List[Path]:
    """Return the entrypoint + transitively-included YAML config files that exist.

    Resolves from ``liquifai.context.get_context().included_paths`` (populated by
    the CLI bootstrap), falling back to ``ctx.config_path``. Returns ``[]`` when no
    liquifai context is active (e.g. a programmatic run).
    """
    try:
        from liquifai.context import get_context
    except ImportError:  # pragma: no cover - liquifai always present in the workspace
        return []
    ctx = get_context()
    if ctx is None:
        return []
    paths: List[Path] = list(getattr(ctx, "included_paths", None) or [])
    config_path = getattr(ctx, "config_path", None)
    if not paths and config_path is not None:
        paths = [config_path]
    return [p for p in paths if p.exists()]


def config_script_path() -> Optional[str]:
    """Return the entrypoint config path as a string, or ``None`` outside a liquifai run."""
    try:
        from liquifai.context import get_context
    except ImportError:  # pragma: no cover
        return None
    ctx = get_context()
    if ctx is not None and getattr(ctx, "config_path", None):
        return str(ctx.config_path)
    return None


def resolve_log_file() -> Optional[Path]:
    """Return the active loggair output file path, or ``None`` when not configured."""
    try:
        from loggair import get_active_config
    except ImportError:  # pragma: no cover
        return None
    cfg = get_active_config()
    log_file = cfg.get("log_file")
    if not log_file:
        return None
    path = Path(log_file)
    return path if path.exists() else None


def log_run_context(
    trackers: Sequence[Tracker],
    params: Optional[Mapping[str, Any]] = None,
) -> None:
    """Push run provenance to every tracker: hyperparams, git hash, and the config tree.

    Args:
        trackers: The tracking backends to push to.
        params: Optional hyperparameters (only JSON-scalar values are forwarded).
            The git commit and config-script path are added automatically.
    """
    if not trackers:
        return
    clean: dict = {}
    if params:
        clean = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
    script = config_script_path()
    if script is not None:
        clean["config_script"] = script

    commit = get_git_commit_hash()
    tags = {GIT_COMMIT_TAG: commit} if commit else {}
    if commit:
        clean.setdefault(GIT_COMMIT_TAG, commit)

    config_paths = included_config_paths()
    for tracker in trackers:
        try:
            if clean:
                tracker.log_hyperparams(clean)
            if tags:
                tracker.set_tags(tags)
            for path in config_paths:
                tracker.log_artifact(path, "config")
        except Exception as exc:  # pragma: no cover - backend-specific
            logger.debug(f"log_run_context failed on {type(tracker).__name__}: {exc}")


def capture_log_file(trackers: Sequence[Tracker]) -> None:
    """Upload the active loggair log file as a ``logs/`` artifact to every tracker."""
    if not trackers:
        return
    log_file = resolve_log_file()
    if log_file is None:
        return
    for tracker in trackers:
        try:
            tracker.log_artifact(log_file, "logs")
        except Exception as exc:  # pragma: no cover - backend-specific
            logger.debug(f"capture_log_file failed on {type(tracker).__name__}: {exc}")


__all__ = [
    "GIT_COMMIT_TAG",
    "capture_log_file",
    "config_script_path",
    "get_git_commit_hash",
    "included_config_paths",
    "log_run_context",
    "resolve_log_file",
]

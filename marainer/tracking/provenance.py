"""Run-provenance capture: git hash, config artifacts, and the log file.

The one place that answers "what produced this run?" and pushes the answer to the
tracking backends, independent of any training framework:

* :func:`get_git_commit_hash` — ``git rev-parse HEAD`` (+ ``-dirty``) for the CWD.
* :func:`git_state` / :func:`code_state` — the same question asked of a SPECIFIC
  directory, and then of every editable distribution in the environment. A run's
  behaviour is set by every package on ``sys.path``, not by whichever repository
  the shell happened to be sitting in, so the manifest records a LIST.
* :func:`included_config_paths` — the entrypoint YAML + every transitively
  ``include:``-d file (from ``liquifai.context``), so the whole config tree is
  captured as an artifact.
* :func:`config_params` — the flat ``{dotted-path: value}`` view of a configured
  object graph. It lives HERE rather than in each consumer, because two consumers
  deriving it themselves is two different answers to "what were the hyperparameters?".
* :func:`environment_snapshot` — the relevant environment variables, with anything
  that looks like a credential redacted (this ends up in an uploaded artifact).
* :func:`accelerator_state` — what the run computed on, read from the already-loaded
  framework only; this module never imports one.
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

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from loggair import get_logger

from marainer.tracking.base import Tracker

if TYPE_CHECKING:  # type-only: `manifest` imports THIS module, so a runtime import would cycle.
    from marainer.tracking.manifest import RunManifest, RunStatus

logger = get_logger(__name__)

GIT_COMMIT_TAG = "git.commit"

#: Environment-variable name prefixes worth recording — tracking backends, accelerator
#: selection, dataset caches, and thread pools all change a run's behaviour or its
#: destination. Callers extend this rather than replacing it when they have their own.
DEFAULT_ENV_PREFIXES: Tuple[str, ...] = (
    "AIM_",
    "CLEARML_",
    "COMET_",
    "CUDA_",
    "HF_",
    "HUGGINGFACE_",
    "KERAS_",
    "MKL_",
    "MLFLOW_",
    "NEPTUNE_",
    "OMP_",
    "PYTORCH_",
    "TF_",
    "TORCH_",
    "WANDB_",
)

#: Substrings that mark a variable as a credential. The manifest is uploaded as a run
#: artifact, so a captured ``WANDB_API_KEY`` would be a secret published to every
#: tracking backend — matched names keep their NAME (that a token was set is itself
#: provenance) and lose their value.
SECRET_MARKERS: Tuple[str, ...] = ("TOKEN", "KEY", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH")

REDACTED = "***"


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


def git_state(path: Path) -> Optional[Dict[str, Any]]:
    """Return ``{commit, dirty, branch, remote, root}`` for the repository at ``path``.

    ``None`` when ``path`` is not inside a git repository (or git is unavailable) — a
    package installed from a wheel has no repository, and that is not an error.
    ``dirty`` is ``None`` rather than ``False`` when the status probe itself failed:
    "unknown" and "clean" mean very different things when reproducing a run.
    """

    def git(*args: str) -> Optional[str]:
        try:
            out = subprocess.check_output(["git", "-C", str(path), *args], stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return None
        return out.decode("utf-8", errors="replace").strip()

    commit = git("rev-parse", "HEAD")
    if not commit:
        return None
    status = git("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "remote": git("remote", "get-url", "origin"),
        "root": git("rev-parse", "--show-toplevel"),
    }


def editable_source_paths() -> Dict[str, Path]:
    """Map ``distribution name -> source directory`` for every editable install.

    An editable install is the only kind whose code a developer can be *editing*, so
    it is the only kind whose git state is worth probing. PEP 610 records the source
    directory in each distribution's ``direct_url.json``; a wheel install has none.
    """
    from urllib.parse import unquote, urlparse

    found: Dict[str, Path] = {}
    for dist in importlib.metadata.distributions():
        try:
            raw = dist.read_text("direct_url.json")
            if not raw:
                continue
            info = json.loads(raw)
            if not info.get("dir_info", {}).get("editable"):
                continue
            parsed = urlparse(info.get("url", ""))
            if parsed.scheme != "file":
                continue
            name = dist.metadata["Name"]
            source = Path(unquote(parsed.path))
            if name and source.is_dir():
                found[str(name)] = source
        except Exception as exc:  # pragma: no cover - a malformed dist must not break capture
            logger.debug(f"editable_source_paths skipped a distribution: {exc}")
    return found


def code_state() -> List[Dict[str, Any]]:
    """Return one record per repository whose code this run can execute.

    Editable distributions first (their name is the informative label), then the
    working directory — which covers a consumer project run straight from a checkout
    without being installed. Deduplicated by git *toplevel*, so a package living in a
    subdirectory of a repository does not report that repository twice.
    """
    sources: List[Tuple[Optional[str], Path]] = sorted(
        ((name, path) for name, path in editable_source_paths().items()), key=lambda item: item[0] or ""
    )
    sources.append((None, Path.cwd()))

    entries: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for name, path in sources:
        state = git_state(path)
        if state is None:
            continue
        root = state.pop("root", None) or str(path)
        if root in seen:
            continue
        seen.add(root)
        entries.append({"name": name or Path(root).name, "path": root, **state})
    return entries


def file_digest(path: Path) -> Optional[str]:
    """SHA-256 of ``path``, or ``None`` when it cannot be read.

    Config files are referenced from the manifest by path *and* digest, so a later
    reader can tell whether the file still on disk is the one the run actually used.
    """
    try:
        with Path(path).open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()
    except OSError as exc:
        logger.debug(f"file_digest failed for {path}: {exc}")
        return None


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def config_params(obj: Any) -> Dict[str, Any]:
    """Flat ``{dotted-path: value}`` of a configured object graph, scalars only.

    Walks the LIVE object (not the class signature), so it sees defaults the user never
    wrote into YAML plus anything applied post-construction. Non-scalar values are
    deliberately dropped rather than stringified: an object's ``repr`` carries a memory
    address, which would make two identical runs diff — the object graph is captured
    properly by the resolved-configuration dump instead. Lists/tuples survive when every
    element is a scalar (an image size, a class list).

    Returns ``{}`` for anything that is not a configured object; deriving parameters must
    never take down a run.
    """
    try:
        from confluid import get_hierarchy_from_instance

        hierarchy = get_hierarchy_from_instance(obj)
    except Exception as exc:
        logger.debug(f"config_params could not walk {type(obj).__name__}: {exc}")
        return {}

    params: Dict[str, Any] = {}
    for path, entry in hierarchy.items():
        value = entry[1]
        if _is_scalar(value):
            params[path] = value
        elif isinstance(value, (list, tuple)) and all(_is_scalar(v) for v in value):
            params[path] = list(value)
    return params


def environment_snapshot(
    prefixes: Optional[Sequence[str]] = None,
    names: Sequence[str] = (),
) -> Dict[str, str]:
    """Return the run-relevant environment variables, credentials redacted.

    Args:
        prefixes: Name prefixes to capture; ``None`` uses :data:`DEFAULT_ENV_PREFIXES`.
        names: Additional exact names to capture whatever their prefix.
    """
    wanted = tuple(prefixes) if prefixes is not None else DEFAULT_ENV_PREFIXES
    exact = set(names)
    snapshot: Dict[str, str] = {}
    for name, value in sorted(os.environ.items()):
        if not (name.startswith(wanted) or name in exact):
            continue
        upper = name.upper()
        snapshot[name] = REDACTED if any(marker in upper for marker in SECRET_MARKERS) else value
    return snapshot


def accelerator_state() -> Optional[Dict[str, Any]]:
    """What this run computes on, read from an ALREADY-IMPORTED framework.

    Deliberately ``sys.modules`` rather than an import: marainer's core install has no
    torch, and importing one here would make a light tracking call pay a multi-second
    framework import — or pull in a framework the process deliberately did not choose.
    A process that trains has already imported its framework, so nothing is missed.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    state: Dict[str, Any] = {"kind": "cpu", "count": 0, "name": None, "torch_version": torch.__version__}
    try:
        if torch.cuda.is_available():
            state.update(kind="cuda", count=torch.cuda.device_count(), name=torch.cuda.get_device_name(0))
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            state.update(kind="mps", count=1, name="Apple MPS")
    except Exception as exc:  # pragma: no cover - a probe must never break capture
        logger.debug(f"accelerator_state probe failed: {exc}")
    return state


def platform_state() -> Dict[str, Any]:
    """Interpreter and machine identity — the parts of "where did this run" that never vary."""
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": platform.node(),
    }


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
    *,
    runnable: Any = None,
    manifest: Optional["RunManifest"] = None,
    capture_code: bool = True,
) -> Optional["RunManifest"]:
    """Capture a run's whole context and push it — the ONE call at the start of a run.

    Fills a :class:`~marainer.tracking.manifest.RunManifest` (environment, code state,
    configuration, hyperparameters) and projects it onto every tracker, plus the
    marainer-native ``git.commit`` tag and the config tree as ``config/`` artifacts.

    Consumers previously had to choose between this function and driving a manifest
    themselves, which meant two overlapping answers to "what produced this run?". There
    is now one: call this, keep the returned manifest, and pass it to
    :func:`capture_log_file` at the end.

    Args:
        trackers: The tracking backends to push to.
        params: Hyperparameters the caller knows. Merged OVER anything derived from
            ``runnable``, so an explicit value always wins. Only JSON scalars survive.
        runnable: The configured object about to run. Given one, the resolved
            configuration is dumped and its hyperparameters derived — do not hand-build
            a params dict for something this can read.
        manifest: Reuse an existing manifest instead of building one. Without it, one is
            built from the FIRST tracker's identity and ``log_dir``, so the manifest lands
            beside that backend's own artifacts.
        capture_code: Probe every editable install's git state. Costs roughly a second in
            a large workspace (one ``git`` call set per repository, once per run); set
            ``False`` where that matters more than knowing which code ran.

    Returns:
        The manifest, so the caller can go on recording outputs and finish it. ``None``
        when there is nothing to do — no trackers and no manifest supplied — which keeps
        a tracker-less run from writing files nobody asked for.
    """
    # Local import: this module is the capture LAYER that `manifest` builds on, so a
    # module-level import back into it would cycle.
    from marainer.tracking.manifest import RunManifest

    if manifest is None:
        if not trackers:
            return None
        first = trackers[0]
        manifest = RunManifest(
            run_dir=first.log_dir,
            experiment_name=first.experiment_name,
            run_name=first.run_name,
            # Read off the runnable rather than left blank: `task` is what tells a training
            # manifest from an evaluation one when comparing two runs of the SAME config, and
            # by the time this is called a CLI verb has already overridden any YAML value.
            task=getattr(runnable, "task", None),
        )

    manifest.capture_environment()
    if capture_code:
        manifest.capture_code()
    manifest.capture_config(runnable)

    # Caller-supplied params merge over the derived ones; the git hash and config script
    # ride along as params too, which is where consumers have always read them.
    extra: Dict[str, Any] = {}
    if params:
        extra.update({k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))})
    script = config_script_path()
    if script is not None:
        extra["config_script"] = script
    commit = get_git_commit_hash()
    if commit:
        extra.setdefault(GIT_COMMIT_TAG, commit)
    if extra:
        manifest.set("params", extra)

    tags = {GIT_COMMIT_TAG: commit} if commit else {}
    config_paths = included_config_paths()
    for tracker in trackers:
        try:
            if tags:
                tracker.set_tags(tags)
            for path in config_paths:
                tracker.log_artifact(path, "config")
            # Last: the manifest is complete by now, and this is what projects the
            # hyperparameters onto every backend (base `log_manifest`).
            tracker.log_manifest(manifest)
        except Exception as exc:  # pragma: no cover - backend-specific
            logger.debug(f"log_run_context failed on {type(tracker).__name__}: {exc}")
    return manifest


def capture_log_file(
    trackers: Sequence[Tracker],
    manifest: Optional["RunManifest"] = None,
    status: "RunStatus" = "success",
    finalize: bool = True,
) -> None:
    """Close a run out: upload the log file, finish + re-project the manifest, END the run.

    The end-of-run counterpart to :func:`log_run_context`, and the ONE place a run's terminal
    status is stated. Passing the manifest it returned stamps that status and pushes the final
    state, so the record a backend holds matches the one on disk.

    **The tracker is finalized here, and that is why this is one function rather than a call
    every consumer makes for itself.** Until 2026-08-02 this finished the MANIFEST and left the
    tracker running: a real MLflow sweep of thirteen runs showed eleven as ``RUNNING`` with no
    end time — so the same run was ``success`` on disk and in-flight in the UI, and no backend
    could report a duration. Nothing called :meth:`Tracker.finalize` at all; the two runs that
    looked right did so by accident, because a Lightning consumer had adapted its trackers into
    ``pl.Logger``s and Lightning finalizes those itself. A per-consumer call would have been
    eight chances to forget; here it is one, and every consumer already reaches it.

    Finalizing is LAST, after the artifact and manifest pushes: a backend with an explicit run
    lifecycle (MLflow, W&B) terminates the run in :meth:`~Tracker.finalize`, and a terminated
    run rejects further writes.

    Args:
        trackers: The tracking backends to push to.
        manifest: The manifest from :func:`log_run_context`. Omit it to upload only the
            log file (the historical behaviour).
        status: Terminal status for the run AND the manifest. The manifest keeps the FIRST
            terminal status, so an already-finished one is left alone.
        finalize: End the run on each tracker. Pass ``False`` to upload the log file and push
            the manifest WITHOUT closing the run — for a caller that reaches here more than
            once (a Lightning fit-then-test pass ends twice) and has a backend that cannot
            take a second ending. MLflow can: terminating twice is idempotent and a write
            after the first still lands, with the status staying ``FINISHED`` (measured).
            ``WandbTracker`` cannot, because its ``finish()`` drops the run handle and the
            next access starts a NEW run.
    """
    log_file = resolve_log_file()
    if manifest is not None:
        manifest.finish(status)
    for tracker in trackers:
        try:
            if log_file is not None:
                tracker.log_artifact(log_file, "logs")
            if manifest is not None:
                tracker.log_manifest(manifest)
            if finalize:
                tracker.finalize(status)
        except Exception as exc:  # pragma: no cover - backend-specific
            logger.debug(f"capture_log_file failed on {type(tracker).__name__}: {exc}")


__all__ = [
    "DEFAULT_ENV_PREFIXES",
    "GIT_COMMIT_TAG",
    "REDACTED",
    "SECRET_MARKERS",
    "accelerator_state",
    "capture_log_file",
    "code_state",
    "config_params",
    "config_script_path",
    "editable_source_paths",
    "environment_snapshot",
    "file_digest",
    "get_git_commit_hash",
    "git_state",
    "included_config_paths",
    "log_run_context",
    "platform_state",
    "resolve_log_file",
]

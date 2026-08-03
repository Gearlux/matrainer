"""Run-provenance capture: git hash, config artifacts, and the log file.

The one place that answers "what produced this run?" and pushes the answer to the
tracking backends, independent of any training framework:

* :func:`get_git_commit_hash` — ``git rev-parse HEAD`` (+ ``-dirty``) for the CWD.
* :func:`command_line` — the invocation as a shell snippet someone can PASTE to run it
  again. ``sys.argv`` alone is not that: the workspace's runner is launched with ``-m``,
  whose ``argv[0]`` is the module's file path and does not re-execute.
* :func:`package_freeze` — every installed distribution and its version, in pip's
  ``requirements.txt`` format. The code state above covers the packages being *edited*;
  a run also depends on the several hundred wheels around them.
* :func:`git_state` / :func:`code_state` — the same question asked of a SPECIFIC
  directory, and then of every editable distribution in the environment. A run's
  behaviour is set by every package on ``sys.path``, not by whichever repository
  the shell happened to be sitting in, so the manifest records a LIST.
* :func:`included_config_paths` — the entrypoint YAML + every transitively
  ``include:``-d file (from ``liquifai.context``), so the whole config tree is
  captured as an artifact.
* :func:`configured_slots` — every OBJECT wired into a configured graph, as
  ``(slot path, object)``. Deliberately different from the leaf-scalar walk below:
  :func:`configured_datasets` and :func:`configured_models` need the objects, because
  what identifies a dataset or a model is a source or a wrapper, not one of its settings.
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
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

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

#: ``argv[0]`` values that carry no re-executable entry point: an interactive interpreter
#: (``""``), a script piped on stdin (``"-"``), and ``python -c`` — whose code lives in an
#: argument the interpreter has already consumed and never puts in ``sys.argv``.
UNREPRODUCIBLE_ARGV0: Tuple[str, ...] = ("", "-", "-c")

#: How long the package freeze may take before it is abandoned. Generous for a subprocess
#: measured in tenths of a second, and bounded because tracking must never hang a run.
FREEZE_TIMEOUT_SECONDS = 120.0

#: Environment variables naming the machine-specific roots a run's paths hang off. Where one is
#: SET, absolute paths under it are recorded as ``$NAME/...`` instead, so a manifest read on a
#: machine that keeps its code, data or models somewhere else still points at the right things.
#: Ordered most-specific-first only for readability — :func:`configured_roots` sorts by length.
PORTABLE_ROOT_VARS: Tuple[str, ...] = ("PROJECT_ROOT", "DATA_ROOT", "MODEL_ROOT")


def configured_roots() -> List[Tuple[str, str]]:
    """The set roots as ``(variable name, absolute path)``, LONGEST path first.

    Longest first is what makes nesting work: with ``MODEL_ROOT`` inside ``DATA_ROOT``, a model
    path must contract to the more specific of the two rather than to whichever was listed first.
    An unset or empty variable contributes nothing — contraction is opt-in by configuration.
    """
    found = []
    for name in PORTABLE_ROOT_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            found.append((name, os.path.abspath(os.path.expanduser(value)).rstrip("/")))
    return sorted(found, key=lambda item: len(item[1]), reverse=True)


def contract_roots(text: str, roots: Optional[Sequence[Tuple[str, str]]] = None) -> str:
    """Replace any configured root PREFIX in ``text`` with its ``$NAME``.

    Works on a bare path (``/data/set/x`` -> ``$DATA_ROOT/x``), on a ``file://`` URI
    (``file:///data/set/x`` -> ``file://$DATA_ROOT/x``) and on a command line, because the
    substitution is a literal one on a path BOUNDARY — every occurrence, wherever in the string it
    sits, provided the next character is ``/`` or the match ends the segment. The boundary is what
    keeps ``/data/setup`` from being read as ``$DATA_ROOT`` + ``up``.

    **``os.path.expandvars`` is the exact inverse**, which is the property that makes this safe to
    apply to a whole document: nothing is lost, and a reader who has the same variables set gets
    the original string back. The manifest records the values it used under ``env.roots`` so a
    reader who does NOT can still resolve them by hand.

    Args:
        text: The string to contract.
        roots: Pre-computed roots; ``None`` reads them from the environment. Pass the result of
            :func:`configured_roots` when contracting many strings, so the environment is read once.
    """
    for name, root in configured_roots() if roots is None else roots:
        index = text.find(root)
        while index != -1:
            after = index + len(root)
            if after == len(text) or text[after] == "/":
                text = text[:index] + f"${name}" + text[after:]
                index = text.find(root, index + len(name) + 1)
            else:
                index = text.find(root, after)
    return text


def portable(value: Any, skip_keys: Sequence[str] = ()) -> Any:
    """Recursively contract every string in ``value`` against the configured roots.

    Applied to the whole manifest document rather than at each capture site, so a section added
    later is portable without anyone remembering to make it so — and so there is one answer to
    "is this path absolute?" rather than one per section.

    Args:
        value: Any plain data structure (the manifest document, or one section of it).
        skip_keys: Mapping keys whose subtree is left VERBATIM. ``env.roots`` uses this: it holds
            the expansion table itself, and contracting ``/data/set`` to ``$DATA_ROOT`` there
            would replace the definition with a reference to itself.
    """
    roots = configured_roots()
    if not roots:
        return value

    def walk(node: Any) -> Any:
        if isinstance(node, str):
            return contract_roots(node, roots)
        if isinstance(node, Mapping):
            return {k: (v if k in skip_keys else walk(v)) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return walk(value)


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


#: How deep :func:`configured_slots` descends into an object graph. Dataset slots sit close
#: to the runnable (``train_set``, ``train_set.source``); the cap is what keeps the walk from
#: wandering into an arbitrarily deep model or optimizer state.
MAX_SLOT_DEPTH = 4

#: Values that are never a dataset and never contain one — skipped without inspection, so the
#: walk does not pay for the scalars that make up most of a configuration.
_LEAF_TYPES = (str, bytes, int, float, bool, complex, Path)

#: Private ``__dict__`` entries that are really a registry of PUBLICLY-named children.
#: ``torch.nn.Module.__setattr__`` diverts every Module-valued assignment into ``_modules``
#: keyed by the attribute name the caller used, so ``self.model = TimmModel(...)`` leaves
#: ``model`` absent from ``__dict__`` entirely — and a walk that reads only public
#: ``__dict__`` keys cannot see the model of ANY torch-based runnable.
_CHILD_REGISTRIES = ("_modules",)


def _public_slots(attributes: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """The public attributes of one object, including framework-held child objects.

    Underscore-prefixed names are internal state and stay skipped — descending into them
    reaches a framework's parameter and buffer dictionaries, which is neither cheap nor
    meaningful. The exception is a CHILD REGISTRY (:data:`_CHILD_REGISTRIES`): a private
    mapping whose KEYS are the public attribute names the caller assigned. Its public entries
    are yielded under those names, so a slot reads the same whether the framework kept it in
    ``__dict__`` or diverted it.

    Detected STRUCTURALLY — a `_modules` attribute that is a Mapping — rather than by an
    ``isinstance`` check against ``torch.nn.Module``, because that would make a light tracking
    call import a framework the process may deliberately not have chosen (the same rule
    :func:`accelerator_state` follows).
    """
    for name, value in list(attributes.items()):
        if not name.startswith("_"):
            yield name, value
    for registry in _CHILD_REGISTRIES:
        children = attributes.get(registry)
        if not isinstance(children, Mapping):
            continue
        for name, value in list(children.items()):
            if isinstance(name, str) and not name.startswith("_") and name not in attributes:
                yield name, value


def configured_slots(obj: Any, max_depth: int = MAX_SLOT_DEPTH) -> Iterator[Tuple[str, Any]]:
    """Yield ``(slot path, object)`` for every object wired into a configured graph.

    A deliberately different walk from the one :func:`config_params` uses: that one flattens a
    graph into its LEAF values (the scalars that are hyperparameters), while this one needs the
    OBJECTS themselves, because what identifies a dataset is a source object rather than any of
    its settings.

    Public attributes only — a leading underscore marks internal state, and descending into it
    would reach a framework's private module/parameter dictionaries. Lists, tuples and dicts are
    entered with an ``[index]`` / ``[key]`` suffix so a slot holding several sources names each
    one. Cycle-safe by object identity, and depth-capped: a runnable holds a trained model, and
    an unbounded walk of one is neither cheap nor meaningful.
    """
    seen: Set[int] = set()

    def walk(node: Any, prefix: str, depth: int) -> Iterator[Tuple[str, Any]]:
        if node is None or depth > max_depth or isinstance(node, _LEAF_TYPES):
            return
        if id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, Mapping):
            for key, value in node.items():
                yield from walk(value, f"{prefix}[{key}]", depth + 1)
            return
        if isinstance(node, (list, tuple, set)):
            for index, value in enumerate(node):
                yield from walk(value, f"{prefix}[{index}]", depth + 1)
            return
        attributes = getattr(node, "__dict__", None)
        if not isinstance(attributes, dict):
            return
        for name, value in list(_public_slots(attributes)):
            path = f"{prefix}.{name}" if prefix else name
            yield path, value
            yield from walk(value, path, depth + 1)

    yield from walk(obj, "", 0)


def configured_datasets(obj: Any) -> List[Dict[str, Any]]:
    """Every dataset a configured object graph identifies, one entry each, deduplicated.

    Walks the graph with :func:`configured_slots` and asks each wired object to name the data
    it reads. An object that can (a dataset source, or a view wrapping one) contributes
    ``kind`` / ``name`` / ``uri`` / ``url`` / ``split`` plus the ``slot`` it occupies; an
    object that cannot contributes nothing.

    ``kind``, ``name`` and ``split`` are DERIVED from the URI — its scheme, its path, its
    ``split`` parameter — rather than read off the source, so this function knows nothing
    about any particular kind of dataset and a new source type needs no change here.

    Deduplication is by URI and keeps the FIRST occurrence, which is the outermost wiring: a
    trainer's ``train_set`` typically wraps the real source in a stream or a split view and all
    of them name the same dataset, so the outer slot is the one a reader of the config
    recognises.

    Returns ``[]`` for anything that cannot be walked; identifying datasets must never take
    down a run.
    """
    from urllib.parse import parse_qs, urlparse

    try:
        from recordstream import dataset_uri, dataset_url

        slots = list(configured_slots(obj))
    except Exception as exc:
        logger.debug(f"configured_datasets could not walk {type(obj).__name__}: {exc}")
        return []

    entries: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for path, value in slots:
        try:
            uri = dataset_uri(value)
        except Exception as exc:  # pragma: no cover - a source whose identity property raises
            logger.debug(f"configured_datasets skipped {path}: {exc}")
            continue
        if not uri or uri in seen:
            continue
        seen.add(uri)
        parsed = urlparse(uri)
        splits = parse_qs(parsed.query).get("split") or []
        entries.append(
            {
                "kind": parsed.scheme or "unknown",
                # A `file:` URI's path is absolute and must stay so; a scheme with an authority
                # (`hf://datasets/<id>`) leaves a leading separator that is not part of the name.
                "name": parsed.path if parsed.scheme == "file" else parsed.path.lstrip("/"),
                "uri": uri,
                "url": dataset_url(value),
                "split": splits[0] if splits else None,
                # Size + digest for a LOCAL file, so a reader can tell whether what is on disk is
                # what ran. A remote or directory reference contributes nothing (see the helper).
                **local_file_facts(uri),
                "slot": path,
            }
        )
    return entries


#: Largest file the manifest will SHA-256. Measured at ~1790 MB/s on this workspace's disk, so the
#: default costs at most ~1.2 s — worth it for a checkpoint, not worth it for a multi-tens-of-GB
#: one on a network filesystem, where the read alone could take minutes. Above the cap the size is
#: still recorded, which already distinguishes most "is this the same file?" cases.
MAX_DIGEST_BYTES = 2 * 1024**3


def local_file_facts(uri: str, max_digest_bytes: int = MAX_DIGEST_BYTES) -> Dict[str, Any]:
    """``{size, digest}`` for a ``file://`` URI pointing at a regular file; ``{}`` otherwise.

    This is what lets a later reader tell whether the checkpoint still on disk is the one the run
    loaded — the same question :func:`file_digest` answers for the configuration tree, asked of a
    weights file. A remote artifact (``hf://``, ``s3://``) contributes nothing: it is identified by
    its URI and revision, and fetching gigabytes to fingerprint it would be absurd.

    A DIRECTORY also contributes nothing. A SavedModel or an imagefolder has no single content
    hash, and walking one to invent a composite would be an unbounded amount of work at the start
    of every run — the size and digest fields stay absent rather than guessed at.

    ``size`` is free (one ``stat``) and is recorded even when the digest is skipped, so a file over
    ``max_digest_bytes`` is still distinguishable from a different one. Any failure — the file
    moved between the run reading it and this call, a permission error — yields ``{}`` rather than
    raising; provenance must never take down a run.
    """
    from urllib.parse import unquote, urlparse

    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return {}
    try:
        path = Path(unquote(parsed.path))
        if not path.is_file():
            return {}
        size = path.stat().st_size
    except OSError as exc:
        logger.debug(f"local_file_facts could not stat {uri}: {exc}")
        return {}
    facts: Dict[str, Any] = {"size": size}
    if size <= max_digest_bytes:
        digest = file_digest(path)
        if digest is not None:
            facts["digest"] = digest
    else:
        logger.debug(f"local_file_facts: {path} is {size} bytes, above the digest cap; recording size only")
    return facts


def configured_models(obj: Any) -> List[Dict[str, Any]]:
    """Every model artifact a configured object graph loads, one entry each, deduplicated.

    The model twin of :func:`configured_datasets`, over the same :func:`configured_slots` walk.
    The difference is arity and it is deliberate: a source reads ONE dataset, while a model
    wrapper composes artifacts — a pretrained backbone with a checkpoint over it used two — so a
    wrapper reports a LIST of :class:`~marainer.uri.ModelRef`, each carrying the ``role`` it
    played, and one slot may contribute several entries.

    ``kind`` and ``name`` are DERIVED from the URI (its scheme, its path) exactly as on the dataset
    side, so this function knows nothing about any particular zoo and a new model wrapper needs no
    change here.

    Deduplication is by URI across the whole graph, so two wrappers loading the same backbone
    record it once. Returns ``[]`` for anything that cannot be walked; identifying models must
    never take down a run.
    """
    from urllib.parse import urlparse

    try:
        from marainer.uri import model_refs

        slots = list(configured_slots(obj))
    except Exception as exc:
        logger.debug(f"configured_models could not walk {type(obj).__name__}: {exc}")
        return []

    entries: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for path, value in slots:
        try:
            refs = model_refs(value)
        except Exception as exc:  # pragma: no cover - a wrapper whose property raises
            logger.debug(f"configured_models skipped {path}: {exc}")
            continue
        for ref in refs:
            if ref.uri in seen:
                continue
            seen.add(ref.uri)
            parsed = urlparse(ref.uri)
            entries.append(
                {
                    "kind": parsed.scheme or "unknown",
                    # A `file:` URI's path is absolute and must stay so; a scheme with an authority
                    # (`hf://models/<id>`) leaves a leading separator that is not part of the name.
                    "name": parsed.path if parsed.scheme == "file" else parsed.path.lstrip("/"),
                    "uri": ref.uri,
                    "url": ref.url,
                    "role": ref.role,
                    **local_file_facts(ref.uri),
                    "slot": path,
                }
            )
    return entries


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


def _entry_point_words() -> Optional[List[str]]:
    """The leading words of the pasteable command — how this process was ENTERED.

    Three launch shapes, each detected by the one fact that distinguishes it:

    * ``python -m pkg.mod`` — ``__main__.__spec__`` is set (it is ``None`` for a plain
      script and for a console script). ``argv[0]`` here is the module's *file* path, which
      is why re-pasting ``argv`` verbatim does not work; the spec's ``name`` is the module.
      A trailing ``.__main__`` is stripped, because ``python -m pkg`` reports ``pkg.__main__``
      while ``python -m pkg.__main__`` is not how anyone types it.
    * a console script (``.venv/bin/sonair``) — an executable file that is not ``.py``. It
      carries its own ``#!`` interpreter, so it pastes back exactly as it stands.
    * a plain script — needs the interpreter in front of it.

    ``sys.executable`` is the absolute interpreter, so every form runs without activating
    an environment first. ``None`` when there is no honest answer (see
    :data:`UNREPRODUCIBLE_ARGV0`).
    """
    argv0 = sys.argv[0] if sys.argv else ""
    if argv0 in UNREPRODUCIBLE_ARGV0:
        return None
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    module = getattr(spec, "name", None)
    if module:
        return [sys.executable, "-m", module[: -len(".__main__")] if module.endswith(".__main__") else module]
    script = Path(argv0)
    if script.suffix != ".py" and os.access(argv0, os.X_OK):
        return [argv0]
    return [sys.executable, argv0]


def command_line() -> Optional[str]:
    """This run's invocation as a shell snippet that can be PASTED to run it again.

    The working directory is folded in, so the snippet reproduces the run from anywhere —
    a config referenced by a relative path (the normal case) resolves the same way it did
    originally. Every word is shell-quoted, so a path containing spaces survives.

    Returns ``None`` for an invocation that cannot be reconstructed — an interactive
    session, or ``python -c``, whose code the interpreter never puts in ``sys.argv``. The
    raw ``argv`` is recorded beside this either way, so nothing is lost by declining.

    Example::

        'cd /w/sonair && /w/.venv/bin/python -m recordstream.cli run config.yaml'
    """
    words = _entry_point_words()
    if words is None:
        return None
    roots = configured_roots()
    parts = [_quote(contract_roots(word, roots)) for word in (str(Path.cwd()), *words, *sys.argv[1:])]
    return f"cd {parts[0]} && {' '.join(parts[1:])}"


def _quote(word: str) -> str:
    """Shell-quote ``word``, keeping any ``$VAR`` in it EXPANDABLE.

    ``shlex.quote`` wraps anything that needs quoting in SINGLE quotes, which is exactly right for
    a literal — and exactly wrong here, because it would turn a contracted ``$PROJECT_ROOT/my
    runs`` into a literal dollar sign and the pasted line would look for a directory that does not
    exist. A word carrying a variable is therefore double-quoted instead (the shell expands inside
    those), with the three characters double quotes do not neutralise escaped by hand.
    """
    if "$" not in word:
        return shlex.quote(word)
    # Always double-quoted, never bare: the expansion itself may contain a space, and a bare
    # `$PROJECT_ROOT/x` would then split into two arguments on the machine where it does.
    escaped = word.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
    return f'"{escaped}"'


def package_freeze() -> Optional[Tuple[str, str]]:
    """Every installed distribution and its version, in ``requirements.txt`` format.

    Returns ``(text, tool)`` — the freeze output and the name of the program that produced
    it — or ``None`` when neither program is usable.

    ``uv`` is preferred when it is on ``PATH`` purely because it is fast: measured against
    a 730-distribution environment, ``uv pip freeze`` took 0.06 s against ``pip freeze``'s
    0.96 s, and this runs on every tracked run. ``--python`` pins the environment
    explicitly, so the answer does not depend on the working directory or on which
    environment happens to be active. Nothing is lost by preferring it: the two disagree
    only on how they spell an editable install, and the git commit, branch and *dirty*
    flag for every editable are already in the manifest's ``code`` section — which pip's
    frozen commit cannot express, since it has no notion of uncommitted changes.

    Every failure mode — no ``pip`` module in the environment, a non-zero exit, a hung
    subprocess — is a debug log and ``None``. Tracking must never abort a run.
    """
    candidates: List[Tuple[str, List[str]]] = []
    uv = shutil.which("uv")
    if uv:
        candidates.append(("uv", [uv, "pip", "freeze", "--python", sys.executable]))
    candidates.append(("pip", [sys.executable, "-m", "pip", "freeze"]))

    for tool, command in candidates:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=FREEZE_TIMEOUT_SECONDS, check=True)
        except (subprocess.SubprocessError, OSError) as exc:
            logger.debug(f"package_freeze: {tool} could not produce a freeze: {exc}")
            continue
        text = result.stdout.strip()
        if text:
            return text + "\n", tool
        logger.debug(f"package_freeze: {tool} produced no output")
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
    *,
    runnable: Any = None,
    manifest: Optional["RunManifest"] = None,
    capture_code: bool = True,
) -> Optional["RunManifest"]:
    """Capture a run's whole context and push it — the ONE call at the start of a run.

    Fills a :class:`~marainer.tracking.manifest.RunManifest` (environment, code state,
    configuration, hyperparameters, installed packages, datasets) and projects it onto every
    tracker, plus the marainer-native ``git.commit`` tag and the config tree as ``config/``
    artifacts.

    Consumers previously had to choose between this function and driving a manifest
    themselves, which meant two overlapping answers to "what produced this run?". There
    is now one: call this, keep the returned manifest, and pass it to
    :func:`capture_log_file` at the end.

    Args:
        trackers: The tracking backends to push to.
        params: Hyperparameters the caller knows. Merged OVER anything derived from
            ``runnable``, so an explicit value always wins. Only JSON scalars survive.
        runnable: The configured object about to run. Given one, the resolved
            configuration is dumped, its hyperparameters derived, and every dataset its
            sources identify recorded — do not hand-build a params dict, or a dataset
            record, for something this can read.
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
    manifest.capture_packages()
    manifest.capture_datasets(runnable)
    manifest.capture_models(runnable)

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
    "FREEZE_TIMEOUT_SECONDS",
    "GIT_COMMIT_TAG",
    "MAX_DIGEST_BYTES",
    "MAX_SLOT_DEPTH",
    "PORTABLE_ROOT_VARS",
    "REDACTED",
    "SECRET_MARKERS",
    "UNREPRODUCIBLE_ARGV0",
    "accelerator_state",
    "capture_log_file",
    "code_state",
    "command_line",
    "config_params",
    "config_script_path",
    "configured_roots",
    "contract_roots",
    "configured_datasets",
    "configured_models",
    "configured_slots",
    "editable_source_paths",
    "environment_snapshot",
    "file_digest",
    "get_git_commit_hash",
    "git_state",
    "included_config_paths",
    "local_file_facts",
    "log_run_context",
    "package_freeze",
    "platform_state",
    "portable",
    "resolve_log_file",
]

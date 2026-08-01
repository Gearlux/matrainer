"""``RunManifest`` — the crash-safe, backend-neutral record of what produced a run.

The manifest is the **union** of everything static about a run — identity, command
line, code state, configuration, hyperparameters, environment, datasets, models and
output files. A tracking backend's native slot (a hosted server's dataset-input API,
an artifact-lineage graph, a hyperparameter plugin) is a **projection** of it, never
an alternative: every fact lands here unconditionally, a backend that has a native
home for a fact *also* writes it there, and no fact is ever native-only. That is what
keeps two runs comparable when one tracked to a server and the other to a local
directory. See ``docs/architecture.md`` record 3.

Two properties are load-bearing and easy to break:

* **Crash-safety is a WRITE POLICY, not a shutdown hook.** Every mutation re-emits the
  whole document atomically (temp file + :func:`os.replace`), so an interrupted run
  leaves a complete record of everything gathered up to that point. The ``atexit`` /
  ``SIGINT`` / ``SIGTERM`` hooks only stamp the terminal :data:`RunStatus`; if they
  never run, the manifest is still on disk and still valid, merely marked ``running``.
  Never buffer until run end.
* **YAML and JSON, from one write.** ``manifest.yaml`` is the artifact a human reads
  (``yaml.safe_dump``, tag-free, fixed section order so two runs diff cleanly);
  ``manifest.json`` is the same document for tooling that should not need a YAML
  parser. Because they are written together, an interrupted run cannot leave them
  disagreeing.

Nothing here re-owns another library's serialization format. A resolved configuration
dump belongs in its own artifact file, referenced from the ``config`` section by path
and digest — the same rule :mod:`marainer.artifacts` follows for the label vocabulary.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import signal
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union, get_args

import yaml
from loggair import get_logger

from marainer.run_names import generate_run_name
from marainer.tracking import provenance

logger = get_logger(__name__)

#: Bumped when the on-disk layout changes incompatibly, so a reader can refuse politely.
MANIFEST_VERSION = 1

MANIFEST_YAML = "manifest.yaml"
MANIFEST_JSON = "manifest.json"

#: Where the resolved-configuration dump lands, relative to the run directory. It stays a
#: SEPARATE file because it is the configuration library's format (``!class:`` / ``!ref``
#: tags), which the manifest itself must never carry.
CONFIG_DIR = "config"
FINAL_CONFIG_FILE = "final.yaml"

#: A run's terminal state. ``running`` until :meth:`RunManifest.finish` says otherwise.
RunStatus = Literal["running", "success", "failed", "interrupted"]

#: The manifest's sections, **in the order they are emitted** — the file reads
#: top-down from "which run is this" to "what did it produce", and two runs diff
#: cleanly instead of scrambling alphabetically.
ManifestSection = Literal[
    "run",
    "command",
    "code",
    "config",
    "params",
    "env",
    "datasets",
    "models",
    "outputs",
]

#: Runtime tuple derived FROM the annotation — one source of truth for the section set.
SECTIONS: Tuple[ManifestSection, ...] = get_args(ManifestSection)

#: Sections holding a list of entries (append via :meth:`RunManifest.add`); every other
#: section is a mapping merged via :meth:`RunManifest.set`.
LIST_SECTIONS: Tuple[ManifestSection, ...] = ("code", "datasets", "models", "outputs")

#: The complement, derived — never restate the partition.
MAPPING_SECTIONS: Tuple[ManifestSection, ...] = tuple(s for s in SECTIONS if s not in LIST_SECTIONS)


def _now() -> str:
    """UTC timestamp in ISO-8601, the one time format the manifest uses."""
    return datetime.now(timezone.utc).isoformat()


def _plain(value: Any) -> Any:
    """Coerce ``value`` into something both ``yaml.safe_dump`` and ``json.dumps`` accept.

    A manifest must never abort a run because a caller handed it an exotic object, so
    anything unrecognised degrades to its ``str()`` rather than raising. Scalars that
    know how to unwrap themselves (numpy/torch zero-dim values, via ``.item()``) do so
    first, since ``str(np.float32(0.1))`` would be a lie dressed as a number.

    Builtin scalars are rebuilt rather than passed through, because ``yaml.safe_dump``
    refuses a *subclass* of a type it knows — ``torch.__version__`` is a ``str``
    subclass, and passing it straight through failed the whole write with the
    unhelpful ``('cannot represent an object', '2.13.0')``. An ``isinstance`` check is
    what lets a subclass in, so the conversion has to be unconditional.
    """
    # `type(... ) is bool` rather than isinstance, and FIRST: bool is an int subclass, so an
    # isinstance-int check below would otherwise turn True into 1. There is no bool-subclass
    # case to handle — Python forbids subclassing bool.
    if value is None or type(value) is bool:
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _plain(item())
        except Exception:  # pragma: no cover - a multi-element array has no scalar form
            pass
    return str(value)


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically — temp file in the same directory, then rename.

    Same-directory placement matters: :func:`os.replace` is only atomic within one
    filesystem, and it is what guarantees a reader never observes a half-written
    manifest (the exact moment a crashed run is most likely to be inspected).
    """
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class RunManifest:
    """Accumulate a run's static facts and keep them on disk, continuously and atomically.

    Zero-arg constructible per the workspace lazy-init rule: the constructor stores
    values only, and the run directory is created on first :attr:`directory` access.
    Every mutating call (:meth:`set`, :meth:`add`, the ``record_*`` helpers,
    :meth:`finish`) re-emits both files, so there is no flush to forget and no window
    in which an interrupted run has nothing on disk.

    Args:
        run_dir: Directory the manifest is written to. ``None`` (default) derives
            ``runs/<experiment_name>/<run_name>`` — usually you pass a tracker's
            ``log_dir`` so the manifest lands beside that backend's own artifacts.
        experiment_name: The experiment this run belongs to.
        run_name: This run's name; ``None`` draws a random slug via
            :func:`marainer.run_names.generate_run_name`, matching ``Tracker``.
        task: What the run is doing (``fit`` / ``evaluate`` / ``test`` / ``predict``),
            when the caller knows.
        handle_interrupts: Install ``SIGINT``/``SIGTERM`` handlers that stamp
            ``interrupted`` before chaining to the previous handler. Set ``False`` when
            the embedding application owns signal handling; ``atexit`` still applies.
    """

    def __init__(
        self,
        run_dir: Optional[Union[str, Path]] = None,
        experiment_name: str = "marainer",
        run_name: Optional[str] = None,
        task: Optional[str] = None,
        handle_interrupts: bool = True,
    ) -> None:
        self.run_dir = run_dir
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix="run")
        self.task = task
        self.handle_interrupts = handle_interrupts
        self.status: RunStatus = "running"
        self.started_at = _now()
        self.stopped_at: Optional[str] = None
        self._sections: Dict[str, Any] = {s: ([] if s in LIST_SECTIONS else {}) for s in SECTIONS}
        # Reentrant: a signal handler runs on the thread it interrupts, so a plain Lock
        # would deadlock against a mutation already in progress on that same thread.
        self._lock = threading.RLock()
        self._hooks_installed = False
        self._previous_handlers: Dict[int, Any] = {}
        self._write_failed = False

    # -- location ---------------------------------------------------------------------------

    @property
    def directory(self) -> Path:
        """The run directory, created on access. Recomputed from current inputs, never cached."""
        base = Path(self.run_dir) if self.run_dir is not None else Path("runs") / self.experiment_name / self.run_name
        base.mkdir(parents=True, exist_ok=True)
        return base

    @property
    def path(self) -> Path:
        """Path of ``manifest.yaml`` — the readable artifact."""
        return self.directory / MANIFEST_YAML

    @property
    def json_path(self) -> Path:
        """Path of ``manifest.json`` — the same document, for tooling."""
        return self.directory / MANIFEST_JSON

    # -- accumulation -----------------------------------------------------------------------

    def set(self, section: ManifestSection, values: Mapping[str, Any]) -> None:
        """Merge ``values`` into a mapping ``section``, then re-emit both files.

        Raises:
            ValueError: On an unknown section, or a list section (use :meth:`add`).
                A mistyped section is a programming error, not a runtime condition —
                silently creating a new key would hide it until someone read the file.
        """
        self._require(section, MAPPING_SECTIONS)
        with self._lock:
            self._sections[section].update(dict(values))
        self.write()

    def add(self, section: ManifestSection, entry: Mapping[str, Any]) -> None:
        """Append ``entry`` to a list ``section``, then re-emit both files.

        Raises:
            ValueError: On an unknown section, or a mapping section (use :meth:`set`).
        """
        self._require(section, LIST_SECTIONS)
        with self._lock:
            self._sections[section].append(dict(entry))
        self.write()

    def record_dataset(
        self,
        kind: str,
        name: str,
        revision: Optional[str] = None,
        split: Optional[str] = None,
        count: Optional[int] = None,
        digest: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Record one dataset the run consumed (``kind`` = where it came from, e.g. ``hf``)."""
        self.add(
            "datasets",
            {
                "kind": kind,
                "name": name,
                "revision": revision,
                "split": split,
                "count": count,
                "digest": digest,
                **extra,
            },
        )

    def record_model(
        self,
        source: str,
        name: str,
        revision: Optional[str] = None,
        digest: Optional[str] = None,
        **extra: Any,
    ) -> None:
        """Record one model the run used (``source`` = zoo / checkpoint / registry / …)."""
        self.add("models", {"source": source, "name": name, "revision": revision, "digest": digest, **extra})

    def record_output(
        self,
        path: Union[str, Path],
        kind: str = "checkpoint",
        **extra: Any,
    ) -> None:
        """Record one file the run produced — a checkpoint, an export, a report."""
        self.add("outputs", {"kind": kind, "path": str(path), **extra})

    # -- capture ----------------------------------------------------------------------------

    def capture_environment(
        self,
        env_prefixes: Optional[Sequence[str]] = None,
        env_names: Sequence[str] = (),
    ) -> None:
        """Fill ``command`` and ``env`` — how the run was invoked and where it ran.

        Args:
            env_prefixes: Environment-variable prefixes to record; ``None`` uses
                :data:`~marainer.tracking.provenance.DEFAULT_ENV_PREFIXES`.
            env_names: Extra exact variable names to record. Credential-looking names
                are redacted whichever way they were selected — the manifest becomes an
                uploaded artifact.
        """
        self.set("command", {"argv": list(sys.argv), "cwd": str(Path.cwd()), "executable": sys.executable})
        self.set(
            "env",
            {
                **provenance.platform_state(),
                "accelerator": provenance.accelerator_state(),
                "variables": provenance.environment_snapshot(env_prefixes, env_names),
            },
        )

    def capture_code(self) -> None:
        """Fill ``code`` — one record per repository whose code this run can execute.

        Appended in one shot rather than through :meth:`add` per entry, so a ten-package
        workspace costs one write instead of ten.
        """
        entries = provenance.code_state()
        if not entries:
            return
        with self._lock:
            self._sections["code"].extend(entries)
        self.write()

    def capture_config(self, runnable: Any = None) -> None:
        """Fill ``config`` (and ``params`` when given a ``runnable``).

        The *input* side is the entrypoint config plus its transitively included tree,
        each recorded by path and SHA-256. The *resolved* side — what the configuration
        actually became after materialization and post-construction — is dumped to
        ``config/final.yaml`` beside the manifest and referenced the same way, because
        that dump is the configuration library's format and the manifest must stay
        readable by a generic YAML tool.

        Args:
            runnable: The configured object the run is about to execute. Omit it (the
                default) to record only the input configuration — a run driven by code
                rather than by a config file still has a command line and a code state.
        """
        config: Dict[str, Any] = {}
        script = provenance.config_script_path()
        if script is not None:
            config["script"] = script
        includes = [{"path": str(p), "sha256": provenance.file_digest(p)} for p in provenance.included_config_paths()]
        if includes:
            config["includes"] = includes
        if runnable is not None:
            final = self._write_final_config(runnable)
            if final is not None:
                config["final"] = final
            params = provenance.config_params(runnable)
            if params:
                self.set("params", params)
        self.set("config", config)

    def _write_final_config(self, runnable: Any) -> Optional[Dict[str, Any]]:
        """Dump the resolved configuration beside the manifest; return its path + digest."""
        try:
            from confluid import dump

            text = dump(runnable)
        except Exception as exc:
            logger.debug(f"resolved-configuration dump failed for {type(runnable).__name__}: {exc}")
            return None
        try:
            directory = self.directory
            target = directory / CONFIG_DIR / FINAL_CONFIG_FILE
            target.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(target, text)
        except OSError as exc:
            logger.debug(f"resolved-configuration dump could not be written: {exc}")
            return None
        # Relative, so a run directory stays portable when it is copied or archived.
        return {"path": str(target.relative_to(directory)), "sha256": hashlib.sha256(text.encode()).hexdigest()}

    # -- reading (what a native projection uses) --------------------------------------------

    def section(self, name: ManifestSection) -> Any:
        """A COPY of one section — a ``dict`` for mapping sections, a ``list`` for list ones.

        This is how a backend's native projection reads the manifest: never through the
        private store, so a projection cannot mutate what it is projecting. Values come back
        :func:`_plain`-coerced, the same as :meth:`to_dict` writes them — a backend handing
        them to its own API must not have to re-do that, and a stray ``Path`` reaching a
        native slot fails there rather than here.
        """
        self._require_known(name)
        with self._lock:
            value = self._sections[name]
            if name == "run":
                return _plain({**self._identity(), **value})
            return _plain(list(value) if isinstance(value, list) else dict(value))

    def primary_code(self) -> Optional[Dict[str, Any]]:
        """The one code record that best answers "which repository is THIS run's?".

        The ``code`` section is a list on purpose, but some backends have exactly one
        native slot for a source commit, and something has to go in it. Four steps, each
        defensible on its own, and ``None`` rather than a guess when none of them applies:

        1. the repository containing the working directory — the project being worked on;
        2. the repository containing the entry script, for a run launched from elsewhere;
        3. the only record, when there is exactly one — not a choice at all;
        4. ``None``. With several unrelated repositories and nothing tying the run to any
           of them, there IS no primary; the list stays authoritative and the backend's
           single slot is better left empty than filled with an arbitrary pick. (Returning
           the first record instead once stamped an unrelated workspace package's commit
           on a run as ``mlflow.source.git.commit`` — a wrong answer looks exactly like a
           right one in a UI.)
        """
        entries = self.section("code")
        if not entries:
            return None
        command = self.section("command")
        argv = command.get("argv") or []
        for candidate in (command.get("cwd") or str(Path.cwd()), self.section("config").get("script"), *argv[:1]):
            match = self._code_containing(entries, candidate)
            if match is not None:
                return match
        return dict(entries[0]) if len(entries) == 1 else None

    @staticmethod
    def _code_containing(entries: List[Dict[str, Any]], location: Optional[str]) -> Optional[Dict[str, Any]]:
        """The record whose repository contains ``location``, comparing whole path segments."""
        if not location:
            return None
        target = str(Path(location).absolute())
        for entry in entries:
            path = entry.get("path")
            # Segment-wise, so `/w/project-old` does not "contain" `/w/project`.
            if path and (target == path or target.startswith(path.rstrip("/") + "/")):
                return dict(entry)
        return None

    def _require_known(self, section: str) -> None:
        if section not in SECTIONS:
            raise ValueError(f"unknown manifest section {section!r}; expected one of {', '.join(SECTIONS)}")

    def _require(self, section: str, allowed: Tuple[ManifestSection, ...]) -> None:
        self._require_known(section)
        if section not in allowed:
            # Derived from the section itself rather than from `allowed`, so the message
            # stays right however the partition is spelled at the call site.
            correct, used = ("add", "set") if section in LIST_SECTIONS else ("set", "add")
            raise ValueError(f"manifest section {section!r} takes {correct}(), not {used}()")

    # -- serialization ----------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """The whole manifest as plain data, in emission order and safe for both dumpers."""
        with self._lock:
            doc: Dict[str, Any] = {"manifest_version": MANIFEST_VERSION, "status": self.status}
            for section in SECTIONS:
                value = self._sections[section]
                if section == "run":
                    doc[section] = {**self._identity(), **value}
                elif isinstance(value, list):
                    doc[section] = list(value)
                else:
                    doc[section] = dict(value)
            return _plain(doc)  # type: ignore[no-any-return]

    def _identity(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "task": self.task,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
        }

    def write(self) -> None:
        """Re-emit ``manifest.yaml`` and ``manifest.json`` atomically. Never raises.

        Tracking must not be able to abort a training run, so a write failure is
        reported and swallowed — loudly the first time (an unwritable run directory is
        something the operator must fix), quietly afterwards so a per-mutation write
        cannot flood the log with the same message.
        """
        try:
            doc = self.to_dict()
            directory = self.directory
            self._install_hooks()
            _atomic_write(directory / MANIFEST_YAML, yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
            _atomic_write(directory / MANIFEST_JSON, json.dumps(doc, indent=2) + "\n")
            self._write_failed = False
        except Exception as exc:
            if not self._write_failed:
                self._write_failed = True
                logger.warning(f"RunManifest could not be written to {self.run_dir or 'runs/'}: {exc}")
            else:
                logger.debug(f"RunManifest write still failing: {exc}")

    # -- lifecycle --------------------------------------------------------------------------

    def finish(self, status: RunStatus = "success") -> None:
        """Stamp the terminal ``status`` + stop time and re-emit. The FIRST terminal status wins.

        That ordering is what makes the shutdown hooks safe: a run that already called
        ``finish("success")`` is not relabelled ``interrupted`` by the ``atexit`` hook
        firing moments later during normal interpreter shutdown.
        """
        with self._lock:
            if self.status != "running":
                return
            self.status = status
            self.stopped_at = _now()
        self.write()

    def _install_hooks(self) -> None:
        """Register the shutdown hooks once, on the first successful write."""
        if self._hooks_installed:
            return
        # Set BEFORE registering: a failure below must not make every subsequent write retry.
        self._hooks_installed = True
        atexit.register(self._on_exit)
        if not self.handle_interrupts:
            return
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous_handlers[signum] = signal.signal(signum, self._on_signal)
            except (ValueError, OSError) as exc:
                # ValueError: not the main thread. OSError: signal unsupported here.
                logger.debug(f"RunManifest could not install a handler for signal {signum}: {exc}")

    def _on_exit(self) -> None:
        """Interpreter shutdown: a run still marked ``running`` never reached its own end."""
        self.finish("interrupted")

    def _on_signal(self, signum: int, frame: Optional[FrameType]) -> None:
        """Stamp ``interrupted``, then hand the signal back to whoever had it.

        Chaining rather than consuming is the point: the caller's own Ctrl-C behaviour
        must survive the fact that a manifest is being kept.
        """
        self.finish("interrupted")
        previous = self._previous_handlers.get(signum, signal.SIG_DFL)
        if callable(previous):
            previous(signum, frame)
        elif previous == signal.SIG_DFL:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        # signal.SIG_IGN: the process chose to ignore this signal — do the same.


__all__: List[str] = [
    "LIST_SECTIONS",
    "MANIFEST_JSON",
    "MANIFEST_VERSION",
    "MANIFEST_YAML",
    "MAPPING_SECTIONS",
    "ManifestSection",
    "RunManifest",
    "RunStatus",
    "SECTIONS",
]

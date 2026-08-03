"""The :class:`Tracker` ABC and the :class:`TrackerGroup` fan-out composite.

A ``Tracker`` is matrainer's platform-neutral experiment-tracking backend. Its
method surface deliberately mirrors ``pytorch_lightning.loggers.Logger``
(``name`` / ``version`` / ``experiment`` / ``log_hyperparams`` / ``log_metrics`` /
``save`` / ``finalize``) so that:

* a Lightning training subpackage can adapt a ``Tracker`` to an ``L.Logger`` (and
  vice-versa) with a thin wrapper, and
* a NON-Lightning training platform (a TensorFlow loop, an Ultralytics trainer, a
  hand-rolled loop) drives the SAME tracker objects directly.

On top of the Lightning shape it adds the extensions every experiment needs but
that ``Logger`` leaves backend-specific: ``set_tags`` / ``log_artifact`` /
``log_figure`` / ``log_text`` / ``log_dir``.

Concrete backends live one-per-file (``mlflow_backend`` / ``tensorboard_backend`` /
``csv_backend`` / ``wandb_backend``) and lazily import their library through
:func:`matrainer._imports.require` — importing this module (or the whole
``matrainer.tracking`` package) pulls in NONE of them.

Every ``Tracker`` obeys the workspace lazy / zero-arg construction rule: the
constructor stores config only; the native backend handle is created on first
:attr:`experiment` access.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Union, cast

from confluid import configurable
from loggair import get_logger

from matrainer.run_names import generate_run_name

if TYPE_CHECKING:  # `manifest` imports `provenance`, which imports THIS module — type-only, no cycle.
    from matrainer.tracking.manifest import RunManifest

logger = get_logger(__name__)


def as_tracker_list(value: Any) -> Optional[List["Tracker"]]:
    """Normalize a ``trackers`` slot value to a list (or ``None`` when nothing was wired).

    A consuming trainer declares its slot as ``Union[Tracker, Iterable[Tracker]]`` so that a
    YAML key can be a single tracker OR a list, and — the part that actually matters —
    so confluid will broadcast a LIST-valued top-level ``trackers:`` key into it at all. That
    gate reads the ANNOTATION: a list only reaches a slot whose declared type admits one, which
    is why the same key silently did nothing on a slot typed ``Optional[List[Tracker]]`` and
    assigned after construction.

    Values pass through un-flowed: a slot may still hold deferred ``Fluid`` markers at
    construction time, and it is the trainer's job to flow them when it runs.
    """
    if value is None:
        return None
    # `str`/`bytes` are Iterable but are never a list OF trackers — without the guard a
    # stray string would explode into one "tracker" per character.
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return [cast("Tracker", value)]
    return [cast("Tracker", item) for item in value]


def text_file_stem(name: str, step: Optional[int] = None) -> str:
    """The filename stem a :meth:`Tracker.log_text` value is stored under.

    Shared so every backend files the same value under the same name: a metric name is
    ``val/confusion``-shaped and ``/`` is a path separator, and a per-step value needs the
    step in the NAME (a backend whose text channel has no step axis would otherwise keep
    only the last one).
    """
    stem = name.replace("/", "_").strip("_") or "text"
    return f"{stem}_{step}" if step is not None else stem


@configurable
class Tracker(ABC):
    """A platform-neutral experiment-tracking backend (``pytorch_lightning.loggers.Logger`` shaped).

    Subclasses implement the two abstract members — :attr:`experiment` (the lazily
    created native handle) and :meth:`log_metrics` — and typically override
    :meth:`log_hyperparams`, :meth:`log_artifact`, :meth:`log_figure`,
    :meth:`set_tags`, and :meth:`finalize`. The extension methods default to a
    debug-logged no-op so a backend that cannot serve one degrades quietly rather
    than raising mid-run (tracking must never abort a real training run).

    Args:
        experiment_name: The experiment / project this run belongs to.
        run_name: This run's name. ``None`` (default) draws a random human-friendly
            slug (e.g. ``run_funny_rabbit``) via
            :func:`matrainer.run_names.generate_run_name`.
        save_dir: Local directory a file-backed backend writes under.
        tags: Optional key→value tags stamped on the run at creation.
    """

    def __init__(
        self,
        experiment_name: str = "matrainer",
        run_name: Optional[str] = None,
        save_dir: str = "./runs",
        tags: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name if run_name is not None else generate_run_name(prefix="run")
        self.save_dir = save_dir
        self.tags = dict(tags) if tags else {}

    # -- pytorch_lightning.loggers.Logger shape -----------------------------------------------

    @property
    def name(self) -> str:
        """The experiment name (Lightning ``Logger.name``)."""
        return self.experiment_name

    @property
    def version(self) -> Optional[str]:
        """The run identifier / version (Lightning ``Logger.version``); ``run_name`` by default."""
        return self.run_name

    @property
    @abstractmethod
    def experiment(self) -> Any:
        """The lazily created native backend handle (Lightning ``Logger.experiment``).

        Building it here — not in ``__init__`` — keeps construction cheap and
        side-effect-free (the workspace zero-arg rule). The first access creates
        the run; subsequent accesses return the cached handle.
        """
        raise NotImplementedError

    @abstractmethod
    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        """Record run hyperparameters / config (Lightning ``Logger.log_hyperparams``)."""
        raise NotImplementedError

    @abstractmethod
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        """Record scalar metrics at an optional ``step`` (Lightning ``Logger.log_metrics``)."""
        raise NotImplementedError

    def save(self) -> None:
        """Flush any buffered state to disk (Lightning ``Logger.save``). Default no-op."""

    def finalize(self, status: str = "success") -> None:
        """Finish the run with a terminal ``status`` (Lightning ``Logger.finalize``).

        Default flushes via :meth:`save`; backends with an explicit run lifecycle
        (MLflow, W&B) override to terminate the run.
        """
        self.save()

    # -- matrainer extensions (default: debug-log + skip, never raise) --------------------------

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Attach key→value tags to the run. Default: merge into :attr:`tags` only."""
        self.tags.update(tags)

    def log_artifact(self, path: Union[str, Path], artifact_dir: str = "") -> None:
        """Persist a local file as a run artifact under ``artifact_dir``. Default no-op."""
        logger.debug(f"{type(self).__name__}.log_artifact({path!r}) not supported; skipping.")

    def log_figure(self, name: str, figure: Any, step: Optional[int] = None) -> None:
        """Persist a matplotlib ``figure`` under ``name``. Default no-op."""
        logger.debug(f"{type(self).__name__}.log_figure({name!r}) not supported; skipping.")

    def log_text(self, name: str, text: str, step: Optional[int] = None) -> None:
        """Persist a block of text under ``name``. Default: a ``.md`` file in this backend's tree.

        The channel for a run value that is neither a number nor an image — a confusion
        matrix rendered as a table, a classification report, a diff. :meth:`log_metrics`
        cannot carry it (one float per name) and :meth:`log_figure` needs a plotting
        library to have drawn it first.

        Unlike :meth:`log_figure` this default is REAL work, not a skip: text is trivially
        serializable, so every backend can hold it and a caller never has to ask whether
        this one can. A backend with a native text channel overrides and uses it INSTEAD —
        note the contrast with :meth:`log_manifest`, where an override must write native
        slots *in addition* to the artifact. There the artifact is a union guarantee that
        keeps two differently-tracked runs comparable; here the text IS the payload, so one
        durable home per backend is the whole requirement and a second copy is just a
        second copy.

        ``step`` is part of the FILENAME rather than a version of one file, so a value
        logged once an epoch keeps its history on a backend whose native channel has no
        step axis.
        """
        directory = self.log_dir or Path(self.save_dir)
        path = Path(directory) / "text" / f"{text_file_stem(name, step)}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        # A file already inside this backend's own tree is where its reader looks; copying it
        # into an `artifacts/` subdirectory of the same tree would just duplicate it (the same
        # rule `log_manifest` applies). A remote-only backend has no tree, so it uploads.
        if self.log_dir is None:
            self.log_artifact(path, artifact_dir="text")

    def log_manifest(self, manifest: "RunManifest") -> None:
        """Push the run manifest to this backend. Default: both files + the hyperparameters.

        This is the *projection* half of the manifest contract: the manifest is the union
        of everything static about the run, and every backend gets it whole, as an
        artifact, for free. A backend with native homes for more of it (a dataset-input
        API, source-control tags, a model registry) overrides this to ALSO write them
        there — never *instead*, or two runs tracked differently stop being comparable.
        See ``docs/architecture.md`` record 3.

        The ``params`` section is projected here rather than in each override because
        :meth:`log_hyperparams` is the one native slot EVERY backend has; a backend that
        only implements the abstract surface still gets its hyperparameters where its UI
        expects them.

        The manifest's SIDE ARTIFACTS go with it (:meth:`~matrainer.tracking.manifest.RunManifest.artifacts`
        — the resolved-configuration dump, the installed-package freeze). Those files hold
        formats the manifest deliberately does not re-own, so it carries only a path and a
        digest for each; uploading the manifest without them would leave a remote-only backend
        holding references to files that exist on exactly one machine, which is rule 3 of the
        contract (*no fact is ever native-only*) broken by omission. They are logged under
        their own subdirectory so ``config/final.yaml`` and ``environment/requirements.txt``
        stay apart on a backend that flattens artifact names.

        A file already inside this backend's own :attr:`log_dir` is skipped rather than
        copied next to itself — the common case, since the manifest is usually pointed at
        that directory in the first place.
        """
        manifest.write()
        params = manifest.section("params")
        if params:
            self.log_hyperparams(params)
        log_dir = self.log_dir
        for path in (manifest.path, manifest.json_path):
            if log_dir is not None and path.is_relative_to(log_dir):
                continue
            self.log_artifact(path)
        for path in manifest.artifacts():
            if log_dir is not None and path.is_relative_to(log_dir):
                continue
            self.log_artifact(path, artifact_dir=path.parent.name)

    @property
    def log_dir(self) -> Optional[Path]:
        """Local directory this backend writes to, or ``None`` for remote-only backends."""
        return None


class TrackerGroup(Tracker):
    """A composite :class:`Tracker` that fans every call out to a list of trackers.

    The multi-backend analogue of a Lightning logger *list*: log once, land in
    MLflow AND TensorBoard AND CSV. A failure in one member is logged and
    swallowed so the others still receive the call (tracking must never abort a
    run). ``experiment`` returns the list of member handles; ``log_dir`` returns
    the first member's directory that exists.

    Not ``@configurable`` for canvas discovery — it is assembled programmatically
    from a config's ``trackers:`` list, not wired as a node.

    Args:
        trackers: The member trackers to fan out to.
    """

    def __init__(self, trackers: Optional[List[Tracker]] = None) -> None:
        super().__init__()
        self.trackers: List[Tracker] = list(trackers) if trackers else []

    def _fan_out(self, method: str, *args: Any, **kwargs: Any) -> None:
        for tracker in self.trackers:
            try:
                getattr(tracker, method)(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - backend-specific failures
                logger.debug(f"{type(tracker).__name__}.{method} failed: {exc}")

    @property
    def experiment(self) -> List[Any]:
        return [t.experiment for t in self.trackers]

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        self._fan_out("log_hyperparams", params)

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        self._fan_out("log_metrics", metrics, step)

    def save(self) -> None:
        self._fan_out("save")

    def finalize(self, status: str = "success") -> None:
        self._fan_out("finalize", status)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self._fan_out("set_tags", tags)

    def log_artifact(self, path: Union[str, Path], artifact_dir: str = "") -> None:
        self._fan_out("log_artifact", path, artifact_dir)

    def log_figure(self, name: str, figure: Any, step: Optional[int] = None) -> None:
        self._fan_out("log_figure", name, figure, step)

    def log_text(self, name: str, text: str, step: Optional[int] = None) -> None:
        self._fan_out("log_text", name, text, step)

    def log_manifest(self, manifest: "RunManifest") -> None:
        self._fan_out("log_manifest", manifest)

    @property
    def log_dir(self) -> Optional[Path]:
        for tracker in self.trackers:
            d = tracker.log_dir
            if d is not None:
                return d
        return None


__all__ = ["Tracker", "TrackerGroup", "as_tracker_list", "text_file_stem"]

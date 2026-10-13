"""Model packages — a model's whole inference graph, its weights and its class names in one folder.

A checkpoint alone does not say how to run it: the model answers correctly only on inputs made the
way its training inputs were made, and its raw outputs mean something only after the steps that
turn them back into the caller's terms. Both live in a flow graph (preprocessing steps, the
predictor, postprocessing steps), and every consumer that ran the checkpoint had to rebuild that
graph by hand. A package is that graph, written once, beside the weights it loads:

* :class:`PackageModel` — a runnable (``runnable:`` of a config) that writes a package from a flow
  graph file and the weights files its predictor slots load;
* :class:`ModelPackage` — the op that runs a package on a record, one step in any flow graph;
* :func:`read_package` — the package file as plain data.

The folder::

    package.yaml        what is in it: the entry graph, every graph file and weights file with its
                        digest, the class names, the installed versions of the graph's packages
    graphs/...          (graphs: copy) the entry graph and every file it includes, in their layout
    weights/...         (weight_files: copy) each weights file
    class_names.json    (when class names are given) recordstream's LabelMap format

``copy`` freezes a package: a later edit of the source graph does not change what it runs.
``reference`` records relative paths to the sources instead, so the package follows them. The two
are chosen separately for the graphs and the weights.

The weights are NOT written into the graph: ``package.yaml`` maps each predictor slot (a dotted key
into the graph, ``flow.detect.op.model.checkpoint_path``) to its file, and :class:`ModelPackage`
fills the slot with the file's absolute path when it loads the graph. The graph therefore stays the
file its author wrote, and a package moves as one folder. Loading is confluid's own: the entry graph
included, the weight slots and the caller's ``overrides`` written after it, the last spec winning —
no second rule. Architecture rationale: ``docs/architecture.md`` ("Model packages").
"""

import copy
import os
import shutil
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, NamedTuple, Optional, Sequence, Set, Tuple, Union, get_args

import confluid
import yaml
from confluid import configurable
from confluid.fluid import Fluid
from loggair import get_logger
from recordstream.flow.subgraph import Subgraph
from recordstream.items import Record

from matrainer.artifacts import save_class_names
from matrainer.tracking.provenance import command_line, contract_roots, file_digest

logger = get_logger(__name__)

#: How a package holds a source file: ``copy`` puts it in the package (frozen), ``reference``
#: records its path relative to the package (the package follows edits of the source).
PackageFiles = Literal["copy", "reference"]
_PACKAGE_FILES: Tuple[str, ...] = get_args(PackageFiles)

#: The file that makes a folder a package; :class:`ModelPackage` reads it first.
PACKAGE_FILE = "package.yaml"
#: The layout :func:`read_package`'s readers understand. Bumped when a field changes meaning.
PACKAGE_FORMAT = 1
#: Where ``copy`` puts the graph files and the weights files inside a package.
GRAPHS_DIR = "graphs"
WEIGHTS_DIR = "weights"

#: Distributions every package names whatever its graph holds: the ones that load it.
_LOADERS = ("confluid", "recordstream", "matrainer")

__all__ = [
    "GRAPHS_DIR",
    "PACKAGE_FILE",
    "PACKAGE_FORMAT",
    "WEIGHTS_DIR",
    "LoadedPackage",
    "ModelPackage",
    "PackageFiles",
    "PackageModel",
    "load_package_graph",
    "read_package",
]


def _expand(path: str) -> Path:
    """A path as a config writes it — ``$PROJECT_ROOT/...`` or ``~/...`` — made usable."""
    return Path(os.path.expanduser(os.path.expandvars(path)))


def _absolute(path: Path) -> Path:
    """``path`` made absolute WITHOUT following symlinks (``..`` folded): the path a reader typed stays recognisable."""
    return Path(os.path.normpath(os.path.abspath(path)))


def read_package(path: Union[str, Path]) -> Dict[str, Any]:
    """The package file of the package at ``path``, as plain data (JSON-serializable).

    Args:
        path: The package folder (``$VAR`` and ``~`` expanded).

    Raises:
        ValueError: ``path`` is not a folder holding :data:`PACKAGE_FILE`, naming what is missing.

    Example::

        info = read_package("runs/signal_detection/a/package")
        info["weights"]["flow.detect.op.model.checkpoint_path"]["sha256"]
    """
    root = _absolute(_expand(str(path)))
    if not root.is_dir():
        raise ValueError(f"no package at {root}")
    package_file = root / PACKAGE_FILE
    if not package_file.is_file():
        raise ValueError(f"no package at {root}: {PACKAGE_FILE} is missing")
    info = yaml.safe_load(package_file.read_text())
    if not isinstance(info, dict):
        raise ValueError(f"{package_file} is not a package file: it holds no mapping")
    return info


class LoadedPackage(NamedTuple):
    """A package's graph, loaded the one way: ``Subgraph(steps=document["flow"], result=result)`` runs it.

    ``request`` is what ``confluid.load`` was given — the entry graph included, each weight slot, the
    caller's overrides, in that order — so a reader that needs the graph in another form (its markers
    with their file locations, through ``confluid.load(request, until="document")``) loads the same thing.
    """

    root: Path  # the package folder, absolute
    info: Dict[str, Any]  # its package file, as :func:`read_package` returns it
    request: Dict[str, Any]  # the document ``confluid.load`` was given
    document: Dict[str, Any]  # the loaded graph, its ops built: ``flow``, ``outputs``, the included blocks
    result: str  # the one step the package returns


def load_package_graph(
    path: str, overrides: Optional[Mapping[str, Any]] = None, *, owner: str = "load_package_graph"
) -> LoadedPackage:
    """Load the graph of the package at ``path`` with its weights in their slots and ``overrides`` written last.

    The ONE loader of a package: the op that runs a package and a compiler that reads one both call it,
    so the loading rule — confluid's document order, the last spec winning — is written once.

    Args:
        path: The package folder (``$VAR`` and ``~`` expanded).
        overrides: Dotted keys into the package's graph and their values, applied after it and its weight slots.
        owner: Who asks, as a refusal names it (``"ModelPackage"``).

    Raises:
        ValueError: ``path`` is blank, names no package, a package of another format, or a graph that
            returns more than one step.

    Example::

        loaded = load_package_graph("runs/a/package", {"flow.detect.op.device": "mps"})
        Subgraph(steps=loaded.document["flow"], result=loaded.result)(record)
    """
    if not path:
        raise ValueError(f"{owner}: 'path' is blank — name the package folder")
    try:
        info = read_package(path)
    except ValueError as refusal:
        raise ValueError(f"{owner}: {refusal}") from refusal
    root = _absolute(_expand(path))
    if info.get("format") != PACKAGE_FORMAT:
        raise ValueError(
            f"{owner}: {root} is a package of format {info.get('format')}, this matrainer reads format {PACKAGE_FORMAT}"
        )
    request: Dict[str, Any] = {"include": [str(_absolute(root / info["graph"]))]}
    for slot, entry in (info.get("weights") or {}).items():
        request[slot] = str(_absolute(root / entry["path"]))
    request.update(copy.deepcopy(dict(overrides or {})))
    document = confluid.load(copy.deepcopy(request))
    outputs = document.get("outputs") or ""
    if isinstance(outputs, (list, tuple)):
        if len(outputs) != 1:
            raise ValueError(f"{owner}: {root}'s graph returns {list(outputs)} — a package returns ONE step")
        outputs = outputs[0]
    return LoadedPackage(root=root, info=info, request=request, document=document, result=str(outputs))


def _targets(node: Any, found: Set[str]) -> None:
    """Every ``!class:`` target written anywhere in a document (its markers, nested in mappings and lists)."""
    if isinstance(node, Fluid):
        if isinstance(node.target, str):
            found.add(node.target)
        _targets(node.kwargs, found)
    elif isinstance(node, dict):
        for value in node.values():
            _targets(value, found)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _targets(value, found)


def _package_versions(document: Any) -> Dict[str, str]:
    """The installed version of every distribution a class of ``document`` comes from, plus the loaders.

    What a reader needs to run the package as it was made: the classes are named by import path, so
    the distributions behind their top-level modules are the code the package depends on. A top-level
    package several distributions install into (a namespace whose children ship inside it) lists them
    all: the module's own distribution cannot be told from an editable install's metadata, and a list
    that names one too many is still a list that misses none.
    """
    found: Set[str] = set()
    _targets(document, found)
    providers = metadata.packages_distributions()
    names = set(_LOADERS)
    for target in found:
        names.update(providers.get(target.split(".", 1)[0], ()))
    versions: Dict[str, str] = {}
    for name in sorted(names, key=str.lower):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:  # a top-level module whose distribution is not installed
            logger.debug(f"PackageModel: no installed distribution named {name!r}")
    return versions


@configurable(category="runnable", group="model")
class PackageModel:
    """Write a model package: a flow graph, every file it includes, its weights and its class names.

    Run as a config's ``runnable:`` (``recordstream run package.yaml``). An ``output_dir`` that
    already holds a package is replaced; one holding anything else is refused untouched — so
    pointing it at a run folder full of checkpoints by mistake costs nothing. The new package is
    written beside the old one and moved into place at the end, so a failed run leaves the old one.

    Args:
        graph: The flow graph file — a document with ``flow:`` (and ``outputs:``) — that runs the model
            on one record: its preprocessing, the predictor, its postprocessing.
        weights: Predictor slot -> the weights file it loads. A slot is a dotted key into the graph
            (``flow.detect.op.model.checkpoint_path``); the package fills it with the file's path.
        output_dir: The package folder to write.
        graphs: ``copy`` puts the graph and every file it includes into the package (an edit of the
            source later changes nothing); ``reference`` records their paths relative to the package.
        weight_files: The same choice for the weights files.
        class_names: The model's classes, id order. Written as ``class_names.json`` and listed in the
            package file; empty = the package names none.
        description: One line saying what the model is, for a reader of the package file.

    Example::

        runnable: !class:matrainer.model_package.PackageModel
          graph: config/model.yaml
          weights: {flow.detect.op.model.checkpoint_path: runs/a/checkpoint_best_total.pth}
          output_dir: ./runs/a/package
    """

    def __init__(
        self,
        graph: str = "",
        weights: Optional[Dict[str, str]] = None,
        output_dir: str = "",
        graphs: PackageFiles = "copy",
        weight_files: PackageFiles = "copy",
        class_names: Optional[List[str]] = None,
        description: str = "",
    ) -> None:
        self.graph = graph
        self.weights = weights
        self.output_dir = output_dir
        self.graphs = graphs
        self.weight_files = weight_files
        self.class_names = class_names
        self.description = description

    def run(self) -> str:
        """Write the package; returns its folder (the path a :class:`ModelPackage` takes)."""
        if not self.graph:
            raise ValueError("PackageModel: 'graph' is blank — name the flow graph file the package runs")
        if not self.output_dir:
            raise ValueError("PackageModel: 'output_dir' is blank — name the package folder to write")
        for name, value in (("graphs", self.graphs), ("weight_files", self.weight_files)):
            if value not in _PACKAGE_FILES:
                raise ValueError(f"PackageModel: {name}: {value!r} is not one of {', '.join(_PACKAGE_FILES)}")
        entry = _absolute(_expand(self.graph))
        if not entry.is_file():
            raise ValueError(f"PackageModel: the graph {self.graph} does not exist (looked for {entry})")
        written = dict(self.weights or {})
        weights = {slot: _absolute(_expand(path)) for slot, path in written.items()}
        for slot, path in weights.items():
            if not path.is_file():
                raise ValueError(f"PackageModel: the weights of {slot}, {written[slot]}, do not exist")
        if self.weight_files == "copy":
            self._refuse_shared_names(weights)
        out = _absolute(_expand(self.output_dir))
        if out.exists() and any(out.iterdir()) and not (out / PACKAGE_FILE).is_file():
            raise ValueError(
                f"PackageModel: {out} holds files and is not a package (it has no {PACKAGE_FILE}) — "
                "name an empty or new folder"
            )
        document, paths = confluid.load(str(entry), until="document", return_paths=True)
        if not isinstance(document, dict) or "flow" not in document:
            raise ValueError(f"PackageModel: the graph {entry} has no `flow:` — a package runs a flow graph")

        staging = out.with_name(out.name + ".partial")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        files, graph_path = self._place_graphs(entry, [_absolute(p) for p in paths], staging, out)
        info: Dict[str, Any] = {
            "format": PACKAGE_FORMAT,
            "description": self.description,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": command_line(),
            "graph": graph_path,
            "graphs": self.graphs,
            "weight_files": self.weight_files,
            "files": files,
            "weights": self._place_weights(weights, staging, out),
            "class_names": list(self.class_names) if self.class_names else None,
            "packages": _package_versions(document),
        }
        if self.class_names:
            save_class_names(list(self.class_names), staging)
        (staging / PACKAGE_FILE).write_text(
            _header(info["command"]) + yaml.safe_dump(info, sort_keys=False, allow_unicode=True)
        )
        if out.exists():
            shutil.rmtree(out)
        staging.rename(out)
        logger.info(f"PackageModel: wrote the package {out} ({len(files)} graph files, {len(weights)} weights files)")
        return str(out)

    @staticmethod
    def _refuse_shared_names(weights: Dict[str, Path]) -> None:
        """Copied weights land in ONE folder by their file names, so two of one name would overwrite each other."""
        seen: Dict[str, str] = {}
        for slot, path in weights.items():
            if path.name in seen:
                raise ValueError(
                    f"PackageModel: two weights files are both named {path.name} ({seen[path.name]} and {slot}) — "
                    "copied, one would replace the other; rename one or reference them (weight_files: reference)"
                )
            seen[path.name] = slot

    def _place_graphs(
        self, entry: Path, paths: Sequence[Path], staging: Path, out: Path
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Copy (or reference) every graph file; returns their entries and the entry graph's package path.

        Copied files keep their layout below the folder holding them all, so each relative ``include:``
        resolves inside the package exactly as it did at the source — no file is rewritten.
        """
        files: List[Dict[str, Any]] = []
        if self.graphs == "reference":
            for path in paths:
                files.append({"path": os.path.relpath(path, out), "sha256": file_digest(path)})
            return files, os.path.relpath(entry, out)
        common = Path(os.path.commonpath([str(path.parent) for path in paths]))
        for path in paths:
            relative = Path(GRAPHS_DIR) / path.relative_to(common)
            (staging / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, staging / relative)
            files.append(
                {
                    "path": relative.as_posix(),
                    "sha256": file_digest(staging / relative),
                    "source": contract_roots(str(path)),
                }
            )
        return files, (Path(GRAPHS_DIR) / entry.relative_to(common)).as_posix()

    def _place_weights(self, weights: Dict[str, Path], staging: Path, out: Path) -> Dict[str, Dict[str, Any]]:
        """Copy (or reference) each weights file; returns slot -> its entry (path, size, digest)."""
        placed: Dict[str, Dict[str, Any]] = {}
        for slot, path in weights.items():
            facts = {"size": path.stat().st_size, "sha256": file_digest(path)}
            if self.weight_files == "reference":
                placed[slot] = {"path": os.path.relpath(path, out), **facts}
                continue
            (staging / WEIGHTS_DIR).mkdir(exist_ok=True)
            shutil.copy2(path, staging / WEIGHTS_DIR / path.name)
            placed[slot] = {"path": f"{WEIGHTS_DIR}/{path.name}", **facts, "source": contract_roots(str(path))}
        return placed


def _header(command: Optional[str]) -> str:
    """The generated-file header of a package file: what wrote it and how to write it again."""
    how = command or "PackageModel(...).run() — called from Python, so no command line to repeat"
    rule = "# " + "=" * 73
    return (
        f"{rule}\n"
        "# AUTO-GENERATED FILE — DO NOT EDIT BY HAND\n"
        f"# Generated by: {how}\n"
        "# Source: matrainer/model_package.py (PackageModel)\n"
        f"# To regenerate: {how}   (it replaces this package)\n"
        f"{rule}\n"
    )


@configurable(category="op", group="model")
class ModelPackage:
    """Run a model package on a record: its graph's steps, with the weights inside the package.

    One op wherever a flow graph or a stream takes one. The package is read on first use (never in
    the constructor) and read again when ``path`` or ``overrides`` change. It is a SEALED op: a
    step-through tracer sees it as one step, not the package's inner steps — the package is a unit
    a caller tunes through ``overrides``, not a graph it edits.

    Args:
        path: The package folder (``$VAR`` and ``~`` expanded).
        overrides: Dotted keys into the package's graph and their values, applied AFTER it and its
            weight slots (the last spec wins): ``{flow.detect.op.device: mps}``.

    Example::

        flow:
          model:
            op: !class:matrainer.model_package.ModelPackage
              path: $PROJECT_ROOT/runs/a/package
              overrides: {flow.detect.op.model.score_threshold: 0.3}
    """

    def __init__(self, path: str = "", overrides: Optional[Dict[str, Any]] = None) -> None:
        self.path = path
        # `Any` values: an override replaces any setting of any class the package's graph holds.
        self.overrides: Dict[str, Any] = dict(overrides) if overrides else {}
        self._inner: Optional[Subgraph] = None
        self._loaded_with: Optional[Tuple[str, Dict[str, Any]]] = None

    def _subgraph(self) -> Subgraph:
        """The package's graph as a subgraph, loaded on first use and again after ``path``/``overrides`` change."""
        wanted = (self.path, self.overrides)
        if self._inner is not None and self._loaded_with == wanted:
            return self._inner
        loaded = load_package_graph(self.path, self.overrides, owner="ModelPackage")
        self._inner = Subgraph(steps=loaded.document["flow"], result=loaded.result)
        self._loaded_with = (self.path, copy.deepcopy(self.overrides))
        logger.debug(f"ModelPackage: loaded {loaded.root}")
        return self._inner

    def __call__(self, record: Record) -> Optional[Record]:
        """The package's graph run on ``record``; ``None`` = a step inside dropped it."""
        return self._subgraph()(record)

    @property
    def consumes(self) -> Any:  # the package graph's own declaration: by name or by type (Subgraph)
        """What the package's graph needs from the record (read off its steps)."""
        return self._subgraph().consumes

    @property
    def produces(self) -> Any:  # the same shape as consumes
        """What the package's graph adds to the record."""
        return self._subgraph().produces

    @property
    def flags(self) -> Tuple[str, ...]:
        """The flags the package's graph raises on the record it returns."""
        return self._subgraph().flags

    def __getstate__(self) -> Dict[str, Any]:
        """Pickled (a worker process) without the loaded graph: the worker reads the package itself."""
        state = dict(self.__dict__)
        state["_inner"] = None
        state["_loaded_with"] = None
        return state

"""The declarative sweep SPEC — the half of a sweep that belongs to the consuming project.

A project ships ``examples/sweep/sweep.yaml`` describing its MATRIX: which backends exist,
what each needs installed (``requires`` — `find_spec` probes), which model variants each runs,
any per-row environment pin (deltaid's keras row needs ``KERAS_BACKEND=tensorflow`` — a
measured per-pixel fact, not a preference), which trackers a sweep may write into, and any
per-tracker setup command (starting the workspace MLflow is the project's knowledge — this
package never names the tool that does it). Everything that is the SAME wherever a sweep runs
— probing, subprocess isolation, experiment naming, the report — is
:mod:`matrainer.sweep.runner`'s.

The spec is plain YAML, deliberately NOT a confluid document: it does not configure objects,
it points at the confluid overlay every cell runs (``config:``) and enumerates the cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

__all__ = ["FrameworkRow", "ModelVariant", "SweepSpec", "load_spec"]


@dataclass(frozen=True)
class FrameworkRow:
    """One backend row of the matrix."""

    name: str
    #: Import probes — EVERY one must resolve via ``find_spec`` or the row is skipped (reported,
    #: never fatal: a machine with three extras runs three backends and says so). Defaults to
    #: the row's own name; a backend whose distribution differs names it (``lightning`` →
    #: ``pytorch_lightning``, ``jax`` → ``flax``).
    requires: Sequence[str] = ()
    #: Environment pinned onto THIS row's subprocess (scoped there, never exported).
    env: Mapping[str, str] = field(default_factory=dict)
    #: Which model variants this backend runs, by name (see :class:`ModelVariant`). Empty means
    #: the spec has no model dimension and the row is a single cell named after the backend.
    models: Sequence[str] = ()


@dataclass(frozen=True)
class ModelVariant:
    """One entry of the optional model dimension.

    The FIRST declared variant is the config's default and appends no flags; a non-default
    variant says what reaches the command line (``["--model", "dncnn"]`` — the bound dimension
    flag the config's ``!scope:model=…`` block answers to).
    """

    name: str
    flags: Sequence[str] = ()


@dataclass(frozen=True)
class SweepSpec:
    """The whole spec, paths resolved. Built by :func:`load_spec`, never by hand-parsing."""

    #: The confluid overlay every cell runs — resolved against the spec file's directory, so a
    #: spec is invocable from anywhere in the project.
    config: Path
    frameworks: Sequence[FrameworkRow]
    models: Sequence[ModelVariant]
    trackers: Sequence[str]
    #: Default ``--max_epochs`` for every cell; ``None`` leaves the config's own value.
    epochs: Optional[int] = None
    #: Optional per-tracker setup command, run ONCE before that tracker's first cell.
    tracker_setup: Mapping[str, Sequence[str]] = field(default_factory=dict)
    #: Optional override of the per-cell command prefix. Default (``None``) is the generic
    #: ``matrainer train`` runner; a project whose CLI does pre-work the generic runner lacks
    #: names its own (``[distilaide, train]``).
    command: Optional[Sequence[str]] = None

    def framework(self, name: str) -> FrameworkRow:
        for row in self.frameworks:
            if row.name == name:
                return row
        raise KeyError(f"sweep spec declares no framework {name!r} (has: {', '.join(r.name for r in self.frameworks)})")

    def variant(self, name: str) -> ModelVariant:
        for variant in self.models:
            if variant.name == name:
                return variant
        raise KeyError(f"sweep spec declares no model variant {name!r} (has: {', '.join(v.name for v in self.models)})")


_KNOWN_KEYS = {"config", "command", "epochs", "frameworks", "models", "trackers", "tracker_setup"}
_KNOWN_FRAMEWORK_KEYS = {"name", "requires", "env", "models"}
_KNOWN_MODEL_KEYS = {"name", "flags"}


def load_spec(path: Path) -> SweepSpec:
    """Load and validate a sweep spec, LOUDLY — a typo'd key is named, never ignored.

    Silent tolerance is the wrong default for a matrix file: a misspelt ``modles:`` would
    quietly halve the sweep, and the report would look complete.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"sweep spec not found: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: a sweep spec is a mapping, got {type(raw).__name__}")
    _reject_unknown(path, raw, _KNOWN_KEYS, context="")

    config = raw.get("config")
    if not config:
        raise ValueError(f"{path}: 'config:' is required — the confluid overlay every cell runs")

    frameworks = [_framework_row(path, entry) for entry in _required_list(path, raw, "frameworks")]
    models = [_model_variant(path, entry) for entry in raw.get("models") or []]
    trackers = list(raw.get("trackers") or [])
    if not trackers:
        raise ValueError(f"{path}: 'trackers:' is required — name at least one (e.g. [csv])")

    declared = {variant.name for variant in models}
    for row in frameworks:
        unknown = [m for m in row.models if m not in declared]
        if unknown:
            raise ValueError(f"{path}: framework {row.name!r} names undeclared model variant(s) {unknown}")
        # A row naming NO variants runs the first (default) one — what a bare-string entry means.

    setup = {str(name): [str(c) for c in cmd] for name, cmd in (raw.get("tracker_setup") or {}).items()}
    command = list(raw["command"]) if raw.get("command") else None
    epochs = raw.get("epochs")
    return SweepSpec(
        config=(path.parent / str(config)).resolve(),
        frameworks=frameworks,
        models=models,
        trackers=trackers,
        epochs=int(epochs) if epochs is not None else None,
        tracker_setup=setup,
        command=command,
    )


def _required_list(path: Path, raw: Mapping[str, Any], key: str) -> List[Any]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: '{key}:' is required and must be a non-empty list")
    return value


def _framework_row(path: Path, entry: Any) -> FrameworkRow:
    if isinstance(entry, str):
        return FrameworkRow(name=entry, requires=(entry,))
    if not isinstance(entry, dict):
        raise ValueError(f"{path}: a framework entry is a name or a mapping, got {type(entry).__name__}")
    _reject_unknown(path, entry, _KNOWN_FRAMEWORK_KEYS, context="framework entry ")
    name = entry.get("name")
    if not name:
        raise ValueError(f"{path}: a framework mapping needs a 'name:'")
    return FrameworkRow(
        name=str(name),
        requires=tuple(entry.get("requires") or (str(name),)),
        env={str(k): str(v) for k, v in (entry.get("env") or {}).items()},
        models=tuple(entry.get("models") or ()),
    )


def _model_variant(path: Path, entry: Any) -> ModelVariant:
    if isinstance(entry, str):
        return ModelVariant(name=entry)
    if not isinstance(entry, dict):
        raise ValueError(f"{path}: a model entry is a name or a mapping, got {type(entry).__name__}")
    _reject_unknown(path, entry, _KNOWN_MODEL_KEYS, context="model entry ")
    name = entry.get("name")
    if not name:
        raise ValueError(f"{path}: a model mapping needs a 'name:'")
    return ModelVariant(name=str(name), flags=tuple(str(f) for f in entry.get("flags") or ()))


def _reject_unknown(path: Path, mapping: Mapping[str, Any], known: set, context: str) -> None:
    unknown = sorted(set(mapping) - known)
    if unknown:
        raise ValueError(f"{path}: unknown {context}key(s) {unknown} — known: {sorted(known)}")

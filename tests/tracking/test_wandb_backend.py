"""Tests for WandbTracker with wandb mocked via sys.modules injection."""

import sys
import types
from pathlib import Path
from typing import Any, List

import pytest

from matrainer.tracking.wandb_backend import WandbTracker


class _FakeConfig:
    def __init__(self) -> None:
        self.data: dict = {}

    def update(self, d: dict, allow_val_change: bool = False) -> None:
        self.data.update(d)


class _FakeRun:
    def __init__(self) -> None:
        self.id = "wb-1"
        self.dir = "/tmp/wb"
        self.config = _FakeConfig()
        self.logged: List[tuple] = []
        self.saved: List[str] = []
        self.used: List[Any] = []
        self.finished: Any = None

    def log(self, data: dict, step: Any = None) -> None:
        self.logged.append((data, step))

    def save(self, path: str, base_path: str = "") -> None:
        self.saved.append(path)

    def use_artifact(self, artifact: Any) -> Any:
        self.used.append(artifact)
        return artifact

    def finish(self, exit_code: int = 0) -> None:
        self.finished = exit_code


class _FakeArtifact:
    """Stands in for wandb.Artifact, enforcing the real name rule so the sanitizer is tested."""

    def __init__(self, name: str, type: str, metadata: Any = None, **kw: Any) -> None:
        import re

        if re.search(r"[^A-Za-z0-9._-]", name):
            raise ValueError(
                f"Artifact name may only contain alphanumeric characters, dashes, "
                f"underscores, and dots. Invalid name: {name}"
            )
        self.name = name
        self.type = type
        self.metadata = dict(metadata or {})
        self.references: List[str] = []

    def add_reference(self, uri: str, **kw: Any) -> Any:
        # The real library refuses an unknown scheme without an explicit entry name; the tracker
        # must never reach here with one (measured against wandb 0.28).
        if not uri.startswith(("file://", "http://", "https://", "s3://", "gs://", "az://")):
            raise ValueError("You must pass name=... when tracking references with unknown schemes.")
        # It also VALIDATES eagerly: a `file://` path not on this disk raises `ValueError`
        # ("Path ... must be a valid file or directory path"). The fake enforces that because
        # not enforcing it is what let a real defect ship — a failed reference used to skip
        # `use_artifact` too, dropping the entry from the lineage graph entirely, and a fake
        # that accepted any path could never notice.
        if uri.startswith("file://") and not Path(uri[len("file://") :]).exists():
            raise ValueError(f"Path '{uri}' must be a valid file or directory path")
        self.references.append(uri)
        return []


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> Any:
    mod = types.ModuleType("wandb")
    runs: List[_FakeRun] = []

    def init(**kwargs: Any) -> _FakeRun:
        r = _FakeRun()
        runs.append(r)
        return r

    mod.init = init  # type: ignore[attr-defined]
    mod.Artifact = _FakeArtifact  # type: ignore[attr-defined]
    mod.runs = runs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", mod)
    return mod


def test_import_error_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_require(module: str, extra: str, feature: str = "") -> Any:
        raise ImportError(f"pip install matrainer[{extra}]")

    monkeypatch.setattr("matrainer.tracking.wandb_backend.require", fake_require)
    t = WandbTracker()
    with pytest.raises(ImportError, match=r"pip install matrainer\[wandb\]"):
        _ = t.experiment


def test_run_created_lazily(fake_wandb: Any) -> None:
    t = WandbTracker(experiment_name="proj", run_name="r")
    assert not fake_wandb.runs
    _ = t.experiment
    assert len(fake_wandb.runs) == 1
    assert t.version == "wb-1"


def test_log_metrics_and_hparams(fake_wandb: Any) -> None:
    t = WandbTracker(run_name="r")
    t.log_hyperparams({"lr": 0.1})
    t.log_metrics({"loss": 0.4}, step=2)
    run = fake_wandb.runs[0]
    assert run.config.data == {"lr": 0.1}
    assert run.logged == [({"loss": 0.4}, 2)]


def test_finalize_finishes_run(fake_wandb: Any) -> None:
    t = WandbTracker(run_name="r")
    _ = t.experiment
    t.finalize("failed")
    assert fake_wandb.runs[0].finished == 1


# --- artifact lineage --------------------------------------------------------------------------


def _manifest_with_lineage(tmp_path: Any) -> Any:
    from matrainer.tracking.manifest import RunManifest

    m = RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    m.record_dataset(kind="hf", name="ylecun/mnist", uri="hf://datasets/ylecun/mnist?split=train", split="train")
    m.record_model(
        source="hf", name="timm/resnet50.a1_in1k", uri="hf://models/timm/resnet50.a1_in1k", role="pretrained"
    )
    ckpt = Path(tmp_path) / "last.ckpt"
    ckpt.write_text("weights")
    m.record_model(source="file", name=str(ckpt), uri=f"file://{ckpt}", role="checkpoint")
    return m


def test_datasets_and_models_are_marked_used_with_their_own_types(fake_wandb: Any, tmp_path: Any) -> None:
    """Marking an artifact USED is what puts the run on W&B's lineage graph."""
    WandbTracker().log_manifest(_manifest_with_lineage(tmp_path))
    used = fake_wandb.runs[0].used
    assert [a.type for a in used] == ["dataset", "model", "model"]
    assert [a.name for a in used][:2] == ["ylecun_mnist", "timm_resnet50.a1_in1k"]
    # The checkpoint's name derives from its absolute path, so assert the SHAPE rather than a
    # tmp-dir literal: sanitised, and still ending in the file it names.
    assert used[2].name.endswith("last.ckpt") and "/" not in used[2].name


def test_the_artifact_name_is_sanitized_but_the_identity_survives_in_metadata(fake_wandb: Any, tmp_path: Any) -> None:
    """W&B names allow no `/`, which every real dataset and model identifier contains."""
    WandbTracker().log_manifest(_manifest_with_lineage(tmp_path))
    dataset = fake_wandb.runs[0].used[0]
    assert "/" not in dataset.name
    assert dataset.metadata["uri"] == "hf://datasets/ylecun/mnist?split=train"
    assert dataset.metadata["name"] == "ylecun/mnist"


def test_an_addressable_uri_becomes_a_reference(fake_wandb: Any, tmp_path: Any) -> None:
    """A reference records WHERE the bytes live without uploading them."""
    WandbTracker().log_manifest(_manifest_with_lineage(tmp_path))
    checkpoint = fake_wandb.runs[0].used[2]
    assert checkpoint.references == [f"file://{Path(tmp_path) / 'last.ckpt'}"]


def test_a_zoo_scheme_is_metadata_only_rather_than_a_rejected_reference(fake_wandb: Any, tmp_path: Any) -> None:
    """`add_reference` refuses a scheme it cannot address; the tracker must not hand it one."""
    WandbTracker().log_manifest(_manifest_with_lineage(tmp_path))
    hub_model = fake_wandb.runs[0].used[1]
    assert hub_model.references == []
    assert hub_model.metadata["uri"] == "hf://models/timm/resnet50.a1_in1k"


def test_a_manifest_with_no_lineage_marks_nothing(fake_wandb: Any, tmp_path: Any) -> None:
    from matrainer.tracking.manifest import RunManifest

    WandbTracker().log_manifest(RunManifest(run_dir=tmp_path / "run", handle_interrupts=False))
    assert fake_wandb.runs[0].used == []


def test_a_failing_artifact_does_not_break_the_run(fake_wandb: Any, tmp_path: Any) -> None:
    """Tracking must never abort a run — a lost lineage link beats a dead training job."""

    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("wandb said no")

    fake_wandb.Artifact = boom
    WandbTracker().log_manifest(_manifest_with_lineage(tmp_path))  # no raise
    assert fake_wandb.runs[0].used == []


def test_the_manifest_artifact_is_still_uploaded(fake_wandb: Any, tmp_path: Any) -> None:
    """Rule 3: the native slots are ADDITIONAL, never instead of the union artifact."""
    manifest = _manifest_with_lineage(tmp_path)
    WandbTracker().log_manifest(manifest)
    assert any(name.endswith("manifest.yaml") for name in fake_wandb.runs[0].saved)


def test_an_unresolvable_reference_still_reaches_the_lineage_graph(fake_wandb: Any, tmp_path: Any) -> None:
    """A failed `add_reference` must degrade to metadata-only, never drop the artifact.

    `add_reference` VALIDATES eagerly — a `file://` path not on this disk raises, and an
    `http(s)://` URI is fetched (a 404 raises). That call used to share the caller's `try` with
    `use_artifact`, so a checkpoint whose file had moved vanished from the lineage graph
    entirely instead of appearing as the metadata record it still is.
    """
    from matrainer.tracking.manifest import RunManifest

    m = RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    m.record_model(source="file", name="gone.ckpt", uri="file:///nowhere/gone.ckpt", role="checkpoint")
    # Even with raise_errors on: a reference is an optimisation on top of the artifact, never a failure.
    WandbTracker(raise_errors=True).log_manifest(m)

    used = fake_wandb.runs[0].used
    assert [a.name for a in used] == ["gone.ckpt"], "the entry was dropped instead of degrading"
    assert used[0].references == []
    assert used[0].metadata["uri"] == "file:///nowhere/gone.ckpt"


def test_a_portable_uri_is_expanded_before_it_is_referenced(
    fake_wandb: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`file://$MODEL_ROOT/...` must resolve to the bytes on THIS disk.

    The manifest is portable: with `MODEL_ROOT` set, `section()` contracts a checkpoint's URI to
    `file://$MODEL_ROOT/last.ckpt`, which is deliberately not a path anywhere. Referencing it
    verbatim could therefore NEVER resolve on a machine that has the roots configured — which is
    every machine that sets them. `expandvars` is the documented exact inverse, so the reference
    names the real file while the metadata keeps the portable spelling.
    """
    from matrainer.tracking.manifest import RunManifest

    root = Path(tmp_path) / "models"
    root.mkdir()
    (root / "last.ckpt").write_text("weights")
    monkeypatch.setenv("MODEL_ROOT", str(root))

    m = RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    m.record_model(source="file", name="last.ckpt", uri=f"file://{root}/last.ckpt", role="checkpoint")
    WandbTracker().log_manifest(m)

    artifact = fake_wandb.runs[0].used[0]
    assert artifact.metadata["uri"] == "file://$MODEL_ROOT/last.ckpt", "the manifest must stay portable"
    assert artifact.references == [f"file://{root}/last.ckpt"], "the reference must name the real file"


# --- raise_errors: a refused call is a debug line, and raised as well when asked ----------------

SAID_NO = RuntimeError("wandb said no")


def _said_no(*args: Any, **kwargs: Any) -> Any:
    raise SAID_NO


def _file(tmp_path: Path) -> Path:
    path = tmp_path / "a.txt"
    path.write_text("A")
    return path


#: (run method refused, the tracker call that sends one request to it)
_ONE_REQUEST = {
    "save": lambda t, tmp: t.log_artifact(_file(tmp)),
    "log": lambda t, tmp: t.log_figure("val/cm", object()),
    "finish": lambda t, tmp: t.finalize("success"),
}


@pytest.mark.parametrize("raise_errors", [False, True])
@pytest.mark.parametrize("method", sorted(_ONE_REQUEST))
def test_a_refused_request_is_raised_only_when_asked(
    fake_wandb: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, raise_errors: bool
) -> None:
    tracker = WandbTracker(raise_errors=raise_errors)
    monkeypatch.setattr(tracker.experiment, method, _said_no)
    if raise_errors:
        with pytest.raises(RuntimeError, match="wandb said no"):
            _ONE_REQUEST[method](tracker, tmp_path)
    else:
        _ONE_REQUEST[method](tracker, tmp_path)


@pytest.mark.parametrize("raise_errors", [False, True])
def test_a_refused_finish_still_lets_go_of_the_run(
    fake_wandb: Any, monkeypatch: pytest.MonkeyPatch, raise_errors: bool
) -> None:
    tracker = WandbTracker(raise_errors=raise_errors)
    monkeypatch.setattr(tracker.experiment, "finish", _said_no)
    try:
        tracker.finalize("success")
    except RuntimeError:
        assert raise_errors
    assert tracker._run is None


@pytest.mark.parametrize("raise_errors", [False, True])
def test_a_refused_lineage_entry_does_not_stop_the_others(
    fake_wandb: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raise_errors: bool
) -> None:
    tracker = WandbTracker(raise_errors=raise_errors)
    run = tracker.experiment
    using = run.use_artifact

    def use(artifact: Any) -> Any:
        if artifact.type == "dataset":
            raise SAID_NO
        return using(artifact)

    monkeypatch.setattr(run, "use_artifact", use)
    if raise_errors:
        with pytest.raises(RuntimeError, match="wandb said no"):
            tracker.log_manifest(_manifest_with_lineage(tmp_path))
    else:
        tracker.log_manifest(_manifest_with_lineage(tmp_path))
    assert [a.type for a in run.used] == ["model", "model"]

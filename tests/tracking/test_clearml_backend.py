"""Tests for ClearMLTracker — mocked for the mapping, real offline ClearML for the round trip."""

import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pytest

from matrainer.tracking.clearml_backend import HPARAMS_SECTION, ClearMLTracker


class _FakeLogger:
    def __init__(self) -> None:
        self.scalars: List[tuple] = []
        self.figures: List[tuple] = []

    def report_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        self.scalars.append((title, series, value, iteration))

    def report_matplotlib_figure(self, title: str, series: str, figure: Any, iteration: int) -> None:
        self.figures.append((title, figure, iteration))


class _FakeTask:
    """Records calls; stands in for ``clearml.Task``."""

    instances: List["_FakeTask"] = []
    offline_calls: List[bool] = []
    offline: bool = False

    def __init__(self, **kw: Any) -> None:
        self.kwargs = kw
        self.id = "task-1"
        self.connected: Dict[str, Any] = {}
        self.configurations: List[tuple] = []
        self.tags: List[str] = list(kw.get("tags") or [])
        self.artifacts: List[tuple] = []
        self.logger = _FakeLogger()
        self.completed = False
        self.failed: Optional[str] = None
        self.closed = False
        _FakeTask.instances.append(self)

    # -- the classmethod surface matrainer uses -------------------------------------------
    @classmethod
    def init(cls, **kw: Any) -> "_FakeTask":
        return cls(**kw)

    @classmethod
    def set_offline(cls, value: bool) -> None:
        cls.offline_calls.append(value)

    @classmethod
    def is_offline(cls) -> bool:
        return cls.offline

    # -- the instance surface -------------------------------------------------------------
    def connect(self, mutable: Any, name: Optional[str] = None) -> Any:
        # The real `Task.connect` takes an OBJECT as readily as a mapping (an InputModel is the
        # case that matters here), so the fake must not assume it can be dict()-ed.
        self.connected[str(name)] = dict(mutable) if isinstance(mutable, Mapping) else mutable
        return mutable

    def connect_configuration(self, configuration: Any, name: Optional[str] = None) -> Any:
        self.configurations.append((name, configuration))
        return configuration

    def add_tags(self, tags: List[str]) -> None:
        self.tags.extend(tags)

    def upload_artifact(self, name: str, artifact_object: Any) -> None:
        self.artifacts.append((name, str(artifact_object)))

    def get_logger(self) -> _FakeLogger:
        return self.logger

    def mark_completed(self) -> None:
        self.completed = True

    def mark_failed(self, status_reason: Optional[str] = None) -> None:
        self.failed = status_reason

    def close(self) -> None:
        self.closed = True


class _FakeInputModel:
    """Stands in for ``clearml.InputModel``; records what a server would have registered."""

    imported: List[Dict[str, Any]] = []

    def __init__(self, **kw: Any) -> None:
        self.id = f"model-{len(_FakeInputModel.imported)}"
        self.kwargs = kw

    @classmethod
    def import_model(cls, **kw: Any) -> "_FakeInputModel":
        model = cls(**kw)
        cls.imported.append(kw)
        return model


@pytest.fixture
def fake_clearml(monkeypatch: pytest.MonkeyPatch) -> Any:
    _FakeTask.instances = []
    _FakeTask.offline_calls = []
    _FakeTask.offline = False
    _FakeInputModel.imported = []
    mod = types.ModuleType("clearml")
    mod.Task = _FakeTask  # type: ignore[attr-defined]
    mod.InputModel = _FakeInputModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "clearml", mod)
    return _FakeTask


def test_import_error_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_require(module: str, extra: str, feature: str = "") -> Any:
        raise ImportError(f"{feature} requires the '{module}' package. Install with: pip install matrainer[{extra}]")

    monkeypatch.setattr("matrainer.tracking.clearml_backend.require", fake_require)
    with pytest.raises(ImportError, match=r"pip install matrainer\[clearml\]"):
        _ = ClearMLTracker().experiment


def test_task_is_created_lazily(fake_clearml: Any) -> None:
    t = ClearMLTracker(experiment_name="proj", run_name="r")
    assert not fake_clearml.instances  # nothing built at construction
    task = t.experiment
    assert len(fake_clearml.instances) == 1
    assert task.kwargs["project_name"] == "proj" and task.kwargs["task_name"] == "r"


def test_frameworks_are_never_auto_connected(fake_clearml: Any) -> None:
    """ClearML otherwise monkey-patches torch/matplotlib/argparse in the measured process."""
    assert ClearMLTracker().experiment.kwargs["auto_connect_frameworks"] is False


def test_resource_monitoring_follows_the_flag(fake_clearml: Any) -> None:
    assert ClearMLTracker(log_system_metrics=True).experiment.kwargs["auto_resource_monitoring"] is True
    assert ClearMLTracker(log_system_metrics=False).experiment.kwargs["auto_resource_monitoring"] is False


def test_offline_is_requested_before_init(fake_clearml: Any) -> None:
    ClearMLTracker(offline=True).experiment
    assert fake_clearml.offline_calls == [True]


def test_offline_is_not_touched_by_default(fake_clearml: Any) -> None:
    """A default tracker must not flip a process-wide ClearML mode nobody asked for."""
    ClearMLTracker().experiment
    assert fake_clearml.offline_calls == []


def test_version_is_the_task_id(fake_clearml: Any) -> None:
    t = ClearMLTracker(run_name="r")
    assert t.version == "r"  # before the task exists
    _ = t.experiment
    assert t.version == "task-1"


def test_log_hyperparams_merges_across_calls(fake_clearml: Any) -> None:
    t = ClearMLTracker()
    t.log_hyperparams({"lr": 0.1})
    t.log_hyperparams({"epochs": 3})
    assert fake_clearml.instances[0].connected[HPARAMS_SECTION] == {"lr": 0.1, "epochs": 3}


def test_log_metrics_splits_the_key_into_title_and_series(fake_clearml: Any) -> None:
    """`train/loss` becomes ClearML's `train` graph with a `loss` line — how its UI groups."""
    ClearMLTracker().log_metrics({"train/loss": 0.5}, step=4)
    assert fake_clearml.instances[0].logger.scalars == [("train", "loss", 0.5, 4)]


def test_an_unprefixed_metric_titles_itself(fake_clearml: Any) -> None:
    ClearMLTracker().log_metrics({"accuracy": 0.9})
    assert fake_clearml.instances[0].logger.scalars == [("accuracy", "accuracy", 0.9, 0)]


def test_constructor_tags_reach_task_init(fake_clearml: Any) -> None:
    ClearMLTracker(tags={"stage": "smoke"}).experiment
    assert fake_clearml.instances[0].kwargs["tags"] == ["stage=smoke"]


def test_set_tags_before_the_task_exists_does_not_double_add(fake_clearml: Any) -> None:
    """Same ordering trap as the Aim backend: task creation applies self.tags."""
    t = ClearMLTracker()
    assert not fake_clearml.instances
    t.set_tags({"stage": "smoke"})
    assert fake_clearml.instances[0].tags == ["stage=smoke"]


def test_constructor_and_later_tags_both_land_once(fake_clearml: Any) -> None:
    t = ClearMLTracker(tags={"a": "1"})
    t.set_tags({"b": "2"})
    assert sorted(fake_clearml.instances[0].tags) == ["a=1", "b=2"]


def test_log_artifact_namespaces_by_directory(fake_clearml: Any, tmp_path: Path) -> None:
    """ClearML files artifacts by NAME, so same-named files in two dirs would overwrite."""
    f = tmp_path / "train.yaml"
    f.write_text("x: 1")
    t = ClearMLTracker()
    t.log_artifact(f, "config")
    t.log_artifact(f, "logs")
    assert [name for name, _ in fake_clearml.instances[0].artifacts] == [
        "config/train.yaml",
        "logs/train.yaml",
    ]


def test_log_artifact_skips_a_missing_file(fake_clearml: Any, tmp_path: Path) -> None:
    ClearMLTracker().log_artifact(tmp_path / "gone.yaml")
    assert fake_clearml.instances == []  # never even opened the task


def test_log_figure_reports_a_matplotlib_figure(fake_clearml: Any) -> None:
    ClearMLTracker().log_figure("cm", "FIGURE", step=2)
    assert fake_clearml.instances[0].logger.figures == [("cm", "FIGURE", 2)]


def test_log_text_uploads_an_artifact_rather_than_reporting_to_the_console(fake_clearml: Any) -> None:
    """`report_text` writes to the task's console stream, where a per-epoch table is unusable.

    An artifact is addressable by name and comparable between tasks; the step is IN the name
    because ClearML files artifacts by name and would otherwise keep only the last one.
    """
    ClearMLTracker().log_text("val/confusion", "|a|b|", step=2)

    assert fake_clearml.instances[0].artifacts == [("text/val_confusion_2.md", "|a|b|")]


def test_finalize_completes_or_fails(fake_clearml: Any) -> None:
    ok = ClearMLTracker()
    ok.experiment
    ok.finalize("success")
    bad = ClearMLTracker()
    bad.experiment
    bad.finalize("interrupted")
    assert fake_clearml.instances[0].completed and fake_clearml.instances[0].closed
    assert fake_clearml.instances[1].failed == "interrupted"


def test_finalize_without_a_task_is_a_noop(fake_clearml: Any) -> None:
    ClearMLTracker().finalize("success")
    assert fake_clearml.instances == []


def test_log_dir_is_none(fake_clearml: Any) -> None:
    """Server-backed: the manifest must be UPLOADED, not assumed to be in a local dir."""
    assert ClearMLTracker().log_dir is None


# --------------------------------------------------------------------------- #
# manifest projection
# --------------------------------------------------------------------------- #


def build_manifest(tmp_path: Path) -> Any:
    from matrainer.tracking.manifest import RunManifest

    m = RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    m.set("params", {"lr": 0.1})
    m.record_dataset(kind="hf", name="org/set", split="train", count=12)
    m.record_model(source="zoo", name="efficientvit_b0")
    return m


def test_log_manifest_connects_the_manifest_as_a_configuration(fake_clearml: Any, tmp_path: Path) -> None:
    """ClearML's Configuration tab shows it inline and diffs it between tasks."""
    t = ClearMLTracker()
    t.log_manifest(build_manifest(tmp_path))
    names = [name for name, _ in fake_clearml.instances[0].configurations]
    assert "manifest" in names


def test_log_manifest_connects_datasets_and_models_as_sections(fake_clearml: Any, tmp_path: Path) -> None:
    t = ClearMLTracker()
    t.log_manifest(build_manifest(tmp_path))
    connected = fake_clearml.instances[0].connected
    assert connected["datasets"]["0.name"] == "org/set"
    assert connected["datasets"]["0.split"] == "train"
    assert connected["models"]["0.name"] == "efficientvit_b0"


def test_log_manifest_skips_empty_sections(fake_clearml: Any, tmp_path: Path) -> None:
    from matrainer.tracking.manifest import RunManifest

    t = ClearMLTracker()
    t.log_manifest(RunManifest(run_dir=tmp_path / "run", handle_interrupts=False))
    connected = fake_clearml.instances[0].connected
    assert "datasets" not in connected and "models" not in connected


def test_log_manifest_still_does_the_generic_projection(fake_clearml: Any, tmp_path: Path) -> None:
    """Native slots are ADDITIONAL — the artifact and the params must land regardless."""
    t = ClearMLTracker()
    t.log_manifest(build_manifest(tmp_path))
    task = fake_clearml.instances[0]
    assert task.connected[HPARAMS_SECTION]["lr"] == 0.1
    assert sorted(name for name, _ in task.artifacts) == ["manifest.json", "manifest.yaml"]


# --------------------------------------------------------------------------- #
# real ClearML
# --------------------------------------------------------------------------- #


def test_round_trip_against_real_clearml_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Integration: only real ClearML proves the API accepts what we build.

    Runs in OFFLINE mode, which needs no server and no credentials — the same mode the
    backend exposes so a laptop can use it before anyone stands one up.

    The mode is undone by letting ``monkeypatch`` restore ``CLEARML_OFFLINE_MODE``, NOT by
    calling ``Task.set_offline(False)``: turning it back OFF validates the ClearML config and
    raises ``MissingConfigError`` on an unconfigured machine, so the obvious cleanup fails the
    test after its body has already passed. (Measured — that is exactly how this test first
    failed.)
    """
    pytest.importorskip("clearml")
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
    cfg = tmp_path / "train.yaml"
    cfg.write_text("lr: 0.1")

    tracker = ClearMLTracker(experiment_name="matrainer-itest", run_name="r", offline=True, log_system_metrics=False)
    tracker.log_hyperparams({"lr": 0.1})
    tracker.log_hyperparams({"epochs": 5})  # merged, not replaced
    tracker.log_metrics({"train/loss": 0.5}, step=0)
    tracker.set_tags({"stage": "smoke"})
    tracker.log_artifact(cfg, "config")
    tracker.log_manifest(build_manifest(tmp_path))
    assert tracker.version and tracker.version.startswith("offline-")
    tracker.finalize("success")


# --- model lineage (InputModel) ------------------------------------------------------------------


def _manifest_with_models(tmp_path: Any) -> Any:
    from matrainer.tracking.manifest import RunManifest

    m = RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    m.add(
        "models",
        {
            "kind": "hf",
            "name": "timm/resnet50.a1_in1k",
            "uri": "hf://models/timm/resnet50.a1_in1k",
            "role": "pretrained",
            "slot": "model",
        },
    )
    m.add(
        "models",
        {"kind": "file", "name": "/w/last.ckpt", "uri": "file:///w/last.ckpt", "role": "checkpoint", "slot": "model"},
    )
    return m


def test_models_are_registered_as_input_models(fake_clearml: Any, tmp_path: Any) -> None:
    """The Models tab and the lineage graph, which a flat key/value section cannot reach."""
    ClearMLTracker(log_system_metrics=False).log_manifest(_manifest_with_models(tmp_path))
    imported = _FakeInputModel.imported
    # The `hf://` entry is deliberately absent — ClearML cannot address that scheme and would
    # store a CWD-relative path for it (see the zoo-URI test below).
    assert [m["weights_url"] for m in imported] == ["file:///w/last.ckpt"]
    assert imported[0]["name"] == "/w/last.ckpt"
    assert imported[0]["framework"] == "file"  # `kind`; the manual `record_model` writes `source`
    assert imported[0]["tags"] == ["checkpoint"]
    assert imported[0]["comment"] == "checkpoint"


def test_the_import_is_by_url_so_no_weights_are_uploaded(fake_clearml: Any, tmp_path: Any) -> None:
    ClearMLTracker(log_system_metrics=False).log_manifest(_manifest_with_models(tmp_path))
    assert all("weights_url" in m for m in _FakeInputModel.imported)
    assert not any("weights" in m and m.get("weights") for m in _FakeInputModel.imported)


def test_a_second_log_manifest_does_not_register_the_models_twice(fake_clearml: Any, tmp_path: Any) -> None:
    """`log_manifest` runs at least twice per run; a re-import would be a second entry."""
    tracker = ClearMLTracker(log_system_metrics=False)
    manifest = _manifest_with_models(tmp_path)
    tracker.log_manifest(manifest)
    tracker.log_manifest(manifest)
    assert len(_FakeInputModel.imported) == 1  # the addressable one, registered once


def test_offline_skips_registration_but_keeps_the_comparable_section(fake_clearml: Any, tmp_path: Any) -> None:
    """`import_model` needs a server — measured: it dies offline with an AttributeError.

    So the flat hyperparameter section is not merely a fallback, it is the ONLY projection an
    offline run gets, which is why it stays rather than being replaced.
    """
    _FakeTask.offline = True
    tracker = ClearMLTracker(offline=True, log_system_metrics=False)
    tracker.log_manifest(_manifest_with_models(tmp_path))
    assert _FakeInputModel.imported == []
    assert "models" in _FakeTask.instances[0].connected  # the section still landed


def test_a_manifest_with_no_models_registers_nothing(fake_clearml: Any, tmp_path: Any) -> None:
    from matrainer.tracking.manifest import RunManifest

    ClearMLTracker(log_system_metrics=False).log_manifest(
        RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    )
    assert _FakeInputModel.imported == []


def test_a_failing_registration_does_not_break_the_run(
    fake_clearml: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tracking must never abort a run — a lost lineage link beats a dead training job."""

    def boom(**kwargs: Any) -> Any:
        raise RuntimeError("server said no")

    monkeypatch.setattr(_FakeInputModel, "import_model", boom)
    ClearMLTracker(log_system_metrics=False).log_manifest(_manifest_with_models(tmp_path))  # no raise


def test_the_weights_url_is_expanded_so_it_can_actually_resolve(
    fake_clearml: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`manifest.section` returns the PORTABLE spelling — `file://$MODEL_ROOT/last.ckpt`.

    A tag may stay portable (a person reads it), but `weights_url` is meant to RESOLVE, so it
    needs the bytes on this disk. Without the expansion ClearML would be handed a path containing
    a literal `$MODEL_ROOT`, which exists nowhere — the same defect the W&B reference had.
    """
    from matrainer.tracking.manifest import RunManifest

    weights = tmp_path / "last.ckpt"
    weights.write_text("w")
    monkeypatch.setenv("MODEL_ROOT", str(tmp_path))

    m = RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    m.add("models", {"kind": "file", "name": str(weights), "uri": weights.as_uri(), "role": "checkpoint"})
    assert "$MODEL_ROOT" in m.section("models")[0]["uri"]  # the section really is contracted

    ClearMLTracker(log_system_metrics=False).log_manifest(m)
    imported = _FakeInputModel.imported[0]
    assert "$MODEL_ROOT" not in imported["weights_url"]
    assert imported["weights_url"] == weights.as_uri()
    # The TAG keeps the portable spelling — that is the whole point of contracting it.
    assert "$MODEL_ROOT" in str(_FakeTask.instances[0].connected["models"]["0.uri"])


def test_a_zoo_uri_is_not_registered_as_an_input_model(fake_clearml: Any, tmp_path: Any) -> None:
    """ClearML REWRITES an unaddressable scheme instead of rejecting it — so it must not see one.

    Measured against a live clearml 2.1 server: `StorageHelper.conform_url` treats an unknown
    scheme as a relative path and resolves it against the CWD, so `hf://models/timm/x` was
    registered with `url=file://<cwd>/hf:/models/timm/x` — a local file that does not exist,
    pointing wherever the run happened to start. A broken link in the Models tab is worse than
    no link: it looks like a right answer.

    The model is not lost. It stays in the `models` hyperparameter section, which is exactly why
    that section is kept beside this registration rather than replaced by it.
    """
    from matrainer.tracking.manifest import RunManifest

    weights = tmp_path / "last.ckpt"
    weights.write_text("w")
    m = RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    m.record_model(source="hf", name="timm/x.r224", uri="hf://models/timm/x.r224", role="pretrained")
    m.record_model(source="file", name=str(weights), uri=weights.as_uri(), role="checkpoint")

    ClearMLTracker(log_system_metrics=False).log_manifest(m)

    assert [i["weights_url"] for i in _FakeInputModel.imported] == [weights.as_uri()]
    section = _FakeTask.instances[0].connected["models"]
    assert section["0.uri"] == "hf://models/timm/x.r224", "the zoo model must stay in the section"
    assert section["1.uri"] == weights.as_uri()


# --- raise_errors: a refused call is a debug line, and raised as well when asked ----------------

SAID_NO = RuntimeError("clearml said no")


def _said_no(*args: Any, **kwargs: Any) -> Any:
    raise SAID_NO


def _file(tmp_path: Path) -> Path:
    path = tmp_path / "a.txt"
    path.write_text("A")
    return path


#: (where the refusal is put, its method, the tracker call that sends one request there)
_ONE_REQUEST = [
    ("task", "connect", lambda t, tmp: t.log_hyperparams({"lr": 0.1})),
    ("task", "add_tags", lambda t, tmp: t.set_tags({"stage": "smoke"})),
    ("task", "upload_artifact", lambda t, tmp: t.log_artifact(_file(tmp))),
    ("task", "upload_artifact", lambda t, tmp: t.log_text("val/report", "|a|")),
    ("logger", "report_matplotlib_figure", lambda t, tmp: t.log_figure("val/cm", object())),
    ("task", "close", lambda t, tmp: t.finalize("success")),
]


@pytest.mark.parametrize("raise_errors", [False, True])
@pytest.mark.parametrize(
    "where, method, call", _ONE_REQUEST, ids=[f"{m}-{i}" for i, (_, m, _) in enumerate(_ONE_REQUEST)]
)
def test_a_refused_request_is_raised_only_when_asked(
    fake_clearml: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    where: str,
    method: str,
    call: Any,
    raise_errors: bool,
) -> None:
    tracker = ClearMLTracker(log_system_metrics=False, raise_errors=raise_errors)
    task = tracker.experiment
    monkeypatch.setattr(task if where == "task" else task.logger, method, _said_no)
    if raise_errors:
        with pytest.raises(RuntimeError, match="clearml said no"):
            call(tracker, tmp_path)
    else:
        call(tracker, tmp_path)


@pytest.mark.parametrize("raise_errors", [False, True])
def test_a_refused_metric_does_not_stop_the_others(
    fake_clearml: Any, monkeypatch: pytest.MonkeyPatch, raise_errors: bool
) -> None:
    tracker = ClearMLTracker(log_system_metrics=False, raise_errors=raise_errors)
    reporter = tracker.experiment.logger
    reporting = reporter.report_scalar

    def report(**kwargs: Any) -> None:
        if kwargs["series"] == "b":
            raise SAID_NO
        reporting(**kwargs)

    monkeypatch.setattr(reporter, "report_scalar", report)
    if raise_errors:
        with pytest.raises(RuntimeError, match="clearml said no"):
            tracker.log_metrics({"train/a": 1.0, "train/b": 2.0, "train/c": 3.0}, step=1)
    else:
        tracker.log_metrics({"train/a": 1.0, "train/b": 2.0, "train/c": 3.0}, step=1)
    assert [series for _, series, _, _ in reporter.scalars] == ["a", "c"]


@pytest.mark.parametrize("raise_errors", [False, True])
def test_a_refused_configuration_does_not_stop_the_sections(
    fake_clearml: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raise_errors: bool
) -> None:
    tracker = ClearMLTracker(log_system_metrics=False, raise_errors=raise_errors)
    task = tracker.experiment
    monkeypatch.setattr(task, "connect_configuration", _said_no)
    if raise_errors:
        with pytest.raises(RuntimeError, match="clearml said no"):
            tracker.log_manifest(build_manifest(tmp_path))
    else:
        tracker.log_manifest(build_manifest(tmp_path))
    assert {"datasets", "models"} <= set(task.connected)


def _manifest_with_two_files(tmp_path: Any) -> Any:
    from matrainer.tracking.manifest import RunManifest

    m = RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    for name in ("a", "b"):
        m.add("models", {"kind": "file", "name": f"/w/{name}.ckpt", "uri": f"file:///w/{name}.ckpt", "role": name})
    return m


@pytest.mark.parametrize("raise_errors", [False, True])
def test_a_refused_registration_does_not_stop_the_others(
    fake_clearml: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raise_errors: bool
) -> None:
    importing = _FakeInputModel.import_model

    def import_model(**kwargs: Any) -> Any:
        if kwargs["weights_url"] == "file:///w/a.ckpt":
            raise SAID_NO
        return importing(**kwargs)

    monkeypatch.setattr(_FakeInputModel, "import_model", import_model)
    tracker = ClearMLTracker(log_system_metrics=False, raise_errors=raise_errors)
    if raise_errors:
        with pytest.raises(RuntimeError, match="clearml said no"):
            tracker.log_manifest(_manifest_with_two_files(tmp_path))
    else:
        tracker.log_manifest(_manifest_with_two_files(tmp_path))
    assert [m["weights_url"] for m in _FakeInputModel.imported] == ["file:///w/b.ckpt"]


@pytest.mark.parametrize("raise_errors", [False, True])
def test_a_refused_model_link_keeps_the_registration(
    fake_clearml: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raise_errors: bool
) -> None:
    tracker = ClearMLTracker(log_system_metrics=False, raise_errors=raise_errors)
    task = tracker.experiment
    connecting = task.connect

    def connect(mutable: Any, name: Optional[str] = None) -> Any:
        if str(name).startswith("model:"):
            raise SAID_NO
        return connecting(mutable, name=name)

    monkeypatch.setattr(task, "connect", connect)
    if raise_errors:
        with pytest.raises(RuntimeError, match="clearml said no"):
            tracker.log_manifest(_manifest_with_models(tmp_path))
    else:
        tracker.log_manifest(_manifest_with_models(tmp_path))
    assert list(tracker._input_models) == ["file:///w/last.ckpt"]  # a retry will not register it again


@pytest.mark.parametrize("raise_errors", [False, True])
def test_a_refused_section_does_not_stop_the_others(
    fake_clearml: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raise_errors: bool
) -> None:
    tracker = ClearMLTracker(log_system_metrics=False, raise_errors=raise_errors)
    task = tracker.experiment
    connecting = task.connect

    def connect(mutable: Any, name: Optional[str] = None) -> Any:
        if name == "datasets":
            raise SAID_NO
        return connecting(mutable, name=name)

    monkeypatch.setattr(task, "connect", connect)
    if raise_errors:
        with pytest.raises(RuntimeError, match="clearml said no"):
            tracker.log_manifest(build_manifest(tmp_path))
    else:
        tracker.log_manifest(build_manifest(tmp_path))
    assert "models" in task.connected and "datasets" not in task.connected

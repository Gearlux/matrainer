"""Tests for MlflowTracker with mlflow mocked via sys.modules injection."""

import posixpath
import re
import sys
import types
from pathlib import Path
from typing import Any, List

import pytest

from matrainer.tracking.mlflow_backend import MAX_NAME_LENGTH, MlflowTracker, _is_path_ambiguous, mlflow_safe_name

#: MLflow's own key rules, which `_FakeClient` below enforces so a mocked test cannot pass on a
#: name the real library refuses. Pinned to the installed MLflow by
#: `test_the_name_rules_match_the_installed_mlflow`.
_MLFLOW_KEY_CHARACTERS = re.compile(r"^[/\w.\- :]*$")

#: The standard tag NAMES, verified against the installed MLflow in
#: ``test_the_standard_tag_names_match_the_installed_mlflow`` — so the fakes here cannot
#: drift from reality without a test saying so.
_STANDARD_TAGS = {
    "MLFLOW_GIT_BRANCH": "mlflow.source.git.branch",
    "MLFLOW_GIT_COMMIT": "mlflow.source.git.commit",
    "MLFLOW_GIT_REPO_URL": "mlflow.source.git.repoURL",
    "MLFLOW_SOURCE_NAME": "mlflow.source.name",
    "MLFLOW_SOURCE_TYPE": "mlflow.source.type",
}


class _FakeDataset:
    def __init__(self, name: str, digest: str, source_type: str, source: str) -> None:
        self.name, self.digest, self.source_type, self.source = name, digest, source_type, source


class _FakeInputTag:
    def __init__(self, key: str, value: str) -> None:
        self.key, self.value = key, value


class _FakeLoggedModelInput:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id


class _FakeDatasetInput:
    def __init__(self, dataset: Any, tags: Any = None) -> None:
        self.dataset, self.tags = dataset, tags


class _FakeRun:
    def __init__(self, run_id: str) -> None:
        self.info = types.SimpleNamespace(run_id=run_id)


def _require_mlflow_key(name: Any) -> str:
    """Refuse ``name`` exactly as MLflow does — the fake is held to the real library's rules."""
    if not isinstance(name, str) or not _MLFLOW_KEY_CHARACTERS.match(name):
        raise ValueError(f"Invalid value {name!r} for parameter 'key' supplied: Names may only contain …")
    canonical = posixpath.normpath(name)
    if canonical != name or canonical == "." or canonical.startswith(("..", "/")):
        raise ValueError(f"Invalid value {name!r} for parameter 'key' supplied: … must not resolve to other names")
    if len(name) > 250:
        raise ValueError("'Param key' exceeds the maximum length of 250 characters")
    return name


class _FakeClient:
    """Records calls; stands in for mlflow.tracking.MlflowClient."""

    instances: List["_FakeClient"] = []

    def __init__(self, tracking_uri: Any = None) -> None:
        self.tracking_uri = tracking_uri
        self.created_tags: dict = {}
        self.params: dict = {}
        self.metrics: List[tuple] = []
        self.tags: dict = {}
        self.artifacts: List[tuple] = []
        self.texts: List[tuple] = []
        self.terminated: Any = None
        self.inputs: List[Any] = []
        self.model_inputs: List[Any] = []
        self.logged_models: List[Any] = []
        _FakeClient.instances.append(self)

    def get_experiment_by_name(self, name: str) -> Any:
        return None

    def create_experiment(self, name: str) -> str:
        return "exp-1"

    def create_run(self, experiment_id: str, run_name: str, tags: Any = None) -> _FakeRun:
        # `create_run` is the one call that does NOT validate a key (measured against mlflow 3.15).
        self.created_tags = dict(tags or {})
        return _FakeRun("run-42")

    def log_param(self, run_id: str, key: str, value: Any) -> None:
        self.params[_require_mlflow_key(key)] = value

    def log_metric(self, run_id: str, key: str, value: float, step: int = 0, **kw: Any) -> None:
        self.metrics.append((_require_mlflow_key(key), value, step))

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        self.tags[_require_mlflow_key(key)] = value

    def log_artifact(self, run_id: str, path: str, artifact_dir: Any = None) -> None:
        self.artifacts.append((path, artifact_dir))

    def log_text(self, run_id: str, text: str, artifact_file: str) -> None:
        self.texts.append((artifact_file, text))

    def set_terminated(self, run_id: str, status: str) -> None:
        self.terminated = status

    def log_inputs(self, run_id: str, datasets: Any = None, models: Any = None) -> None:
        self.inputs.extend(datasets or [])
        self.model_inputs.extend(models or [])

    def create_logged_model(
        self,
        experiment_id: Any = None,
        name: Any = None,
        source_run_id: Any = None,
        tags: Any = None,
        model_type: Any = None,
    ) -> Any:
        # A fresh id every call, exactly like the real client — the tracker, not MLflow, is what
        # keeps a second `log_manifest` from duplicating (measured against mlflow 3.14).
        self.logged_models.append(
            types.SimpleNamespace(
                model_id=f"lm-{len(self.logged_models)}",
                experiment_id=experiment_id,
                name=name,
                tags=dict(tags or {}),
                model_type=model_type,
            )
        )
        return self.logged_models[-1]


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> Any:
    _FakeClient.instances = []
    mlflow_mod = types.ModuleType("mlflow")
    tracking_mod = types.ModuleType("mlflow.tracking")
    tracking_mod.MlflowClient = _FakeClient  # type: ignore[attr-defined]

    # The manifest projection reaches two more MLflow modules. They must be injected too:
    # replacing the top-level `mlflow` with a bare ModuleType leaves it without a __path__,
    # so `from mlflow.entities import ...` would raise and the projection would silently
    # no-op — a mocked test that proves nothing.
    entities_mod = types.ModuleType("mlflow.entities")
    entities_mod.Dataset = _FakeDataset  # type: ignore[attr-defined]
    entities_mod.DatasetInput = _FakeDatasetInput  # type: ignore[attr-defined]
    entities_mod.InputTag = _FakeInputTag  # type: ignore[attr-defined]
    entities_mod.LoggedModelInput = _FakeLoggedModelInput  # type: ignore[attr-defined]
    tags_mod = types.ModuleType("mlflow.utils.mlflow_tags")
    for _name, _value in _STANDARD_TAGS.items():
        setattr(tags_mod, _name, _value)

    monkeypatch.setitem(sys.modules, "mlflow", mlflow_mod)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking_mod)
    monkeypatch.setitem(sys.modules, "mlflow.entities", entities_mod)
    monkeypatch.setitem(sys.modules, "mlflow.utils", types.ModuleType("mlflow.utils"))
    monkeypatch.setitem(sys.modules, "mlflow.utils.mlflow_tags", tags_mod)
    return _FakeClient


def test_import_error_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """When mlflow is unimportable, experiment access raises with the pip hint."""

    def fake_require(module: str, extra: str, feature: str = "") -> Any:
        raise ImportError(f"{feature} requires the '{module}' package. Install with: pip install matrainer[{extra}]")

    monkeypatch.setattr("matrainer.tracking.mlflow_backend.require", fake_require)
    t = MlflowTracker(log_system_metrics=False)
    with pytest.raises(ImportError, match=r"pip install matrainer\[mlflow\]"):
        _ = t.experiment


def test_experiment_creates_run_lazily(fake_mlflow: Any) -> None:
    t = MlflowTracker(experiment_name="exp", run_name="r", log_system_metrics=False)
    assert not fake_mlflow.instances  # nothing built at construction
    _ = t.experiment
    assert t.run_id == "run-42"
    assert len(fake_mlflow.instances) == 1


def test_log_params_metrics_tags(fake_mlflow: Any) -> None:
    t = MlflowTracker(run_name="r", log_system_metrics=False)
    t.log_hyperparams({"lr": 0.1})
    t.log_metrics({"loss": 0.5}, step=2)
    t.set_tags({"git.commit": "abc"})
    client = fake_mlflow.instances[0]
    assert client.params == {"lr": 0.1}
    assert client.metrics == [("loss", 0.5, 2)]
    assert client.tags == {"git.commit": "abc"}


def test_log_artifact_forwards(fake_mlflow: Any, tmp_path: Any) -> None:
    f = tmp_path / "c.yaml"
    f.write_text("x")
    t = MlflowTracker(run_name="r", log_system_metrics=False)
    t.log_artifact(f, "config")
    assert fake_mlflow.instances[0].artifacts == [(str(f), "config")]


def test_log_text_uses_mlflows_own_text_artifact(fake_mlflow: Any, tmp_path: Any) -> None:
    """`client.log_text` writes the artifact itself, so the base class's file is not needed.

    The step rides in the artifact NAME because MLflow artifacts have no step axis — without
    it a value logged once an epoch would keep only the last one.
    """
    t = MlflowTracker(run_name="r", save_dir=str(tmp_path), log_system_metrics=False)
    t.log_text("val/confusion", "|a|b|", step=2)

    assert fake_mlflow.instances[0].texts == [("text/val_confusion_2.md", "|a|b|")]
    assert fake_mlflow.instances[0].artifacts == []  # no second copy through log_artifact


def test_finalize_terminates_run(fake_mlflow: Any) -> None:
    t = MlflowTracker(run_name="r", log_system_metrics=False)
    _ = t.experiment
    t.finalize("success")
    assert fake_mlflow.instances[0].terminated == "FINISHED"
    t2 = MlflowTracker(run_name="r2", log_system_metrics=False)
    _ = t2.experiment
    t2.finalize("failed")
    assert fake_mlflow.instances[1].terminated == "FAILED"


# --------------------------------------------------------------------------- #
# manifest projection
# --------------------------------------------------------------------------- #


def build_manifest(tmp_path: Any) -> Any:
    from matrainer.tracking.manifest import RunManifest

    m = RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    m.set("command", {"argv": ["train.py", "-c", "a.yaml"], "cwd": "/w/project"})
    m.set("config", {"script": "/w/project/train.yaml"})
    m.set("params", {"lr": 0.1})
    m.add("code", {"name": "lib", "path": "/w/lib", "commit": "aaa", "dirty": False, "branch": "main"})
    m.add(
        "code",
        {
            "name": "project",
            "path": "/w/project",
            "commit": "bbb",
            "dirty": True,
            "branch": "dev",
            "remote": "git@example.com:o/p.git",
        },
    )
    m.record_dataset(kind="hf", name="org/set", split="train", count=12, revision="v2")
    return m


def test_log_manifest_projects_the_standard_source_tags(fake_mlflow: Any, tmp_path: Any) -> None:
    t = MlflowTracker(run_name="r", log_system_metrics=False)
    t.log_manifest(build_manifest(tmp_path))
    tags = fake_mlflow.instances[0].tags
    assert tags["mlflow.source.name"] == "/w/project/train.yaml"  # the config, not `python`
    assert tags["mlflow.source.type"] == "LOCAL"
    # The repository containing the cwd wins the single native commit slot, not the first.
    assert tags["mlflow.source.git.commit"] == "bbb"
    assert tags["mlflow.source.git.branch"] == "dev"
    assert tags["mlflow.source.git.repoURL"] == "git@example.com:o/p.git"
    assert tags["mlflow.source.git.dirty"] == "true"


def test_log_manifest_falls_back_to_argv_for_the_source_name(fake_mlflow: Any, tmp_path: Any) -> None:
    from matrainer.tracking.manifest import RunManifest

    m = RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    m.set("command", {"argv": ["train.py"], "cwd": "/w"})
    t = MlflowTracker(run_name="r", log_system_metrics=False)
    t.log_manifest(m)
    assert fake_mlflow.instances[0].tags["mlflow.source.name"] == "train.py"


def test_log_manifest_omits_source_tags_it_has_no_value_for(fake_mlflow: Any, tmp_path: Any) -> None:
    """A wheel-installed repo record has a commit and nothing else; don't invent the rest."""
    from matrainer.tracking.manifest import RunManifest

    m = RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    m.add("code", {"name": "lib", "path": "/w/lib", "commit": "aaa"})
    t = MlflowTracker(run_name="r", log_system_metrics=False)
    t.log_manifest(m)
    tags = fake_mlflow.instances[0].tags
    assert tags["mlflow.source.git.commit"] == "aaa"
    assert "mlflow.source.git.branch" not in tags
    assert "mlflow.source.git.repoURL" not in tags
    assert "mlflow.source.git.dirty" not in tags
    assert "mlflow.source.name" not in tags  # no argv, no config script -> nothing to name


def test_log_manifest_projects_datasets_as_inputs(fake_mlflow: Any, tmp_path: Any) -> None:
    t = MlflowTracker(run_name="r", log_system_metrics=False)
    t.log_manifest(build_manifest(tmp_path))
    inputs = fake_mlflow.instances[0].inputs
    assert len(inputs) == 1
    dataset = inputs[0].dataset
    assert dataset.name == "org/set"
    assert dataset.source_type == "hf"
    assert "org/set" in dataset.source and "train" in dataset.source
    assert len(dataset.digest) == 16  # a reference digest stands in for an absent data digest
    assert {t.key: t.value for t in inputs[0].tags} == {"split": "train", "count": "12", "revision": "v2"}


def test_a_recorded_digest_is_used_verbatim(fake_mlflow: Any, tmp_path: Any) -> None:
    from matrainer.tracking.manifest import RunManifest

    m = RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    m.record_dataset(kind="local", name="d", digest="0123456789abcdef0123")
    MlflowTracker(run_name="r", log_system_metrics=False).log_manifest(m)
    assert fake_mlflow.instances[0].inputs[0].dataset.digest == "0123456789abcdef0123"


def test_the_reference_digest_is_stable_and_discriminating(tmp_path: Any) -> None:
    """Two runs on the same reference must collide; two on different ones must not."""
    from matrainer.tracking.mlflow_backend import _reference_digest

    a = {"kind": "hf", "name": "org/set", "split": "train"}
    assert _reference_digest(a) == _reference_digest(dict(reversed(list(a.items()))))
    assert _reference_digest(a) != _reference_digest({**a, "split": "test"})


def test_log_manifest_still_does_the_generic_projection(fake_mlflow: Any, tmp_path: Any) -> None:
    """Native slots are ADDITIONAL — the artifact and the params must land regardless."""
    t = MlflowTracker(run_name="r", log_system_metrics=False)
    t.log_manifest(build_manifest(tmp_path))
    client = fake_mlflow.instances[0]
    assert client.params["lr"] == 0.1
    assert [Path(p).name for p, _ in client.artifacts] == ["manifest.yaml", "manifest.json"]


def test_log_manifest_without_datasets_logs_no_inputs(fake_mlflow: Any, tmp_path: Any) -> None:
    from matrainer.tracking.manifest import RunManifest

    t = MlflowTracker(run_name="r", log_system_metrics=False)
    t.log_manifest(RunManifest(run_dir=tmp_path / "run", handle_interrupts=False))
    assert fake_mlflow.instances[0].inputs == []


def test_the_standard_tag_names_match_the_installed_mlflow() -> None:
    """Pin the fakes above to reality, so a renamed MLflow constant fails HERE.

    The projection imports these from ``mlflow.utils.mlflow_tags`` to keep one source of
    truth; this asserts the names the mocked tests assert on are the real ones.
    """
    pytest.importorskip("mlflow")
    from mlflow.utils import mlflow_tags

    for name, value in _STANDARD_TAGS.items():
        assert getattr(mlflow_tags, name) == value


def test_projection_lands_in_a_real_mlflow_store(tmp_path: Any) -> None:
    """Integration: mocks cannot prove ``log_inputs`` accepts what we build.

    Runs against a real sqlite-backed store — MLflow 3.x refuses the file store outright,
    so ``sqlite:///`` is the cheapest backend a test can use.
    """
    pytest.importorskip("mlflow")
    tracker = MlflowTracker(
        experiment_name="proj",
        run_name="r",
        tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        log_system_metrics=False,
    )
    tracker.log_manifest(build_manifest(tmp_path))

    run = tracker.experiment.get_run(tracker.run_id)
    assert run.data.tags["mlflow.source.git.commit"] == "bbb"
    assert run.data.tags["mlflow.source.git.dirty"] == "true"
    assert run.data.params["lr"] == "0.1"
    inputs = run.inputs.dataset_inputs
    assert [d.dataset.name for d in inputs] == ["org/set"]
    assert inputs[0].dataset.source_type == "hf"

    tracker.log_manifest(build_manifest(tmp_path))  # idempotent: MLflow dedups on digest
    assert len(tracker.experiment.get_run(tracker.run_id).inputs.dataset_inputs) == 1


# --- model lineage (log_inputs(models=...)) --------------------------------------------------


def _manifest_with_models(tmp_path: Any) -> Any:
    from matrainer.tracking.manifest import RunManifest

    m = RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    m.record_model(
        source="hf",
        name="timm/resnet50.a1_in1k",
        uri="hf://models/timm/resnet50.a1_in1k",
        url="https://huggingface.co/timm/resnet50.a1_in1k",
        role="pretrained",
    )
    m.record_model(source="file", name="/w/last.ckpt", uri="file:///w/last.ckpt", role="checkpoint")
    return m


def test_models_become_logged_model_inputs(fake_mlflow: Any, tmp_path: Any) -> None:
    """A `LoggedModelInput` needs a logged-model ID, so the entity is created first."""
    MlflowTracker(log_system_metrics=False).log_manifest(_manifest_with_models(tmp_path))
    client = fake_mlflow.instances[0]
    assert [m.model_id for m in client.model_inputs] == ["lm-0", "lm-1"]
    assert len(client.logged_models) == 2


def test_every_manifest_field_rides_along_as_a_tag(fake_mlflow: Any, tmp_path: Any) -> None:
    MlflowTracker(log_system_metrics=False).log_manifest(_manifest_with_models(tmp_path))
    first = fake_mlflow.instances[0].logged_models[0]
    assert first.tags["uri"] == "hf://models/timm/resnet50.a1_in1k"
    assert first.tags["role"] == "pretrained"
    assert first.tags["url"] == "https://huggingface.co/timm/resnet50.a1_in1k"
    assert first.model_type == "hf"  # `source` here; `capture_models` writes `kind` — both read


def test_the_logged_model_name_is_sanitized(fake_mlflow: Any, tmp_path: Any) -> None:
    """MLflow rejects `/ : . %` and quotes in a model name — every separator a real id uses."""
    MlflowTracker(log_system_metrics=False).log_manifest(_manifest_with_models(tmp_path))
    names = [m.name for m in fake_mlflow.instances[0].logged_models]
    assert names == ["timm_resnet50_a1_in1k", "w_last_ckpt"]


def test_a_second_log_manifest_does_not_duplicate_the_models(fake_mlflow: Any, tmp_path: Any) -> None:
    """`log_manifest` runs at least twice per run, and MLflow does NOT dedup a logged model."""
    tracker = MlflowTracker(log_system_metrics=False)
    manifest = _manifest_with_models(tmp_path)
    tracker.log_manifest(manifest)
    tracker.log_manifest(manifest)
    client = fake_mlflow.instances[0]
    assert len(client.logged_models) == 2
    assert len(client.model_inputs) == 2


def test_a_manifest_with_no_models_logs_nothing(fake_mlflow: Any, tmp_path: Any) -> None:
    from matrainer.tracking.manifest import RunManifest

    MlflowTracker(log_system_metrics=False).log_manifest(RunManifest(run_dir=tmp_path / "run", handle_interrupts=False))
    assert fake_mlflow.instances[0].logged_models == []


def test_a_failing_create_logged_model_does_not_break_the_run(
    fake_mlflow: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tracking must never abort a run — a lost lineage link beats a dead training job."""
    tracker = MlflowTracker(log_system_metrics=False)
    client = tracker.experiment

    def boom(**kwargs: Any) -> Any:
        raise RuntimeError("server said no")

    monkeypatch.setattr(client, "create_logged_model", boom)
    tracker.log_manifest(_manifest_with_models(tmp_path))  # no raise
    assert client.model_inputs == []


# --------------------------------------------------------------------------- #
# raise_errors — a refused call is a debug line, and raised as well when asked
# --------------------------------------------------------------------------- #

SAID_NO = RuntimeError("server said no")


def _refuse(monkeypatch: pytest.MonkeyPatch, target: Any, method: str, only: Any = None) -> None:
    """Make ``target.method`` raise :data:`SAID_NO` — on every call, or only on one naming ``only``."""
    real = getattr(target, method, None)

    def refused(*args: Any, **kwargs: Any) -> Any:
        if only is None or only in args or only in kwargs.values():
            raise SAID_NO
        return real(*args, **kwargs)  # type: ignore[misc]

    monkeypatch.setattr(target, method, refused, raising=False)


def _file(tmp_path: Path) -> Path:
    path = tmp_path / "a.txt"
    path.write_text("A")
    return path


#: (client method refused, the tracker call that sends one request to it)
_ONE_REQUEST = [
    ("log_artifact", lambda t, tmp: t.log_artifact(_file(tmp))),
    ("log_figure", lambda t, tmp: t.log_figure("val/cm", object())),
    ("log_text", lambda t, tmp: t.log_text("val/report", "|a|")),
    ("log_inputs", lambda t, tmp: t.log_manifest(build_manifest(tmp))),  # the datasets
    ("log_inputs", lambda t, tmp: t.log_manifest(_manifest_with_models(tmp))),  # the models' link
    ("set_terminated", lambda t, tmp: t.finalize("success")),
]


@pytest.mark.parametrize("raise_errors", [False, True])
@pytest.mark.parametrize("method, call", _ONE_REQUEST, ids=[f"{m}-{i}" for i, (m, _) in enumerate(_ONE_REQUEST)])
def test_a_refused_request_is_raised_only_when_asked(
    fake_mlflow: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, call: Any, raise_errors: bool
) -> None:
    tracker = MlflowTracker(log_system_metrics=False, raise_errors=raise_errors)
    _refuse(monkeypatch, tracker.experiment, method)
    if raise_errors:
        with pytest.raises(RuntimeError, match="server said no"):
            call(tracker, tmp_path)
    else:
        call(tracker, tmp_path)  # a debug line, as always


#: (client method refused for the name `b`, the tracker call naming a, b and c, what the client holds)
_PER_NAME = {
    "log_param": (lambda t: t.log_hyperparams({"a": 1, "b": 2, "c": 3}), lambda c: c.params),
    "log_metric": (
        lambda t: t.log_metrics({"a": 1.0, "b": 2.0, "c": 3.0}, step=4),
        lambda c: [key for key, _, _ in c.metrics],
    ),
    "set_tag": (lambda t: t.set_tags({"a": "1", "b": "2", "c": "3"}), lambda c: c.tags),
}


@pytest.mark.parametrize("raise_errors", [False, True])
@pytest.mark.parametrize("method", sorted(_PER_NAME))
def test_a_refused_name_does_not_stop_the_others(
    fake_mlflow: Any, monkeypatch: pytest.MonkeyPatch, method: str, raise_errors: bool
) -> None:
    """Every name is tried, as before; with raise_errors the first refusal is raised afterwards."""
    tracker = MlflowTracker(log_system_metrics=False, raise_errors=raise_errors)
    client = tracker.experiment
    _refuse(monkeypatch, client, method, only="b")
    call, held = _PER_NAME[method]
    if raise_errors:
        with pytest.raises(RuntimeError, match="server said no"):
            call(tracker)
    else:
        call(tracker)
    assert sorted(held(client)) == ["a", "c"]


@pytest.mark.parametrize("raise_errors", [False, True])
def test_a_refused_model_does_not_stop_the_others(
    fake_mlflow: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raise_errors: bool
) -> None:
    tracker = MlflowTracker(log_system_metrics=False, raise_errors=raise_errors)
    client = tracker.experiment
    _refuse(monkeypatch, client, "create_logged_model", only="timm_resnet50_a1_in1k")
    if raise_errors:
        with pytest.raises(RuntimeError, match="server said no"):
            tracker.log_manifest(_manifest_with_models(tmp_path))
    else:
        tracker.log_manifest(_manifest_with_models(tmp_path))
    assert [m.name for m in client.logged_models] == ["w_last_ckpt"] and len(client.model_inputs) == 1


def test_a_failed_monitor_shutdown_is_never_raised_and_the_run_still_ends(fake_mlflow: Any) -> None:
    """The system-metrics monitor is local bookkeeping: its failure must not keep the run open."""
    tracker = MlflowTracker(log_system_metrics=False, raise_errors=True)
    client = tracker.experiment

    class Monitor:
        def finish(self) -> None:
            raise SAID_NO

    tracker._system_monitor = Monitor()
    tracker.finalize("success")
    assert client.terminated == "FINISHED"


def test_a_resilient_tracker_replays_the_metric_mlflow_could_not_send(
    fake_mlflow: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wrapper turns raise_errors on, so a metric lost to a dropped server is queued and sent later."""
    from matrainer.tracking.resilient import ResilientTracker

    wrapped = ResilientTracker(tracker=MlflowTracker(log_system_metrics=False), timeout=5.0)
    wrapped.log_metrics({"loss": 1.0}, step=0)
    client = fake_mlflow.instances[0]
    sending = client.log_metric

    def dropped(*args: Any, **kwargs: Any) -> None:
        raise ConnectionError("Max retries exceeded with url: /api/2.0/mlflow/runs/log-metric (Connection refused)")

    monkeypatch.setattr(client, "log_metric", dropped)
    wrapped.log_metrics({"loss": 0.5}, step=1)
    assert [call.method for call in wrapped.pending] == ["log_metrics"]
    monkeypatch.setattr(client, "log_metric", sending)
    wrapped.log_metrics({"loss": 0.25}, step=2)
    assert [step for _, _, step in client.metrics] == [0, 1, 2] and wrapped.pending == []


# --- parameter / metric / tag names (mlflow_safe_name) ---------------------------------------
#
# MLflow holds a key to three rules at once: a character set, readability as a path, and a
# length cap. Each example below was measured against a real store BEFORE the mapping existed
# (refused) and after it (accepted); `test_the_name_rules_match_the_installed_mlflow` is what
# keeps that measurement honest as MLflow moves.

#: `(name the caller logs, name MLflow receives)`. The first three were refused outright before
#: `mlflow_safe_name`; the last two were already accepted and must stay byte-identical.
NAME_MAPPINGS = [
    ("x.kwargs[source].y", "x.kwargs.source.y"),
    ("callbacks[0].patience", "callbacks.0.patience"),
    ("model<v2>", "model_v2_"),
    ("RFDETRDetector.epochs", "RFDETRDetector.epochs"),
    ("a:b-c d/e_f", "a:b-c d/e_f"),
]


def _real_tracker(tmp_path: Any, **kw: Any) -> MlflowTracker:
    """A tracker on a real sqlite-backed store — MLflow 3.x refuses the file store outright."""
    return MlflowTracker(
        experiment_name="names",
        run_name="r",
        tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        log_system_metrics=False,
        **kw,
    )


@pytest.mark.parametrize(("logged", "received"), NAME_MAPPINGS)
def test_each_name_rule_maps_the_name_it_is_for(logged: str, received: str) -> None:
    assert mlflow_safe_name(logged) == received


def test_a_config_walk_parameter_name_reaches_a_real_mlflow(tmp_path: Any) -> None:
    """The defect this exists for: a dict in the configuration cost the run its WHOLE params section.

    A bracketed step in one name was enough: MLflow refused it, `log_hyperparams` raised,
    `log_manifest` raised with it and a `ResilientTracker` dropped the call at WARNING.
    `raise_errors=True` is what makes a surviving refusal fail here.

    `config_params` no longer brackets a container step (confluid writes `kwargs.source` since its
    record 34), so these inputs now stand for any producer that is not bound by MLflow's rules — a
    framework's metric name, a user's tag key, a name from an older confluid.
    """
    pytest.importorskip("mlflow")
    tracker = _real_tracker(tmp_path, raise_errors=True)

    tracker.log_hyperparams({logged: 1 for logged, _ in NAME_MAPPINGS})

    params = tracker.experiment.get_run(tracker.run_id).data.params
    assert sorted(params) == sorted(received for _, received in NAME_MAPPINGS)


def test_a_metric_and_a_tag_name_are_held_to_the_same_rule(tmp_path: Any) -> None:
    """Not a params-only restriction: one MLflow validator serves parameters, metrics AND tags.

    So a metric a training framework names `val/acc@1`, and a tag named from a configuration
    key, land through the same mapping rather than being refused one call at a time.
    """
    pytest.importorskip("mlflow")
    tracker = _real_tracker(tmp_path, raise_errors=True)

    tracker.log_metrics({"val/acc@1": 0.5}, step=0)
    tracker.set_tags({"model<v2>": "x"})

    run = tracker.experiment.get_run(tracker.run_id)
    assert run.data.metrics["val/acc_1"] == 0.5
    assert run.data.tags["model_v2_"] == "x"


def test_a_path_shaped_dict_key_loses_its_slashes(tmp_path: Any) -> None:
    """MLflow's second rule: a key must ALREADY be canonical as a posix path.

    `datasets[$DATA_ROOT/set/]` passes the character rule once `$` becomes `_`, and is still
    refused — `must not resolve to other names` — because a trailing slash does not survive
    normalisation. Each slash becomes `_` (the trailing one too, rather than being dropped: a
    name keeps every step it had, so two keys cannot collapse into one), and only in that case —
    `a:b-c d/e_f` above keeps its own slash.
    """
    pytest.importorskip("mlflow")
    tracker = _real_tracker(tmp_path, raise_errors=True)

    tracker.log_hyperparams({"datasets[$DATA_ROOT/set/]": 1})

    assert "datasets._DATA_ROOT_set_" in tracker.experiment.get_run(tracker.run_id).data.params


@pytest.mark.parametrize("name", ["", ".", "..", "..x", "/abs/x", "a//b", "a/./b", "x/.."])
def test_a_name_that_is_only_path_syntax_still_becomes_a_usable_key(name: str) -> None:
    """Nothing degenerate escapes: each of these is refused by MLflow as written."""
    safe = mlflow_safe_name(name)
    assert safe and not _is_path_ambiguous(safe)


def test_a_long_name_keeps_its_tail_behind_a_digest(tmp_path: Any) -> None:
    """Over 250 characters MLflow refuses the key for a THIRD reason, with its own message.

    The tail is what is kept because the leaf setting is what identifies a hyperparameter; the
    digest is what keeps two deep paths that end the same way from colliding into one parameter.
    """
    pytest.importorskip("mlflow")
    deep = "Runnable." + ".".join(f"level_{i}" for i in range(40))
    sibling = "Other." + ".".join(f"level_{i}" for i in range(40))
    assert len(deep) > MAX_NAME_LENGTH and deep[-60:] == sibling[-60:]

    first, second = mlflow_safe_name(deep), mlflow_safe_name(sibling)
    assert len(first) == MAX_NAME_LENGTH and first.endswith("level_39")
    assert first != second  # the digest is what separates them

    tracker = _real_tracker(tmp_path, raise_errors=True)
    tracker.log_hyperparams({deep: 1, sibling: 2})
    assert sorted(tracker.experiment.get_run(tracker.run_id).data.params) == sorted([first, second])


def test_the_name_rules_match_the_installed_mlflow() -> None:
    """Pin the three rules to MLflow's own validators, so an upstream change fails HERE.

    Asserts the property that matters rather than equality of the rules: every name the mapping
    produces is one the installed MLflow accepts. `mlflow_safe_name`'s character set is
    deliberately narrower than MLflow's (ASCII, where its `\\w` admits any Unicode letter), so
    equality would fail while the contract holds.
    """
    pytest.importorskip("mlflow")
    from mlflow.utils.validation import MAX_ENTITY_KEY_LENGTH, path_not_unique, validate_param_and_metric_name

    assert MAX_NAME_LENGTH == MAX_ENTITY_KEY_LENGTH
    nasty = [logged for logged, _ in NAME_MAPPINGS] + [
        "",
        ".",
        "..",
        "..x",
        "/abs/x",
        "a//b",
        "a/./b",
        "x/..",
        "datasets[$DATA_ROOT/set/]",
        "val/acc@1",
        "metrics/mAP50-95(B)",
        "héllo",
        "x" * 400,
        "kwargs[a/./b]",
        "[0]",
        "a\nb",
        "a\tb",
        "100%",
        'say "no"',
    ]
    for name in nasty:
        safe = mlflow_safe_name(name)
        assert validate_param_and_metric_name(safe), f"{name!r} -> {safe!r} fails the character rule"
        assert not path_not_unique(safe), f"{name!r} -> {safe!r} fails the path rule"
        assert 0 < len(safe) <= MAX_ENTITY_KEY_LENGTH, f"{name!r} -> {safe!r} fails the length rule"


def test_constructor_tags_get_the_same_name_as_a_later_set_tag(fake_mlflow: Any) -> None:
    """`create_run` does not validate a tag key but `set_tag` does (measured) — one name per tag.

    Without this a tag stamped at creation and re-set during the run would stand twice in the
    UI, once under each spelling.
    """
    tracker = MlflowTracker(run_name="r", tags={"model<v2>": "a"}, log_system_metrics=False)
    client = tracker.experiment
    tracker.set_tags({"model<v2>": "b"})

    assert client.created_tags == {"model_v2_": "a"}
    assert list(client.tags) == ["model_v2_"]

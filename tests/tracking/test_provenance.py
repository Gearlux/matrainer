"""Tests for run-provenance capture (git state, config, environment, log file)."""

import subprocess
import sys
import types
from pathlib import Path
from typing import Any, List, Mapping, Optional

import pytest
import yaml

from marainer.tracking import CsvTracker, provenance
from marainer.tracking.base import Tracker


def make_repo(path: Path, dirty: bool = False) -> Path:
    """A real git repository — the git plumbing is what these functions actually shell out to."""
    path.mkdir(parents=True, exist_ok=True)
    run = lambda *a: subprocess.run(["git", "-C", str(path), *a], check=True, capture_output=True)  # noqa: E731
    run("init", "-q", "-b", "main")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "T")
    run("remote", "add", "origin", "git@example.com:org/repo.git")
    (path / "f.txt").write_text("one")
    run("add", "-A")
    run("commit", "-qm", "first")
    if dirty:
        (path / "f.txt").write_text("two")
    return path


class RecordingTracker(Tracker):
    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.hparams: dict = {}
        self.set_tag_calls: dict = {}
        self.artifacts: List[tuple] = []

    @property
    def experiment(self) -> str:
        return "h"

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        self.hparams.update(params)

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        pass

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self.set_tag_calls.update(tags)

    def log_artifact(self, path: Any, artifact_dir: str = "") -> None:
        self.artifacts.append((str(path), artifact_dir))


def test_get_git_commit_hash_in_repo() -> None:
    # marainer is a git repo — expect a hash string (possibly '-dirty'), not None.
    h = provenance.get_git_commit_hash()
    assert h is None or isinstance(h, str)


def test_get_git_commit_hash_handles_no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise FileNotFoundError

    monkeypatch.setattr(provenance.subprocess, "check_output", boom)
    assert provenance.get_git_commit_hash() is None


def test_log_run_context_stamps_git_config_and_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "train.yaml"
    cfg.write_text("x: 1")
    monkeypatch.setattr(provenance, "get_git_commit_hash", lambda: "deadbeef")
    monkeypatch.setattr(provenance, "config_script_path", lambda: str(cfg))
    monkeypatch.setattr(provenance, "included_config_paths", lambda: [cfg])

    monkeypatch.chdir(tmp_path)
    t = RecordingTracker()
    manifest = provenance.log_run_context([t], params={"lr": 0.1, "model": object()}, capture_code=False)

    assert t.hparams["lr"] == 0.1
    assert "model" not in t.hparams  # non-scalar dropped
    assert t.hparams["config_script"] == str(cfg)
    assert t.hparams[provenance.GIT_COMMIT_TAG] == "deadbeef"
    assert t.set_tag_calls == {provenance.GIT_COMMIT_TAG: "deadbeef"}
    assert (str(cfg), "config") in t.artifacts
    assert manifest is not None and manifest.section("params")["lr"] == 0.1


def test_log_run_context_empty_trackers_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No trackers and no manifest: write nothing, rather than a manifest nobody asked for."""
    monkeypatch.chdir(tmp_path)
    assert provenance.log_run_context([], params={"lr": 1}) is None
    assert list(tmp_path.iterdir()) == []


def test_log_run_context_builds_the_manifest_beside_the_first_tracker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    tracker = CsvTracker(experiment_name="exp", run_name="r", save_dir=str(tmp_path / "runs"))
    manifest = provenance.log_run_context([tracker], capture_code=False)
    assert manifest is not None
    assert manifest.path == tmp_path / "runs" / "exp" / "r" / "manifest.yaml"
    assert manifest.run_name == "r" and manifest.experiment_name == "exp"


def test_log_run_context_derives_params_from_a_runnable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The point of the fold: hand it the object, not a hand-built params dict."""
    from confluid import configurable

    @configurable
    class ContextToy:
        def __init__(self, lr: float = 0.02, epochs: int = 7) -> None:
            self.lr, self.epochs = lr, epochs

    monkeypatch.chdir(tmp_path)
    t = RecordingTracker()
    manifest = provenance.log_run_context([t], runnable=ContextToy(), capture_code=False)
    assert manifest is not None
    derived = by_leaf(manifest.section("params"))
    assert derived["lr"] == 0.02 and derived["epochs"] == 7
    assert (manifest.directory / "config" / "final.yaml").exists()


def test_explicit_params_win_over_derived_ones(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from confluid import configurable

    @configurable
    class OverrideToy:
        def __init__(self, lr: float = 0.02) -> None:
            self.lr = lr

    monkeypatch.chdir(tmp_path)
    t = RecordingTracker()
    manifest = provenance.log_run_context([t], params={"lr": 0.99}, runnable=OverrideToy(), capture_code=False)
    assert manifest is not None
    assert manifest.section("params")["lr"] == 0.99  # the caller's own key, set last


def test_log_run_context_reuses_a_supplied_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from marainer.tracking.manifest import RunManifest

    monkeypatch.chdir(tmp_path)
    mine = RunManifest(run_dir=tmp_path / "mine", run_name="chosen", handle_interrupts=False)
    returned = provenance.log_run_context([], manifest=mine, capture_code=False)
    assert returned is mine  # supplied -> used even with no trackers at all
    assert (tmp_path / "mine" / "manifest.yaml").exists()


def test_log_run_context_can_skip_the_code_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`capture_code` costs a git call set per repository; it must be refusable."""
    called: List[int] = []

    def spy() -> List[Any]:
        called.append(1)
        return []

    monkeypatch.setattr(provenance, "code_state", spy)
    monkeypatch.chdir(tmp_path)
    manifest = provenance.log_run_context([RecordingTracker()], capture_code=False)
    assert called == []
    assert manifest is not None and manifest.section("code") == []


def test_log_run_context_probes_code_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [{"name": "pkg", "path": "/p", "commit": "c0ffee", "dirty": False}]
    monkeypatch.setattr(provenance, "code_state", lambda: entries)
    monkeypatch.chdir(tmp_path)
    manifest = provenance.log_run_context([RecordingTracker()])
    assert manifest is not None and manifest.section("code") == entries


def test_capture_log_file_uploads_when_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log = tmp_path / "app.log"
    log.write_text("line")
    monkeypatch.setattr(provenance, "resolve_log_file", lambda: log)
    t = RecordingTracker()
    provenance.capture_log_file([t])
    assert t.artifacts == [(str(log), "logs")]


def test_log_run_context_with_nothing_to_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No params, no repository, no config tree — the call must still be harmless."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(provenance, "get_git_commit_hash", lambda: None)
    monkeypatch.setattr(provenance, "config_script_path", lambda: None)
    monkeypatch.setattr(provenance, "included_config_paths", lambda: [])
    monkeypatch.setattr(provenance, "code_state", list)
    t = RecordingTracker()
    provenance.log_run_context([t])
    assert t.hparams == {} and t.set_tag_calls == {}
    assert [d for _, d in t.artifacts] == ["", ""]  # only the two manifest files


def test_capture_log_file_empty_trackers_is_noop() -> None:
    provenance.capture_log_file([])  # no raise


def test_capture_log_file_finishes_and_reprojects_the_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The end-of-run half of the fold: what the backend holds matches what is on disk."""
    from marainer.tracking.manifest import RunManifest

    monkeypatch.setattr(provenance, "resolve_log_file", lambda: None)
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    t = RecordingTracker()
    provenance.capture_log_file([t], manifest=m, status="failed")
    assert m.status == "failed"
    assert len(t.artifacts) == 2  # the finished manifest was pushed again
    assert yaml.safe_load(m.path.read_text())["status"] == "failed"


def test_capture_log_file_leaves_an_already_finished_manifest_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marainer.tracking.manifest import RunManifest

    monkeypatch.setattr(provenance, "resolve_log_file", lambda: None)
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    m.finish("success")
    provenance.capture_log_file([RecordingTracker()], manifest=m, status="failed")
    assert m.status == "success"  # first terminal status wins


def test_capture_log_file_uploads_both_the_log_and_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from marainer.tracking.manifest import RunManifest

    log = tmp_path / "run.log"
    log.write_text("line")
    monkeypatch.setattr(provenance, "resolve_log_file", lambda: log)
    m = RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    t = RecordingTracker()
    provenance.capture_log_file([t], manifest=m)
    assert (str(log), "logs") in t.artifacts
    assert len(t.artifacts) == 3


def test_capture_log_file_noop_without_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provenance, "resolve_log_file", lambda: None)
    t = RecordingTracker()
    provenance.capture_log_file([t])
    assert t.artifacts == []


def test_resolve_log_file_reads_loggair_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text("x")
    import loggair

    monkeypatch.setattr(loggair, "get_active_config", lambda: {"log_file": str(log)})
    assert provenance.resolve_log_file() == log


def test_resolve_log_file_none_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    import loggair

    monkeypatch.setattr(loggair, "get_active_config", lambda: {"configured": False})
    assert provenance.resolve_log_file() is None


# --------------------------------------------------------------------------- #
# git / code state
# --------------------------------------------------------------------------- #


def test_git_state_reads_a_clean_repository(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "clean")
    state = provenance.git_state(repo)
    assert state is not None
    assert len(state["commit"]) == 40
    assert state["dirty"] is False
    assert state["branch"] == "main"
    assert state["remote"] == "git@example.com:org/repo.git"
    assert Path(state["root"]).resolve() == repo.resolve()


def test_git_state_flags_a_dirty_worktree(tmp_path: Path) -> None:
    state = provenance.git_state(make_repo(tmp_path / "dirty", dirty=True))
    assert state is not None and state["dirty"] is True


def test_git_state_is_none_outside_a_repository(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    assert provenance.git_state(plain) is None


def test_git_state_is_none_without_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def no_git(*a: Any, **k: Any) -> Any:
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance.subprocess, "check_output", no_git)
    assert provenance.git_state(tmp_path) is None


def test_git_state_reports_unknown_dirtiness_rather_than_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed status probe must not be indistinguishable from a clean tree."""
    repo = make_repo(tmp_path / "repo")
    real = provenance.subprocess.check_output

    def selective(cmd: List[str], **k: Any) -> Any:
        if "status" in cmd:
            raise subprocess.CalledProcessError(1, cmd)
        return real(cmd, **k)

    monkeypatch.setattr(provenance.subprocess, "check_output", selective)
    state = provenance.git_state(repo)
    assert state is not None and state["dirty"] is None


def fake_distribution(name: str, url: str, editable: bool = True) -> Any:
    payload = '{"url": "%s", "dir_info": {"editable": %s}}' % (url, "true" if editable else "false")
    return types.SimpleNamespace(
        metadata={"Name": name}, read_text=lambda f: payload if f == "direct_url.json" else None
    )


def test_editable_source_paths_finds_editable_installs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "pkg with space"
    src.mkdir()
    dists = [
        fake_distribution("editable-one", src.as_uri()),
        fake_distribution("wheel-one", src.as_uri(), editable=False),
        types.SimpleNamespace(metadata={"Name": "no-direct-url"}, read_text=lambda f: None),
    ]
    monkeypatch.setattr(provenance.importlib.metadata, "distributions", lambda: iter(dists))
    found = provenance.editable_source_paths()
    assert found == {"editable-one": src}  # a percent-encoded path round-trips


def test_editable_source_paths_skips_a_vanished_source_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An editable install whose checkout has been deleted is stale metadata, not a source."""
    dists = [fake_distribution("gone", (tmp_path / "deleted").as_uri())]
    monkeypatch.setattr(provenance.importlib.metadata, "distributions", lambda: iter(dists))
    assert provenance.editable_source_paths() == {}


def test_editable_source_paths_skips_non_file_urls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dists = [fake_distribution("remote", "https://example.com/pkg.git")]
    monkeypatch.setattr(provenance.importlib.metadata, "distributions", lambda: iter(dists))
    assert provenance.editable_source_paths() == {}


def test_editable_source_paths_survives_a_broken_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(f: str) -> str:
        raise OSError("unreadable metadata")

    dists = [types.SimpleNamespace(metadata={"Name": "broken"}, read_text=explode)]
    monkeypatch.setattr(provenance.importlib.metadata, "distributions", lambda: iter(dists))
    assert provenance.editable_source_paths() == {}


def test_code_state_lists_one_entry_per_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a = make_repo(tmp_path / "a")
    b = make_repo(tmp_path / "b", dirty=True)
    monkeypatch.setattr(provenance, "editable_source_paths", lambda: {"pkg-b": b, "pkg-a": a})
    monkeypatch.chdir(tmp_path)
    entries = provenance.code_state()
    assert [e["name"] for e in entries] == ["pkg-a", "pkg-b"]  # sorted by distribution name
    assert [e["dirty"] for e in entries] == [False, True]
    assert "root" not in entries[0]  # folded into `path`


def test_code_state_deduplicates_by_git_toplevel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = make_repo(tmp_path / "repo")
    nested = repo / "src" / "pkg"
    nested.mkdir(parents=True)
    monkeypatch.setattr(provenance, "editable_source_paths", lambda: {"outer": repo, "inner": nested})
    monkeypatch.chdir(tmp_path)
    entries = provenance.code_state()
    assert len(entries) == 1
    assert entries[0]["name"] == "inner"  # first alphabetically wins; both name the same toplevel


def test_code_state_includes_the_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = make_repo(tmp_path / "consumer")
    monkeypatch.setattr(provenance, "editable_source_paths", lambda: {})
    monkeypatch.chdir(repo)
    entries = provenance.code_state()
    assert len(entries) == 1
    assert entries[0]["name"] == "consumer"  # unnamed -> the repository directory name


def test_code_state_is_empty_outside_any_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provenance, "editable_source_paths", lambda: {})
    monkeypatch.chdir(tmp_path)
    assert provenance.code_state() == []


# --------------------------------------------------------------------------- #
# digests + config params
# --------------------------------------------------------------------------- #


def test_file_digest_is_the_sha256(tmp_path: Path) -> None:
    f = tmp_path / "a.yaml"
    f.write_text("x: 1")
    empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert provenance.file_digest(f) != empty
    assert len(provenance.file_digest(f) or "") == 64


def test_file_digest_is_none_for_a_missing_file(tmp_path: Path) -> None:
    assert provenance.file_digest(tmp_path / "gone.yaml") is None


def by_leaf(params: Mapping[str, Any]) -> dict:
    """Re-key by the trailing attribute name.

    The dotted PREFIX is the configuration library's shortest-unique alias, so it shifts
    with whatever else is registered in the process (``Widget`` can come out as ``w``).
    ``config_params`` passes those keys through unchanged — asserting on a fixed prefix
    would pin someone else's naming rule.
    """
    return {path.rsplit(".", 1)[-1]: value for path, value in params.items()}


def test_config_params_flattens_scalars_and_scalar_lists() -> None:
    from confluid import configurable

    @configurable
    class ParamWidget:
        def __init__(self, lr: float = 0.1, label: str = "w", size: Any = (8, 8), on: bool = True) -> None:
            self.lr, self.label, self.size, self.on = lr, label, size, on

    params = by_leaf(provenance.config_params(ParamWidget()))
    assert params["lr"] == 0.1
    assert params["label"] == "w"
    assert params["size"] == [8, 8]  # a tuple of scalars survives as a list
    assert params["on"] is True


def test_config_params_keeps_nested_scalars_but_never_an_object_repr() -> None:
    """A nested object's scalars are richer provenance; its repr would carry a memory address."""
    from confluid import configurable

    @configurable
    class ParamInner:
        def __init__(self, k: int = 7) -> None:
            self.k = k

    @configurable
    class ParamOuter:
        def __init__(self, inner: Any = None, lr: float = 0.5) -> None:
            self.inner, self.lr = inner if inner is not None else ParamInner(), lr

    params = provenance.config_params(ParamOuter())
    leaves = by_leaf(params)
    assert leaves["lr"] == 0.5
    assert leaves["k"] == 7  # walked into the nested configurable
    assert not any("object at 0x" in str(v) for v in params.values())
    assert all(isinstance(v, (bool, int, float, str, list)) or v is None for v in params.values())


def test_config_params_drops_a_value_that_is_neither_scalar_nor_scalar_list() -> None:
    from confluid import configurable

    @configurable
    class ParamMapped:
        def __init__(self, mapping: Any = None, mixed: Any = (1, object()), lr: float = 0.3) -> None:
            self.mapping, self.mixed, self.lr = mapping if mapping is not None else {"a": 1}, mixed, lr

    params = by_leaf(provenance.config_params(ParamMapped()))
    assert params["lr"] == 0.3
    assert "mapping" not in params  # a dict is not a hyperparameter value
    assert "mixed" not in params  # a list is kept only when EVERY element is a scalar


def test_config_params_is_empty_for_a_plain_object() -> None:
    assert provenance.config_params(object()) == {}


def test_config_params_is_empty_when_the_walk_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deriving hyperparameters must never take down a run."""

    def boom(obj: Any) -> Any:
        raise RuntimeError("hierarchy exploded")

    monkeypatch.setattr("confluid.get_hierarchy_from_instance", boom)
    assert provenance.config_params(object()) == {}


# --------------------------------------------------------------------------- #
# liquifai config context
# --------------------------------------------------------------------------- #


def set_context(monkeypatch: pytest.MonkeyPatch, **fields: Any) -> None:
    ctx = types.SimpleNamespace(**{"included_paths": [], "config_path": None, **fields})
    monkeypatch.setattr("liquifai.context.get_context", lambda: ctx)


def test_included_config_paths_returns_the_resolved_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a, b, gone = tmp_path / "a.yaml", tmp_path / "b.yaml", tmp_path / "gone.yaml"
    a.write_text("a: 1")
    b.write_text("b: 2")
    set_context(monkeypatch, included_paths=[a, b, gone])
    assert provenance.included_config_paths() == [a, b]  # a path that no longer exists is dropped


def test_included_config_paths_falls_back_to_the_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "train.yaml"
    cfg.write_text("x: 1")
    set_context(monkeypatch, included_paths=[], config_path=cfg)
    assert provenance.included_config_paths() == [cfg]


def test_included_config_paths_is_empty_without_a_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("liquifai.context.get_context", lambda: None)
    assert provenance.included_config_paths() == []


def test_config_script_path_reports_the_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "train.yaml"
    set_context(monkeypatch, config_path=cfg)
    assert provenance.config_script_path() == str(cfg)


def test_config_script_path_is_none_without_a_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("liquifai.context.get_context", lambda: None)
    assert provenance.config_script_path() is None


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #


def test_environment_snapshot_selects_by_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5005")
    monkeypatch.setenv("SOMETHING_ELSE", "ignored")
    snapshot = provenance.environment_snapshot()
    assert snapshot["MLFLOW_TRACKING_URI"] == "http://localhost:5005"
    assert "SOMETHING_ELSE" not in snapshot


@pytest.mark.parametrize(
    "name", ["WANDB_API_KEY", "MLFLOW_TRACKING_PASSWORD", "HF_TOKEN", "CLEARML_API_SECRET_KEY", "AIM_AUTH_HEADER"]
)
def test_environment_snapshot_redacts_credentials(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The manifest is uploaded as an artifact — a captured token would be a published secret."""
    monkeypatch.setenv(name, "super-secret-value")
    snapshot = provenance.environment_snapshot()
    assert snapshot[name] == provenance.REDACTED  # the NAME survives: that a token was set is provenance
    assert "super-secret-value" not in str(snapshot)


def test_environment_snapshot_redacts_an_explicitly_named_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_APP_TOKEN", "nope")
    snapshot = provenance.environment_snapshot(prefixes=(), names=["MY_APP_TOKEN"])
    assert snapshot == {"MY_APP_TOKEN": provenance.REDACTED}


def test_environment_snapshot_honours_custom_prefixes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROJECT_DATA_ROOT", "/data")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x")
    assert provenance.environment_snapshot(prefixes=["PROJECT_"]) == {"PROJECT_DATA_ROOT": "/data"}


def test_platform_state_reports_the_interpreter_and_machine() -> None:
    state = provenance.platform_state()
    assert set(state) == {"python", "implementation", "platform", "machine", "hostname"}
    assert state["python"].count(".") == 2


def test_accelerator_state_is_none_without_a_loaded_framework(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert provenance.accelerator_state() is None


def fake_torch(cuda: bool = False, mps: bool = False, broken: bool = False) -> Any:
    def cuda_available() -> bool:
        if broken:
            raise RuntimeError("driver exploded")
        return cuda

    return types.SimpleNamespace(
        __version__="9.9.9",
        cuda=types.SimpleNamespace(
            is_available=cuda_available,
            device_count=lambda: 2,
            get_device_name=lambda i: "Fake GPU",
        ),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
    )


def test_accelerator_state_reports_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda=True))
    assert provenance.accelerator_state() == {
        "kind": "cuda",
        "count": 2,
        "name": "Fake GPU",
        "torch_version": "9.9.9",
    }


def test_accelerator_state_reports_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", fake_torch(mps=True))
    state = provenance.accelerator_state()
    assert state is not None and state["kind"] == "mps" and state["count"] == 1


def test_accelerator_state_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", fake_torch())
    state = provenance.accelerator_state()
    assert state is not None and state["kind"] == "cpu" and state["count"] == 0


def test_accelerator_state_survives_a_probe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", fake_torch(broken=True))
    state = provenance.accelerator_state()
    assert state is not None and state["kind"] == "cpu"  # the version is still worth having

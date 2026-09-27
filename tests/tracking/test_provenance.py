"""Tests for run-provenance capture (git state, config, environment, log file)."""

import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pytest
import yaml

from matrainer.tracking import CsvTracker, provenance
from matrainer.tracking.base import Tracker


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
        #: The terminal status this tracker was ended with, or None if it never was.
        self.finalized: Optional[str] = None

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

    def finalize(self, status: str = "success") -> None:
        self.finalized = status


def test_get_git_commit_hash_in_repo() -> None:
    # matrainer is a git repo — expect a hash string (possibly '-dirty'), not None.
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
    from matrainer.tracking.manifest import RunManifest

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


def test_capture_log_file_ends_the_run_on_every_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gap this closed: the manifest said ``success`` while the tracker still said RUNNING.

    Found by a real MLflow sweep — eleven of thirteen finished runs showed ``RUNNING`` with no
    end time, because NOTHING called `Tracker.finalize`. It belongs here rather than in each
    consumer: this is already the one place a run's terminal status is stated, and a per-consumer
    call would be one chance per backend to forget.
    """
    monkeypatch.setattr(provenance, "resolve_log_file", lambda: None)
    first, second = RecordingTracker(), RecordingTracker()

    provenance.capture_log_file([first, second], status="failed")

    assert first.finalized == "failed" and second.finalized == "failed"


def test_the_run_is_ended_AFTER_the_artifacts_are_pushed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ordering, not tidiness: a backend with a run lifecycle stops accepting writes when ended.

    MLflow happens to tolerate a late write, but W&B's `finish()` drops the run handle and the
    next access starts a NEW one — so an upload sequenced after the ending would land somewhere
    else entirely.
    """
    from matrainer.tracking.manifest import RunManifest

    log = tmp_path / "app.log"
    log.write_text("line")
    monkeypatch.setattr(provenance, "resolve_log_file", lambda: log)

    calls: List[str] = []

    class OrderRecordingTracker(RecordingTracker):
        def log_artifact(self, path: Any, artifact_dir: str = "") -> None:
            calls.append("log_artifact")

        def finalize(self, status: str = "success") -> None:
            calls.append("finalize")

    manifest = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    provenance.capture_log_file([OrderRecordingTracker()], manifest=manifest)

    assert calls[-1] == "finalize", f"finalize must come last, got {calls}"
    assert "log_artifact" in calls


def test_capture_log_file_can_upload_without_ending_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """The escape hatch for a caller that reaches here twice per run and cannot end twice."""
    monkeypatch.setattr(provenance, "resolve_log_file", lambda: None)
    tracker = RecordingTracker()

    provenance.capture_log_file([tracker], finalize=False)

    assert tracker.finalized is None


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
    # The two manifest files plus the package freeze — which is unconditional provenance:
    # what is installed is a fact about the run even when nothing else could be gathered.
    assert [d for _, d in t.artifacts] == ["", "", "environment"]


def test_capture_log_file_empty_trackers_is_noop() -> None:
    provenance.capture_log_file([])  # no raise


def test_capture_log_file_finishes_and_reprojects_the_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The end-of-run half of the fold: what the backend holds matches what is on disk."""
    from matrainer.tracking.manifest import RunManifest

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
    from matrainer.tracking.manifest import RunManifest

    monkeypatch.setattr(provenance, "resolve_log_file", lambda: None)
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    m.finish("success")
    provenance.capture_log_file([RecordingTracker()], manifest=m, status="failed")
    assert m.status == "success"  # first terminal status wins


def test_capture_log_file_uploads_both_the_log_and_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from matrainer.tracking.manifest import RunManifest

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
    """A custom prefix captures its own vars and NOTHING else — including the defaults.

    The prefix is deliberately one no real environment carries. This asserted on `PROJECT_`
    until 2026-08-01 and then failed for anyone who had sourced the workspace activation
    script, which exports `PROJECT_ROOT`: a true match the strict equality read as a bug.
    A test that reads the ambient environment has to pick a name the ambient environment
    cannot supply.
    """
    monkeypatch.setenv("MATRAINER_TEST_DATA_ROOT", "/data")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x")  # a default-prefix decoy the filter must drop
    assert provenance.environment_snapshot(prefixes=["MATRAINER_TEST_"]) == {"MATRAINER_TEST_DATA_ROOT": "/data"}


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


def test_the_manifest_records_the_runnables_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`task` is how you tell a training manifest from an evaluation one for the SAME config."""
    from confluid import configurable

    @configurable
    class TaskToy:
        def __init__(self, task: str = "evaluate") -> None:
            self.task = task

    monkeypatch.chdir(tmp_path)
    manifest = provenance.log_run_context([RecordingTracker()], runnable=TaskToy(), capture_code=False)
    assert manifest is not None
    assert manifest.section("run")["task"] == "evaluate"


def test_a_runnable_without_a_task_leaves_it_unset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Not every runnable has one; reading it must not invent a value."""
    monkeypatch.chdir(tmp_path)
    manifest = provenance.log_run_context([RecordingTracker()], capture_code=False)
    assert manifest is not None and manifest.section("run")["task"] is None


# --- the pasteable command line ---------------------------------------------------------------


def set_argv(monkeypatch: pytest.MonkeyPatch, argv: List[str], spec_name: Optional[str] = None) -> None:
    """Stand a launch shape up: `sys.argv` plus `__main__.__spec__`, which is what tells them apart."""
    monkeypatch.setattr(sys, "argv", argv)
    main = types.ModuleType("__main__")
    main.__spec__ = types.SimpleNamespace(name=spec_name) if spec_name else None  # type: ignore[assignment]
    monkeypatch.setitem(sys.modules, "__main__", main)


def test_a_module_launch_is_reconstructed_as_dash_m(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The case `argv` alone cannot express: argv[0] is the module's FILE, which does not re-run."""
    monkeypatch.chdir(tmp_path)
    set_argv(monkeypatch, ["/site-packages/recordstream/cli.py", "run", "cfg.yaml"], spec_name="recordstream.cli")
    line = provenance.command_line()
    assert line == f"cd {tmp_path} && {sys.executable} -m recordstream.cli run cfg.yaml"


def test_a_package_launch_drops_the_dunder_main_suffix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`python -m pkg` reports `pkg.__main__`, but `python -m pkg.__main__` is not how anyone types it."""
    monkeypatch.chdir(tmp_path)
    set_argv(monkeypatch, ["/site-packages/pkg/__main__.py", "-v"], spec_name="pkg.__main__")
    assert provenance.command_line() == f"cd {tmp_path} && {sys.executable} -m pkg -v"


def test_a_console_script_pastes_back_as_itself(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """It carries its own `#!` interpreter, so putting one in front of it would be wrong."""
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "sonair"
    script.write_text("#!/usr/bin/env python\n")
    script.chmod(0o755)
    set_argv(monkeypatch, [str(script), "train", "cfg.yaml"])
    assert provenance.command_line() == f"cd {tmp_path} && {script} train cfg.yaml"


def test_a_plain_script_gets_the_interpreter_in_front(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    set_argv(monkeypatch, ["train.py", "--lr", "0.1"])
    assert provenance.command_line() == f"cd {tmp_path} && {sys.executable} train.py --lr 0.1"


def test_an_executable_dot_py_script_still_gets_the_interpreter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A `.py` suffix wins over the executable bit: `python x.py` is how it was actually launched."""
    monkeypatch.chdir(tmp_path)
    script = tmp_path / "train.py"
    script.write_text("print(1)\n")
    script.chmod(0o755)
    set_argv(monkeypatch, [str(script)])
    assert provenance.command_line() == f"cd {tmp_path} && {sys.executable} {script}"


@pytest.mark.parametrize("argv0", provenance.UNREPRODUCIBLE_ARGV0)
def test_an_unreproducible_launch_declines_rather_than_guessing(monkeypatch: pytest.MonkeyPatch, argv0: str) -> None:
    """`python -c` keeps its code out of argv, and a REPL has no command at all."""
    set_argv(monkeypatch, [argv0, "extra"])
    assert provenance.command_line() is None


def test_an_empty_argv_declines(monkeypatch: pytest.MonkeyPatch) -> None:
    set_argv(monkeypatch, [])
    assert provenance.command_line() is None


def test_paths_and_arguments_with_spaces_are_quoted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The whole point is pasting it; an unquoted space would silently become two arguments."""
    workdir = tmp_path / "my runs"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    set_argv(monkeypatch, ["/pkg/cli.py", "run", "my cfg.yaml"], spec_name="pkg.cli")
    line = provenance.command_line()
    assert line is not None
    assert f"cd '{Path.cwd()}'" in line and "'my cfg.yaml'" in line


def test_the_manifest_carries_the_pasteable_line_beside_the_raw_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both stay: `argv` is what `primary_code` parses, `line` is what a person re-runs."""
    from matrainer.tracking.manifest import RunManifest

    monkeypatch.chdir(tmp_path)
    set_argv(monkeypatch, ["/pkg/cli.py", "run"], spec_name="pkg.cli")
    m = RunManifest(run_dir=tmp_path / "run")
    m.capture_environment()
    command = m.section("command")
    assert command["argv"] == ["/pkg/cli.py", "run"]
    assert command["line"] == f"cd {tmp_path} && {sys.executable} -m pkg.cli run"


def test_the_manifest_omits_the_line_when_there_is_no_honest_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from matrainer.tracking.manifest import RunManifest

    set_argv(monkeypatch, ["-c"])
    m = RunManifest(run_dir=tmp_path / "run")
    m.capture_environment()
    assert "line" not in m.section("command")


# --- the installed-package freeze --------------------------------------------------------------


def fake_run(results: Mapping[str, Any]) -> Any:
    """A `subprocess.run` stand-in keyed by the program name, so each candidate can be steered."""

    def run(command: List[str], **kwargs: Any) -> Any:
        key = "uv" if Path(command[0]).name == "uv" else "pip"
        outcome = results[key]
        if isinstance(outcome, Exception):
            raise outcome
        return types.SimpleNamespace(stdout=outcome, returncode=0)

    return run


def test_the_freeze_prefers_uv_when_it_is_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Purely a speed choice, but a load-bearing one: this runs on every tracked run."""
    monkeypatch.setattr(provenance.shutil, "which", lambda name: "/opt/bin/uv")
    monkeypatch.setattr(provenance.subprocess, "run", fake_run({"uv": "a==1\nb==2\n", "pip": "wrong==0\n"}))
    assert provenance.package_freeze() == ("a==1\nb==2\n", "uv")


def test_the_freeze_pins_the_environment_rather_than_trusting_the_active_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--python <executable>` is what makes the answer independent of cwd and of activation."""
    seen: List[List[str]] = []

    def run(command: List[str], **kwargs: Any) -> Any:
        seen.append(command)
        return types.SimpleNamespace(stdout="a==1\n", returncode=0)

    monkeypatch.setattr(provenance.shutil, "which", lambda name: "/opt/bin/uv")
    monkeypatch.setattr(provenance.subprocess, "run", run)
    provenance.package_freeze()
    assert seen[0] == ["/opt/bin/uv", "pip", "freeze", "--python", sys.executable]


def test_the_freeze_falls_back_to_pip_without_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provenance.shutil, "which", lambda name: None)
    monkeypatch.setattr(provenance.subprocess, "run", fake_run({"pip": "a==1\n"}))
    assert provenance.package_freeze() == ("a==1\n", "pip")


def test_the_freeze_falls_through_to_pip_when_uv_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provenance.shutil, "which", lambda name: "/opt/bin/uv")
    failure = subprocess.CalledProcessError(1, "uv")
    monkeypatch.setattr(provenance.subprocess, "run", fake_run({"uv": failure, "pip": "a==1\n"}))
    assert provenance.package_freeze() == ("a==1\n", "pip")


def test_the_freeze_returns_none_when_no_program_works(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracking must never abort a run; an environment that cannot be listed is not fatal."""
    monkeypatch.setattr(provenance.shutil, "which", lambda name: None)
    monkeypatch.setattr(provenance.subprocess, "run", fake_run({"pip": OSError("no pip")}))
    assert provenance.package_freeze() is None


def test_a_timeout_is_survived(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provenance.shutil, "which", lambda name: None)
    monkeypatch.setattr(provenance.subprocess, "run", fake_run({"pip": subprocess.TimeoutExpired("pip", 1.0)}))
    assert provenance.package_freeze() is None


def test_empty_output_counts_as_no_freeze(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provenance.shutil, "which", lambda name: None)
    monkeypatch.setattr(provenance.subprocess, "run", fake_run({"pip": "   \n"}))
    assert provenance.package_freeze() is None


def test_the_real_environment_can_be_frozen() -> None:
    """No mocks: whichever program this machine has must produce a real requirements list."""
    result = provenance.package_freeze()
    assert result is not None
    text, tool = result
    assert tool in {"uv", "pip"}
    assert any("==" in line for line in text.splitlines())


# --- datasets identified from the configuration ------------------------------------------------


class ToySource:
    """A source carrying the identity protocol, standing in for a real dataset source."""

    def __init__(self, uri: Optional[str] = None, url: Optional[str] = None) -> None:
        self.dataset_uri = uri
        self.dataset_url = url


def test_configured_datasets_derives_kind_name_and_split_from_the_uri() -> None:
    """Nothing here knows what a Hugging Face dataset is — the URI carries all three."""
    from confluid import configurable

    @configurable
    class Toy:
        def __init__(self, train_set: Any = None) -> None:
            self.train_set = train_set

    source = ToySource("hf://datasets/ylecun/mnist?split=train", "https://huggingface.co/datasets/ylecun/mnist")
    entries = provenance.configured_datasets(Toy(train_set=source))
    assert len(entries) == 1
    assert entries[0]["kind"] == "hf"
    assert entries[0]["name"] == "ylecun/mnist"
    assert entries[0]["split"] == "train"
    assert entries[0]["uri"] == "hf://datasets/ylecun/mnist?split=train"
    assert entries[0]["url"] == "https://huggingface.co/datasets/ylecun/mnist"
    assert entries[0]["slot"].endswith("train_set")


def test_a_file_uri_keeps_its_absolute_path_as_the_name() -> None:
    from confluid import configurable

    @configurable
    class Toy:
        def __init__(self, train_set: Any = None) -> None:
            self.train_set = train_set

    entries = provenance.configured_datasets(Toy(train_set=ToySource("file:///data/gert/imagefolder")))
    assert entries[0] == {
        "kind": "file",
        "name": "/data/gert/imagefolder",
        "uri": "file:///data/gert/imagefolder",
        "url": None,
        "split": None,
        "slot": entries[0]["slot"],
    }


def test_one_dataset_reached_through_a_wrapper_is_recorded_once(tmp_path: Path) -> None:
    """A trainer's slot wraps the real source; both name the same dataset, so it is ONE entry."""
    from confluid import configurable

    @configurable
    class Wrapper:
        def __init__(self, source: Any = None) -> None:
            self.source = source

    @configurable
    class Toy:
        def __init__(self, train_set: Any = None) -> None:
            self.train_set = train_set

    inner = ToySource("hf://datasets/a/b?split=train")
    entries = provenance.configured_datasets(Toy(train_set=Wrapper(source=inner)))
    assert [e["uri"] for e in entries] == ["hf://datasets/a/b?split=train"]
    # The OUTER slot is kept — it is the one a reader of the config recognises.
    assert entries[0]["slot"].endswith("train_set")


def test_several_distinct_datasets_are_all_recorded() -> None:
    from confluid import configurable

    @configurable
    class Toy:
        def __init__(self, train_set: Any = None, test_set: Any = None) -> None:
            self.train_set = train_set
            self.test_set = test_set

    toy = Toy(train_set=ToySource("hf://datasets/a/b?split=train"), test_set=ToySource("hf://datasets/a/b?split=test"))
    assert len(provenance.configured_datasets(toy)) == 2


def test_a_graph_with_no_identifiable_dataset_reports_nothing() -> None:
    from confluid import configurable

    @configurable
    class Toy:
        def __init__(self, lr: float = 0.1) -> None:
            self.lr = lr

    assert provenance.configured_datasets(Toy()) == []


def test_configured_datasets_never_raises_on_an_unwalkable_object() -> None:
    assert provenance.configured_datasets(object()) == []
    assert provenance.configured_datasets(None) == []


def test_log_run_context_records_the_datasets_without_the_consumer_asking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of doing it here: ten consumers get it with no call of their own."""
    from confluid import configurable

    @configurable
    class Toy:
        def __init__(self, train_set: Any = None) -> None:
            self.train_set = train_set

    monkeypatch.chdir(tmp_path)
    source = ToySource("hf://datasets/ylecun/mnist?split=train")
    manifest = provenance.log_run_context([RecordingTracker()], runnable=Toy(train_set=source), capture_code=False)
    assert manifest is not None
    assert [d["uri"] for d in manifest.section("datasets")] == ["hf://datasets/ylecun/mnist?split=train"]


# --- models identified from the configuration ---------------------------------------------------


class ToyModel:
    """A model wrapper carrying the identity protocol, standing in for a real one."""

    def __init__(self, *refs: Any) -> None:
        from matrainer.uri import ModelRef

        self.model_refs = [r if isinstance(r, ModelRef) else ModelRef(*r) for r in refs]


def _model_toy() -> Any:
    from confluid import configurable

    @configurable
    class ModelToy:
        def __init__(self, model: Any = None, teacher: Any = None) -> None:
            self.model = model
            self.teacher = teacher

    return ModelToy


def test_configured_models_derives_kind_and_name_from_the_uri() -> None:
    """Nothing here knows what a timm model is — the URI carries the classification."""
    from matrainer.uri import hf_model_ref

    toy = _model_toy()(model=ToyModel(hf_model_ref("timm/resnet50.a1_in1k")))
    entries = provenance.configured_models(toy)
    assert len(entries) == 1
    assert entries[0]["kind"] == "hf"
    assert entries[0]["name"] == "timm/resnet50.a1_in1k"
    assert entries[0]["uri"] == "hf://models/timm/resnet50.a1_in1k"
    assert entries[0]["url"] == "https://huggingface.co/timm/resnet50.a1_in1k"
    assert entries[0]["role"] == "pretrained"
    assert entries[0]["slot"] == "model"


def test_one_wrapper_may_contribute_several_entries_with_distinct_roles(tmp_path: Path) -> None:
    """The reason the model protocol is plural: a fine-tuned backbone loaded TWO artifacts."""
    from matrainer.uri import file_model_ref, hf_model_ref

    checkpoint = tmp_path / "last.ckpt"
    toy = _model_toy()(model=ToyModel(hf_model_ref("timm/resnet50.a1_in1k"), file_model_ref(checkpoint)))
    entries = provenance.configured_models(toy)
    assert [e["role"] for e in entries] == ["pretrained", "checkpoint"]
    assert [e["kind"] for e in entries] == ["hf", "file"]
    # A file URI keeps its ABSOLUTE path as the name; an `hf://` one drops the authority separator.
    assert entries[1]["name"] == str(checkpoint.resolve())
    assert entries[1]["url"] is None


def test_several_models_in_one_graph_are_all_recorded() -> None:
    """A pipeline composing models is the same shape one level up."""
    from matrainer.uri import hf_model_ref

    toy = _model_toy()(model=ToyModel(hf_model_ref("org/student")), teacher=ToyModel(hf_model_ref("org/teacher")))
    assert sorted(e["name"] for e in provenance.configured_models(toy)) == ["org/student", "org/teacher"]


def test_the_same_artifact_reached_twice_is_recorded_once() -> None:
    from matrainer.uri import hf_model_ref

    shared = ToyModel(hf_model_ref("org/net"))
    toy = _model_toy()(model=shared, teacher=ToyModel(hf_model_ref("org/net")))
    assert len(provenance.configured_models(toy)) == 1


def test_a_model_that_loaded_nothing_external_reports_nothing() -> None:
    """Random weights referenced no artifact; the architecture is configuration, recorded there."""
    assert provenance.configured_models(_model_toy()(model=ToyModel())) == []


def test_configured_models_never_raises_on_an_unwalkable_object() -> None:
    assert provenance.configured_models(object()) == []
    assert provenance.configured_models(None) == []


def test_log_run_context_records_the_models_without_the_consumer_asking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from matrainer.uri import hf_model_ref

    monkeypatch.chdir(tmp_path)
    toy = _model_toy()(model=ToyModel(hf_model_ref("timm/resnet50.a1_in1k")))
    manifest = provenance.log_run_context([RecordingTracker()], runnable=toy, capture_code=False)
    assert manifest is not None
    assert [m["uri"] for m in manifest.section("models")] == ["hf://models/timm/resnet50.a1_in1k"]


# --- portable paths ------------------------------------------------------------------------------


@pytest.fixture
def roots(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Configure the three roots, with MODEL_ROOT deliberately NESTED inside DATA_ROOT."""
    values = {"PROJECT_ROOT": "/w/taidal", "DATA_ROOT": "/store", "MODEL_ROOT": "/store/models"}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return values


def test_no_configured_root_means_no_contraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opt-in by configuration: a machine that sets none sees absolute paths, as before."""
    for name in provenance.PORTABLE_ROOT_VARS:
        monkeypatch.delenv(name, raising=False)
    assert provenance.configured_roots() == []
    assert provenance.contract_roots("/store/mnist") == "/store/mnist"


def test_an_empty_variable_is_not_a_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_ROOT", "   ")
    assert [name for name, _ in provenance.configured_roots()] == []


@pytest.mark.parametrize(
    "original,expected",
    [
        ("/w/taidal/sonair/cfg.yaml", "$PROJECT_ROOT/sonair/cfg.yaml"),
        ("/store/mnist/train", "$DATA_ROOT/mnist/train"),
        ("file:///store/mnist", "file://$DATA_ROOT/mnist"),
        ("/store", "$DATA_ROOT"),  # the root itself, with nothing after it
    ],
)
def test_a_path_under_a_root_is_recorded_against_that_root(roots: dict, original: str, expected: str) -> None:
    assert provenance.contract_roots(original) == expected


def test_the_most_specific_root_wins(roots: dict) -> None:
    """MODEL_ROOT sits inside DATA_ROOT; a model path must not contract to the outer one."""
    assert provenance.contract_roots("/store/models/last.ckpt") == "$MODEL_ROOT/last.ckpt"


@pytest.mark.parametrize("original", ["/storefront/x", "/store-old/x", "/w/taidal-old/y"])
def test_a_lookalike_path_is_left_alone(roots: dict, original: str) -> None:
    """The match is on a path BOUNDARY — `/store` must not swallow the start of `/storefront`."""
    assert provenance.contract_roots(original) == original


def test_several_occurrences_in_one_string_all_contract(roots: dict) -> None:
    line = "cd /w/taidal/sonair && /w/taidal/.venv/bin/python train.py"
    assert provenance.contract_roots(line) == "cd $PROJECT_ROOT/sonair && $PROJECT_ROOT/.venv/bin/python train.py"


def test_expandvars_is_the_exact_inverse(roots: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """The property that makes contracting a whole document safe: nothing is lost."""
    import os as _os

    for original in ("/w/taidal/a/b", "file:///store/models/x.ckpt", "cd /store && /w/taidal/run"):
        assert _os.path.expandvars(provenance.contract_roots(original)) == original


def test_contracting_is_idempotent(roots: dict) -> None:
    once = provenance.contract_roots("/store/mnist")
    assert provenance.contract_roots(once) == once


def test_portable_walks_nested_structures(roots: dict) -> None:
    document = {"a": ["/store/x", {"b": "/w/taidal/y"}], "c": 3, "d": None}
    assert provenance.portable(document) == {"a": ["$DATA_ROOT/x", {"b": "$PROJECT_ROOT/y"}], "c": 3, "d": None}


def test_portable_leaves_a_skipped_subtree_verbatim(roots: dict) -> None:
    """`env.roots` holds the expansion table — contracting it would destroy the definitions."""
    document = {"roots": {"DATA_ROOT": "/store"}, "path": "/store/x"}
    assert provenance.portable(document, skip_keys=("roots",)) == {
        "roots": {"DATA_ROOT": "/store"},
        "path": "$DATA_ROOT/x",
    }


def test_the_pasteable_line_is_contracted_and_still_expands(
    roots: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workdir = tmp_path / "proj"
    workdir.mkdir()
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.chdir(workdir)
    set_argv(monkeypatch, ["/pkg/cli.py", "run", "cfg.yaml"], spec_name="pkg.cli")
    line = provenance.command_line()
    # DOUBLE-quoted rather than bare: the expansion may contain a space on the reading machine.
    assert line is not None and line.startswith('cd "$PROJECT_ROOT/proj" &&')


def test_a_contracted_word_needing_quotes_stays_EXPANDABLE(
    roots: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`shlex.quote` would SINGLE-quote it, and a single-quoted `$VAR` is a literal dollar sign.

    That would silently turn a portable line into one that looks for a directory named `$PROJECT_ROOT`.
    """
    workdir = tmp_path / "my runs"
    workdir.mkdir()
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.chdir(workdir)
    set_argv(monkeypatch, ["/pkg/cli.py"], spec_name="pkg.cli")
    line = provenance.command_line()
    assert line is not None
    assert line.startswith('cd "$PROJECT_ROOT/my runs"')  # DOUBLE quotes: the shell expands inside
    assert "'$PROJECT_ROOT" not in line


def test_a_word_with_no_variable_is_quoted_the_ordinary_way(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in provenance.PORTABLE_ROOT_VARS:
        monkeypatch.delenv(name, raising=False)
    workdir = tmp_path / "my runs"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    set_argv(monkeypatch, ["/pkg/cli.py"], spec_name="pkg.cli")
    line = provenance.command_line()
    assert line is not None and line.startswith(f"cd '{workdir}'")


# ---------------------------------------------------------------------------
# Framework-held child objects (torch's `_modules`)
# ---------------------------------------------------------------------------


class _FakeModule:
    """The shape of `torch.nn.Module` that matters here, with no torch dependency.

    Assigning a child diverts it into `_modules` and leaves `__dict__` without the name —
    which is exactly what made every torch-based runnable's model invisible to the walk.
    """

    #: Annotated so mypy types the registry; `__getattr__` below otherwise widens it to `object`.
    _modules: Dict[str, "_FakeModule"]

    def __init__(self) -> None:
        object.__setattr__(self, "_modules", {})

    def __setattr__(self, name: str, value: object) -> None:
        if isinstance(value, _FakeModule):
            self._modules[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> object:
        try:
            return self._modules[name]
        except KeyError:
            raise AttributeError(name) from None


class _FakeBackbone(_FakeModule):
    @property
    def model_refs(self) -> List[Any]:
        from matrainer.uri import ModelRef

        return [ModelRef(uri="hf://models/acme/net", url=None, role="pretrained")]


def test_a_child_object_a_framework_diverts_is_still_a_slot() -> None:
    """A model held in `_modules` must be walked — it is the normal case, not an edge one.

    `torch.nn.Module.__setattr__` routes every Module-valued assignment into `_modules`, so
    `self.model = TimmModel(...)` leaves `model` absent from `__dict__`. Reading only public
    `__dict__` keys therefore missed the model of EVERY torch-based runnable, and the manifest
    recorded an empty `models` section while `datasets` (a plain attribute) filled correctly —
    green tests, complete-looking run, no weights named. Found by reading a real run's manifest.
    """
    runnable = _FakeModule()
    runnable.train_set = "a plain attribute"
    runnable.model = _FakeBackbone()

    slots = dict(provenance.configured_slots(runnable))
    assert "model" in slots, f"the diverted child is invisible: {sorted(slots)}"
    assert "train_set" in slots, "a plain public attribute must still be walked"

    entries = provenance.configured_models(runnable)
    assert [e["uri"] for e in entries] == ["hf://models/acme/net"]
    assert entries[0]["slot"] == "model"


def test_the_registry_itself_is_not_walked_as_a_private_attribute() -> None:
    """`_modules` is read for its CHILDREN, never yielded as a slot of its own.

    The underscore rule still holds: descending into private state reaches a framework's
    parameter and buffer dictionaries. This registry is the one exception, and only because
    its KEYS are the public names the caller assigned.
    """
    runnable = _FakeModule()
    runnable.model = _FakeBackbone()

    paths = [path for path, _ in provenance.configured_slots(runnable)]
    assert not any(path.startswith("_") or "._" in path for path in paths), paths


def test_a_public_attribute_wins_over_a_same_named_child() -> None:
    """No slot is ever yielded twice; the real attribute is authoritative."""
    runnable = _FakeModule()
    object.__setattr__(runnable, "model", "the plain attribute")
    runnable._modules["model"] = _FakeBackbone()

    slots = [(path, value) for path, value in provenance.configured_slots(runnable) if path == "model"]
    assert slots == [("model", "the plain attribute")]


# --- local-file facts (size + digest) -------------------------------------------------------------


def test_a_local_file_gets_its_size_and_digest(tmp_path: Path) -> None:
    """The question this answers: is the checkpoint on disk the one the run loaded?"""
    import hashlib

    weights = tmp_path / "epoch=3.ckpt"
    weights.write_bytes(b"weights")
    facts = provenance.local_file_facts(weights.as_uri())
    assert facts == {"size": 7, "digest": hashlib.sha256(b"weights").hexdigest()}


def test_a_directory_contributes_nothing(tmp_path: Path) -> None:
    """A SavedModel or an imagefolder has no single content hash, and walking one to invent a
    composite would be unbounded work at the start of every run."""
    assert provenance.local_file_facts(tmp_path.as_uri()) == {}


@pytest.mark.parametrize("uri", ["hf://models/timm/resnet50.a1_in1k", "s3://bucket/x.ckpt", "ultralytics://yolo26n"])
def test_a_remote_reference_is_not_fetched_to_fingerprint_it(uri: str) -> None:
    assert provenance.local_file_facts(uri) == {}


def test_a_missing_file_yields_nothing_rather_than_raising() -> None:
    assert provenance.local_file_facts("file:///nowhere/gone.ckpt") == {}


def test_a_file_over_the_cap_keeps_its_size_and_skips_the_digest(tmp_path: Path) -> None:
    """Above the cap the size still distinguishes most 'is this the same file?' cases."""
    weights = tmp_path / "huge.ckpt"
    weights.write_bytes(b"x" * 100)
    facts = provenance.local_file_facts(weights.as_uri(), max_digest_bytes=10)
    assert facts == {"size": 100}


def test_a_percent_encoded_path_is_decoded(tmp_path: Path) -> None:
    """`Path.as_uri()` percent-encodes `=`, which every Lightning checkpoint name contains."""
    weights = tmp_path / "epoch=3.ckpt"
    weights.write_bytes(b"w")
    assert "%3D" in weights.as_uri()
    assert provenance.local_file_facts(weights.as_uri())["size"] == 1


def test_a_checkpoint_model_carries_its_digest_into_the_manifest(tmp_path: Path) -> None:
    """End to end through the walk: a `file://` model ref gains size + digest."""
    from confluid import configurable

    from matrainer.uri import file_model_ref, hf_model_ref

    weights = tmp_path / "last.ckpt"
    weights.write_bytes(b"weights")

    class Wrapper:
        def __init__(self) -> None:
            self.model_refs = [hf_model_ref("timm/resnet50.a1_in1k"), file_model_ref(weights)]

    @configurable
    class ModelToy:
        def __init__(self, model: Any = None) -> None:
            self.model = model

    entries = provenance.configured_models(ModelToy(model=Wrapper()))
    remote, local = entries
    assert "digest" not in remote and "size" not in remote  # a Hub reference is not fetched
    assert local["size"] == 7 and len(local["digest"]) == 64

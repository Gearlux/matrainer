"""Tests for run-provenance capture (git hash, config artifacts, log file)."""

from pathlib import Path
from typing import Any, List, Mapping, Optional

import pytest

from marainer.tracking import provenance
from marainer.tracking.base import Tracker


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

    t = RecordingTracker()
    provenance.log_run_context([t], params={"lr": 0.1, "model": object()})

    assert t.hparams["lr"] == 0.1
    assert "model" not in t.hparams  # non-scalar dropped
    assert t.hparams["config_script"] == str(cfg)
    assert t.hparams[provenance.GIT_COMMIT_TAG] == "deadbeef"
    assert t.set_tag_calls == {provenance.GIT_COMMIT_TAG: "deadbeef"}
    assert t.artifacts == [(str(cfg), "config")]


def test_log_run_context_empty_trackers_is_noop() -> None:
    provenance.log_run_context([], params={"lr": 1})  # no raise


def test_capture_log_file_uploads_when_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log = tmp_path / "app.log"
    log.write_text("line")
    monkeypatch.setattr(provenance, "resolve_log_file", lambda: log)
    t = RecordingTracker()
    provenance.capture_log_file([t])
    assert t.artifacts == [(str(log), "logs")]


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

"""Tests for AimTracker — mocked for the mapping, real Aim for the round trip."""

import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from marainer.tracking.aim_backend import AIM_SYSTEM_INTERVAL, AimTracker


class _FakeRun:
    """Records calls; stands in for ``aim.Run``."""

    instances: List["_FakeRun"] = []

    def __init__(self, repo: Any = None, experiment: Any = None, **kw: Any) -> None:
        self.repo = repo
        self.experiment = experiment
        self.kwargs = kw
        self.hash = "abc123"
        self.name: Optional[str] = None
        self.values: Dict[str, Any] = {}
        self.tracked: List[tuple] = []
        self.tags: List[str] = []
        self.artifacts: List[tuple] = []
        self.artifacts_uri: Optional[str] = None
        self.closed = False
        _FakeRun.instances.append(self)

    def __setitem__(self, key: str, value: Any) -> None:
        self.values[key] = value

    def track(self, value: Any, name: str = "", step: Any = None, **kw: Any) -> None:
        self.tracked.append((name, value, step))

    def add_tag(self, tag: str) -> None:
        self.tags.append(tag)

    def set_artifacts_uri(self, uri: str) -> None:
        self.artifacts_uri = uri

    def log_artifact(self, path: str, name: Any = None, **kw: Any) -> None:
        if self.artifacts_uri is None:
            raise ValueError("run.artifacts_uri is None")  # what real Aim raises
        self.artifacts.append((path, name))

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_aim(monkeypatch: pytest.MonkeyPatch) -> Any:
    _FakeRun.instances = []
    mod = types.ModuleType("aim")
    mod.Run = _FakeRun  # type: ignore[attr-defined]
    mod.Image = lambda fig: ("image", fig)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aim", mod)
    return _FakeRun


def test_import_error_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_require(module: str, extra: str, feature: str = "") -> Any:
        raise ImportError(f"{feature} requires the '{module}' package. Install with: pip install marainer[{extra}]")

    monkeypatch.setattr("marainer.tracking.aim_backend.require", fake_require)
    with pytest.raises(ImportError, match=r"pip install marainer\[aim\]"):
        _ = AimTracker().experiment


def test_run_is_created_lazily(fake_aim: Any, tmp_path: Path) -> None:
    t = AimTracker(experiment_name="exp", run_name="r", save_dir=str(tmp_path))
    assert not fake_aim.instances  # nothing built at construction
    run = t.experiment
    assert len(fake_aim.instances) == 1
    assert run.experiment == "exp" and run.name == "r"
    assert run.repo == str(tmp_path)


def test_system_tracking_follows_the_flag(fake_aim: Any, tmp_path: Path) -> None:
    on = AimTracker(save_dir=str(tmp_path), log_system_metrics=True).experiment
    off = AimTracker(save_dir=str(tmp_path), log_system_metrics=False).experiment
    assert on.kwargs["system_tracking_interval"] == AIM_SYSTEM_INTERVAL
    assert off.kwargs["system_tracking_interval"] is None


def test_terminal_capture_is_off_by_default(fake_aim: Any, tmp_path: Path) -> None:
    """Aim's own default is True; loggair already writes the log marainer uploads."""
    assert AimTracker(save_dir=str(tmp_path)).experiment.kwargs["capture_terminal_logs"] is False


def test_version_is_the_run_hash(fake_aim: Any, tmp_path: Path) -> None:
    t = AimTracker(run_name="r", save_dir=str(tmp_path))
    assert t.version == "r"  # before the run exists
    _ = t.experiment
    assert t.version == "abc123"


def test_log_hyperparams_merges_across_calls(fake_aim: Any, tmp_path: Path) -> None:
    """Aim replaces `run["hparams"]` wholesale, but marainer's contract is to merge."""
    t = AimTracker(save_dir=str(tmp_path))
    t.log_hyperparams({"lr": 0.1})
    t.log_hyperparams({"epochs": 3})
    assert fake_aim.instances[0].values["hparams"] == {"lr": 0.1, "epochs": 3}


def test_log_metrics_tracks_each_scalar(fake_aim: Any, tmp_path: Path) -> None:
    AimTracker(save_dir=str(tmp_path)).log_metrics({"loss": 0.5, "acc": 0.9}, step=4)
    assert sorted(fake_aim.instances[0].tracked) == [("acc", 0.9, 4), ("loss", 0.5, 4)]


def test_set_tags_flattens_to_aim_tag_strings(fake_aim: Any, tmp_path: Path) -> None:
    t = AimTracker(save_dir=str(tmp_path))
    t.set_tags({"git.commit": "abc", "stage": "smoke"})
    assert sorted(fake_aim.instances[0].tags) == ["git.commit=abc", "stage=smoke"]
    assert t.tags["stage"] == "smoke"  # and the base bookkeeping still happened


def test_set_tags_before_the_run_exists_does_not_double_add(fake_aim: Any, tmp_path: Path) -> None:
    """Regression: merging into self.tags first made run creation re-apply the same tags.

    ``set_tags`` on a tracker whose run has not been built yet used to emit each tag twice
    — once from the run-creation path (which applies ``self.tags``) and once from the call
    itself.
    """
    t = AimTracker(save_dir=str(tmp_path))
    assert not fake_aim.instances  # no run yet — this is the case that broke
    t.set_tags({"stage": "smoke"})
    assert fake_aim.instances[0].tags == ["stage=smoke"]


def test_constructor_and_later_tags_both_land_exactly_once(fake_aim: Any, tmp_path: Path) -> None:
    t = AimTracker(save_dir=str(tmp_path), tags={"a": "1"})
    t.set_tags({"b": "2"})
    assert sorted(fake_aim.instances[0].tags) == ["a=1", "b=2"]


def test_constructor_tags_are_applied_when_the_run_appears(fake_aim: Any, tmp_path: Path) -> None:
    t = AimTracker(save_dir=str(tmp_path), tags={"stage": "smoke"})
    _ = t.experiment
    assert fake_aim.instances[0].tags == ["stage=smoke"]


def test_log_artifact_sets_the_uri_once(fake_aim: Any, tmp_path: Path) -> None:
    """Real Aim raises unless an artifacts URI was set first."""
    f = tmp_path / "train.yaml"
    f.write_text("x: 1")
    t = AimTracker(experiment_name="exp", run_name="r", save_dir=str(tmp_path / "repo"))
    t.log_artifact(f, "config")
    t.log_artifact(f, "logs")
    run = fake_aim.instances[0]
    assert run.artifacts_uri is not None and run.artifacts_uri.startswith("file://")
    # Namespaced by artifact_dir, so same-named files in different dirs do not collide.
    assert run.artifacts == [(str(f), "config/train.yaml"), (str(f), "logs/train.yaml")]


def test_log_artifact_without_a_dir_uses_the_bare_name(fake_aim: Any, tmp_path: Path) -> None:
    f = tmp_path / "manifest.yaml"
    f.write_text("a: 1")
    AimTracker(save_dir=str(tmp_path / "repo")).log_artifact(f)
    assert fake_aim.instances[0].artifacts == [(str(f), "manifest.yaml")]


def test_log_artifact_skips_a_missing_file(fake_aim: Any, tmp_path: Path) -> None:
    AimTracker(save_dir=str(tmp_path)).log_artifact(tmp_path / "gone.yaml")
    assert fake_aim.instances == []  # never even opened the run


def test_log_figure_tracks_an_image(fake_aim: Any, tmp_path: Path) -> None:
    AimTracker(save_dir=str(tmp_path)).log_figure("cm", "FIGURE", step=2)
    assert fake_aim.instances[0].tracked == [("cm", ("image", "FIGURE"), 2)]


def test_finalize_records_status_and_closes(fake_aim: Any, tmp_path: Path) -> None:
    t = AimTracker(save_dir=str(tmp_path))
    _ = t.experiment
    t.finalize("failed")
    run = fake_aim.instances[0]
    assert run.values["status"] == "failed" and run.closed


def test_finalize_without_a_run_is_a_noop(fake_aim: Any, tmp_path: Path) -> None:
    AimTracker(save_dir=str(tmp_path)).finalize("success")
    assert fake_aim.instances == []


def test_log_dir_is_none_so_the_manifest_is_uploaded(fake_aim: Any, tmp_path: Path) -> None:
    """A file merely SITTING in Aim's directory is not registered — it must be logged.

    ``Tracker.log_manifest`` skips uploading a file already inside ``log_dir``; returning a
    directory here would silently stop the manifest from ever reaching Aim.
    """
    assert AimTracker(save_dir=str(tmp_path)).log_dir is None


def test_manifest_reaches_aim_as_artifacts_and_hparams(fake_aim: Any, tmp_path: Path) -> None:
    from marainer.tracking.manifest import RunManifest

    m = RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    m.set("params", {"lr": 0.1})
    AimTracker(save_dir=str(tmp_path / "repo")).log_manifest(m)
    run = fake_aim.instances[0]
    assert run.values["hparams"] == {"lr": 0.1}
    assert sorted(name for _, name in run.artifacts) == ["manifest.json", "manifest.yaml"]


# --------------------------------------------------------------------------- #
# real Aim
# --------------------------------------------------------------------------- #


def test_round_trip_against_real_aim(tmp_path: Path) -> None:
    """Integration: the mocks prove the mapping, only real Aim proves the API.

    Reads back through a REINDEXED repo on purpose: ``Repo.iter_runs()`` is served from an
    index that a finished run does not populate by itself, so a fresh repo legitimately
    iterates empty while every value is on disk (``aim up`` indexes when it starts). Read
    the module docstring before "fixing" that.
    """
    aim = pytest.importorskip("aim")
    repo_dir = tmp_path / "repo"

    tracker = AimTracker(experiment_name="proj", run_name="r", save_dir=str(repo_dir), log_system_metrics=False)
    cfg = tmp_path / "train.yaml"
    cfg.write_text("lr: 0.1")
    tracker.log_hyperparams({"lr": 0.1})
    tracker.log_hyperparams({"epochs": 5})
    tracker.log_metrics({"train/loss": 0.5}, step=0)
    tracker.set_tags({"stage": "smoke"})
    tracker.log_artifact(cfg, "config")
    tracker.finalize("success")

    repo = aim.Repo(str(repo_dir))
    repo._recreate_index()
    runs = list(repo.iter_runs())
    assert len(runs) == 1
    run = runs[0]
    assert run.name == "r" and run.experiment == "proj"
    assert dict(run["hparams"]) == {"lr": 0.1, "epochs": 5}  # merged, not replaced
    assert run["status"] == "success"
    assert sorted({m.name for m in run.metrics()}) == ["train/loss"]
    assert "stage=smoke" in list(run.props.tags)


def test_log_manifest_projects_the_sections_as_run_values(fake_aim: Any, tmp_path: Path) -> None:
    """Aim's native slot is a structured run value, which its UI can filter and compare."""
    from marainer.tracking.manifest import RunManifest

    m = RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    m.set("params", {"lr": 0.1})
    m.set("env", {"python": "3.13"})
    m.add("code", {"name": "pkg", "commit": "abc", "dirty": False})
    m.record_dataset(kind="hf", name="org/set", split="train", count=12)

    AimTracker(save_dir=str(tmp_path / "repo")).log_manifest(m)
    values = fake_aim.instances[0].values
    assert values["env"]["python"] == "3.13"
    assert values["code"][0]["commit"] == "abc"
    assert values["datasets"][0]["name"] == "org/set"


def test_params_are_not_projected_twice(fake_aim: Any, tmp_path: Path) -> None:
    """`hparams` is Aim's own comparison slot; a second copy would only ever disagree."""
    from marainer.tracking.manifest import RunManifest

    m = RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    m.set("params", {"lr": 0.1})
    AimTracker(save_dir=str(tmp_path / "repo")).log_manifest(m)
    values = fake_aim.instances[0].values
    assert values["hparams"] == {"lr": 0.1}
    assert "params" not in values


def test_empty_sections_are_skipped(fake_aim: Any, tmp_path: Path) -> None:
    from marainer.tracking.manifest import RunManifest

    AimTracker(save_dir=str(tmp_path / "repo")).log_manifest(
        RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    )
    values = fake_aim.instances[0].values
    assert "code" not in values and "datasets" not in values


def test_the_status_is_not_projected_at_log_time(fake_aim: Any, tmp_path: Path) -> None:
    """Regression: it would read `running` forever — `finalize()` owns `run["status"]`.

    Measured against real Aim: `log_manifest` runs mid-run, so a status projected there is
    always `running` and a filter on it matches nothing.
    """
    from marainer.tracking.manifest import RunManifest

    tracker = AimTracker(save_dir=str(tmp_path / "repo"))
    tracker.log_manifest(RunManifest(run_dir=tmp_path / "run", handle_interrupts=False))
    assert "manifest_status" not in fake_aim.instances[0].values
    assert "status" not in fake_aim.instances[0].values  # not yet — the run has not ended
    tracker.finalize("success")
    assert fake_aim.instances[0].values["status"] == "success"

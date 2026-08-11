"""Tests for :mod:`matrainer.tracking.manifest` — the crash-safe run manifest.

The three properties worth pinning, all easy to regress:

* both files are written from every mutation, atomically, and agree with each other;
* the section order in the YAML is the declared one (a diff between two runs must line
  up, so the order is contract, not cosmetics);
* the terminal status is stamped by whichever shutdown path fires FIRST, and the signal
  handler hands the signal back rather than swallowing it.
"""

import atexit
import hashlib
import json
import signal
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, get_args

import pytest
import yaml

from matrainer.tracking import CsvTracker, Tracker, TrackerGroup
from matrainer.tracking import manifest as manifest_module
from matrainer.tracking import provenance
from matrainer.tracking.manifest import (
    LIST_SECTIONS,
    MANIFEST_JSON,
    MANIFEST_VERSION,
    MANIFEST_YAML,
    MAPPING_SECTIONS,
    SECTIONS,
    ManifestSection,
    RunManifest,
    _atomic_write,
    _plain,
)


@pytest.fixture(autouse=True)
def no_real_hooks(monkeypatch: pytest.MonkeyPatch) -> Dict[str, List[Any]]:
    """Keep the test process clean: record hook registrations instead of installing them.

    Without this every manifest built here would leave a live ``atexit`` hook writing to
    a deleted ``tmp_path`` at interpreter shutdown, and would replace pytest's own
    signal handling.
    """
    calls: Dict[str, List[Any]] = {"atexit": [], "signal": []}

    def fake_register(fn: Any, *a: Any, **k: Any) -> Any:
        calls["atexit"].append(fn)
        return fn

    def fake_signal(num: Any, handler: Any) -> Any:
        calls["signal"].append((num, handler))
        return signal.SIG_DFL

    monkeypatch.setattr(atexit, "register", fake_register)
    monkeypatch.setattr(manifest_module.signal, "signal", fake_signal)
    return calls


def read_yaml(m: RunManifest) -> Dict[str, Any]:
    doc: Dict[str, Any] = yaml.safe_load(m.path.read_text())
    return doc


# --------------------------------------------------------------------------- #
# section declaration
# --------------------------------------------------------------------------- #


def test_sections_are_derived_from_the_literal() -> None:
    assert SECTIONS == get_args(ManifestSection)


def test_mapping_and_list_sections_partition_the_declaration() -> None:
    assert set(MAPPING_SECTIONS) | set(LIST_SECTIONS) == set(SECTIONS)
    assert not set(MAPPING_SECTIONS) & set(LIST_SECTIONS)


# --------------------------------------------------------------------------- #
# construction + location
# --------------------------------------------------------------------------- #


def test_zero_arg_construction_works() -> None:
    m = RunManifest()
    assert m.status == "running"
    assert m.run_name  # a random slug was drawn
    assert m.stopped_at is None


def test_constructor_does_no_io(tmp_path: Path) -> None:
    target = tmp_path / "not-created-yet"
    RunManifest(run_dir=target)
    assert not target.exists()


def test_directory_is_created_on_access(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "run"
    m = RunManifest(run_dir=target)
    assert m.directory == target
    assert target.is_dir()


def test_directory_recomputes_from_current_inputs(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path / "first")
    assert m.directory.name == "first"
    m.run_dir = tmp_path / "second"  # post-construction change must not be stale
    assert m.directory.name == "second"


def test_default_directory_derives_from_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    m = RunManifest(experiment_name="exp", run_name="brave_otter")
    assert m.directory == Path("runs") / "exp" / "brave_otter"


# --------------------------------------------------------------------------- #
# dual write
# --------------------------------------------------------------------------- #


def test_write_emits_both_files_with_identical_content(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path, experiment_name="exp", run_name="r", task="fit")
    m.write()
    assert m.path.name == MANIFEST_YAML and m.json_path.name == MANIFEST_JSON
    assert yaml.safe_load(m.path.read_text()) == json.loads(m.json_path.read_text())


def test_yaml_is_tag_free_and_readable(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path)
    m.set("params", {"lr": 0.001, "model": Path("/tmp/weights.pt")})
    text = m.path.read_text()
    assert "!!python" not in text and "!class" not in text and "!ref" not in text
    assert "lr: 0.001" in text


def test_sections_are_emitted_in_declaration_order(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path)
    m.write()
    keys = list(read_yaml(m))
    assert keys[:2] == ["manifest_version", "status"]
    assert tuple(keys[2:]) == SECTIONS


def test_every_section_is_present_even_when_empty(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path)
    m.write()
    doc = read_yaml(m)
    assert doc["manifest_version"] == MANIFEST_VERSION
    for section in LIST_SECTIONS:
        assert doc[section] == []
    for section in MAPPING_SECTIONS:
        assert isinstance(doc[section], dict)


def test_run_section_carries_the_identity(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path, experiment_name="exp", run_name="r", task="evaluate")
    m.write()
    run = read_yaml(m)["run"]
    assert run["experiment_name"] == "exp"
    assert run["run_name"] == "r"
    assert run["task"] == "evaluate"
    assert run["started_at"] and run["stopped_at"] is None


def test_write_leaves_no_temp_files_behind(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path)
    m.write()
    assert sorted(p.name for p in tmp_path.iterdir()) == [MANIFEST_JSON, MANIFEST_YAML]


def test_every_mutation_rewrites(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path)
    m.set("env", {"python": "3.13"})
    assert read_yaml(m)["env"] == {"python": "3.13"}  # no explicit write() call
    m.add("code", {"name": "matrainer", "commit": "abc", "dirty": False})
    assert read_yaml(m)["code"][0]["commit"] == "abc"


def test_write_failure_is_reported_once_and_never_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: List[str] = []
    debugs: List[str] = []
    monkeypatch.setattr(manifest_module.logger, "warning", lambda msg: warnings.append(msg))
    monkeypatch.setattr(manifest_module.logger, "debug", lambda msg: debugs.append(msg))

    def boom(path: Path, text: str) -> None:
        raise OSError("read-only filesystem")

    monkeypatch.setattr(manifest_module, "_atomic_write", boom)
    m = RunManifest(run_dir=tmp_path)
    m.write()
    m.write()
    m.write()
    assert len(warnings) == 1  # loud once — an unwritable run dir is actionable
    assert len(debugs) == 2  # quiet thereafter — a per-mutation write must not flood


def test_write_recovers_after_a_transient_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: List[str] = []
    monkeypatch.setattr(manifest_module.logger, "warning", lambda msg: warnings.append(msg))
    real = manifest_module._atomic_write
    state = {"fail": True}

    def flaky(path: Path, text: str) -> None:
        if state["fail"]:
            raise OSError("transient")
        real(path, text)

    monkeypatch.setattr(manifest_module, "_atomic_write", flaky)
    m = RunManifest(run_dir=tmp_path)
    m.write()  # outage -> warning
    state["fail"] = False
    m.write()  # recovered -> the flag resets
    state["fail"] = True
    m.write()  # a NEW outage is loud again rather than silently swallowed
    assert len(warnings) == 2


def test_atomic_write_replaces_existing_content(tmp_path: Path) -> None:
    target = tmp_path / "f.txt"
    _atomic_write(target, "first")
    _atomic_write(target, "second")
    assert target.read_text() == "second"
    assert not (tmp_path / "f.txt.tmp").exists()


# --------------------------------------------------------------------------- #
# accumulation
# --------------------------------------------------------------------------- #


def test_set_merges_rather_than_replaces(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path)
    m.set("params", {"lr": 0.1})
    m.set("params", {"batch_size": 32})
    assert read_yaml(m)["params"] == {"lr": 0.1, "batch_size": 32}


def test_add_appends(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path)
    m.add("outputs", {"kind": "checkpoint", "path": "a.ckpt"})
    m.add("outputs", {"kind": "checkpoint", "path": "b.ckpt"})
    assert [o["path"] for o in read_yaml(m)["outputs"]] == ["a.ckpt", "b.ckpt"]


def test_set_on_a_list_section_is_rejected(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path)
    with pytest.raises(ValueError, match="takes add\\(\\), not set\\(\\)"):
        m.set("datasets", {"name": "x"})


def test_add_on_a_mapping_section_is_rejected(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path)
    with pytest.raises(ValueError, match="takes set\\(\\), not add\\(\\)"):
        m.add("params", {"lr": 0.1})


@pytest.mark.parametrize("call", ["set", "add"])
def test_unknown_section_is_rejected(tmp_path: Path, call: str) -> None:
    m = RunManifest(run_dir=tmp_path)
    with pytest.raises(ValueError, match="unknown manifest section 'nope'"):
        getattr(m, call)("nope", {"a": 1})


def test_mutations_do_not_alias_the_caller_s_dict(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path)
    payload = {"lr": 0.1}
    m.set("params", payload)
    payload["lr"] = 999
    assert m.to_dict()["params"]["lr"] == 0.1


def test_record_dataset_entry_shape(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path)
    m.record_dataset(kind="hf", name="org/set", revision="abc123", split="train", count=12, note="smoke")
    entry = read_yaml(m)["datasets"][0]
    assert entry == {
        "kind": "hf",
        "name": "org/set",
        "uri": None,
        "url": None,
        "revision": "abc123",
        "split": "train",
        "count": 12,
        "digest": None,
        "note": "smoke",
    }


def test_record_model_entry_shape(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path)
    m.record_model(source="zoo", name="org/net", uri="hf://models/org/net", role="pretrained", note="smoke")
    assert read_yaml(m)["models"][0] == {
        "source": "zoo",
        "name": "org/net",
        "uri": "hf://models/org/net",
        "url": None,
        "role": "pretrained",
        "revision": None,
        "digest": None,
        "note": "smoke",
    }


def test_record_output_defaults_to_checkpoint_and_stringifies_the_path(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path)
    m.record_output(tmp_path / "best.ckpt", monitor="val/acc", value=0.91)
    entry = read_yaml(m)["outputs"][0]
    assert entry["kind"] == "checkpoint"
    assert entry["path"] == str(tmp_path / "best.ckpt")
    assert entry["monitor"] == "val/acc"


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #


def test_finish_stamps_status_and_stop_time(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path)
    m.finish("success")
    doc = read_yaml(m)
    assert doc["status"] == "success"
    assert doc["run"]["stopped_at"]


def test_first_terminal_status_wins(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path)
    m.finish("success")
    stopped = read_yaml(m)["run"]["stopped_at"]
    m.finish("failed")
    m._on_exit()
    doc = read_yaml(m)
    assert doc["status"] == "success"
    assert doc["run"]["stopped_at"] == stopped


def test_atexit_marks_an_unfinished_run_interrupted(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path)
    m.write()
    m._on_exit()
    assert read_yaml(m)["status"] == "interrupted"


def test_hooks_are_installed_once_on_the_first_write(tmp_path: Path, no_real_hooks: Dict[str, List[Any]]) -> None:
    m = RunManifest(run_dir=tmp_path)
    m.write()
    m.write()
    assert len(no_real_hooks["atexit"]) == 1
    assert [num for num, _ in no_real_hooks["signal"]] == [signal.SIGINT, signal.SIGTERM]


def test_handle_interrupts_false_installs_atexit_only(tmp_path: Path, no_real_hooks: Dict[str, List[Any]]) -> None:
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    m.write()
    assert len(no_real_hooks["atexit"]) == 1
    assert no_real_hooks["signal"] == []


def test_signal_install_failure_is_tolerated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def not_main_thread(num: int, handler: Any) -> Any:
        raise ValueError("signal only works in main thread")

    monkeypatch.setattr(manifest_module.signal, "signal", not_main_thread)
    m = RunManifest(run_dir=tmp_path)
    m.write()  # must not raise
    assert m.path.exists()


def test_signal_handler_stamps_then_chains_to_the_previous_handler(tmp_path: Path) -> None:
    seen: List[Tuple[int, Any]] = []
    m = RunManifest(run_dir=tmp_path)
    m.write()
    m._previous_handlers[signal.SIGINT] = lambda num, frame: seen.append((num, frame))
    m._on_signal(signal.SIGINT, None)
    assert read_yaml(m)["status"] == "interrupted"
    assert seen == [(signal.SIGINT, None)]


def test_signal_handler_re_raises_when_the_previous_handler_was_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    killed: List[Tuple[int, int]] = []
    restored: List[Tuple[int, Any]] = []
    monkeypatch.setattr(manifest_module.os, "kill", lambda pid, num: killed.append((pid, num)))
    monkeypatch.setattr(manifest_module.signal, "signal", lambda num, handler: restored.append((num, handler)))
    # handle_interrupts=False so `restored` records only what the HANDLER does, not the
    # two installs `write()` would otherwise make through the same patched function.
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    m.write()
    m._previous_handlers[signal.SIGTERM] = signal.SIG_DFL
    m._on_signal(signal.SIGTERM, None)
    assert read_yaml(m)["status"] == "interrupted"
    assert restored == [(signal.SIGTERM, signal.SIG_DFL)]
    assert [num for _, num in killed] == [signal.SIGTERM]


def test_signal_handler_honours_an_ignored_signal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    killed: List[Any] = []
    monkeypatch.setattr(manifest_module.os, "kill", lambda pid, num: killed.append(num))
    m = RunManifest(run_dir=tmp_path)
    m.write()
    m._previous_handlers[signal.SIGINT] = signal.SIG_IGN
    m._on_signal(signal.SIGINT, None)
    assert read_yaml(m)["status"] == "interrupted"
    assert killed == []  # the process chose to ignore it; so do we


def test_mutation_after_finish_still_persists(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path)
    m.finish("success")
    m.record_output(tmp_path / "late.ckpt")
    doc = read_yaml(m)
    assert doc["status"] == "success"
    assert doc["outputs"][0]["path"].endswith("late.ckpt")


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT cannot be delivered to a child on Windows")
def test_a_real_sigint_lands_as_interrupted_without_swallowing_ctrl_c(tmp_path: Path) -> None:
    """End-to-end in a REAL process — the mocked handler tests cannot prove either half.

    Two things must hold at once and they pull against each other: the manifest must
    record ``interrupted``, AND the caller's Ctrl-C must still behave like Ctrl-C. A
    handler that stamped the status and returned would satisfy the first and silently
    break the second (the training loop would sail on through a Ctrl-C).
    """
    child = tmp_path / "child.py"
    out = tmp_path / "out"
    child.write_text(
        "import sys, time\n"
        "from matrainer.tracking import RunManifest\n"
        f"m = RunManifest(run_dir={str(out)!r}, run_name='r')\n"
        "m.set('params', {'lr': 0.1})\n"
        "print('READY', flush=True)\n"
        "time.sleep(60)\n"
        "m.finish('success')\n"
    )
    proc = subprocess.Popen([sys.executable, str(child)], stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "READY"  # sync on output, never on a sleep
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=30) != 0  # KeyboardInterrupt still ends the process
    finally:
        proc.kill()
        if proc.stdout is not None:
            proc.stdout.close()

    doc = yaml.safe_load((out / MANIFEST_YAML).read_text())
    assert doc["status"] == "interrupted"
    assert doc["run"]["stopped_at"]
    assert doc["params"] == {"lr": 0.1}  # everything gathered before the signal survived
    assert json.loads((out / MANIFEST_JSON).read_text()) == doc


def test_lock_is_reentrant_so_a_handler_can_interrupt_a_mutation() -> None:
    m = RunManifest(handle_interrupts=False)
    assert isinstance(m._lock, type(threading.RLock()))


# --------------------------------------------------------------------------- #
# capture
# --------------------------------------------------------------------------- #


def test_capture_environment_fills_command_and_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", "a.yaml"])
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5005")
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    m.capture_environment()
    doc = read_yaml(m)
    assert doc["command"]["argv"] == ["train.py", "--config", "a.yaml"]
    assert doc["command"]["cwd"] and doc["command"]["executable"]
    assert doc["env"]["python"].count(".") == 2
    assert doc["env"]["variables"]["MLFLOW_TRACKING_URI"] == "http://localhost:5005"


def test_capture_environment_redacts_secrets_into_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The manifest is an uploaded artifact — the raw value must not reach the disk at all."""
    monkeypatch.setenv("WANDB_API_KEY", "super-secret-value")
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    m.capture_environment()
    assert "super-secret-value" not in m.path.read_text()
    assert "super-secret-value" not in m.json_path.read_text()


def test_capture_environment_accepts_custom_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROJECT_DATA_ROOT", "/data")
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    m.capture_environment(env_prefixes=(), env_names=["PROJECT_DATA_ROOT"])
    assert read_yaml(m)["env"]["variables"] == {"PROJECT_DATA_ROOT": "/data"}


def test_capture_code_writes_once_for_many_repositories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [{"name": f"pkg{i}", "path": f"/p{i}", "commit": "c", "dirty": False} for i in range(5)]
    monkeypatch.setattr(manifest_module.provenance, "code_state", lambda: entries)
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    writes: List[int] = []
    real_write = m.write

    def counting_write() -> None:
        writes.append(1)
        real_write()

    monkeypatch.setattr(m, "write", counting_write)
    m.capture_code()
    assert len(writes) == 1  # one write for five repositories, not five
    assert [e["name"] for e in read_yaml(m)["code"]] == [f"pkg{i}" for i in range(5)]


def test_capture_code_is_a_noop_outside_any_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manifest_module.provenance, "code_state", list)
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    m.capture_code()
    assert not m.path.exists()  # nothing to say, nothing written


def test_capture_config_records_the_input_tree_by_path_and_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = tmp_path / "train.yaml"
    cfg.write_text("x: 1")
    monkeypatch.setattr(manifest_module.provenance, "config_script_path", lambda: str(cfg))
    monkeypatch.setattr(manifest_module.provenance, "included_config_paths", lambda: [cfg])
    m = RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    m.capture_config()
    config = read_yaml(m)["config"]
    assert config["script"] == str(cfg)
    assert config["includes"] == [{"path": str(cfg), "sha256": provenance.file_digest(cfg)}]
    assert "final" not in config  # no runnable -> nothing resolved to dump


def test_capture_config_dumps_the_resolved_config_to_its_own_file(tmp_path: Path) -> None:
    """The dump keeps the config library's tags, so it must NOT be inlined in the manifest."""
    from confluid import configurable

    @configurable
    class CaptureToy:
        def __init__(self, lr: float = 0.25, epochs: int = 3) -> None:
            self.lr, self.epochs = lr, epochs

    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    m.capture_config(CaptureToy())

    final = read_yaml(m)["config"]["final"]
    assert final["path"] == "config/final.yaml"  # relative, so the run dir stays portable
    dumped = (tmp_path / "config" / "final.yaml").read_text()
    assert "_target_:" in dumped  # the config markers live THERE ...
    assert "_target_:" not in m.path.read_text()  # ... and never in the manifest
    assert final["sha256"] == hashlib.sha256(dumped.encode()).hexdigest()

    params = {k.rsplit(".", 1)[-1]: v for k, v in read_yaml(m)["params"].items()}
    assert params["lr"] == 0.25 and params["epochs"] == 3


def test_capture_config_survives_an_undumpable_runnable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(obj: Any) -> str:
        raise TypeError("cannot represent")

    monkeypatch.setattr("confluid.dump", boom)
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    m.capture_config(object())  # must not raise
    assert "final" not in read_yaml(m)["config"]


def test_capture_config_survives_an_unwritable_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("confluid.dump", lambda obj: "a: 1")

    def boom(path: Path, text: str) -> None:
        raise OSError("read-only")

    monkeypatch.setattr(manifest_module, "_atomic_write", boom)
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    m.capture_config(object())  # must not raise
    assert m._sections["config"] == {}


# --------------------------------------------------------------------------- #
# reading (what a projection uses)
# --------------------------------------------------------------------------- #


def test_section_returns_a_copy_not_the_store(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    m.set("params", {"lr": 0.1})
    got = m.section("params")
    got["lr"] = 999
    assert m.section("params")["lr"] == 0.1  # a projection cannot mutate what it projects


def test_section_run_includes_the_identity(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path, experiment_name="exp", run_name="r", task="fit", handle_interrupts=False)
    m.set("run", {"note": "extra"})
    run = m.section("run")
    assert run["experiment_name"] == "exp" and run["run_name"] == "r" and run["task"] == "fit"
    assert run["note"] == "extra"


def test_section_rejects_an_unknown_name(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    with pytest.raises(ValueError, match="unknown manifest section 'nope'"):
        m.section("nope")  # type: ignore[arg-type]  # the Literal rejects it statically too


def test_primary_code_prefers_the_repository_holding_the_cwd(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    m.set("command", {"cwd": "/w/project/subdir"})
    m.add("code", {"name": "lib", "path": "/w/lib", "commit": "aaa"})
    m.add("code", {"name": "project", "path": "/w/project", "commit": "bbb"})
    primary = m.primary_code()
    assert primary is not None and primary["commit"] == "bbb"


def test_primary_code_does_not_match_a_sibling_by_prefix(tmp_path: Path) -> None:
    """`/w/project-old` must not match a cwd of `/w/project` on a bare string prefix."""
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    m.set("command", {"cwd": "/w/project"})
    m.add("code", {"name": "other", "path": "/w/project-old", "commit": "aaa"})
    primary = m.primary_code()
    assert primary is not None and primary["name"] == "other"  # falls back, not a prefix match


def test_primary_code_uses_the_entry_script_when_the_cwd_is_elsewhere(tmp_path: Path) -> None:
    """`python /w/project/train.py` from /tmp still has an obvious primary repository."""
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    m.set("command", {"cwd": "/somewhere/else", "argv": ["/w/project/train.py"]})
    m.add("code", {"name": "lib", "path": "/w/lib", "commit": "aaa"})
    m.add("code", {"name": "project", "path": "/w/project", "commit": "bbb"})
    primary = m.primary_code()
    assert primary is not None and primary["commit"] == "bbb"


def test_primary_code_prefers_the_config_script_over_argv(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    m.set("command", {"cwd": "/nowhere", "argv": ["/w/lib/runner.py"]})
    m.set("config", {"script": "/w/project/train.yaml"})
    m.add("code", {"name": "lib", "path": "/w/lib", "commit": "aaa"})
    m.add("code", {"name": "project", "path": "/w/project", "commit": "bbb"})
    primary = m.primary_code()
    assert primary is not None and primary["commit"] == "bbb"


def test_primary_code_uses_a_lone_record_even_with_no_match(tmp_path: Path) -> None:
    """One repository is not a choice — there is nothing else it could be."""
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    m.set("command", {"cwd": "/somewhere/else"})
    m.add("code", {"name": "lib", "path": "/w/lib", "commit": "aaa"})
    primary = m.primary_code()
    assert primary is not None and primary["commit"] == "aaa"


def test_primary_code_is_none_when_nothing_ties_the_run_to_a_repository(tmp_path: Path) -> None:
    """Regression: this used to return the alphabetically-first workspace package.

    Found by running against a real MLflow server — a run launched from outside every
    checkout got an unrelated package's commit stamped as `mlflow.source.git.commit`,
    and a wrong answer is indistinguishable from a right one in a UI.
    """
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    m.set("command", {"cwd": "/somewhere/else", "argv": ["-"]})
    m.add("code", {"name": "aisland", "path": "/w/aisland", "commit": "aaa"})
    m.add("code", {"name": "project", "path": "/w/project", "commit": "bbb"})
    assert m.primary_code() is None


def test_primary_code_is_none_without_any_code(tmp_path: Path) -> None:
    assert RunManifest(run_dir=tmp_path, handle_interrupts=False).primary_code() is None


# --------------------------------------------------------------------------- #
# projection onto a tracker
# --------------------------------------------------------------------------- #


class RecordingTracker(Tracker):
    def __init__(self, log_dir: Optional[Path] = None, **kw: Any) -> None:
        super().__init__(**kw)
        self._dir = log_dir
        self.artifacts: List[Tuple[str, str]] = []

    @property
    def experiment(self) -> str:
        return "handle"

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        pass

    def log_artifact(self, path: Any, artifact_dir: str = "") -> None:
        self.artifacts.append((str(path), artifact_dir))

    @property
    def log_dir(self) -> Optional[Path]:
        return self._dir


def test_log_manifest_uploads_both_files(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    tracker = RecordingTracker()
    tracker.log_manifest(m)
    assert [Path(p).name for p, _ in tracker.artifacts] == [MANIFEST_YAML, MANIFEST_JSON]


def test_log_manifest_flushes_before_uploading(tmp_path: Path) -> None:
    """A tracker must never upload a manifest older than the caller's last mutation."""
    m = RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    tracker = RecordingTracker()
    tracker.log_manifest(m)
    assert m.path.exists() and m.json_path.exists()


def test_log_manifest_does_not_copy_a_file_next_to_itself(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    tracker = RecordingTracker(log_dir=tmp_path)
    tracker.log_manifest(m)
    assert tracker.artifacts == []  # already inside this backend's own directory


def test_log_manifest_uploads_when_the_manifest_lives_elsewhere(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path / "manifest-home", handle_interrupts=False)
    tracker = RecordingTracker(log_dir=tmp_path / "tracker-home")
    (tmp_path / "tracker-home").mkdir()
    tracker.log_manifest(m)
    assert len(tracker.artifacts) == 2


def test_log_manifest_uploads_the_side_artifacts_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The manifest references these by path + digest — a remote backend must receive them.

    Without this a run tracked to a remote-only backend holds a manifest full of paths to
    files that exist on exactly one machine, which is the union contract broken by omission.
    """
    monkeypatch.setattr(provenance, "package_freeze", lambda: ("a==1\n", "uv"))
    m = RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    m.capture_packages()
    tracker = RecordingTracker()
    tracker.log_manifest(m)
    assert [(Path(p).name, d) for p, d in tracker.artifacts] == [
        (MANIFEST_YAML, ""),
        (MANIFEST_JSON, ""),
        ("requirements.txt", "environment"),
    ]


def test_side_artifacts_inside_the_backends_own_tree_are_not_copied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same rule the manifest pair follows — a file-backed backend must not duplicate them."""
    monkeypatch.setattr(provenance, "package_freeze", lambda: ("a==1\n", "uv"))
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    m.capture_packages()
    tracker = RecordingTracker(log_dir=tmp_path)
    tracker.log_manifest(m)
    assert tracker.artifacts == []


def test_tracker_group_fans_the_manifest_out(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path / "run", handle_interrupts=False)
    a, b = RecordingTracker(), RecordingTracker()
    TrackerGroup([a, b]).log_manifest(m)
    assert len(a.artifacts) == 2 and len(b.artifacts) == 2


def test_csv_tracker_round_trips_a_manifest(tmp_path: Path) -> None:
    """The end-to-end shape a consumer actually writes: manifest beside the tracker's artifacts."""
    tracker = CsvTracker(experiment_name="exp", run_name="r", save_dir=str(tmp_path))
    m = RunManifest(
        run_dir=tracker.log_dir,
        experiment_name=tracker.experiment_name,
        run_name=tracker.run_name,
        task="fit",
    )
    m.set("params", {"lr": 0.1})
    tracker.log_manifest(m)
    m.finish("success")
    doc = yaml.safe_load((tmp_path / "exp" / "r" / MANIFEST_YAML).read_text())
    assert doc["status"] == "success"
    assert doc["run"]["run_name"] == "r"


# --------------------------------------------------------------------------- #
# value coercion
# --------------------------------------------------------------------------- #


class _Scalar:
    def item(self) -> float:
        return 0.25


class _Unrepresentable:
    def __str__(self) -> str:
        return "<opaque>"


class _BadScalar:
    def item(self) -> Any:
        raise ValueError("more than one element")

    def __str__(self) -> str:
        return "<array>"


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (True, True),
        (3, 3),
        (1.5, 1.5),
        ("s", "s"),
        (Path("/a/b"), "/a/b"),
        ((1, 2), [1, 2]),
        ([1, 2], [1, 2]),
        (_Scalar(), 0.25),
        (_Unrepresentable(), "<opaque>"),
        (_BadScalar(), "<array>"),
    ],
)
def test_plain_coercion(value: Any, expected: Any) -> None:
    assert _plain(value) == expected


class _VersionString(str):
    """A ``str`` subclass, exactly like ``torch.__version__`` (``TorchVersion``)."""


class _Count(int):
    pass


class _Ratio(float):
    pass


class _Flag(int):
    """An int-like that is not a bool but reads like one."""


@pytest.mark.parametrize(
    "value,expected_type",
    [(_VersionString("2.13.0"), str), (_Count(3), int), (_Ratio(0.5), float), (_Flag(1), int)],
)
def test_plain_rebuilds_builtin_subclasses(value: Any, expected_type: type) -> None:
    """Regression: ``yaml.safe_dump`` refuses a SUBCLASS of a type it knows.

    Found by running against a real torch: ``torch.__version__`` is a ``str`` subclass, and
    passing it through on an ``isinstance`` check failed the entire write with
    ``('cannot represent an object', '2.13.0')`` — taking the whole manifest with it.
    """
    coerced = _plain(value)
    assert type(coerced) is expected_type


def test_a_builtin_subclass_survives_the_round_trip(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path, handle_interrupts=False)
    m.set("env", {"torch_version": _VersionString("2.13.0")})
    assert read_yaml(m)["env"]["torch_version"] == "2.13.0"


def test_plain_coerces_datetimes_to_iso() -> None:
    stamp = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    assert _plain(stamp) == stamp.isoformat()


def test_plain_recurses_into_nested_structures() -> None:
    assert _plain({"a": {"b": [Path("/x"), {"c": _Scalar()}]}}) == {"a": {"b": ["/x", {"c": 0.25}]}}


def test_plain_stringifies_non_string_keys() -> None:
    assert _plain({1: "a"}) == {"1": "a"}


def test_a_set_survives_the_round_trip(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path)
    m.set("env", {"flags": {"b", "a"}})
    assert sorted(read_yaml(m)["env"]["flags"]) == ["a", "b"]


# --- the installed-package freeze ---------------------------------------------------------------


def test_capture_packages_writes_the_freeze_beside_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pip's format goes to its OWN file; the manifest keeps only a reference to it."""
    monkeypatch.setattr(provenance, "package_freeze", lambda: ("a==1\nb==2\n", "uv"))
    m = RunManifest(run_dir=tmp_path)
    m.capture_packages()

    written = tmp_path / "environment" / "requirements.txt"
    assert written.read_text() == "a==1\nb==2\n"
    packages = read_yaml(m)["env"]["packages"]
    assert packages["path"] == "environment/requirements.txt"  # RELATIVE — a run directory travels
    assert packages["sha256"] == hashlib.sha256(b"a==1\nb==2\n").hexdigest()
    assert packages["count"] == 2
    assert packages["tool"] == "uv"


def test_capture_packages_is_a_noop_when_nothing_can_be_frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provenance, "package_freeze", lambda: None)
    m = RunManifest(run_dir=tmp_path)
    m.capture_packages()
    # `section` rather than the file: a no-op writes nothing at all, which is the point.
    assert "packages" not in m.section("env")
    assert not (tmp_path / "environment").exists()


def test_capture_packages_leaves_the_rest_of_env_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`set` merges, so the freeze reference must not displace the platform/accelerator facts."""
    monkeypatch.setattr(provenance, "package_freeze", lambda: ("a==1\n", "pip"))
    m = RunManifest(run_dir=tmp_path)
    m.capture_environment()
    m.capture_packages()
    env = read_yaml(m)["env"]
    assert env["packages"]["tool"] == "pip"
    assert "python" in env and "variables" in env


# --- side artifacts ------------------------------------------------------------------------------


def test_artifacts_lists_the_side_files_the_manifest_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manifest points at these by path + digest, so a backend holding it must reach them."""
    from confluid import configurable

    @configurable
    class ArtifactToy:
        def __init__(self, lr: float = 0.1) -> None:
            self.lr = lr

    monkeypatch.setattr(provenance, "package_freeze", lambda: ("a==1\n", "uv"))
    m = RunManifest(run_dir=tmp_path)
    m.capture_config(ArtifactToy())
    m.capture_packages()
    assert sorted(p.relative_to(tmp_path).as_posix() for p in m.artifacts()) == [
        "config/final.yaml",
        "environment/requirements.txt",
    ]


def test_artifacts_is_empty_before_anything_is_captured(tmp_path: Path) -> None:
    assert RunManifest(run_dir=tmp_path).artifacts() == []


def test_artifacts_skips_a_file_that_has_been_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller must need no existence guard, so a deleted artifact simply drops out."""
    monkeypatch.setattr(provenance, "package_freeze", lambda: ("a==1\n", "uv"))
    m = RunManifest(run_dir=tmp_path)
    m.capture_packages()
    (tmp_path / "environment" / "requirements.txt").unlink()
    assert m.artifacts() == []


def test_rewriting_an_artifact_does_not_duplicate_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provenance, "package_freeze", lambda: ("a==1\n", "uv"))
    m = RunManifest(run_dir=tmp_path)
    m.capture_packages()
    m.capture_packages()
    assert len(m.artifacts()) == 1


def test_an_unwritable_artifact_is_survived(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracking must never abort a run — a lost reference beats a dead training job."""

    def boom(path: Path, text: str) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(provenance, "package_freeze", lambda: ("a==1\n", "uv"))
    monkeypatch.setattr(manifest_module, "_atomic_write", boom)
    m = RunManifest(run_dir=tmp_path)
    m.capture_packages()  # no raise
    assert m.artifacts() == []


# --- datasets ------------------------------------------------------------------------------------


class _IdentifiedSource:
    def __init__(self, uri: str, url: Optional[str] = None) -> None:
        self.dataset_uri = uri
        self.dataset_url = url


def test_capture_datasets_records_what_the_configuration_identifies(tmp_path: Path) -> None:
    from confluid import configurable

    @configurable
    class DataToy:
        def __init__(self, train_set: Any = None) -> None:
            self.train_set = train_set

    m = RunManifest(run_dir=tmp_path)
    m.capture_datasets(DataToy(train_set=_IdentifiedSource("hf://datasets/a/b?split=train")))
    datasets = read_yaml(m)["datasets"]
    assert [d["uri"] for d in datasets] == ["hf://datasets/a/b?split=train"]
    assert datasets[0]["kind"] == "hf" and datasets[0]["name"] == "a/b" and datasets[0]["split"] == "train"


def test_capture_datasets_writes_the_whole_list_in_one_pass(tmp_path: Path) -> None:
    """One write for N datasets, the same reason `capture_code` extends in a single shot."""
    from confluid import configurable

    @configurable
    class DataToy:
        def __init__(self, train_set: Any = None, test_set: Any = None) -> None:
            self.train_set = train_set
            self.test_set = test_set

    writes: List[int] = []
    m = RunManifest(run_dir=tmp_path)
    original = m.write

    def counting_write() -> None:
        writes.append(1)
        original()

    m.write = counting_write  # type: ignore[method-assign]
    m.capture_datasets(DataToy(_IdentifiedSource("hf://datasets/a/b"), _IdentifiedSource("hf://datasets/c/d")))
    assert len(writes) == 1
    assert len(read_yaml(m)["datasets"]) == 2


def test_capture_datasets_without_a_runnable_is_a_noop(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path)
    m.capture_datasets()
    assert m.section("datasets") == []


def test_capture_datasets_with_nothing_identifiable_is_a_noop(tmp_path: Path) -> None:
    """A run reading from a source with no stable handle is not an error."""
    from confluid import configurable

    @configurable
    class PlainToy:
        def __init__(self, lr: float = 0.1) -> None:
            self.lr = lr

    m = RunManifest(run_dir=tmp_path)
    m.capture_datasets(PlainToy())
    assert m.section("datasets") == []


# --- models ---------------------------------------------------------------------------------------


class _IdentifiedModel:
    def __init__(self, *refs: Any) -> None:
        from matrainer.uri import ModelRef

        self.model_refs = [r if isinstance(r, ModelRef) else ModelRef(*r) for r in refs]


def test_capture_models_records_what_the_configuration_loads(tmp_path: Path) -> None:
    from confluid import configurable

    from matrainer.uri import hf_model_ref

    @configurable
    class ModelToy:
        def __init__(self, model: Any = None) -> None:
            self.model = model

    m = RunManifest(run_dir=tmp_path)
    m.capture_models(ModelToy(model=_IdentifiedModel(hf_model_ref("timm/resnet50.a1_in1k"))))
    models = read_yaml(m)["models"]
    assert [x["uri"] for x in models] == ["hf://models/timm/resnet50.a1_in1k"]
    assert models[0]["kind"] == "hf" and models[0]["role"] == "pretrained" and models[0]["slot"] == "model"


def test_capture_models_writes_the_whole_list_in_one_pass(tmp_path: Path) -> None:
    """One write for N artifacts — the same reason `capture_datasets` and `capture_code` do."""
    from confluid import configurable

    from matrainer.uri import file_model_ref, hf_model_ref

    @configurable
    class ModelToy:
        def __init__(self, model: Any = None) -> None:
            self.model = model

    writes: List[int] = []
    m = RunManifest(run_dir=tmp_path)
    original = m.write

    def counting_write() -> None:
        writes.append(1)
        original()

    m.write = counting_write  # type: ignore[method-assign]
    m.capture_models(ModelToy(model=_IdentifiedModel(hf_model_ref("org/net"), file_model_ref(tmp_path / "a.ckpt"))))
    assert len(writes) == 1
    assert len(read_yaml(m)["models"]) == 2


def test_capture_models_without_a_runnable_is_a_noop(tmp_path: Path) -> None:
    m = RunManifest(run_dir=tmp_path)
    m.capture_models()
    assert m.section("models") == []


def test_capture_models_with_nothing_loaded_is_a_noop(tmp_path: Path) -> None:
    """A model trained from scratch referenced no artifact, which is not an error."""
    from confluid import configurable

    @configurable
    class ScratchToy:
        def __init__(self, model: Any = None) -> None:
            self.model = model

    m = RunManifest(run_dir=tmp_path)
    m.capture_models(ScratchToy(model=_IdentifiedModel()))
    assert m.section("models") == []

"""``ResilientTracker`` — time limits, the replay queue, per-call circuit breaker, artifact bytes.

The wrapped tracker here is a scripted stand-in for a remote server: each method pops the next
outcome from its script (an exception to raise, :data:`BLOCK` to hang until the test releases
it) and records the calls that succeeded. A hang is a call blocked on an event, never a sleep.
"""

import threading
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union, cast

import pytest
from confluid import PartialClass, load

from matrainer.tracking import resilient
from matrainer.tracking.base import Tracker
from matrainer.tracking.manifest import RunManifest
from matrainer.tracking.resilient import ResilientTracker, is_network_error, is_server_error, run_with_timeout

#: A script outcome: the call hangs until the test sets the server's `release` event.
BLOCK = "block"

Outcome = Union[BaseException, str]


class ScriptedServer(Tracker):
    """A remote-tracker stand-in whose calls fail or hang as each method's script says."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.script: Dict[str, List[Outcome]] = {}
        self.calls: List[Tuple[str, Tuple[Any, ...]]] = []
        self.release = threading.Event()
        self.started = 0
        self.uploaded: List[Tuple[str, bytes, str]] = []
        self.run_url: Optional[str] = None

    def _next(self, method: str) -> None:
        outcomes = self.script.get(method) or []
        if outcomes:
            outcome = outcomes.pop(0)
            if outcome == BLOCK:
                self.release.wait()
                raise TimeoutError("released after the caller gave up")
            assert isinstance(outcome, BaseException)
            raise outcome

    @property
    def experiment(self) -> str:
        self._next("experiment")
        self.started += 1
        return "handle"

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        self._next("log_hyperparams")
        self.calls.append(("log_hyperparams", (dict(params),)))

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        self._next("log_metrics")
        self.calls.append(("log_metrics", (dict(metrics), step)))

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self._next("set_tags")
        self.calls.append(("set_tags", (dict(tags),)))

    def log_artifact(self, path: Union[str, Path], artifact_dir: str = "") -> None:
        self._next("log_artifact")
        self.uploaded.append((Path(path).name, Path(path).read_bytes(), artifact_dir))

    def log_figure(self, name: str, figure: Any, step: Optional[int] = None) -> None:
        self._next("log_figure")
        self.calls.append(("log_figure", (name, figure, step)))

    def log_text(self, name: str, text: str, step: Optional[int] = None) -> None:
        self._next("log_text")
        self.calls.append(("log_text", (name, text, step)))

    def log_manifest(self, manifest: Any) -> None:
        self._next("log_manifest")
        self.calls.append(("log_manifest", (manifest,)))

    def save(self) -> None:
        self._next("save")
        self.calls.append(("save", ()))

    def finalize(self, status: str = "success") -> None:
        self._next("finalize")
        self.calls.append(("finalize", (status,)))


class Recorder:
    """Stands in for the module logger: keeps every message by level."""

    def __init__(self) -> None:
        self.lines: Dict[str, List[str]] = {"debug": [], "info": [], "warning": []}

    def debug(self, message: str) -> None:
        self.lines["debug"].append(message)

    def info(self, message: str) -> None:
        self.lines["info"].append(message)

    def warning(self, message: str) -> None:
        self.lines["warning"].append(message)


@pytest.fixture
def log(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder()
    monkeypatch.setattr(resilient, "logger", recorder)
    return recorder


@pytest.fixture
def server(tmp_path: Path) -> Iterator[ScriptedServer]:
    s = ScriptedServer(experiment_name="exp", run_name="run", save_dir=str(tmp_path / "runs"))
    yield s
    s.release.set()  # let any thread the test left hanging finish


def wrap(server: ScriptedServer, **kw: Any) -> ResilientTracker:
    kw.setdefault("timeout", 0.05)
    kw.setdefault("finalize_retry_delay", 0.0)
    return ResilientTracker(tracker=server, **kw)


def metrics(server: ScriptedServer) -> List[Any]:
    return [args for name, args in server.calls if name == "log_metrics"]


# --------------------------------------------------------------------------- classification


@pytest.mark.parametrize(
    "error, network",
    [
        (TimeoutError("x"), True),
        (ConnectionError("x"), True),
        (RuntimeError("HTTPSConnectionPool: Max retries exceeded"), True),
        (RuntimeError("Read timed out."), True),
        (RuntimeError("Connection reset by peer"), True),
        (ValueError("invalid metric value"), False),
        (RuntimeError("500 Internal Server Error"), False),
    ],
)
def test_a_network_failure_is_told_from_a_refusal(error: BaseException, network: bool) -> None:
    assert is_network_error(error) is network


class StatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class ResponseError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__("bad")
        self.response = StatusError(status_code)


class DeclaredStatusError(Exception):
    """MLflow's shape: the status is a method's answer, and the text may hold any number."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status

    def get_http_status_code(self) -> int:
        return self.status


#: MLflow refusing a changed parameter whose new value happens to be 500 (measured 2026-10-14).
CHANGED_PARAM = (
    "Changing param values is not allowed. Param with key='epochs' was already logged with value='100'. "
    "Attempted logging new value '500'."
)


@pytest.mark.parametrize(
    "error, server_side",
    [
        (StatusError(503), True),
        (ResponseError(500), True),
        (StatusError(400), False),
        (DeclaredStatusError(503, "database is down"), True),
        (DeclaredStatusError(400, CHANGED_PARAM), False),
        (RuntimeError("500 Internal Server Error"), True),
        (RuntimeError("500 Server Error: Internal Server Error for url: http://host/api"), True),
        (RuntimeError("503 Service Unavailable"), True),
        (RuntimeError(CHANGED_PARAM), False),
        (RuntimeError("bad request"), False),
    ],
)
def test_a_server_error_is_read_from_its_status(error: BaseException, server_side: bool) -> None:
    """A status the error carries decides; a text counts only where it writes a 5xx status, never a bare 500."""
    assert is_server_error(error) is server_side


def test_mlflows_own_exceptions_are_read_by_their_status() -> None:
    exceptions = pytest.importorskip("mlflow.exceptions")
    refused = exceptions.RestException({"error_code": "INVALID_PARAMETER_VALUE", "message": CHANGED_PARAM})
    broken = exceptions.RestException({"error_code": "INTERNAL_ERROR", "message": "database is down"})
    assert not is_server_error(refused) and is_server_error(broken)


def test_run_with_timeout_returns_raises_and_gives_up() -> None:
    assert run_with_timeout(lambda a, b=0: a + b, (1,), {"b": 2}, 1.0) == 3
    with pytest.raises(KeyError):
        empty: Dict[str, int] = {}
        run_with_timeout(lambda: empty["missing"], (), {}, 1.0)
    release = threading.Event()
    try:
        with pytest.raises(TimeoutError, match="timed out after 0.05 s"):
            run_with_timeout(release.wait, (), {}, 0.05)
    finally:
        release.set()


# --------------------------------------------------------------------------- the wrapped tracker


def test_zero_arg_construction_does_nothing_until_used() -> None:
    t = ResilientTracker()
    assert t.tracker is None and t.pending == [] and not t.queueing_only
    with pytest.raises(ValueError, match="needs a `tracker:` to wrap"):
        t.log_metrics({"a": 1.0})


def test_names_directory_and_handle_are_the_wrapped_trackers(server: ScriptedServer) -> None:
    t = wrap(server)
    assert (t.experiment_name, t.run_name, t.name, t.version) == ("exp", "run", "exp", "run")
    assert t.save_dir == server.save_dir and t.log_dir is None
    server.tags["k"] = "v"
    assert t.tags == {"k": "v"}
    assert t.experiment == "handle"
    assert server.started == 2  # once to start the run, once for the handle asked for


def test_a_deferred_marker_is_built_once() -> None:
    # A config's nested `!class:` arrives built; a deferred marker is what a programmatic caller may hand in.
    marker = cast(Tracker, PartialClass(ScriptedServer, experiment_name="deferred"))
    t = ResilientTracker(tracker=marker)
    first = t.inner
    assert isinstance(first, ScriptedServer) and first.experiment_name == "deferred"
    assert t.inner is first  # written back: a second look does not build a second run
    assert first.raise_errors is True


def test_the_wrapped_tracker_is_switched_to_raise_its_errors_on_first_use(server: ScriptedServer) -> None:
    """The wrapper sees only what a call raises; a tracker keeps a failure to itself unless asked."""
    t = wrap(server)
    assert server.raise_errors is False  # building the wrapper touches nothing
    t.log_metrics({"a": 1.0})
    assert server.raise_errors is True


def test_a_config_nests_the_wrapped_tracker() -> None:
    config = load(
        "tracker: !class:matrainer.tracking.resilient.ResilientTracker\n"
        "  timeout: 2.5\n"
        "  tracker: !class:matrainer.tracking.csv_backend.CsvTracker\n"
        "    experiment_name: nested\n"
    )
    t = config["tracker"]
    assert isinstance(t, ResilientTracker) and t.timeout == 2.5
    assert t.experiment_name == "nested"


def test_the_run_starts_once_on_the_first_call(server: ScriptedServer) -> None:
    t = wrap(server)
    assert server.started == 0
    t.log_metrics({"a": 1.0}, step=0)
    t.log_metrics({"a": 2.0}, step=1)
    assert server.started == 1
    assert metrics(server) == [({"a": 1.0}, 0), ({"a": 2.0}, 1)]


@pytest.mark.parametrize("outcome", [RuntimeError("401 unauthorized"), BLOCK])
def test_a_run_that_cannot_start_turns_tracking_off(server: ScriptedServer, log: Recorder, outcome: Outcome) -> None:
    server.script["experiment"] = [outcome]
    t = wrap(server, start_timeout=0.05)
    t.log_metrics({"a": 1.0})
    t.log_artifact(__file__)
    t.finalize()
    assert server.calls == [] and server.uploaded == []
    assert t.experiment is None
    assert len(log.lines["warning"]) == 1 and "is off for this run" in log.lines["warning"][0]


# --------------------------------------------------------------------------- every call, live


def test_every_surface_call_reaches_the_wrapped_tracker(server: ScriptedServer, tmp_path: Path) -> None:
    t = wrap(server)
    artifact = tmp_path / "a.txt"
    artifact.write_text("A")
    t.log_hyperparams({"lr": 0.1})
    t.set_tags({"k": "v"})
    t.log_figure("fig", "FIGURE", step=2)
    t.log_text("note", "hello", step=3)
    t.log_manifest(cast(RunManifest, "MANIFEST"))
    t.save()
    t.log_artifact(artifact, artifact_dir="d")
    t.finalize("failed")
    assert [name for name, _ in server.calls] == [
        "log_hyperparams",
        "set_tags",
        "log_figure",
        "log_text",
        "log_manifest",
        "save",
        "finalize",
    ]
    assert server.calls[-1] == ("finalize", ("failed",))
    assert server.uploaded == [("a.txt", b"A", "d")]


def test_finalize_reports_the_run_url(server: ScriptedServer, log: Recorder) -> None:
    server.run_url = "https://tracking/run/1"
    wrap(server).finalize()
    assert "Tracked run: https://tracking/run/1" in log.lines["info"]


def test_a_failed_finalize_is_logged_not_raised(server: ScriptedServer, log: Recorder) -> None:
    server.script["finalize"] = [RuntimeError("gone")]
    wrap(server).finalize()
    assert any("Ending the ScriptedServer run failed: gone" in line for line in log.lines["warning"])


def test_the_artifact_time_limit_scales_with_its_size() -> None:
    t = ResilientTracker(timeout=5.0, artifact_upload_rate_mbps=5.0)
    assert t.artifact_timeout_for(0) == 60.0
    assert t.artifact_timeout_for(50 * 1024 * 1024) == 70.0
    assert ResilientTracker(timeout=120.0).artifact_timeout_for(0) == 120.0
    assert ResilientTracker(artifact_timeout=7.0).artifact_timeout_for(10**9) == 7.0


# --------------------------------------------------------------------------- the replay queue


def test_a_network_failure_queues_the_call_and_the_next_call_replays_it(server: ScriptedServer, log: Recorder) -> None:
    server.script["log_metrics"] = [ConnectionError("connection refused")]
    t = wrap(server)
    t.log_metrics({"a": 1.0}, step=0)
    assert [c.method for c in t.pending] == ["log_metrics"] and metrics(server) == []
    assert "Network error in log_metrics, queued it (1/10)" in log.lines["warning"][0]

    t.log_metrics({"a": 2.0}, step=1)
    assert metrics(server) == [({"a": 1.0}, 0), ({"a": 2.0}, 1)]  # in order, at their own steps
    assert t.pending == []
    assert "Sent 1 queued tracking call(s) (0 still queued)." in log.lines["info"]


def test_a_hang_is_cut_off_and_queued(server: ScriptedServer) -> None:
    server.script["log_metrics"] = [BLOCK]
    t = wrap(server)
    t.log_metrics({"a": 1.0}, step=0)
    assert [c.method for c in t.pending] == ["log_metrics"]
    server.release.set()
    t.log_metrics({"a": 2.0}, step=1)
    assert metrics(server) == [({"a": 1.0}, 0), ({"a": 2.0}, 1)]


def test_a_call_behind_a_queue_waits_its_turn(server: ScriptedServer) -> None:
    """While replay is failing, a new value queues BEHIND the old ones instead of overtaking them."""
    t = wrap(server)
    server.script["log_metrics"] = [TimeoutError("t")] * 2
    t.log_metrics({"a": 1.0}, step=0)  # live fails: queued (failure 1)
    t.log_metrics({"a": 2.0}, step=1)  # the replay of step 0 fails (failure 2): step 1 queues behind it
    assert [c.kwargs["step"] for c in t.pending] == [0, 1] and metrics(server) == []
    assert t._failures == 2
    t.log_metrics({"a": 3.0}, step=2)
    assert metrics(server) == [({"a": 1.0}, 0), ({"a": 2.0}, 1), ({"a": 3.0}, 2)]


def test_a_replayed_call_refused_for_another_reason_is_dropped(server: ScriptedServer, log: Recorder) -> None:
    server.script["log_metrics"] = [TimeoutError("t"), ValueError("rejected value")]
    t = wrap(server)
    t.log_metrics({"a": float("nan")}, step=0)
    t.log_metrics({"a": 2.0}, step=1)
    assert metrics(server) == [({"a": 2.0}, 1)] and t.pending == []
    assert any("Dropping queued log_metrics" in line for line in log.lines["warning"])


def test_max_retries_failures_in_a_row_stop_live_calls(server: ScriptedServer, log: Recorder) -> None:
    server.script["log_metrics"] = [TimeoutError("t")] * 3
    t = wrap(server, max_retries=2)
    t.log_metrics({"a": 0.0}, step=0)  # live fails: queued, failure 1
    t.log_metrics({"a": 1.0}, step=1)  # the replay fails: failure 2 — live calls stop
    assert t.queueing_only
    assert sum("2 network failures in a row" in line for line in log.lines["warning"]) == 1
    t.log_metrics({"a": 2.0}, step=2)
    assert len(server.script["log_metrics"]) == 1  # no call reached the server since
    assert [c.kwargs["step"] for c in t.pending] == [0, 1, 2]


def test_reaching_max_retries_on_a_live_call_says_so(server: ScriptedServer, log: Recorder) -> None:
    server.script["log_metrics"] = [TimeoutError("t")]
    t = wrap(server, max_retries=1)
    t.log_metrics({"a": 0.0})
    assert t.queueing_only
    message = log.lines["warning"][-1]
    assert "1 network failures in a row" in message and "log_metrics=1" in message
    assert "artifacts backed up under" in message


def test_queueing_only_says_it_is_still_on_every_25_calls(server: ScriptedServer, log: Recorder) -> None:
    server.script["log_metrics"] = [TimeoutError("t")]
    t = wrap(server, max_retries=1, backup_artifacts=False)
    t.log_metrics({"a": 0.0})
    assert "artifacts backed up" not in log.lines["warning"][-1]
    for step in range(1, 25):
        t.log_metrics({"a": float(step)})
    assert sum("Still only queueing: 25" in line for line in log.lines["warning"]) == 1


# --------------------------------------------------------------------------- refusals


def test_a_server_error_turns_off_only_that_kind_of_call(server: ScriptedServer, log: Recorder, tmp_path: Path) -> None:
    server.script["log_metrics"] = [StatusError(500)]
    t = wrap(server)
    artifact = tmp_path / "a.txt"
    artifact.write_text("A")
    t.log_metrics({"a": 1.0})
    t.log_metrics({"a": 2.0})
    t.log_artifact(artifact)
    assert metrics(server) == [] and server.uploaded == [("a.txt", b"A", "")]
    assert sum("Turning off log_metrics" in line for line in log.lines["warning"]) == 1


def test_any_other_refusal_drops_the_call_and_goes_on(server: ScriptedServer, log: Recorder) -> None:
    server.script["log_metrics"] = [ValueError("invalid metric value")]
    t = wrap(server)
    t.log_metrics({"a": 1.0}, step=0)
    t.log_metrics({"a": 2.0}, step=1)
    assert metrics(server) == [({"a": 2.0}, 1)] and t.pending == []
    assert "Dropping log_metrics: invalid metric value" in log.lines["warning"]


def test_a_refusal_naming_500_drops_that_call_only(server: ScriptedServer, log: Recorder) -> None:
    """A refused value of 500 is not a server error: later parameter calls still go through."""
    server.script["log_hyperparams"] = [ValueError(CHANGED_PARAM)]
    t = wrap(server)
    t.log_hyperparams({"epochs": 500})
    t.log_hyperparams({"lr": 0.1})
    assert [args for name, args in server.calls if name == "log_hyperparams"] == [({"lr": 0.1},)]
    assert f"Dropping log_hyperparams: {CHANGED_PARAM}" in log.lines["warning"]
    assert not any("Turning off" in line for line in log.lines["warning"])


# --------------------------------------------------------------------------- artifacts


def test_an_artifact_deleted_after_logging_still_uploads_under_its_name(server: ScriptedServer, tmp_path: Path) -> None:
    server.script["log_artifact"] = [ConnectionError("down")]
    t = wrap(server)
    figure = tmp_path / "confusion.png"
    figure.write_bytes(b"PNG")
    t.log_artifact(figure, artifact_dir="plots")
    figure.unlink()
    backup = t.pending[0].backup
    assert backup is not None and backup.read_bytes() == b"PNG"
    assert backup.parent == tmp_path / "runs" / "exp" / "run" / "pending_artifacts"

    t.log_metrics({"a": 1.0})
    assert server.uploaded == [("confusion.png", b"PNG", "plots")]
    assert not backup.exists()  # removed once uploaded


def test_an_artifact_still_on_disk_is_replayed_from_its_file(server: ScriptedServer, tmp_path: Path) -> None:
    server.script["log_artifact"] = [ConnectionError("down")]
    t = wrap(server, backup_artifacts=False)
    path = tmp_path / "a.txt"
    path.write_bytes(b"A")
    t.log_artifact(path)
    assert t.pending[0].backup is None
    t.save()
    assert server.uploaded == [("a.txt", b"A", "")]


def test_a_missing_artifact_file_is_still_passed_on(server: ScriptedServer, tmp_path: Path, log: Recorder) -> None:
    # The wrapped tracker decides what a missing file means; this one fails reading it.
    t = wrap(server)
    t.log_artifact(tmp_path / "nope.txt")
    assert any("Dropping log_artifact" in line for line in log.lines["warning"])


def test_a_backup_that_cannot_be_written_is_warned_about(server: ScriptedServer, tmp_path: Path, log: Recorder) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    server.save_dir = str(blocker)  # a FILE where the directory must go
    server.script["log_artifact"] = [ConnectionError("down")]
    path = tmp_path / "a.txt"
    path.write_bytes(b"A")
    t = wrap(server)
    t.log_artifact(path)
    assert t.pending[0].backup is None
    assert any("Could not back up queued artifact a.txt" in line for line in log.lines["warning"])


# --------------------------------------------------------------------------- finalize


def test_finalize_sends_the_queue_then_ends_the_run(server: ScriptedServer, log: Recorder) -> None:
    server.script["log_metrics"] = [TimeoutError("t")] * 3
    t = wrap(server, max_retries=2)
    t.log_metrics({"a": 0.0}, step=0)
    t.log_metrics({"a": 1.0}, step=1)  # now queueing only
    assert t.queueing_only
    t.finalize()
    # Try 1 fails on the last scripted timeout, try 2 sends both.
    assert metrics(server) == [({"a": 0.0}, 0), ({"a": 1.0}, 1)]
    assert server.calls[-1] == ("finalize", ("success",))
    assert any("Run ending: sending 2 queued" in line for line in log.lines["info"])


def test_what_finalize_cannot_send_is_reported_and_its_backup_kept(
    server: ScriptedServer, log: Recorder, tmp_path: Path
) -> None:
    server.script["log_artifact"] = [ConnectionError("down")] * 10
    t = wrap(server, max_retries=3)
    path = tmp_path / "model.bin"
    path.write_bytes(b"M")
    t.log_artifact(path)
    backup = t.pending[0].backup
    t.finalize()
    assert server.uploaded == [] and t.pending == []
    assert backup is not None and backup.exists()
    assert any("Lost tracking call log_artifact" in line and str(backup) in line for line in log.lines["warning"])
    assert server.calls[-1] == ("finalize", ("success",))  # the run is still ended


def test_lost_calls_without_a_backup_say_nothing_about_one(server: ScriptedServer, log: Recorder) -> None:
    server.script["log_metrics"] = [ConnectionError("down")] * 10
    t = wrap(server, max_retries=2)
    t.log_metrics({"a": 1.0})
    t.finalize()
    lost = [line for line in log.lines["warning"] if line.startswith("Lost tracking call log_metrics")]
    assert len(lost) == 1 and "copy is kept" not in lost[0]

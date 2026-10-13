"""``ResilientTracker`` — keep a run's tracking alive through a flaky tracking server.

A tracker that talks to a remote server makes training wait on the network: a call that
hangs holds the training loop with it, and a call that fails is gone. This wrapper sits
around ANY :class:`~matrainer.tracking.base.Tracker` and gives it four guarantees:

* **Every call has a time limit** — ``timeout`` seconds, an artifact's scaled with its size
  (:meth:`ResilientTracker.artifact_timeout_for`), the run's start ``start_timeout``.
* **A network failure queues the call** (a timeout, a connection error — see
  :func:`is_network_error`). The queue is replayed in order before the next call; after
  ``max_retries`` failures in a row the wrapper stops calling the server and only queues,
  and :meth:`ResilientTracker.finalize` tries ``max_retries`` more times to send it.
* **An artifact's bytes are read when it is logged**, so a temporary file deleted right after
  (a rendered figure) still uploads on replay; with ``backup_artifacts`` a copy is written
  under the run's own directory so it survives a crash before the queue is sent.
* **A server error (5xx) turns off only that kind of call** — a broken metrics endpoint does
  not stop artifact uploads. Any other error is logged and the call dropped.

A run that cannot be started (the server refuses, or ``start_timeout`` passes) turns the
wrapper off: every later call is skipped and training carries on.

**The wrapped tracker must let its failures RAISE.** The wrapper only sees what the wrapped
call raises or how long it takes: a tracker that catches its own errors (``MlflowTracker``
logs a failed ``log_metric`` at debug level and returns) gets the time limits but nothing
queued. Rationale: ``docs/architecture.md`` → "A resilient wrapper, not a resilient backend".
"""

from __future__ import annotations

import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union

from confluid import configurable, flow
from loggair import get_logger

from matrainer.tracking.base import Tracker

if TYPE_CHECKING:  # type-only: `manifest` reaches `base`, so a runtime import would cycle.
    from matrainer.tracking.manifest import RunManifest

logger = get_logger(__name__)

#: Words in an error's message that mark it as the network failing rather than the server
#: refusing — client libraries wrap a socket error in their own exception types, so the type
#: alone does not tell.
NETWORK_ERROR_WORDS = ("max retries", "timeout", "timed out", "connection")

#: How often (in queued calls) the cache-only mode says it is still on — it makes no live
#: calls, so nothing else would mention it until the run ends.
_STILL_QUEUEING_EVERY = 25


def is_network_error(error: BaseException) -> bool:
    """Whether ``error`` is the network failing (worth a retry) rather than the server refusing."""
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    text = str(error).lower()
    return any(word in text for word in NETWORK_ERROR_WORDS)


def is_server_error(error: BaseException) -> bool:
    """Whether ``error`` carries a 5xx status — the endpoint is broken, not the network."""
    status = getattr(error, "status_code", None) or getattr(getattr(error, "response", None), "status_code", None)
    if isinstance(status, int):
        return 500 <= status < 600
    return "500" in str(error)


def run_with_timeout(func: Callable[..., Any], args: Tuple[Any, ...], kwargs: Mapping[str, Any], timeout: float) -> Any:
    """Call ``func(*args, **kwargs)`` on a daemon thread; raise :class:`TimeoutError` after ``timeout`` s.

    A thread rather than a signal because a signal only works on the main thread, and a
    training loop's logging callback is not always on it. The cost: a call that times out
    keeps running in the background and may still land, so its replay can record the same
    value twice.
    """
    result: List[Any] = [None]
    error: List[Optional[BaseException]] = [None]

    def target() -> None:
        try:
            result[0] = func(*args, **kwargs)
        except BaseException as exc:  # noqa: B902 - handed back to the caller's thread
            error[0] = exc

    thread = threading.Thread(target=target, daemon=True, name=f"tracker-{getattr(func, '__name__', 'call')}")
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        raise TimeoutError(f"{getattr(func, '__name__', 'call')} timed out after {timeout:g} s")
    if error[0] is not None:
        raise error[0]
    return result[0]


@dataclass
class PendingCall:
    """A call to the wrapped tracker queued after a network failure, replayed later in order.

    Attributes:
        method: The wrapped tracker's method name (``log_metrics``, ``log_artifact``, ...).
        args: Its positional arguments.
        kwargs: Its keyword arguments.
        timeout: The time limit it is replayed with.
        artifact: For ``log_artifact``: the file's bytes, read when it was logged.
        backup: For ``log_artifact``: the on-disk copy of those bytes, when one was written.
    """

    method: str
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    timeout: float
    artifact: Optional[bytes] = None
    backup: Optional[Path] = None


@configurable
class ResilientTracker(Tracker):
    """Wrap a tracker so a slow or unreachable server never stops training or loses a value.

    Zero-arg constructible; the wrapped tracker's run is started on the first call. Name, run
    name, output directory and native handle are the wrapped tracker's.

    Args:
        tracker: The tracker to protect. Required to run: the first call without one raises.
        timeout: Time limit in seconds of a metric, parameter, tag, text or figure call.
        start_timeout: Time limit in seconds of starting the run (the wrapped tracker's first
            :attr:`experiment` access — it creates the experiment and the run).
        artifact_timeout: Fixed time limit in seconds of an artifact upload; ``None`` scales it
            with the file's size (:meth:`artifact_timeout_for`).
        artifact_upload_rate_mbps: The slowest upload rate (MB/s) the scaled limit allows for.
        max_retries: Network failures in a row before only queueing; also the number of tries
            :meth:`finalize` makes to send the queue.
        finalize_retry_delay: Seconds between :meth:`finalize`'s tries.
        backup_artifacts: Write a queued artifact's bytes under
            ``<save_dir>/<experiment>/<run>/pending_artifacts/`` too, so it survives a crash;
            removed once uploaded.
    """

    def __init__(
        self,
        tracker: Optional[Tracker] = None,
        timeout: float = 5.0,
        start_timeout: float = 30.0,
        artifact_timeout: Optional[float] = None,
        artifact_upload_rate_mbps: float = 5.0,
        max_retries: int = 10,
        finalize_retry_delay: float = 1.0,
        backup_artifacts: bool = True,
    ) -> None:
        # No `super().__init__()`: the experiment / run / directory / tags are the wrapped
        # tracker's (the read-only properties below), and the base would draw a second run name.
        self.tracker = tracker
        self.timeout = timeout
        self.start_timeout = start_timeout
        self.artifact_timeout = artifact_timeout
        self.artifact_upload_rate_mbps = artifact_upload_rate_mbps
        self.max_retries = max_retries
        self.finalize_retry_delay = finalize_retry_delay
        self.backup_artifacts = backup_artifacts
        self._started = False
        self._disabled = False
        self._disabled_calls: Set[str] = set()
        self._queue: List[PendingCall] = []
        self._failures = 0
        self._backups = 0

    # -- the wrapped tracker ------------------------------------------------------------------

    @property
    def inner(self) -> Tracker:
        """The wrapped tracker, built from its config marker on first use."""
        if self.tracker is None:
            raise ValueError("ResilientTracker needs a `tracker:` to wrap — none was configured.")
        # Written back: re-flowing a deferred marker on every call would build a new tracker
        # (a new run) each time — the reason `flow_trackers` writes back too.
        self.tracker = flow(self.tracker)
        return self.tracker

    @property
    def experiment_name(self) -> str:  # type: ignore[override]
        return self.inner.experiment_name

    @property
    def run_name(self) -> str:  # type: ignore[override]
        return self.inner.run_name

    @property
    def save_dir(self) -> str:  # type: ignore[override]
        return self.inner.save_dir

    @property
    def tags(self) -> Dict[str, str]:  # type: ignore[override]
        return self.inner.tags

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def version(self) -> Optional[str]:
        return self.inner.version

    @property
    def log_dir(self) -> Optional[Path]:
        return self.inner.log_dir

    @property
    def experiment(self) -> Any:
        """The wrapped tracker's native handle, or ``None`` when its run could not be started."""
        return self.inner.experiment if self._start() else None

    @property
    def pending(self) -> List[PendingCall]:
        """The calls queued for replay, oldest first (a copy)."""
        return list(self._queue)

    @property
    def queueing_only(self) -> bool:
        """Whether ``max_retries`` network failures in a row stopped live calls until :meth:`finalize`."""
        return self._failures >= self.max_retries

    def _start(self) -> bool:
        """Start the wrapped tracker's run once, within ``start_timeout``; ``False`` when off."""
        if self._disabled:
            return False
        if not self._started:
            inner = self.inner
            try:
                run_with_timeout(lambda: inner.experiment, (), {}, self.start_timeout)
            except Exception as exc:
                # Nothing queued could ever be replayed without a run to replay it into.
                logger.warning(f"Tracking to {type(inner).__name__} is off for this run: starting it failed: {exc}")
                self._disabled = True
                return False
            self._started = True
        return True

    # -- the queue ----------------------------------------------------------------------------

    def _summary(self) -> str:
        counts: Dict[str, int] = {}
        for call in self._queue:
            counts[call.method] = counts.get(call.method, 0) + 1
        return ", ".join(f"{name}={count}" for name, count in sorted(counts.items())) or "empty"

    def _replay(self, call: PendingCall) -> None:
        """Send one queued call, restoring an artifact whose file is gone from its saved bytes."""
        method = getattr(self.inner, call.method)
        source = Path(call.args[0]) if call.method == "log_artifact" else None
        if source is None or source.exists() or call.artifact is None:
            run_with_timeout(method, call.args, call.kwargs, call.timeout)
            return
        # Same file name in a private directory: the name is what the artifact store shows.
        with tempfile.TemporaryDirectory(prefix="matrainer-artifact-") as tmp:
            restored = Path(tmp) / source.name
            restored.write_bytes(call.artifact)
            run_with_timeout(method, (restored, *call.args[1:]), call.kwargs, call.timeout)

    def _flush(self) -> None:
        """Replay the queue in order, stopping at the first network failure."""
        if not self._queue or self.queueing_only:
            return
        sent = 0
        for index, call in enumerate(self._queue):
            try:
                self._replay(call)
            except Exception as exc:
                if is_network_error(exc):
                    logger.warning(f"Replaying queued calls stopped at {index + 1}/{len(self._queue)}: {exc}")
                    break
                logger.warning(f"Dropping queued {call.method}: replaying it failed for a non-network reason: {exc}")
            sent += 1
            if call.backup is not None:
                call.backup.unlink(missing_ok=True)
        if sent:
            self._queue = self._queue[sent:]
            self._failures = 0
            logger.info(f"Sent {sent} queued tracking call(s) ({len(self._queue)} still queued).")
        else:
            self._failed()

    def _failed(self) -> None:
        """Count one network failure in a row; say so once when live calls stop."""
        self._failures += 1
        if self._failures == self.max_retries:
            where = Path(self.save_dir) / self.experiment_name / self.run_name
            logger.warning(
                f"{self.max_retries} network failures in a row: no more live tracking calls until the run "
                f"ends; {len(self._queue)} queued so far ({self._summary()})"
                + (f", artifacts backed up under {where}" if self.backup_artifacts else "")
                + ". Local checkpoints are unaffected."
            )

    def _enqueue(self, call: PendingCall) -> None:
        if call.artifact is not None and self.backup_artifacts:
            call.backup = self._write_backup(Path(call.args[0]).name, call.artifact)
        self._queue.append(call)

    def _write_backup(self, file_name: str, data: bytes) -> Optional[Path]:
        directory = Path(self.save_dir) / self.experiment_name / self.run_name / "pending_artifacts"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            self._backups += 1
            path = directory / f"{self._backups}_{file_name}"
            path.write_bytes(data)
        except OSError as exc:
            logger.warning(f"Could not back up queued artifact {file_name} under {directory}: {exc}")
            return None
        logger.debug(f"Backed up queued artifact to {path}")
        return path

    def _call(
        self,
        method: str,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        timeout: float,
        artifact: Optional[bytes] = None,
    ) -> None:
        """Send one call — or queue it — after replaying what is already queued."""
        if not self._start() or method in self._disabled_calls:
            return
        self._flush()
        call = PendingCall(method, args, kwargs, timeout, artifact)
        if self._queue:
            # Replay stopped (or live calls are off): behind the queue, so values keep their order.
            self._enqueue(call)
            if self.queueing_only and len(self._queue) % _STILL_QUEUEING_EVERY == 0:
                logger.warning(
                    f"Still only queueing: {len(self._queue)} tracking call(s) waiting ({self._summary()}); "
                    "they are sent when the run ends."
                )
            return
        try:
            run_with_timeout(getattr(self.inner, method), args, kwargs, timeout)
        except Exception as exc:
            if is_network_error(exc):
                self._enqueue(call)
                logger.warning(f"Network error in {method}, queued it ({self._failures + 1}/{self.max_retries}): {exc}")
                self._failed()
            elif is_server_error(exc):
                self._disabled_calls.add(method)
                logger.warning(f"Turning off {method} for this run after a server error (other calls go on): {exc}")
            else:
                logger.warning(f"Dropping {method}: {exc}")

    # -- the Tracker surface --------------------------------------------------------------------

    def artifact_timeout_for(self, size_bytes: int) -> float:
        """The upload time limit of a file of ``size_bytes``: ``artifact_timeout``, else
        ``max(timeout, 60 s + size / artifact_upload_rate_mbps)``."""
        if self.artifact_timeout is not None:
            return self.artifact_timeout
        return max(self.timeout, 60.0 + size_bytes / (1024 * 1024) / self.artifact_upload_rate_mbps)

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        self._call("log_hyperparams", (dict(params),), {}, max(self.timeout, 10.0))

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        self._call("log_metrics", (dict(metrics),), {"step": step}, self.timeout)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self._call("set_tags", (dict(tags),), {}, self.timeout)

    def log_artifact(self, path: Union[str, Path], artifact_dir: str = "") -> None:
        source = Path(path)
        data = source.read_bytes() if source.is_file() else None
        timeout = self.artifact_timeout_for(len(data) if data is not None else 0)
        self._call("log_artifact", (source,), {"artifact_dir": artifact_dir}, timeout, artifact=data)

    def log_figure(self, name: str, figure: Any, step: Optional[int] = None) -> None:
        self._call("log_figure", (name, figure), {"step": step}, max(self.timeout, 30.0))

    def log_text(self, name: str, text: str, step: Optional[int] = None) -> None:
        self._call("log_text", (name, text), {"step": step}, max(self.timeout, 30.0))

    def log_manifest(self, manifest: "RunManifest") -> None:
        self._call("log_manifest", (manifest,), {}, max(self.timeout * 3, 60.0))

    def save(self) -> None:
        self._call("save", (), {}, self.timeout)

    def finalize(self, status: str = "success") -> None:
        """Send what is queued (``max_retries`` tries), then end the wrapped tracker's run."""
        if not self._start():
            return
        if self._queue:
            logger.info(f"Run ending: sending {len(self._queue)} queued tracking call(s) ({self._summary()}).")
            self._failures = 0
            for attempt in range(self.max_retries):
                self._flush()
                if not self._queue:
                    break
                if attempt < self.max_retries - 1:
                    time.sleep(self.finalize_retry_delay)
            for call in self._queue:
                what = f"({call.args[0]})" if call.method == "log_artifact" else ""
                kept = f" — its copy is kept at {call.backup}" if call.backup is not None else ""
                logger.warning(f"Lost tracking call {call.method}{what}: the server never took it{kept}")
            self._queue.clear()
        try:
            run_with_timeout(self.inner.finalize, (status,), {}, max(self.timeout, 10.0))
        except Exception as exc:
            logger.warning(f"Ending the {type(self.inner).__name__} run failed: {exc}")
            return
        run_url = getattr(self.inner, "run_url", None)
        if run_url:
            logger.info(f"Tracked run: {run_url}")


__all__ = ["PendingCall", "ResilientTracker", "is_network_error", "is_server_error", "run_with_timeout"]

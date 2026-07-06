"""Generic DataFlux pipeline runner for the ``marainer process/download/convert`` verbs.

``DatasetProcessor`` orchestrates a :class:`dataflux.core.Flux` from source to
sink — the canonical ``processor:`` object the marainer CLI's ``process`` /
``download`` / ``convert`` verbs dispatch via ``.run()``. It lives here (not in
a domain package) because it is the workspace's generic data-pipeline runner and
depends only on dataflux + confluid; it was moved out of ``waivefront.processing``
to keep waivefront free of this orchestration concern.
"""

from contextlib import nullcontext
from typing import Any, Iterable, Iterator, Optional

from confluid import configurable
from dataflux.core import Flux
from dataflux.sample import Sample
from dataflux.storage.base import Storage
from loggair import get_logger
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

from marainer.progress import ProgressReporting

logger = get_logger(__name__)


def _flux_total(flux: Flux) -> Optional[int]:
    """``len(flux.source)`` when the source is sized, else ``None`` (a glob-based / streaming source)."""
    try:
        return len(flux.source)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return None


@configurable
class DatasetProcessor(ProgressReporting):
    """Orchestrate a DataFlux pipeline from source to sink.

    Args:
        flux: The :class:`~dataflux.core.Flux` to execute.
        sink: Optional sink; when absent, samples are materialized to a list.
        show_progress: If ``True``, wrap iteration with a ``rich.progress`` bar.
            The total is derived from ``len(flux.source)`` when available; an
            unsized source falls back to a count-only bar. Default ``False``.
        progress_desc: Optional label for the progress bar (defaults to
            ``"DatasetProcessor"``). Ignored when ``show_progress`` is ``False``.
    """

    def __init__(
        self,
        flux: Flux,
        sink: Optional[Any] = None,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
    ) -> None:
        self.flux = flux
        self.sink = sink
        self.show_progress = show_progress
        self.progress_desc = progress_desc

    def run(self) -> None:
        logger.info("Starting DatasetProcessor...")
        # Confluid keeps Class kwargs deferred (post-construction paradigm) so
        # when the processor was loaded from YAML, ``self.flux``, its source,
        # its ops, and ``self.sink`` may all be Fluid stubs. Materialize them
        # here so callers don't need to know.
        from confluid import flow
        from confluid.fluid import Fluid

        flux = flow(self.flux) if isinstance(self.flux, Fluid) else self.flux
        if isinstance(flux.source, Fluid):
            flux.source = flow(flux.source)
        flux.ops = [flow(op) if isinstance(op, Fluid) else op for op in flux.ops]
        self.flux = flux
        sink = flow(self.sink) if isinstance(self.sink, Fluid) else self.sink

        iterator = self._wrap_progress(flux)
        # Drive the executor's progress bar (FluxStudio / ComfyUI) per sample — independent of the
        # console ``show_progress`` rich bar; a no-op when no progress callback was injected.
        total = _flux_total(flux)
        desc = self.progress_desc or "DatasetProcessor"

        if sink:
            logger.info(f"Streaming data to sink: {sink.__class__.__name__}")
            # Replicates dataflux.core.Flux.to_sink so we can iterate through
            # our progress wrapper while preserving the Storage context + flush.
            sink_ctx: Any = sink if isinstance(sink, Storage) else nullcontext()
            count = 0
            with sink_ctx:
                for sample in iterator:
                    sink.write(sample)
                    count += 1
                    self._report_progress(count, total, desc)
                sink.flush()
            logger.info(f"Streamed {count} sample(s) to sink.")
        else:
            logger.info("No sink provided. Materializing data in-memory.")
            results = []
            for count, sample in enumerate(iterator, start=1):
                results.append(sample)
                self._report_progress(count, total, desc)
            logger.info(f"Processed {len(results)} samples.")

        logger.info("Processing complete.")

    def _wrap_progress(self, flux: Flux) -> Iterable[Sample]:
        """Wrap the flux iterator in rich.progress when ``show_progress`` is enabled.

        Source length is probed defensively — not every DataSource implements
        ``__len__`` (e.g. glob-based streaming sources). Missing lengths
        degrade to a count-only bar instead of breaking the run.
        """
        if not self.show_progress:
            return flux
        desc = self.progress_desc or "DatasetProcessor"
        return _ProgressIter(flux, total=_flux_total(flux), desc=desc)


class _ProgressIter:
    """Iterable wrapper that opens a ``rich.progress`` bar at ``__iter__`` time.

    Kept as a class (not a generator) so ``DatasetProcessor._wrap_progress``
    can return a value that is truthy to ``bool()`` even when empty, matching
    the contract of raw ``Flux`` which behaves like a ``Sized`` (``Flux``
    subclasses ``torch.utils.data.Dataset``).
    """

    def __init__(self, flux: Flux, total: Optional[int], desc: str) -> None:
        self._flux = flux
        self._total = total
        self._desc = desc

    def __iter__(self) -> Iterator[Sample]:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(self._desc, total=self._total)
            for sample in self._flux:
                yield sample
                progress.update(task, advance=1)

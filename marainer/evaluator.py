"""Marainer Evaluator: run a trained model over a dataset and collect predictions.

Sibling to :class:`~marainer.trainer.LightningTrainer`. The LightningEvaluator streams a dataset
through a model (already weight-loaded by the model itself — see e.g.
:class:`raidar_ultralytics.yolo26.YOLO26Model`'s ``checkpoint_path`` argument),
accumulates metrics that understand the model's prediction contract (see
:mod:`marainer.outputs`), and optionally forwards each prediction to a sink
for persistence.

Task-agnostic by design: the Evaluator does not know whether it is running
classification, detection, or segmentation — it only requires that the
model's eval-mode ``forward`` produces something the configured ``metrics``
and ``predictions_sink`` expect.

YAML example (detection)::

    evaluator: !class:marainer.evaluator.LightningEvaluator
      model: !class:raidar.faster_rcnn.FasterRCNNModel
        num_classes: 2
        checkpoint_path: ./checkpoints/best.ckpt   # the model loads itself
      dataset: !ref:eval_set
      loader: !class:torch.utils.data.DataLoader
        batch_size: 4
        collate_fn: !ref:raidar.detection.detection_collate_fn
      metrics: !class:raidar.detection.DetectionMetricAdapter
        inner: !class:torchmetrics.detection.MeanAveragePrecision
          iou_type: bbox
      metrics_sidecar_path: ./eval_metrics.json   # optional — see below

    cli:  marainer evaluate config.yaml

When ``metrics_sidecar_path`` is set, the final ``metrics.compute()`` dict is
written as JSON after the test loop completes. This is **logger-agnostic**:
it works whether the user wires TensorBoard, MLflow, CSV, or no Lightning
logger at all — the sidecar is a deterministic file at a known path that
downstream tooling (e.g. Navigator's ``read_run_metrics``) can rely on
without scraping logger backends.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as L
from confluid import LazyClass, configurable, flow
from logflow import get_logger
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import Metric, MetricCollection

from marainer.loggers import _render_matrix_as_table, log_figure_to_loggers
from marainer.run_names import generate_run_name

logger = get_logger(__name__)


def _is_square_2d_tensor(value: Any) -> bool:
    """True when ``value`` is a (N, N) tensor with N >= 1.

    Generic shape check used to detect confusion-matrix-shaped metrics
    without string-matching on metric names.
    """
    return isinstance(value, Tensor) and value.dim() == 2 and value.shape[0] == value.shape[1] and value.shape[0] >= 1


def _ensure_class_names(names: Optional[List[str]], n: int) -> List[str]:
    """Return ``names`` if it has length ``n``, else integer-index fallback."""
    if names is not None and len(names) == n:
        return names
    return [str(i) for i in range(n)]


def _coerce_metric_value(value: Any) -> Any:
    """Convert a torchmetrics compute() value into a JSON-safe Python scalar.

    Tensors become floats (or ints for integer-typed scalars); nested dicts /
    lists / tuples recurse. Anything already JSON-safe (str / bool / None /
    int / float) passes through. Used by :meth:`Evaluator._write_metrics_sidecar`
    so the sidecar is human-readable and ``json.load``-able from any tool.
    """
    if isinstance(value, Tensor):
        if value.numel() == 1:
            try:
                return value.item()
            except RuntimeError:
                # Non-numeric (rare) — fall through to string repr.
                return str(value)
        return [_coerce_metric_value(v) for v in value.detach().cpu().tolist()]
    if isinstance(value, dict):
        return {k: _coerce_metric_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_metric_value(v) for v in value]
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


@configurable
class LightningEvaluator(L.LightningModule):
    """Run a model over a dataset, update metrics, and optionally persist predictions.

    Checkpoint loading is the model's responsibility — pass ``checkpoint_path``
    to your model class (e.g. :class:`raidar_ultralytics.yolo26.YOLO26Model`) instead of
    to the Evaluator.

    Follows confluid's minimal-ctor / post-construction discipline (see the
    sibling :class:`~marainer.trainer.LightningTrainer`): the constructor takes
    only ``model`` / ``dataset`` (+ identity ``experiment_name`` / ``run_name``)
    and does no functional work. The deferred ``loader`` / ``lightning`` body
    slots default to ``LazyClass(...)`` and are flowed by :meth:`run` (the loader
    with ``dataset=``); ``to_pydantic`` introspects all body slots, so they stay
    enumerable by navigaitor / MCP / FluxStudio. A deferred slot wired in YAML
    MUST use ``!lazy:`` (a bare ``!class:`` post-init attr is eagerly flowed on
    assignment).

    Args:
        model: The neural network (deferred via Confluid). The model is
            expected to arrive ALREADY weight-loaded — instantiate it with
            its own ``checkpoint_path`` argument if you need to start from
            a saved run.
        dataset: Evaluation dataset or Flux pipeline (deferred).
        experiment_name: MLflow experiment name.
        run_name: MLflow run name. ``None`` (default) auto-generates a random slug like
            ``evaluate_funny_rabbit`` (see :func:`marainer.run_names.generate_run_name`).

    Post-construction body-attribute slots (set via YAML / broadcasting):
        loader: Deferred ``DataLoader`` config. Supply a matching ``collate_fn``
            for task types that need one (e.g.
            :func:`raidar.detection.detection_collate_fn`).
        metrics: Optional :class:`~torchmetrics.MetricCollection` or adapter
            that understands the model's step output. Logged via Lightning's
            ``log_dict`` at epoch end.
        predictions_sink: Optional duck-typed sink exposing
            ``.write(prediction, metadata)`` and ``.close()``. Called once
            per sample within each batch with the per-sample prediction and
            its metadata (from ``batch.metadata``, which is a list of
            per-sample dicts when using :func:`~marainer.collate.collate_fn_with_metadata`).
        lightning: Deferred PyTorch Lightning Trainer. Only ``test`` is invoked,
            so training knobs like ``max_epochs`` are ignored.
        metrics_sidecar_path: Optional path; when set, ``metrics.compute()``
            is also written there as JSON (logger-agnostic).
        class_names: Optional human-readable class labels for confusion-matrix
            rendering. Accepts either an explicit ``List[str]`` or a path
            (``str``) to a JSON file shaped ``{"class_names": [...]}``
            (the format the :class:`~marainer.trainer.Trainer` writes next
            to the checkpoint). When ``None``, auto-resolved from
            ``getattr(dataset, "label_names", None)`` first, then from a
            ``class_names.json`` co-located with the model's checkpoint,
            and finally falls back to integer indices.
    """

    def __init__(
        self,
        model: Any,
        dataset: Any,
        experiment_name: str = "marainer_eval",
        run_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        # Store config ONLY — no functional work. Deferred slots (loader / lightning)
        # are LazyClass; run() flows them. Every slot is configurable post-construction
        # (YAML / broadcasting); a YAML-wired deferred slot must use !lazy:.
        self.model: Any = model
        self.dataset: Any = dataset
        self.loader: Any = LazyClass(DataLoader)
        self.metrics: Any = None
        self.predictions_sink: Any = None
        self.lightning: Any = LazyClass(L.Trainer)
        self.experiment_name = experiment_name
        # A None run_name gets a random, human-friendly slug (e.g. ``evaluate_funny_rabbit``).
        self.run_name = run_name if run_name is not None else generate_run_name(prefix="evaluate")
        self.metrics_sidecar_path: Optional[str] = None
        self.class_names: Optional[Union[List[str], str]] = None
        self._resolved_class_names: Optional[List[str]] = None

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        preds = self(batch.input)
        batch_size = self._infer_batch_size(batch)
        if self.metrics is not None:
            self.metrics.update(preds, batch.target)
            if isinstance(self.metrics, MetricCollection):
                # Filter for scalars to prevent Lightning logging crashes (e.g. for confusion matrices).
                # Non-scalars are still computed and available in compute() for the sidecar.
                scalars = {k: v for k, v in self.metrics.items() if isinstance(v, Tensor) and v.numel() == 1}
                if scalars:
                    self.log_dict(scalars, on_step=False, on_epoch=True, batch_size=batch_size)
        if self.predictions_sink is not None:
            self._write_predictions(preds, batch)
        return preds

    def _infer_batch_size(self, batch: Any) -> int:
        inp = batch.input
        if hasattr(inp, "shape") and len(inp.shape) > 0:
            return int(inp.shape[0])
        if hasattr(inp, "__len__"):
            return len(inp)
        return 1

    def _write_predictions(self, preds: Any, batch: Any) -> None:
        per_sample_meta = None
        meta = getattr(batch, "metadata", None)
        if isinstance(meta, list):
            # New style: metadata is a list of per-sample dicts
            per_sample_meta = meta
        elif isinstance(meta, dict):
            # Legacy style: metadata is a dict with "per_sample" key
            per_sample_meta = meta.get("per_sample")

        # Detection-style: preds is a dict with "predictions": List[DetectionOutput].
        if isinstance(preds, dict) and "predictions" in preds:
            items = preds["predictions"]
        elif isinstance(preds, list):
            items = preds
        else:
            items = [preds]

        n = len(items) if items is not None else 0
        for i in range(n):
            p = items[i]
            m = per_sample_meta[i] if per_sample_meta and i < len(per_sample_meta) else {}
            self.predictions_sink.write(p, m)

    def _resolve_class_names(self, dataset: Any) -> Optional[List[str]]:
        """Locate class names from (in order) the explicit param, the dataset,
        or a ``class_names.json`` next to the model's checkpoint.

        Returns ``None`` only when every source is missing; callers fall back
        to integer indices at render time.
        """
        if isinstance(self.class_names, list):
            return [str(n) for n in self.class_names]
        if isinstance(self.class_names, str):
            path = Path(self.class_names).expanduser()
            try:
                data = json.loads(path.read_text())
                names = data.get("class_names")
                if names:
                    return [str(n) for n in names]
            except Exception as exc:
                logger.warning(f"Could not read class_names from {path}: {exc}")

        names = getattr(dataset, "label_names", None)
        if names:
            return [str(n) for n in names]

        ckpt_path = getattr(self.model, "checkpoint_path", None)
        if ckpt_path:
            candidate = Path(str(ckpt_path)).expanduser().parent / "class_names.json"
            if candidate.is_file():
                try:
                    data = json.loads(candidate.read_text())
                    names = data.get("class_names")
                    if names:
                        return [str(n) for n in names]
                except Exception as exc:
                    logger.debug(f"Could not read {candidate}: {exc}")
        return None

    def _render_and_log_matrices(self) -> None:
        """For every (N, N) tensor in ``metrics.items()``: log a tabulate
        text table and forward the metric's own ``.plot(labels=...)`` figure
        to every wired Lightning logger.

        Iterates over ``self.metrics.items()`` rather than ``compute()``
        output so we can call the metric's built-in ``.plot()`` (e.g.
        :meth:`torchmetrics.classification.MulticlassConfusionMatrix.plot`)
        with class labels. Metrics that lack ``.plot()`` still get the text
        table.
        """
        if not isinstance(self.metrics, MetricCollection):
            return
        for name, metric in self.metrics.items():
            try:
                value = metric.compute() if isinstance(metric, Metric) else None
            except Exception as exc:  # pragma: no cover - defensive log path
                logger.debug(f"Could not compute {name} for rendering: {exc}")
                continue
            if not _is_square_2d_tensor(value):
                continue
            n = int(value.shape[0])
            labels = _ensure_class_names(self._resolved_class_names, n)
            logger.info("\n" + _render_matrix_as_table(name, value, labels))

            plot = getattr(metric, "plot", None)
            if not callable(plot):
                continue
            try:
                result = plot(labels=labels)
            except Exception as exc:
                logger.debug(f"{type(metric).__name__}.plot failed for {name}: {exc}")
                continue
            figure = result[0] if isinstance(result, tuple) else result
            try:
                log_figure_to_loggers(getattr(self, "trainer", None), name, figure)
            finally:
                try:
                    import matplotlib.pyplot as plt

                    plt.close(figure)
                except Exception:  # pragma: no cover - matplotlib state cleanup
                    pass

    def _finalize_metrics(self) -> None:
        """Render non-scalar metrics for humans + optionally write the JSON sidecar.

        Always runs the render pass when metrics are configured (text table +
        heatmap figure to wired Lightning loggers). The JSON sidecar at
        ``metrics_sidecar_path`` is the **logger-agnostic** read path for
        downstream tooling (Navigator's ``read_run_metrics``) and is opt-in.
        """
        if self.metrics is None:
            return
        self._render_and_log_matrices()
        if self.metrics_sidecar_path is None:
            return
        try:
            raw = self.metrics.compute()
        except Exception as exc:  # pragma: no cover - defensive log path
            logger.warning(f"Could not compute final metrics for sidecar dump: {exc}")
            return
        if isinstance(raw, Tensor):
            payload: Any = _coerce_metric_value(raw)
        elif isinstance(raw, dict):
            payload = {k: _coerce_metric_value(v) for k, v in raw.items()}
        else:
            payload = _coerce_metric_value(raw)

        wrapped: Dict[str, Any] = {
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "metrics": payload,
        }
        out_path = Path(self.metrics_sidecar_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(wrapped, indent=2, sort_keys=True))
        logger.info(f"Wrote final metrics sidecar to {out_path}")

    def run(self) -> None:
        """Materialize deferred components and evaluate.

        The model is responsible for its own weight loading — pass
        ``checkpoint_path`` to the model class, not to this Evaluator.
        """
        logger.info(f"Marainer evaluating: {self.experiment_name} ({self.run_name})")

        # Materialize the dataset, model, and metrics into live instances. The
        # evaluator stays task-agnostic: a task subclass (e.g. the classification
        # evaluator) flows the model + metrics with its dimensions in initialize()
        # at the top of run(); flow() here is idempotent on an already-live instance.
        dataset = flow(self.dataset)
        self.model = flow(self.model)
        if self.metrics is not None:
            self.metrics = flow(self.metrics)
        if self.predictions_sink is not None:
            self.predictions_sink = flow(self.predictions_sink)

        self._resolved_class_names = self._resolve_class_names(dataset)
        loader = flow(self.loader, dataset=dataset)
        lightning_trainer = flow(self.lightning)
        from marainer.trainer import log_config_artifacts

        log_config_artifacts(lightning_trainer)
        lightning_trainer.test(model=self, dataloaders=loader)
        if self.predictions_sink is not None and hasattr(self.predictions_sink, "close"):
            self.predictions_sink.close()
        self._finalize_metrics()
        logger.info("Evaluation complete.")

"""Marainer Evaluator: run a trained model over a dataset and collect predictions.

Sibling to :class:`~marainer.trainer.Trainer`. The Evaluator streams a dataset
through a model (already weight-loaded by the model itself — see e.g.
:class:`raidar.yolo26.YOLO26Model`'s ``checkpoint_path`` argument),
accumulates metrics that understand the model's prediction contract (see
:mod:`marainer.outputs`), and optionally forwards each prediction to a sink
for persistence.

Task-agnostic by design: the Evaluator does not know whether it is running
classification, detection, or segmentation — it only requires that the
model's eval-mode ``forward`` produces something the configured ``metrics``
and ``predictions_sink`` expect.

YAML example (detection)::

    evaluator: !class:marainer.evaluator.Evaluator
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
from confluid import Class, Lazy, configurable, flow
from logflow import get_logger
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import Metric, MetricCollection

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


def _render_matrix_as_table(name: str, matrix: Tensor, class_names: List[str]) -> str:
    """Render a square 2D tensor as a github-flavored markdown table.

    Rows are true labels, columns are predicted labels (the torchmetrics
    confusion-matrix convention: ``C[i, j]`` = count of ``true=i, pred=j``).
    """
    from tabulate import tabulate

    arr = matrix.detach().cpu().tolist()
    headers = [""] + [f"pred:{c}" for c in class_names]
    rows = [[f"true:{class_names[i]}", *arr[i]] for i in range(len(arr))]
    rendered: str = tabulate(rows, headers=headers, tablefmt="github")
    return f"{name}:\n" + rendered


def _log_figure_to_loggers(
    lightning_trainer: Optional[L.Trainer],
    name: str,
    figure: Any,
) -> None:
    """Forward a matplotlib Figure to every wired Lightning logger.

    TensorBoard-style loggers expose ``experiment.add_figure(tag, fig, step)``;
    MLflow-style loggers expose ``experiment.log_figure(run_id, fig, path)``.
    Loggers without either are silently skipped.
    """
    if lightning_trainer is None:
        return
    step = getattr(lightning_trainer, "global_step", 0)
    for lg in getattr(lightning_trainer, "loggers", []) or []:
        exp = getattr(lg, "experiment", None)
        if exp is None:
            continue
        add_figure = getattr(exp, "add_figure", None)
        if callable(add_figure):
            try:
                add_figure(name, figure, step)
                continue
            except Exception as exc:  # pragma: no cover - logger backends are external
                logger.debug(f"add_figure failed on {type(lg).__name__}: {exc}")
        log_figure = getattr(exp, "log_figure", None)
        run_id = getattr(lg, "run_id", None)
        if callable(log_figure) and run_id:
            try:
                log_figure(run_id, figure, f"{name.replace('/', '_')}.png")
            except Exception as exc:  # pragma: no cover - logger backends are external
                logger.debug(f"log_figure failed on {type(lg).__name__}: {exc}")


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
class Evaluator(L.LightningModule):
    """Run a model over a dataset, update metrics, and optionally persist predictions.

    Checkpoint loading is the model's responsibility — pass ``checkpoint_path``
    to your model class (e.g. :class:`raidar.yolo26.YOLO26Model`) instead of
    to the Evaluator.

    Args:
        model: The neural network (deferred via Confluid). The model is
            expected to arrive ALREADY weight-loaded — instantiate it with
            its own ``checkpoint_path`` argument if you need to start from
            a saved run.
        dataset: Evaluation dataset or Flux pipeline (deferred).
        loader: DataLoader config (deferred). Supply a matching
            ``collate_fn`` for task types that need one (e.g.
            :func:`raidar.detection.detection_collate_fn`).
        metrics: Optional :class:`~torchmetrics.MetricCollection` or adapter
            that understands the model's step output. Logged via Lightning's
            ``log_dict`` at epoch end.
        predictions_sink: Optional duck-typed sink exposing
            ``.write(prediction, metadata)`` and ``.close()``. Called once
            per sample within each batch with the per-sample prediction and
            its metadata (if the batch carries a ``per_sample`` metadata
            list, as the standard Marainer collate does).
        lightning: PyTorch Lightning Trainer config (deferred). Only ``test``
            is invoked, so training knobs like ``max_epochs`` are ignored.
        experiment_name / run_name: Metadata for loggers (MLflow, TB, etc.).
        loggers: Optional list of Lightning loggers.
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
        loader: Lazy[Any] = Class(DataLoader),
        metrics: Any = None,
        predictions_sink: Any = None,
        lightning: Any = Class(L.Trainer),
        experiment_name: str = "marainer_eval",
        run_name: str = "evaluate",
        metrics_sidecar_path: Optional[str] = None,
        class_names: Optional[Union[List[str], str]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.loader = loader
        self.metrics: Any = metrics
        self._predictions_sink = predictions_sink
        self.lightning = lightning
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.metrics_sidecar_path: Optional[str] = metrics_sidecar_path
        self.class_names: Optional[Union[List[str], str]] = class_names
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
        if self._predictions_sink is not None:
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
        if isinstance(meta, dict):
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
            self._predictions_sink.write(p, m)

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
                _log_figure_to_loggers(getattr(self, "trainer", None), name, figure)
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
        self.model = flow(self.model)
        if self.metrics is not None:
            self.metrics = flow(self.metrics)
        if self._predictions_sink is not None:
            self._predictions_sink = flow(self._predictions_sink)

        dataset = flow(self.dataset)
        self._resolved_class_names = self._resolve_class_names(dataset)
        loader = flow(self.loader, dataset=dataset)
        lightning_trainer = flow(self.lightning)
        lightning_trainer.test(model=self, dataloaders=loader)
        if self._predictions_sink is not None and hasattr(self._predictions_sink, "close"):
            self._predictions_sink.close()
        self._finalize_metrics()
        logger.info("Evaluation complete.")

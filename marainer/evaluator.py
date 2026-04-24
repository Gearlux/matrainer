"""Marainer Evaluator: run a trained model over a dataset and collect predictions.

Sibling to :class:`~marainer.trainer.Trainer`. The Evaluator loads a model
(optionally from a checkpoint), streams a dataset through it, accumulates
metrics that understand the model's prediction contract (see
:mod:`marainer.outputs`), and optionally forwards each prediction to a sink
for persistence.

Task-agnostic by design: the Evaluator does not know whether it is running
classification, detection, or segmentation — it only requires that the
model's eval-mode ``forward`` produces something the configured ``metrics``
and ``predictions_sink`` expect.

YAML example (detection)::

    evaluator: !class:marainer.evaluator.Evaluator
      model: !class:marainer.detection.FasterRCNNModel { num_classes: 2 }
      checkpoint_path: ./checkpoints/best.ckpt
      dataset: !ref:eval_set
      loader: !class:torch.utils.data.DataLoader
        batch_size: 4
        collate_fn: !ref:marainer.detection.detection_collate_fn
      metrics: !class:marainer.detection.DetectionMetricAdapter
        inner: !class:torchmetrics.detection.MeanAveragePrecision
          iou_type: bbox

    cli:  marainer evaluate config.yaml
"""

from typing import Any, Optional

import pytorch_lightning as L
import torch
from confluid import Class, configurable, flow
from logflow import get_logger
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

logger = get_logger(__name__)


@configurable
class Evaluator(L.LightningModule):
    """Run a model over a dataset, update metrics, and optionally persist predictions.

    Args:
        model: The neural network (deferred via Confluid).
        dataset: Evaluation dataset or Flux pipeline (deferred).
        loader: DataLoader config (deferred). Supply a matching
            ``collate_fn`` for task types that need one (e.g.
            :func:`marainer.detection.detection_collate_fn`).
        metrics: Optional :class:`~torchmetrics.MetricCollection` or adapter
            that understands the model's step output. Logged via Lightning's
            ``log_dict`` at epoch end.
        predictions_sink: Optional duck-typed sink exposing
            ``.write(prediction, metadata)`` and ``.close()``. Called once
            per sample within each batch with the per-sample prediction and
            its metadata (if the batch carries a ``per_sample`` metadata
            list, as the standard Marainer collate does).
        checkpoint_path: Optional path to a Lightning checkpoint. When set,
            ``state_dict`` is loaded (``strict=False``) before evaluation.
        lightning: PyTorch Lightning Trainer config (deferred). Only ``test``
            is invoked, so training knobs like ``max_epochs`` are ignored.
        experiment_name / run_name: Metadata for loggers (MLflow, TB, etc.).
        loggers: Optional list of Lightning loggers.
    """

    def __init__(
        self,
        model: Any,
        dataset: Any,
        loader: Any = Class(DataLoader),
        metrics: Any = None,
        predictions_sink: Any = None,
        checkpoint_path: Optional[str] = None,
        lightning: Any = Class(L.Trainer),
        experiment_name: str = "marainer_eval",
        run_name: str = "evaluate",
        loggers: Any = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.loader = loader
        self.metrics: Any = metrics
        self._predictions_sink = predictions_sink
        self.checkpoint_path = checkpoint_path
        self.lightning = lightning
        self.experiment_name = experiment_name
        self.run_name = run_name
        self._loggers_config = loggers

    def forward(self, x: Any) -> Any:
        return self.model(x)

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        preds = self(batch.input)
        batch_size = self._infer_batch_size(batch)
        if self.metrics is not None:
            self.metrics.update(preds, batch.target)
            if isinstance(self.metrics, MetricCollection):
                self.log_dict(self.metrics, on_step=False, on_epoch=True, batch_size=batch_size)
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

    def run(self) -> None:
        """Materialize deferred components, load checkpoint if any, and evaluate."""
        logger.info(f"Marainer evaluating: {self.experiment_name} ({self.run_name})")
        self.model = flow(self.model)
        if self.metrics is not None:
            self.metrics = flow(self.metrics)
        if self._predictions_sink is not None:
            self._predictions_sink = flow(self._predictions_sink)

        if self.checkpoint_path:
            logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            state = torch.load(self.checkpoint_path, map_location="cpu")
            state_dict = state.get("state_dict", state)
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            if missing:
                logger.info(f"Checkpoint missing keys: {len(missing)}")
            if unexpected:
                logger.info(f"Checkpoint unexpected keys: {len(unexpected)}")

        loader = flow(self.loader, dataset=flow(self.dataset))
        lightning_trainer = flow(self.lightning)
        lightning_trainer.test(model=self, dataloaders=loader)
        if self._predictions_sink is not None and hasattr(self._predictions_sink, "close"):
            self._predictions_sink.close()
        logger.info("Evaluation complete.")

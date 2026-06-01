"""Tests for marainer's default ModelCheckpoint injection, per-logger
``log_all_checkpoints`` opt-in, and YAML config-tree artifact logging.

Covers the behaviour added alongside the
``log_config_artifacts`` / ``_marainer_ensure_default_checkpoint`` helpers
in :mod:`marainer.trainer`.
"""

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class, configurable
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy

from marainer.loggers import MarainerCSVLogger, MarainerMLFlowLogger, MarainerTensorBoardLogger
from marainer.trainer import (
    LightningTrainer,
    _is_model_checkpoint_target,
    _log_paths_to_logger,
    _logger_wants_all_checkpoints,
    log_config_artifacts,
)


@configurable
class TinyModel(nn.Module):
    """Smallest possible model so the trainer instantiates without GPU weight init."""

    def __init__(self, input_size: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.fc(x))


def _make_trainer(**overrides: Any) -> LightningTrainer:
    """Build a LightningTrainer, applying overrides as post-construction attributes.

    Mirrors the minimal-ctor pattern: only ``model`` / ``train_set`` /
    ``experiment_name`` / ``run_name`` are constructor params; every other slot
    (``lightning``, ``val_set``, ``val_metrics``, ``log_system_metrics``, …) is
    set on the instance after construction.
    """
    settings: dict[str, Any] = dict(
        model=TinyModel(),
        train_set=[],
        lightning=Class(L.Trainer, accelerator="cpu"),
    )
    settings.update(overrides)
    ctor = {k: settings.pop(k) for k in ("model", "train_set", "experiment_name", "run_name") if k in settings}
    trainer = LightningTrainer(**ctor)
    for key, value in settings.items():
        setattr(trainer, key, value)
    return trainer


# ---------------------------------------------------------------------------
# _is_model_checkpoint_target / _logger_wants_all_checkpoints helpers
# ---------------------------------------------------------------------------


class TestIsModelCheckpointTarget:
    def test_live_instance(self) -> None:
        assert _is_model_checkpoint_target(ModelCheckpoint(monitor="val/loss"))

    def test_class_fluid_with_concrete_target(self) -> None:
        assert _is_model_checkpoint_target(Class(ModelCheckpoint, monitor="val/loss"))

    def test_class_fluid_with_string_target(self) -> None:
        assert _is_model_checkpoint_target(Class("pytorch_lightning.callbacks.ModelCheckpoint"))

    def test_unrelated_callback_is_not_matched(self) -> None:
        from pytorch_lightning.callbacks import TQDMProgressBar

        assert not _is_model_checkpoint_target(TQDMProgressBar())
        assert not _is_model_checkpoint_target(Class(TQDMProgressBar))

    def test_plain_object_is_not_matched(self) -> None:
        assert not _is_model_checkpoint_target("not a callback")


class TestLoggerWantsAllCheckpoints:
    def test_flag_on_instance(self, tmp_path: Path) -> None:
        lg = MarainerTensorBoardLogger(save_dir=str(tmp_path), log_all_checkpoints=True)
        assert _logger_wants_all_checkpoints(lg)

    def test_flag_off_instance_default(self, tmp_path: Path) -> None:
        lg = MarainerTensorBoardLogger(save_dir=str(tmp_path))
        assert not _logger_wants_all_checkpoints(lg)

    def test_flag_on_fluid(self) -> None:
        fluid = Class("marainer.loggers.MarainerMLFlowLogger", log_all_checkpoints=True)
        assert _logger_wants_all_checkpoints(fluid)

    def test_upstream_logger_without_attr_is_false(self, tmp_path: Path) -> None:
        from pytorch_lightning.loggers import TensorBoardLogger

        lg = TensorBoardLogger(save_dir=str(tmp_path))
        assert not _logger_wants_all_checkpoints(lg)


# ---------------------------------------------------------------------------
# MarainerMLFlowLogger log_model wiring
# ---------------------------------------------------------------------------


class TestMarainerMLFlowLogger:
    def test_log_model_true_when_flag_off(self, tmp_path: Path) -> None:
        lg = MarainerMLFlowLogger(tracking_uri=f"file:{tmp_path}", experiment_name="t")
        assert lg._log_model is True

    def test_log_model_all_when_flag_on(self, tmp_path: Path) -> None:
        lg = MarainerMLFlowLogger(tracking_uri=f"file:{tmp_path}", experiment_name="t", log_all_checkpoints=True)
        assert lg._log_model == "all"

    def test_explicit_log_model_kwarg_wins(self, tmp_path: Path) -> None:
        """Caller-supplied ``log_model`` is respected (setdefault semantics)."""
        lg = MarainerMLFlowLogger(
            tracking_uri=f"file:{tmp_path}",
            experiment_name="t",
            log_all_checkpoints=True,
            log_model=False,
        )
        assert lg._log_model is False
        assert lg.log_all_checkpoints is True


# ---------------------------------------------------------------------------
# _resolve_default_checkpoint_monitor: first-metric + mode auto-detect
# ---------------------------------------------------------------------------


class TestResolveDefaultCheckpointMonitor:
    def test_first_val_metric_with_max_mode(self) -> None:
        trainer = _make_trainer(val_set=[1, 2])
        trainer.val_metrics = MetricCollection({"top1": MulticlassAccuracy(num_classes=2)}, prefix="val/")
        monitor, mode = trainer._resolve_default_checkpoint_monitor()
        assert monitor == "val/top1"
        # MulticlassAccuracy.higher_is_better is True → max
        assert mode == "max"

    def test_min_mode_for_higher_is_better_false(self) -> None:
        class FakeMinMetric:
            higher_is_better = False

            def __init__(self) -> None:
                pass

        trainer = _make_trainer(val_set=[1, 2])
        fake_collection = MagicMock(spec=MetricCollection)
        fake_collection.items.return_value = [("val/loss_proxy", FakeMinMetric())]
        fake_collection.prefix = "val/"
        trainer.val_metrics = fake_collection
        monitor, mode = trainer._resolve_default_checkpoint_monitor()
        assert monitor == "val/loss_proxy"
        assert mode == "min"

    def test_falls_back_to_val_loss_when_val_set_present_no_metrics(self) -> None:
        trainer = _make_trainer(val_set=[1, 2])
        assert trainer.val_metrics is None
        monitor, mode = trainer._resolve_default_checkpoint_monitor()
        assert monitor == "val/loss"
        assert mode == "min"

    def test_falls_back_when_metrics_is_empty(self) -> None:
        trainer = _make_trainer(val_set=[1, 2])
        trainer.val_metrics = MetricCollection({}, prefix="val/")
        monitor, mode = trainer._resolve_default_checkpoint_monitor()
        assert monitor == "val/loss"
        assert mode == "min"

    def test_falls_back_to_train_loss_without_val_set(self) -> None:
        """No val set → can't monitor val/loss (Lightning would warn each
        epoch about the missing key). Fall back to train/loss which is
        always logged by ``training_step``."""
        trainer = _make_trainer()  # val_set defaults to None
        assert trainer.val_set is None
        monitor, mode = trainer._resolve_default_checkpoint_monitor()
        assert monitor == "train/loss"
        assert mode == "min"

    def test_prefix_not_doubled_when_key_already_carries_it(self) -> None:
        """``MetricCollection.items()`` already returns prefixed keys; we must
        not prepend the prefix again."""
        trainer = _make_trainer(val_set=[1, 2])
        trainer.val_metrics = MetricCollection({"top1": MulticlassAccuracy(num_classes=2)}, prefix="val/")
        monitor, _ = trainer._resolve_default_checkpoint_monitor()
        assert monitor == "val/top1"
        assert not monitor.startswith("val/val/")


# ---------------------------------------------------------------------------
# _marainer_ensure_default_checkpoint: injection + opt-out + escalation
# ---------------------------------------------------------------------------


class TestEnsureDefaultCheckpoint:
    def _flow_callbacks(self, trainer: LightningTrainer) -> list[Any]:
        """Resolve ``self.lightning.kwargs['callbacks']`` to a concrete list."""
        callbacks = trainer.lightning.kwargs.get("callbacks") or []
        return callbacks if isinstance(callbacks, list) else [callbacks]

    def test_injects_when_no_callback_present(self) -> None:
        trainer = _make_trainer(val_set=[1, 2])
        trainer._marainer_ensure_default_checkpoint()
        callbacks = self._flow_callbacks(trainer)
        cps = [cb for cb in callbacks if isinstance(cb, ModelCheckpoint)]
        assert len(cps) == 1
        cb = cps[0]
        assert cb.monitor == "val/loss"
        assert cb.mode == "min"
        assert cb.save_last is True
        assert cb.save_top_k == 1

    def test_uses_first_val_metric_for_monitor(self) -> None:
        trainer = _make_trainer(val_set=[1, 2])
        trainer.val_metrics = MetricCollection({"top1": MulticlassAccuracy(num_classes=2)}, prefix="val/")
        trainer._marainer_ensure_default_checkpoint()
        cb = next(cb for cb in self._flow_callbacks(trainer) if isinstance(cb, ModelCheckpoint))
        assert cb.monitor == "val/top1"
        assert cb.mode == "max"

    def test_skips_when_user_provides_modelcheckpoint(self) -> None:
        existing = ModelCheckpoint(monitor="train/loss", mode="min", save_top_k=3)
        trainer = _make_trainer(lightning=Class(L.Trainer, accelerator="cpu", callbacks=[existing]))
        trainer._marainer_ensure_default_checkpoint()
        callbacks = self._flow_callbacks(trainer)
        cps = [cb for cb in callbacks if isinstance(cb, ModelCheckpoint)]
        # User's callback untouched, no second one added.
        assert len(cps) == 1
        assert cps[0] is existing
        assert cps[0].monitor == "train/loss"

    def test_skips_when_user_provides_fluid_modelcheckpoint(self) -> None:
        user_cb = Class(ModelCheckpoint, monitor="train/acc")
        trainer = _make_trainer(lightning=Class(L.Trainer, accelerator="cpu", callbacks=[user_cb]))
        trainer._marainer_ensure_default_checkpoint()
        callbacks = self._flow_callbacks(trainer)
        assert callbacks == [user_cb]  # No injection.

    def test_save_top_k_escalates_when_logger_opts_in(self, tmp_path: Path) -> None:
        logger_a = MarainerTensorBoardLogger(save_dir=str(tmp_path), log_all_checkpoints=True)
        logger_b = MarainerCSVLogger(save_dir=str(tmp_path))  # default False
        trainer = _make_trainer(lightning=Class(L.Trainer, accelerator="cpu", logger=[logger_a, logger_b]))
        trainer._marainer_ensure_default_checkpoint()
        cb = next(cb for cb in self._flow_callbacks(trainer) if isinstance(cb, ModelCheckpoint))
        assert cb.save_top_k == -1

    def test_save_top_k_stays_at_one_without_opt_in(self, tmp_path: Path) -> None:
        logger_a = MarainerTensorBoardLogger(save_dir=str(tmp_path))
        logger_b = MarainerCSVLogger(save_dir=str(tmp_path))
        trainer = _make_trainer(lightning=Class(L.Trainer, accelerator="cpu", logger=[logger_a, logger_b]))
        trainer._marainer_ensure_default_checkpoint()
        cb = next(cb for cb in self._flow_callbacks(trainer) if isinstance(cb, ModelCheckpoint))
        assert cb.save_top_k == 1

    def test_save_top_k_escalation_via_fluid_logger(self) -> None:
        """A logger declared as a deferred ``Class`` (the YAML path) is also
        inspected — ``log_all_checkpoints=True`` in its kwargs flips
        ``save_top_k`` to -1 before the L.Trainer is even constructed."""
        deferred_logger = Class(
            "marainer.loggers.MarainerMLFlowLogger",
            tracking_uri="file:./mlruns",
            experiment_name="t",
            log_all_checkpoints=True,
        )
        trainer = _make_trainer(lightning=Class(L.Trainer, accelerator="cpu", logger=[deferred_logger]))
        trainer._marainer_ensure_default_checkpoint()
        cb = next(cb for cb in self._flow_callbacks(trainer) if isinstance(cb, ModelCheckpoint))
        assert cb.save_top_k == -1

    def test_single_logger_not_in_list(self, tmp_path: Path) -> None:
        """Lightning accepts ``logger=<single>``; the helper must normalize."""
        lg = MarainerTensorBoardLogger(save_dir=str(tmp_path), log_all_checkpoints=True)
        trainer = _make_trainer(lightning=Class(L.Trainer, accelerator="cpu", logger=lg))
        trainer._marainer_ensure_default_checkpoint()
        cb = next(cb for cb in self._flow_callbacks(trainer) if isinstance(cb, ModelCheckpoint))
        assert cb.save_top_k == -1

    def test_respects_enable_checkpointing_false(self) -> None:
        """``enable_checkpointing=False`` is the operator's explicit opt-out.
        Lightning would raise ``MisconfigurationException`` if we added a
        ModelCheckpoint alongside it — skip injection entirely."""
        trainer = _make_trainer(lightning=Class(L.Trainer, accelerator="cpu", enable_checkpointing=False))
        trainer._marainer_ensure_default_checkpoint()
        callbacks = self._flow_callbacks(trainer)
        assert not any(isinstance(cb, ModelCheckpoint) for cb in callbacks)


# ---------------------------------------------------------------------------
# log_config_artifacts: per-backend dispatch
# ---------------------------------------------------------------------------


class _FakeLiquifyCtx:
    def __init__(self, config_path: Path, included_paths: list[Path] | None = None) -> None:
        self.config_path = config_path
        self.included_paths = list(included_paths or [])


class TestLogConfigArtifacts:
    def test_mlflow_logger_receives_log_artifact_for_each_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cfg = tmp_path / "main.yaml"
        cfg.write_text("foo: 1")
        inc = tmp_path / "common.yaml"
        inc.write_text("bar: 2")

        ctx = _FakeLiquifyCtx(cfg, [cfg, inc])
        monkeypatch.setattr("liquifai.context.get_context", lambda: ctx)

        mock_exp = MagicMock()
        mock_logger = MagicMock()
        mock_logger.experiment = mock_exp
        mock_logger.run_id = "run-123"

        mock_lightning_trainer = MagicMock(spec=L.Trainer)
        mock_lightning_trainer.loggers = [mock_logger]

        log_config_artifacts(mock_lightning_trainer)

        calls = mock_exp.log_artifact.call_args_list
        assert len(calls) == 2
        assert calls[0].args == ("run-123", str(cfg), "config")
        assert calls[1].args == ("run-123", str(inc), "config")

    def test_tensorboard_logger_receives_file_copy(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        cfg = tmp_path / "main.yaml"
        cfg.write_text("foo: 1")

        log_dir = tmp_path / "lightning_logs"
        log_dir.mkdir()

        ctx = _FakeLiquifyCtx(cfg, [cfg])
        monkeypatch.setattr("liquifai.context.get_context", lambda: ctx)

        tb_logger = MarainerTensorBoardLogger(save_dir=str(tmp_path))
        # Avoid hitting TB's lazy log_dir resolution (it requires materialized
        # tfevents state) — pass via the helper directly.
        _log_paths_to_logger(MagicMock(experiment=None, run_id=None, log_dir=str(log_dir)), [cfg])

        copied = log_dir / "config" / "main.yaml"
        assert copied.is_file()
        assert copied.read_text() == "foo: 1"
        assert tb_logger is not None  # touched to silence unused-var

    def test_falls_back_to_config_path_when_no_included_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cfg = tmp_path / "main.yaml"
        cfg.write_text("foo: 1")

        # Empty included_paths -> falls back to config_path.
        ctx = _FakeLiquifyCtx(cfg, [])
        monkeypatch.setattr("liquifai.context.get_context", lambda: ctx)

        mock_logger = MagicMock()
        mock_logger.experiment = MagicMock()
        mock_logger.run_id = "r"
        mock_lightning_trainer = MagicMock(spec=L.Trainer)
        mock_lightning_trainer.loggers = [mock_logger]

        log_config_artifacts(mock_lightning_trainer)
        mock_logger.experiment.log_artifact.assert_called_once_with("r", str(cfg), "config")

    def test_no_context_is_silent_no_op(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("liquifai.context.get_context", lambda: None)
        mock_lightning_trainer = MagicMock(spec=L.Trainer)
        mock_lightning_trainer.loggers = []
        log_config_artifacts(mock_lightning_trainer)  # must not raise

    def test_missing_files_are_skipped(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        cfg = tmp_path / "ghost.yaml"  # not created
        ctx = _FakeLiquifyCtx(cfg, [cfg])
        monkeypatch.setattr("liquifai.context.get_context", lambda: ctx)

        mock_logger = MagicMock()
        mock_logger.experiment = MagicMock()
        mock_logger.run_id = "r"
        mock_lightning_trainer = MagicMock(spec=L.Trainer)
        mock_lightning_trainer.loggers = [mock_logger]

        log_config_artifacts(mock_lightning_trainer)
        mock_logger.experiment.log_artifact.assert_not_called()


# ---------------------------------------------------------------------------
# _marainer_warn_eager_runtime_slots: actionable warning for misconfigured slots
# ---------------------------------------------------------------------------


class TestWarnEagerRuntimeSlots:
    def test_warns_when_lightning_arrived_materialized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A live (eagerly-flowed, e.g. !class:) lightning slot triggers an actionable warning."""
        import marainer.trainer as trainer_mod

        trainer = _make_trainer()
        trainer.lightning = L.Trainer(accelerator="cpu")  # already-materialized (the !class: footgun)

        warnings: list[str] = []
        monkeypatch.setattr(trainer_mod.logger, "warning", lambda msg, *a, **k: warnings.append(str(msg)))
        trainer._marainer_warn_eager_runtime_slots()

        assert any("lightning" in w and "!lazy:" in w for w in warnings)

    def test_silent_when_slots_stay_deferred(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default deferred (LazyClass) slots — no warning."""
        import marainer.trainer as trainer_mod

        trainer = _make_trainer()
        trainer.lightning = Class(L.Trainer, accelerator="cpu")  # deferred Fluid — fine

        warnings: list[str] = []
        monkeypatch.setattr(trainer_mod.logger, "warning", lambda msg, *a, **k: warnings.append(str(msg)))
        trainer._marainer_warn_eager_runtime_slots()

        assert warnings == []

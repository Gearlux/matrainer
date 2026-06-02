"""Tests for logger integration in the Trainer."""

from typing import cast
from unittest.mock import MagicMock, patch

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class, configurable, materialize
from confluid.loader import load
from dataflux.core import Flux
from dataflux.sample import Sample
from pytorch_lightning.loggers import CSVLogger

from marainer.trainer import LightningTrainer
from marainer.utils import get_git_commit_hash


@configurable
class TinyModel(nn.Module):
    def __init__(self, input_size: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.fc(x))


def _make_samples(n: int = 32, input_size: int = 4) -> Flux:
    return Flux(source=[Sample(input=torch.randn(input_size), target=i % 2, metadata={}) for i in range(n)])


class TestLogHyperparams:
    def test_hyperparams_logged_to_all_loggers(self) -> None:
        """_log_hyperparams calls log_hyperparams on every attached logger."""
        mock_logger1 = MagicMock()
        mock_logger2 = MagicMock()
        mock_trainer = MagicMock(spec=L.Trainer)
        mock_trainer.loggers = [mock_logger1, mock_logger2]

        trainer = LightningTrainer(model=TinyModel(), train_set=_make_samples(16))
        trainer._marainer_log_hyperparams(mock_trainer)

        mock_logger1.log_hyperparams.assert_called_once()
        mock_logger2.log_hyperparams.assert_called_once()

    def test_log_system_metrics_default_true(self) -> None:
        """The opt-in flag is on by default for every Lightning trainer."""
        trainer = LightningTrainer(model=TinyModel(), train_set=_make_samples(16))
        assert trainer.log_system_metrics is True

    def test_log_system_metrics_opt_out_skips_activate(self, tmp_path: object) -> None:
        """``log_system_metrics=False`` short-circuits the activation in ``run()``."""
        trainer = LightningTrainer(model=TinyModel(), train_set=_make_samples(16))
        # Minimal-ctor pattern: configure the deferred slots post-construction.
        trainer.val_set = _make_samples(4)
        trainer.train_loader = Class(torch.utils.data.DataLoader, batch_size=4)
        trainer.val_loader = Class(torch.utils.data.DataLoader, batch_size=4)
        trainer.lightning = Class(
            L.Trainer,
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            default_root_dir=str(tmp_path),
        )
        trainer.log_system_metrics = False
        with patch("marainer.system_metrics.activate_for_lightning") as activate_mock:
            trainer.run()
        activate_mock.assert_not_called()

    def test_lightning_module_class_constants_dont_leak(self) -> None:
        """Inherited ``pytorch_lightning.LightningModule`` / ``torch.nn.Module`` attributes
        must not surface as user hyperparameters. The discovery walker filters out class
        annotations (``training: bool``), class constants (``CHECKPOINT_HYPER_PARAMS_KEY``),
        and ``__init__``-body setattrs (``self.prepare_data_per_node: bool = True``)
        contributed by non-``@configurable`` ancestors. Paths are also rewritten to their
        shortest unique suffix so the noisy ``LightningTrainer.`` prefix is dropped."""
        mock_logger = MagicMock()
        mock_trainer = MagicMock(spec=L.Trainer)
        mock_trainer.loggers = [mock_logger]

        trainer = LightningTrainer(
            model=TinyModel(),
            train_set=_make_samples(16),
            experiment_name="test_exp",
            run_name="test_run",
        )
        trainer._marainer_log_hyperparams(mock_trainer)

        ((logged_params,), _) = mock_logger.log_hyperparams.call_args
        keys = set(logged_params.keys())

        # PyTorch Lightning / nn.Module noise must be absent.
        leaky_names = {
            "CHECKPOINT_HYPER_PARAMS_KEY",
            "CHECKPOINT_HYPER_PARAMS_NAME",
            "CHECKPOINT_HYPER_PARAMS_TYPE",
            "training",
            "dump_patches",
            "prepare_data_per_node",
            "allow_zero_length_dataloader_with_multiple_devices",
            "automatic_optimization",
        }
        assert not (keys & leaky_names), f"leaked Lightning/torch attrs: {keys & leaky_names}"

        # User hyperparameters must be present, with the root-class prefix dropped.
        assert "experiment_name" in keys
        assert "run_name" in keys
        assert logged_params["experiment_name"] == "test_exp"
        assert logged_params["run_name"] == "test_run"
        assert not any(k.startswith("LightningTrainer.") for k in keys)


class TestYamlLoggers:
    def test_yaml_loggers_reach_lightning_trainer(self, tmp_path: object) -> None:
        """End-to-end: loggers declared under ``lightning.logger:`` in YAML
        must end up attached to the real ``L.Trainer`` produced by flowing
        ``trainer.lightning``.

        ``lightning`` is a deferred body slot (minimal-ctor pattern), so it is
        wired ``!lazy:`` — it stays deferred through materialization (a bare
        ``!class:`` post-init attr would be eagerly flowed on assignment) and is
        built by an explicit ``flow()``.
        """
        from confluid import flow

        yaml_text = f"""
model: !class:tests.test_loggers.TinyModel()

train_set: !class:dataflux.core.Flux
  source: []

lightning: !lazy:pytorch_lightning.Trainer
  max_epochs: 1
  accelerator: cpu
  enable_progress_bar: false
  enable_model_summary: false
  logger:
    - !class:pytorch_lightning.loggers.CSVLogger()
        save_dir: "{tmp_path}"
        name: csv_test
"""
        data = load(yaml_text, flow=False)
        trainer = materialize({"_confluid_class_": "LightningTrainer", **data}, context=data)

        # Before flow: the Lightning Trainer stays deferred, but the YAML-declared
        # loggers are already captured in its kwargs (as Instance objects).
        assert isinstance(trainer, LightningTrainer)
        assert isinstance(trainer.lightning, Class)
        assert "logger" in trainer.lightning.kwargs
        assert len(trainer.lightning.kwargs["logger"]) == 1

        # After flow: the L.Trainer exposes the flowed CSVLogger via .loggers.
        lightning_trainer = flow(trainer.lightning)
        assert isinstance(lightning_trainer, L.Trainer)
        attached = list(lightning_trainer.loggers)
        assert len(attached) == 1
        assert isinstance(attached[0], CSVLogger)
        assert attached[0].name == "csv_test"


class TestGetGitHash:
    def test_returns_hash_in_git_repo(self) -> None:
        """get_git_commit_hash returns a hash string in a git repo."""
        result = get_git_commit_hash()
        assert result is not None
        # Should start with 40 hex chars
        assert len(result) >= 40
        sha = result[:40]
        assert all(c in "0123456789abcdef" for c in sha)
        if len(result) > 40:
            assert result.endswith("-dirty")

    def test_returns_none_outside_git(self) -> None:
        """get_git_commit_hash returns None when git is unavailable."""
        with patch("marainer.utils.subprocess.check_output", side_effect=FileNotFoundError):
            assert get_git_commit_hash() is None

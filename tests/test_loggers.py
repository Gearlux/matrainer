"""Tests for logger integration in the Trainer."""

from typing import cast
from unittest.mock import MagicMock, patch

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class, configurable, materialize
from confluid.loader import load
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


def _make_samples(n: int = 32, input_size: int = 4) -> list[Sample]:
    return [Sample(input=torch.randn(input_size), target=i % 2, metadata={}) for i in range(n)]


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


class TestYamlLoggers:
    def test_yaml_loggers_reach_lightning_trainer(self, tmp_path: object) -> None:
        """End-to-end: loggers declared under ``lightning.logger:`` in YAML
        must end up attached to the real ``L.Trainer`` produced by flowing
        ``trainer.lightning``.
        """
        from confluid import flow

        yaml_text = f"""
model: !class:tests.test_loggers.TinyModel()

train_set: []

lightning: !class:pytorch_lightning.Trainer
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

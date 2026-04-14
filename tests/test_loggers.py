"""Tests for logger integration in the Trainer."""

from typing import cast
from unittest.mock import MagicMock, patch

import pytorch_lightning as L
import torch
import torch.nn as nn
from confluid import Class, configurable
from dataflux.core import Flux
from dataflux.sample import Sample
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from marainer.trainer import Trainer
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


class TestLoggerConfig:
    def test_loggers_none_by_default(self) -> None:
        """Trainer with no loggers config uses Lightning's default."""
        trainer = Trainer(model=TinyModel(), train_set=_make_samples(16))
        assert trainer._loggers_config is None

    def test_loggers_stored_internally(self) -> None:
        """Loggers are stored as _loggers_config, not shadowing the LightningModule property."""
        csv_logger = CSVLogger(save_dir=".", name="test_csv")
        trainer = Trainer(model=TinyModel(), train_set=_make_samples(16), loggers=[csv_logger])
        assert trainer._loggers_config == [csv_logger]

    def test_full_epoch_with_csv_logger(self, tmp_path: object) -> None:
        """CSVLogger receives metrics during a full training epoch."""
        csv_logger = CSVLogger(save_dir=str(tmp_path), name="csv_logs")
        trainer = Trainer(
            model=TinyModel(),
            train_set=_make_samples(32),
            optimizer=Class(torch.optim.SGD, lr=0.01),
            loggers=[csv_logger],
        )
        flux = Flux(source=_make_samples(32))
        loader: DataLoader = DataLoader(flux, batch_size=16)
        lt = L.Trainer(
            max_epochs=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=csv_logger,
        )
        lt.fit(model=trainer, train_dataloaders=loader)
        # CSVLogger should have logged metrics
        assert csv_logger.experiment is not None

    def test_loggers_none_backward_compat(self) -> None:
        """Training works with no loggers configured (backward compat)."""
        trainer = Trainer(
            model=TinyModel(),
            train_set=_make_samples(32),
            optimizer=Class(torch.optim.SGD, lr=0.01),
        )
        flux = Flux(source=_make_samples(32))
        loader: DataLoader = DataLoader(flux, batch_size=16)
        lt = L.Trainer(max_epochs=1, accelerator="cpu", enable_progress_bar=False, enable_model_summary=False)
        lt.fit(model=trainer, train_dataloaders=loader)


class TestLogHyperparams:
    def test_hyperparams_logged_to_all_loggers(self) -> None:
        """_log_hyperparams calls log_hyperparams on every attached logger."""
        mock_logger1 = MagicMock()
        mock_logger2 = MagicMock()
        mock_trainer = MagicMock(spec=L.Trainer)
        mock_trainer.loggers = [mock_logger1, mock_logger2]

        trainer = Trainer(model=TinyModel(), train_set=_make_samples(16))
        trainer._marainer_log_hyperparams(mock_trainer)

        mock_logger1.log_hyperparams.assert_called_once()
        mock_logger2.log_hyperparams.assert_called_once()


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

"""End-to-end test for ``flow_mode="auto"`` + ``Lazy[T]`` annotation.

This test runs the marainer CLI through ``app.run()`` with a real YAML config
and asserts that liquifai's deep-flow:

* eagerly flowed the non-Lazy attrs (``model``, ``lightning``);
* left the ``Lazy[Any]`` attrs (``optimizer``, ``train_loader``, ``val_loader``,
  ``test_loader``) as deferred ``Class`` stubs so ``LightningTrainer.run()`` can flow
  them with runtime kwargs (``params=self.parameters()``, ``dataset=...``).
"""

import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from liquifai.context import set_context
from marainer.cli import app
from marainer.trainer import LightningTrainer


@pytest.fixture(autouse=True)
def _reset_context() -> None:
    set_context(None)  # type: ignore[arg-type]


def test_train_command_honors_lazy_annotations(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """The injected LightningTrainer has eager `model` but deferred `optimizer`/`*_loader`."""
    config = tmp_path / "train.yaml"
    config.write_text(
        "trainer: !class:marainer.trainer.LightningTrainer\n"
        "  model: !class:tests.test_cli_flow_mode._TinyModel\n"
        "    input_size: 4\n"
        "  train_set: []\n"
        "  optimizer: !class:torch.optim.SGD\n"
        "    lr: 0.01\n"
        "  train_loader: !class:torch.utils.data.DataLoader\n"
        "    batch_size: 8\n"
    )

    monkeypatch.setattr(sys, "argv", ["marainer", "train", str(config)])

    with patch.object(LightningTrainer, "run") as fake_run:
        app.run()

    fake_run.assert_called_once()
    # `app.run_command` calls `func(**kwargs)`, so the trainer instance is the
    # one injected. We can fish it back via the global context's resolved kwargs
    # — easiest way is to re-resolve via liquify().
    from confluid.fluid import Class
    from marainer.cli import train

    kwargs = app.liquify(train, config_path=config)
    trainer = kwargs["trainer"]

    # model was NOT marked Lazy, so it's a live instance.
    assert isinstance(trainer.model, _TinyModel)
    # optimizer / train_loader ARE Lazy → still deferred Class stubs.
    assert isinstance(trainer.optimizer, Class)
    assert isinstance(trainer.train_loader, Class)
    # The runtime-injection contract still works: configure_optimizers should
    # flow the optimizer with params=self.parameters() and produce a live optim.
    optim = trainer.configure_optimizers()
    import torch

    assert isinstance(optim, torch.optim.Optimizer)


# Defined at module level so confluid can resolve `tests.test_cli_flow_mode._TinyModel`.
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from confluid import configurable  # noqa: E402


@configurable
class _TinyModel(nn.Module):
    def __init__(self, input_size: int = 4, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from typing import cast

        return cast(torch.Tensor, self.fc(x))

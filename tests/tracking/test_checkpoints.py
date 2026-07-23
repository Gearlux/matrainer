"""Tests for the framework-neutral checkpoint policies."""

from marainer.checkpoints import CheckpointPolicy, SaveBest, SaveEveryEpoch, SaveLast


def test_all_zero_arg_constructible() -> None:
    # Workspace zero-arg / no-work-in-__init__ rule.
    assert CheckpointPolicy().dirpath is None
    assert SaveLast().filename is None
    assert SaveEveryEpoch().every_n_epochs == 1
    sb = SaveBest()
    assert (sb.monitor, sb.mode, sb.top_k) == ("val/loss", "min", 1)


def test_save_best_fields() -> None:
    sb = SaveBest(monitor="val/acc", mode="max", top_k=3, dirpath="/ckpts")
    assert sb.monitor == "val/acc"
    assert sb.mode == "max"
    assert sb.top_k == 3
    assert sb.dirpath == "/ckpts"


def test_save_every_epoch_cadence() -> None:
    assert SaveEveryEpoch(every_n_epochs=5).every_n_epochs == 5


def test_policies_are_checkpoint_policies() -> None:
    for cls in (SaveLast, SaveEveryEpoch, SaveBest):
        assert issubclass(cls, CheckpointPolicy)

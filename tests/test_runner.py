"""TorchRunner marker — distinguishes gradient-based runnables (trainers) from inference-only ones.

The marker lets a GUI executor (FluxStudio's RunNode) re-enable autograd when ComfyUI runs the
graph under ``torch.inference_mode()``; trainers carry it, evaluators / processors do not.
"""

from marainer.evaluator import LightningEvaluator
from marainer.processing import DatasetProcessor
from marainer.runner import TorchRunner
from marainer.trainer import LightningTrainer


def test_base_trainer_is_a_torch_runner() -> None:
    assert issubclass(LightningTrainer, TorchRunner)
    assert getattr(LightningTrainer, "__torch_runner__", False) is True


def test_marker_re_exported_from_package_root() -> None:
    import marainer

    assert marainer.TorchRunner is TorchRunner


def test_subclasses_inherit_the_marker() -> None:
    # A task trainer subclassing the base (sonair / deltaid / raidar) must carry the marker without
    # re-declaring it. Use a local subclass to avoid a reverse dependency on those packages.
    class _TaskTrainer(LightningTrainer):
        pass

    assert getattr(_TaskTrainer, "__torch_runner__", False) is True


def test_inference_only_runnables_are_not_torch_runners() -> None:
    # Evaluation / processing don't compute gradients — they must NOT trip the grad restore.
    assert getattr(LightningEvaluator, "__torch_runner__", False) is False
    assert getattr(DatasetProcessor, "__torch_runner__", False) is False


def test_marker_is_stateless() -> None:
    # Pure mixin: no __init__ of its own, so it composes with LightningModule cleanly.
    assert "__init__" not in TorchRunner.__dict__
    assert TorchRunner.__torch_runner__ is True

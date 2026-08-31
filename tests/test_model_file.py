"""TorchModelFile — a COMPLETE saved torch model, loaded from one file.

The counterpart of :func:`load_lightning_checkpoint` for files that hold the whole pickled
module rather than a state dict. The tests use torch's own layer classes as the saved model —
a whole-model pickle stores an import REFERENCE to the defining class, so only always-importable
classes make a test that cannot rot.
"""

from typing import Any

import pytest
import torch
from confluid import marks
from torch import nn

from matrainer.torch import TorchModelFile


def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))


class TestConstructionIsLazy:
    def test_zero_arg_builds_and_loads_nothing(self) -> None:
        assert TorchModelFile().model is None

    def test_an_unset_file_is_reported_at_use(self) -> None:
        with pytest.raises(ValueError, match="file"):
            TorchModelFile().solidify()

    def test_a_missing_file_names_its_path(self, tmp_path: Any) -> None:
        absent = tmp_path / "absent.pt"
        with pytest.raises(FileNotFoundError, match="absent.pt"):
            TorchModelFile(file=str(absent)).solidify()


class TestLoadingAWholeModel:
    def test_the_saved_model_comes_back_and_forward_delegates(self, tmp_path: Any) -> None:
        saved = _tiny_model()
        path = tmp_path / "whole.pt"
        torch.save(saved, path)

        loader = TorchModelFile(file=str(path)).eval()
        with torch.no_grad():
            x = torch.ones(2, 4)
            assert torch.equal(loader(x), saved.eval()(x))

    def test_solidify_loads_once(self, tmp_path: Any) -> None:
        path = tmp_path / "whole.pt"
        torch.save(_tiny_model(), path)
        loader = TorchModelFile(file=str(path))
        built = loader.solidify()
        assert isinstance(built, nn.Module)
        assert loader.solidify() is built, "the load must be cached, not repeated"

    def test_eval_and_to_reach_the_loaded_model(self, tmp_path: Any) -> None:
        path = tmp_path / "whole.pt"
        torch.save(_tiny_model(), path)
        loader = TorchModelFile(file=str(path))
        loader.solidify()
        loader.eval()
        assert not loader.model.training  # type: ignore[union-attr]
        loader.to("cpu")  # must not raise — the loaded model is a registered child


class TestRefusingWhatIsNotAModel:
    def test_a_plain_state_dict_is_refused_with_the_fix_named(self, tmp_path: Any) -> None:
        path = tmp_path / "state.pt"
        torch.save(_tiny_model().state_dict(), path)
        with pytest.raises(ValueError, match="state dict") as err:
            TorchModelFile(file=str(path)).solidify()
        assert "checkpoint_path" in str(err.value)
        assert "state.pt" in str(err.value)

    def test_a_lightning_checkpoint_is_refused_the_same_way(self, tmp_path: Any) -> None:
        path = tmp_path / "epoch=3.ckpt"
        torch.save({"state_dict": _tiny_model().state_dict(), "epoch": 3}, path)
        with pytest.raises(ValueError, match="checkpoint_path"):
            TorchModelFile(file=str(path)).solidify()

    def test_a_non_callable_object_is_refused(self, tmp_path: Any) -> None:
        path = tmp_path / "tensor.pt"
        torch.save(torch.zeros(3), path)
        with pytest.raises(ValueError, match="tensor.pt"):
            TorchModelFile(file=str(path)).solidify()


class TestDiscoveryAndProvenance:
    def test_the_registration_marks_make_it_an_object_node(self) -> None:
        """role="model" is the gate a canvas bridge keys object-member nodes on; framework
        qualifies the socket so the member docks into torch model slots."""
        m = marks(TorchModelFile)
        assert m.role == "model"
        assert m.framework == "torch"

    def test_model_refs_names_the_file_and_nothing_else(self, tmp_path: Any) -> None:
        path = tmp_path / "whole.pt"
        path.touch()
        (ref,) = TorchModelFile(file=str(path)).model_refs
        assert ref.uri.startswith("file://") and ref.url is None
        assert TorchModelFile().model_refs == []

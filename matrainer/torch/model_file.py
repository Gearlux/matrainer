"""Load a COMPLETE saved torch model from one file — the whole-module counterpart of
:func:`~matrainer.torch.checkpoint.load_lightning_checkpoint`.

A checkpoint holds WEIGHTS and needs the architecture rebuilt around it; a whole-model file
(``torch.save(model, path)``) holds the module itself. :class:`TorchModelFile` is the loader for
the second kind: point ``file`` at the pickle and the saved model — its architecture, weights,
and whatever pre/post-processing its class implements — comes back as one callable module.

Two facts to know before reaching for it, both inherent to torch's whole-model format:

* the pickle stores an import REFERENCE to the model's class, not its code — the defining
  package must be installed (and structurally compatible) wherever the file is loaded; and
* un-pickling executes code from the file — load only files you trust
  (``torch.load(weights_only=False)`` is exactly the trusted-file mode).

A file that holds a dict instead (a plain state dict, a training checkpoint) is REFUSED with the
fix named: those files carry no architecture, so they load through a model wrapper's
``checkpoint_path``, never through this class.
"""

import os
from typing import Any, List, Optional

import torch
from confluid import configurable
from torch import nn

from matrainer.uri import ModelRef, file_model_ref


@configurable(role="model", framework="torch")
class TorchModelFile(nn.Module):
    """One saved torch model, loaded lazily from ``file`` and used exactly as saved.

    ``role="model"`` (task-less — the file decides what the model is for) makes the class an
    object-member producer for canvas/discovery surfaces, docking into any torch model slot;
    everything the model does (its output conventions included) travels WITH the file.

    Args:
        file: Path to a whole-model pickle written by ``torch.save(model, path)``. A state-dict
            or checkpoint file is refused, naming the fix.
    """

    def __init__(self, file: str = "") -> None:
        super().__init__()
        self.file = file
        # Partial: config only — the torch.load (and the code it executes) is solidify()'s cost.
        self.model: Optional[nn.Module] = None

    def solidify(self) -> nn.Module:
        """Load (once) and return the saved model — confluid's ``flow()``-time hook."""
        if self.model is None:
            if not self.file:
                raise ValueError("TorchModelFile needs 'file' (a whole-model pickle from torch.save(model, path))")
            if not os.path.exists(self.file):
                raise FileNotFoundError(f"TorchModelFile: no such file: {self.file}")
            loaded = torch.load(self.file, map_location="cpu", weights_only=False)
            if isinstance(loaded, dict):
                kind = "a training checkpoint" if "state_dict" in loaded else "a state dict"
                keys = ", ".join(sorted(str(k) for k in loaded)[:6])
                raise ValueError(
                    f"TorchModelFile: {self.file!r} holds a dict ({kind}; top-level keys: {keys}), "
                    f"not a saved model — it carries no architecture. Load it through a model "
                    f"wrapper's checkpoint_path (the wrapper rebuilds the network), or re-save "
                    f"the whole model with torch.save(model, path)."
                )
            if not callable(loaded):
                raise ValueError(
                    f"TorchModelFile: {self.file!r} holds a {type(loaded).__name__}, not a callable model."
                )
            # Assigning through nn.Module registers the child, so eval()/.to()/state_dict()
            # on THIS object reach the loaded model.
            self.model = loaded
        return self.model

    @property
    def model_refs(self) -> List[ModelRef]:
        """The one artifact this loads — see :mod:`matrainer.uri`. Reads stored config only."""
        return [file_model_ref(self.file)] if self.file else []

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.solidify()(*args, **kwargs)


__all__ = ["TorchModelFile"]

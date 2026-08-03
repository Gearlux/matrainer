"""Where a torch run executes — the ``"auto"`` choice, written out (**Lightning-free**).

A ``Trainer`` picks an accelerator for you. A hand-written training loop has no Trainer, so
somebody has to answer "which device?", and every such loop answers it identically: prefer CUDA,
then Apple's MPS, then CPU. That is one function, and it belongs beside the other things a torch
training platform needs from this package (:mod:`matrainer.torch.metrics`,
:mod:`matrainer.torch.checkpoint`) rather than in whichever loop happened to write it first.

Distinct from :func:`matrainer.tracking.provenance.accelerator_state`, which answers a different
question for a different reader: that one REPORTS what the process is running on (reading an
already-imported framework out of ``sys.modules``, so a light tracking call never pays for a
framework import), while this one DECIDES where a run will execute and therefore imports torch
outright. Do not merge them — a tracking probe must not force a framework import, and a device
choice must not be silently skipped because torch was not imported yet.
"""

from typing import Literal

import torch

#: Where a run executes. ``"auto"`` resolves to the best device actually present; the rest are
#: taken at their word. A closed ``Literal`` rather than a bare ``str`` so a config knob typed
#: with it enumerates its options in every form spec / MCP schema / editor widget.
Device = Literal["auto", "cpu", "cuda", "mps"]

__all__ = ["Device", "resolve_device"]


def resolve_device(choice: Device = "auto") -> torch.device:
    """``"auto"`` -> the best device actually present; anything else taken at its word.

    Explicit rather than clever: an operator who wrote ``device: cpu`` to compare against a GPU
    run gets CPU, and one who wrote ``cuda`` on a machine without it gets torch's own error
    rather than a silent fallback that would make the benchmark meaningless.

    The ``"auto"`` order — CUDA, then MPS, then CPU — is the one a ``Trainer`` would have chosen,
    written out because a hand-written loop has no Trainer to choose for it.

    Args:
        choice: ``"auto"`` (default) or an explicit device name.

    Returns:
        The resolved :class:`torch.device`.
    """
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

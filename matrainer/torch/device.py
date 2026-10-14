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

import os
from typing import Literal, Optional

import torch
from loggair import get_logger

logger = get_logger(__name__)

#: Where a run executes. ``"auto"`` resolves to the best device actually present; the rest are
#: taken at their word. A closed ``Literal`` rather than a bare ``str`` so a config knob typed
#: with it enumerates its options in every form spec / MCP schema / editor widget.
Device = Literal["auto", "cpu", "cuda", "mps"]

#: The variables torch's Apple-GPU (MPS) allocator reads when the GPU is first used in a process.
LOW_WATERMARK_VARIABLE = "PYTORCH_MPS_LOW_WATERMARK_RATIO"
HIGH_WATERMARK_VARIABLE = "PYTORCH_MPS_HIGH_WATERMARK_RATIO"
#: torch's high watermark when its variable is unset — measured (torch 2.13, Apple M3 Pro): a low watermark of 1.69
#: is accepted, 1.71 refused with ``RuntimeError: invalid low watermark ratio 1.71``.
TORCH_DEFAULT_HIGH_WATERMARK = 1.7

__all__ = [
    "Device",
    "HIGH_WATERMARK_VARIABLE",
    "LOW_WATERMARK_VARIABLE",
    "TORCH_DEFAULT_HIGH_WATERMARK",
    "apply_mps_low_watermark",
    "resolve_device",
]


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


def apply_mps_low_watermark(value: Optional[float]) -> None:
    """Hand ``value`` to torch's Apple-GPU allocator as its LOW watermark — above that share of the GPU's recommended
    working set, torch gives back the memory it caches instead of keeping it.

    Why a training run wants it: by default torch keeps what it has cached, and on a Mac the GPU's memory IS the
    machine's. Measured 2026-10-14 (18 GB M3 Pro, an RF-DETR segmentation smoke run, batch 1): unset, the process held
    11 GB and swapped, its epoch took 8 min 42 s; with 0.5 it held 8.8-9.4 GB and the whole run took 2 min 33 s. A
    LIMIT is not the answer — torch's ``set_per_process_memory_fraction(0.6)`` stopped the same run with ``MPS backend
    out of memory``; the low watermark only changes when the cache is released.

    torch reads the variable once, when the GPU is first used in the process, so this must run before the model is
    built. It only sets the variable: on a run without an Apple GPU nothing reads it. A value already in the
    environment (the shell's) is used instead, and said so; a GPU this process already used is warned about, since
    the value can then no longer take effect.

    Args:
        value: The low watermark, ``0`` up to the high watermark (``PYTORCH_MPS_HIGH_WATERMARK_RATIO``, torch's
            default 1.7); ``None`` sets nothing.

    Raises:
        ValueError: A value outside that range — named here, because torch's own refusal
            (``invalid low watermark ratio 1.4``) names neither the setting nor the bound.
    """
    if value is None:
        return
    high_text = os.environ.get(HIGH_WATERMARK_VARIABLE)
    high = float(high_text) if high_text else TORCH_DEFAULT_HIGH_WATERMARK
    if not 0.0 <= value <= high:
        source = HIGH_WATERMARK_VARIABLE if high_text else "torch's default"
        raise ValueError(f"mps_low_watermark {value} must be between 0 and the high watermark {high:g} ({source})")
    current = os.environ.get(LOW_WATERMARK_VARIABLE)
    if current is not None:
        if float(current) != value:
            logger.info(
                f"{LOW_WATERMARK_VARIABLE}={current} from the environment is used; mps_low_watermark={value} is not"
            )
        return
    os.environ[LOW_WATERMARK_VARIABLE] = repr(value)
    # Asking the allocator creates it — with the value just set, so asking is safe; memory already held means it was
    # created before, with the old value.
    if torch.backends.mps.is_available() and torch.mps.current_allocated_memory() > 0:
        logger.warning(
            f"mps_low_watermark={value} cannot take effect: this process already used the GPU, and torch read "
            f"{LOW_WATERMARK_VARIABLE} then. Set it in the environment before the process starts."
        )
        return
    logger.debug(f"{LOW_WATERMARK_VARIABLE}={value} for torch's Apple-GPU allocator")

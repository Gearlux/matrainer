"""Load a model's weights from a saved checkpoint.

Three on-disk formats are tolerated by a single helper:

1. **PyTorch Lightning ``.ckpt``** — produced by ``trainer.save_checkpoint``.
   Top-level ``state_dict`` key whose entries are prefixed with ``model.``
   (Lightning's :class:`pytorch_lightning.LightningModule` wrapper adds it via
   ``self.model = model`` inside :class:`marainer.trainer.LightningTrainer` /
   :class:`marainer.evaluator.LightningEvaluator`). This helper strips that
   prefix so model classes can load their own weights directly in ``__init__``
   without dragging Marainer's Lightning wrapper into the picture.

2. **Ultralytics ``.pt``** — produced by ``ultralytics.YOLO.train`` (and
   ``raidar_ultralytics.ultralytics_trainer.UltralyticsTrainer``). The
   top-level ``model`` key holds a *live* ``DetectionModel`` instance (an
   ``nn.Module``); we pull its ``state_dict()`` and load that. Crucially, the
   ``model.<n>...`` prefix on each weight here belongs to the
   ``DetectionModel``'s own ``nn.Sequential``; it is **not** Lightning's
   wrapping prefix and must not be stripped.

3. **Plain state-dict** — ``torch.save(model.state_dict(), path)``. The
   loaded object IS the state dict.

After format detection, the helper auto-detects whether to retarget the load
to a wrapping module's named child. Raidar-ultralytics'
``raidar_ultralytics.yolo26.YOLO26Model`` exposes the ``DetectionModel`` under
``self.detector``; if the source keys match the detector's state_dict but not
the wrapping ``YOLO26Model``'s, the load is redirected to ``self.detector`` so
the same checkpoint can also drop in via ``YOLO26Model(checkpoint_path=...)``
without the caller hand-prefixing.

This helper lives in Marainer — the common upstream of every task package
(sonair / deltaid / raidar / raidar-ultralytics) — because it loads weights
saved by Marainer's own Lightning wrapper. Downstream model classes import it
across the dependency boundary (``from marainer.checkpoint import
load_lightning_checkpoint``); the Ultralytics branch needs no ``ultralytics``
import (it only ``isinstance``-checks ``nn.Module``), so Marainer stays free of
that dependency.
"""

import os
from typing import Any, Mapping

import torch
import torch.nn as nn
from logflow import get_logger

logger = get_logger(__name__)

_LIGHTNING_PREFIX = "model."


def load_lightning_checkpoint(model: nn.Module, path: str, strict: bool = False) -> None:
    """Load weights from a Lightning, Ultralytics, or plain state-dict checkpoint.

    See module docstring for the three accepted on-disk shapes.

    Args:
        model: The target ``nn.Module``. May be the model itself, or a
            wrapper whose ``named_children()`` includes a child whose
            state_dict keys match the source — the helper picks the right
            target automatically.
        path: Filesystem path to the checkpoint (``.ckpt`` or ``.pt``).
        strict: Forwarded to :meth:`torch.nn.Module.load_state_dict`.
            Defaults to ``False`` so e.g. a re-trained classifier head with
            a different ``num_classes`` doesn't reject the entire load.

    Raises:
        FileNotFoundError: ``path`` doesn't exist.
        RuntimeError: ``strict=True`` and any keys mismatch.
    """
    # Explicit existence check: torch.load delegates straight to open() and
    # surfaces whatever errno the OS returns. On Linux that's a PermissionError
    # when the user can't traverse a parent of a non-existent path (e.g.
    # `/nonexistent/...`), which obscures the real cause. os.path.exists()
    # (unlike Path.exists() in 3.12+) returns False on any os.stat() failure,
    # so it normalises both ENOENT and EACCES into the documented contract.
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    logger.info(f"{type(model).__name__}: loading checkpoint from {path}")
    raw: Any = torch.load(path, map_location="cpu", weights_only=False)

    # 1. Pull a state_dict out of `raw` regardless of on-disk shape.
    state_dict, ultralytics_format = _extract_state_dict(raw)

    # 2. Lightning wraps its module under `self.model` — strip that prefix
    #    so keys match the wrapped module's own state_dict. Skip for the
    #    Ultralytics shape, where `model.<n>...` is a real namespace level
    #    (the `DetectionModel.model` ModuleList).
    if not ultralytics_format:
        state_dict = {
            (k[len(_LIGHTNING_PREFIX) :] if k.startswith(_LIGHTNING_PREFIX) else k): v for k, v in state_dict.items()
        }

    # 3. Auto-retarget to a named child when the wrapper's state_dict doesn't
    #    overlap the source keys but a child's does. Generic via
    #    ``named_children()`` — no hardcoded ``detector`` literal.
    target, retargeted_to = _resolve_load_target(model, state_dict)
    missing, unexpected = target.load_state_dict(state_dict, strict=strict)

    where = f" → child '{retargeted_to}'" if retargeted_to else ""
    if missing:
        logger.info(f"{type(model).__name__}: checkpoint missing {len(missing)} key(s){where} (e.g. {missing[:3]})")
    if unexpected:
        logger.info(
            f"{type(model).__name__}: checkpoint had {len(unexpected)} unexpected key(s){where} "
            f"(e.g. {unexpected[:3]})"
        )


def _extract_state_dict(raw: Any) -> tuple[Mapping[str, Any], bool]:
    """Pull a flat ``{name: tensor}`` mapping out of any of the three shapes.

    Returns ``(state_dict, ultralytics_format)``. The flag tells the caller
    to skip the Lightning prefix-strip step — see module docstring.
    """
    if isinstance(raw, Mapping):
        model_obj = raw.get("model")
        if isinstance(model_obj, nn.Module):
            return dict(model_obj.state_dict()), True
        if "state_dict" in raw:
            return dict(raw["state_dict"]), False
    return raw, False


def _resolve_load_target(model: nn.Module, state_dict: Mapping[str, Any]) -> tuple[nn.Module, str]:
    """Pick the right module to load ``state_dict`` into.

    If the source keys overlap ``model``'s own state_dict, the model itself
    is the target. Otherwise the helper walks ``named_children()`` once and
    picks the first child whose state_dict overlaps. Returns
    ``(target_module, child_name_or_empty_string)`` so the caller can log
    the retarget.

    Falls back to ``model`` (no retarget) when no child matches — the
    subsequent ``load_state_dict`` then fails or no-ops normally, surfacing
    the real issue instead of silently routing into an unrelated child.
    """
    src_keys = set(state_dict.keys())
    if src_keys & set(model.state_dict().keys()):
        return model, ""
    for name, child in model.named_children():
        if src_keys & set(child.state_dict().keys()):
            return child, name
    return model, ""


__all__ = ["load_lightning_checkpoint"]

"""Runnable markers shared across marainer trainers / evaluators / processors."""


class TorchRunner:
    """Mixin marking a runnable whose ``run()`` performs gradient-based optimization.

    Pure marker — no state, no ``__init__`` — so it composes cleanly with
    ``pytorch_lightning.LightningModule`` (and any other base). The point is to let a GUI
    executor distinguish "this run needs autograd" from "this run is inference-only".

    Why it exists: FluxStudio's ``RunNode`` calls ``run()`` from inside ComfyUI's graph
    execution, which ComfyUI wraps in ``torch.inference_mode()`` for cheap, grad-free node
    evaluation. Under inference mode every tensor created — model parameters (model nodes are
    lazy, so the backbone is built *inside* ``run()``), forward activations, the loss — is an
    "inference tensor" with no autograd graph, so ``loss.backward()`` dies with
    ``"element 0 of tensors does not require grad and does not have a grad_fn"``. The executor
    reads the duck-typed ``__torch_runner__`` flag (no hard ``marainer`` import on the GUI side)
    and re-enables normal autograd for the duration of ``run()``.

    Trainers (the base :class:`~marainer.trainer.LightningTrainer` and its task subclasses) inherit
    this. Inference-only runnables (:class:`~marainer.evaluator.LightningEvaluator`,
    :class:`~marainer.processing.DatasetProcessor`) deliberately do NOT — they run as-is under the
    executor's inference mode.
    """

    __torch_runner__: bool = True

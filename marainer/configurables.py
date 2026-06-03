"""Register select third-party runtime classes as confluid configurables (tagged ``lazy``).

These are the runtime members a Lightning runnable defers and flows itself at ``run()`` — a
``torch`` optimizer (needs ``params=model.parameters()``), a ``DataLoader`` (needs ``dataset=…``),
and the ``pytorch_lightning.Trainer``. They are NOT workspace classes, so they have no
``@configurable`` decorator; we ``register`` them here with ``lazy=True`` so:

* confluid stamps ``__confluid_lazy__`` on the class (a deferred / runtime-injection slot), and
* FluxStudio generates an OBJECT node for each (``fluxstudio.bridge._build_object_node``) whose
  ``execute`` yields a deferred ``LazyClass`` the runnable flows with the missing runtime arg — so a
  user can wire a configured optimizer / loader / Lightning Trainer into a trainer node instead of
  relying solely on the trainer's baked defaults.

The classes are bound as module-level names AND listed in ``__all__`` so FluxStudio's discovery
second pass (``register_module_nodes`` — ``@configurable`` classes re-exported via ``__all__``)
surfaces them: ``scan_module`` alone would skip them, since their ``__module__`` is ``torch`` /
``pytorch_lightning``, not this module. Exposed to discovery via the ``confluid.configurables``
entry point so FluxStudio / navigaitor eagerly import it.
"""

from confluid import register
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger, TensorBoardLogger
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader

from marainer.evaluator import LightningEvaluator
from marainer.trainer import LightningTrainer

# NOTE: these are marked ``lazy`` but deliberately carry NO ``category`` — only a presentation-only
# ``group`` (FluxStudio nests them as ``Taidal/Marainer/Object/<group>``). FluxStudio detects them as
# object-member nodes via the ``lazy`` mark; withholding a ``category`` keeps them OUT of navigaitor's
# form-spec slot pickers (which resolve a picker FROM a slot's default-target category and would
# otherwise offer the raw ``params``-taking optimizer instead of navigaitor's curated infra config).

# Optimizers — ``params`` is the runtime-injected slot (the trainer flows
# ``flow(self.optimizer, params=self.parameters())`` in ``configure_optimizers``).
register(Adam, group="optimizer", lazy=True)
register(AdamW, group="optimizer", lazy=True)
register(SGD, group="optimizer", lazy=True)

# DataLoader — ``dataset`` is the runtime-injected slot (the trainer flows
# ``flow(self.train_loader, dataset=train_ds)`` in ``run()``).
register(DataLoader, group="loader", lazy=True)

# The underlying Lightning Trainer — the trainer's ``run()`` flows ``self.lightning`` after the
# auto-``ModelCheckpoint`` / logger wiring; no runtime arg, but kept lazy so its node defers (a live
# eager Trainer in a body slot would skip that pre-flow wiring).
register(Trainer, group="lightning", lazy=True)

# Lightning loggers — fed into the ``pytorch_lightning.Trainer.logger`` slot. Registered with
# ``role="logger"`` (NOT ``lazy``) so FluxStudio builds them as EAGER OBJECT nodes (``role="logger"``
# is in the bridge ``_MEMBER_ROLES``): a logger's required ``save_dir`` stays a configurable widget
# instead of being skipped as a runtime-injected slot, and the live ``instance`` output is a usable
# logger. They must stay ``!class:`` (eager) in YAML — the Lightning Trainer materializes the nested
# logger Fluids in its ``logger`` list when it flows; a deferred ``!lazy:`` logger would never build.
# The stock PyTorch-Lightning loggers are offered directly. No ``category`` (only the presentation
# ``group``) — same rationale as the runtime members.
for _logger_cls in (
    TensorBoardLogger,
    CSVLogger,
    MLFlowLogger,
):
    register(_logger_cls, role="logger", group="logger")

# Base runnables surfaced for the canvas. ``LightningTrainer`` / ``LightningEvaluator`` are first-party
# (already ``@configurable``) — they need no ``register`` call; they are re-exported here purely so
# FluxStudio's ``__all__``-gated discovery pass surfaces them as RUNNABLE nodes (``_is_runnable``),
# the same way the task trainers (sonair / deltaid / raidar) are. This makes a config rooted on the
# generic base trainer (e.g. ``waivefront-rfuav/config/train_efficientvit_b0.yaml``) importable into
# the canvas. Re-exported HERE (an import-only discovery module that already pulls Lightning) rather
# than in ``marainer.__init__`` so a plain ``import marainer`` stays free of the Lightning import cost.

# Re-exported names FluxStudio discovery surfaces (the second, ``__all__``-gated pass).
__all__ = [
    "Adam",
    "AdamW",
    "SGD",
    "DataLoader",
    "Trainer",
    "TensorBoardLogger",
    "CSVLogger",
    "MLFlowLogger",
    "LightningTrainer",
    "LightningEvaluator",
]

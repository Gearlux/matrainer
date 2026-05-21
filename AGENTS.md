# Marainer Mandates

- **Unchanged Step Functions:** `LightningTrainer.training_step` / `validation_step` follow the `preds = self(batch.input); loss = self.loss_fn(preds, batch.target)` pattern. Per-task adapters (raidar, sonair, deltaid) MUST conform to this shape on either side rather than overriding step functions. Subclassing the Trainer is a code smell.
- **Five-Verb Vocabulary:** The CLI exposes exactly five verbs — `train`, `evaluate`, `process` (escape hatch), `download`, `convert`. New verbs require user approval; resist verb proliferation. Use `_run_marainer_subcommand` for thin wrappers mirroring `train_model` / `evaluate_model` / `download_data` / `convert_data` in navigaitor.
- **Plugin Discovery via Entry Points:** Sibling projects register CLI verbs through `[project.entry-points."marainer.commands"]` and a `register(app)` callable. Don't fork the CLI; extend it.
- **Confluid-First Wiring:** Models, losses, datasets, loaders, callbacks, and trainers MUST be Confluid-`@configurable` so a single YAML drives the entire experiment. No code-only configuration.
- **DataFlux Lineage:** All training data flows through `dataflux.Sample(input, target, metadata)`. Never bypass metadata — full sample-to-prediction traceability is mandatory for sinks and downstream analysis.
- **Logger Auto-Wiring:** Lightning loggers are injected into `trainer.loggers` by confluid/liquifai. Trainer code MUST NOT forward loggers manually — verify before flagging as a bug.
- **Predictions Sinks:** `Evaluator.predictions_sink` consumes per-sample dicts. Detection/classification/segmentation models MUST emit a per-sample mapping (not a batched tensor) when used in predict mode.

import sys
from typing import Any

from liquifai import LiquifyApp
from logflow import get_logger

from marainer.evaluator import LightningEvaluator
from marainer.plugins import load_plugins
from marainer.trainer import LightningTrainer

logger = get_logger(__name__)

# Ensure CWD is in path for example/project resolution
if "." not in sys.path:
    sys.path.insert(0, ".")

# Main App Instance
app = LiquifyApp(name="marainer")


@app.script_command(flow_mode="auto")
def train(trainer: LightningTrainer) -> None:
    """Orchestrate a training task."""
    from confluid import dump

    print(dump(trainer))
    logger.info(f"Marainer embarking on training journey: {trainer.experiment_name}")

    if hasattr(trainer, "run"):
        trainer.run()
    else:
        logger.error("The injected trainer object does not implement 'run()'.")


@app.script_command(flow_mode="auto")
def evaluate(evaluator: LightningEvaluator) -> None:
    """Evaluate a trained model over a dataset.

    Loads a checkpoint into the injected :class:`~marainer.evaluator.LightningEvaluator`,
    runs one pass over its dataset, logs metrics, and (if configured) streams
    predictions to a sink.
    """
    if evaluator is None:
        logger.error("'evaluator' was not bound — provide one under 'evaluator:' in your YAML.")
        return
    logger.info(f"Marainer evaluating: {evaluator.experiment_name}")
    if hasattr(evaluator, "run"):
        evaluator.run()
    else:
        logger.error("The injected evaluator object does not implement 'run()'.")


def _dispatch_processor(processor: Any, verb_label: str) -> None:
    """Shared body for ``process`` / ``download`` / ``convert``.

    The three CLI verbs are semantic aliases — same Confluid DI under the
    ``processor:`` key, same ``.run()`` dispatch. ``verb_label`` only shapes
    the log message so the operator can tell which verb was invoked.
    """
    if processor is None:
        logger.error("'processor' was not bound — provide one under 'processor:' in your YAML.")
        return
    from confluid import flow
    from confluid.fluid import Fluid

    if isinstance(processor, Fluid):
        processor = flow(processor)

    label = processor.__class__.__name__
    logger.info(f"Marainer {verb_label}: {label}")
    if hasattr(processor, "run"):
        processor.run()
    else:
        logger.error("The injected processor object does not implement 'run()'.")


@app.script_command(flow_mode="auto")
def process(processor: Any) -> None:
    """Run any Confluid-instantiated processor object that exposes ``.run()``.

    Generic dispatcher (escape hatch). Prefer the more specific ``download``
    or ``convert`` verbs when the intent fits. The YAML config is expected
    to bind the object under the top-level ``processor:`` key.
    ``flow_mode="auto"`` deep-flows the processor under confluid's active
    context so nested ``!ref:`` markers (e.g. ``drone_filter: !ref:drone``)
    resolve against the loaded YAML's top-level keys rather than against an
    empty context.
    """
    _dispatch_processor(processor, "running processor")


@app.script_command(flow_mode="auto")
def download(processor: Any) -> None:
    """Run a data-acquisition processor (HTTP / CCXT / API → dataset on disk).

    Convention: YAML name ``download_*.yaml``, processor class name
    ``Download*``. Semantic alias of ``process`` — the YAML still binds
    under ``processor:`` and dispatch is identical, the verb only signals
    intent and shapes the log message.
    """
    _dispatch_processor(processor, "downloading via")


@app.script_command(flow_mode="auto")
def convert(processor: Any) -> None:
    """Run a format-conversion processor (transform existing data on disk).

    Convention: YAML name ``*_to_*.yaml``, processor class name ``*To*``.
    Semantic alias of ``process`` — the YAML still binds under
    ``processor:`` and dispatch is identical, the verb only signals intent
    and shapes the log message.
    """
    _dispatch_processor(processor, "converting via")


def main() -> None:
    load_plugins(app)
    app.run()


if __name__ == "__main__":
    main()

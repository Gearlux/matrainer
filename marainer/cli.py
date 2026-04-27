import sys
from typing import Any

from liquifai import LiquifyApp
from logflow import get_logger

from marainer.evaluator import Evaluator
from marainer.trainer import Trainer

logger = get_logger(__name__)

# Ensure CWD is in path for example/project resolution
if "." not in sys.path:
    sys.path.insert(0, ".")

# Main App Instance
app = LiquifyApp(name="marainer")


@app.script_command(flow_mode="auto")
def train(trainer: Trainer) -> None:
    """Orchestrate a training task."""
    from confluid import dump

    print(dump(trainer))
    logger.info(f"Marainer embarking on training journey: {trainer.experiment_name}")

    if hasattr(trainer, "run"):
        trainer.run()
    else:
        logger.error("The injected trainer object does not implement 'run()'.")


@app.script_command(flow_mode="auto")
def evaluate(evaluator: Evaluator) -> None:
    """Evaluate a trained model over a dataset.

    Loads a checkpoint into the injected :class:`~marainer.evaluator.Evaluator`,
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


@app.script_command()
def process(processor: Any) -> None:
    """Run any Confluid-instantiated processor object that exposes ``.run()``.

    The YAML config is expected to bind the object under the top-level
    ``processor:`` key. Liquify's DI resolves it from the loaded config —
    when the annotation is ``Any`` the value comes back as a deferred
    ``confluid.Fluid``, so we materialize the top-level here. Any deferred
    deps inside the processor (source / sink / ops) are the processor's
    responsibility to materialize on ``.run()``.
    """
    if processor is None:
        logger.error("'processor' was not bound — provide one under 'processor:' in your YAML.")
        return
    from confluid import flow
    from confluid.fluid import Fluid

    if isinstance(processor, Fluid):
        processor = flow(processor)

    label = processor.__class__.__name__
    logger.info(f"Marainer running processor: {label}")
    if hasattr(processor, "run"):
        processor.run()
    else:
        logger.error("The injected processor object does not implement 'run()'.")


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()

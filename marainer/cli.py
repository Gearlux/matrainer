import sys

from liquify import LiquifyApp
from logflow import get_logger

from marainer.trainer import Trainer

logger = get_logger(__name__)

# Ensure CWD is in path for example/project resolution
if "." not in sys.path:
    sys.path.insert(0, ".")

# Main App Instance
app = LiquifyApp(name="marainer")


@app.script_command()
def train(trainer: Trainer) -> None:
    """Orchestrate a training task."""
    from confluid import dump

    print(dump(trainer))
    logger.info(f"Marainer embarking on training journey: {trainer.experiment_name}")

    if hasattr(trainer, "run"):
        trainer.run()
    else:
        logger.error("The injected trainer object does not implement 'run()'.")


# @app.script_command()
# def process(task: Any, config_path: Path) -> None:
#     """Generic processing task orchestration."""
#     mlflow.set_experiment(getattr(task, "experiment_name", "marainer_processing"))
#     with mlflow.start_run(run_name=getattr(task, "run_name", "process")):
#         log_task_parameters(task)
#         if config_path.exists():
#             mlflow.log_artifact(str(config_path), "config")
#         if hasattr(task, "run"):
#             task.run()


def main() -> None:
    app.run()


if __name__ == "__main__":
    main()

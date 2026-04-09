import sys
from typing import Any

import mlflow
from liquify import LiquifyApp
from liquify.discovery import get_configurable_paths
from logflow import get_logger

from marainer.trainer import Trainer

logger = get_logger(__name__)

# Ensure CWD is in path for example/project resolution
if "." not in sys.path:
    sys.path.insert(0, ".")

# Main App Instance
app = LiquifyApp(name="marainer")


def log_task_parameters(task: Any) -> None:
    """Extracts and logs all configurable parameters of the task to MLflow."""
    # We use get_configurable_paths to extract parameters from the materialized task
    params = get_configurable_paths(task)
    clean_params = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
    mlflow.log_params(clean_params)


@app.script_command()
def train(trainer: Trainer) -> None:
    """
    Orchestrate a training task.
    'task' is an instance that implements 'run()'.
    """
    from confluid import dump

    print(dump(trainer))
    experiment_name = getattr(trainer, "experiment_name", "marainer_generic")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=getattr(trainer, "run_name", "train")):
        logger.info(f"Marainer embarking on training journey: {experiment_name}")

        # Log the task configuration parameters
        log_task_parameters(trainer)

        # Log the manifest for perfect traceability
        # if config_path.exists():
        #     mlflow.log_artifact(str(config_path), "config")

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

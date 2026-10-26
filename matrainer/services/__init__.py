"""The tracking servers as workspace services: the Aim UI (``aim_service``) and the ClearML server
(``clearml_service``).

Each is a class the workspace's service manager finds through an ``aisland.plugins`` entry point (``pyproject.toml``)
and runs as ``aisland aim …`` / ``aisland clearml …``. Neither imports the manager: it is handed a context (the
service's data folder, the port and options asked for, ``run``, ``read_env``/``write_env``, ``choose_command``,
``echo``, ``fail``). Nothing here is imported by ``import matrainer``.
"""

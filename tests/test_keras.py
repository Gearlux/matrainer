"""The Keras tracking helper — and the import ordering it must not get wrong.

Keras 3 reads ``KERAS_BACKEND`` at IMPORT time and defaults to ``tensorflow``, which
``matrainer[keras]`` deliberately does not install. A bare ``import keras`` here therefore died
with ``ModuleNotFoundError: No module named 'tensorflow'`` whenever this callback was the FIRST
keras import in a process — and ``require`` reported it as *"the 'keras' package is not
installed"*, advice that fixes nothing because keras WAS installed. The fix routes the import
through ``recordstream.keras``, which sets the variable to the first engine actually present.

The load-bearing test is :func:`test_the_callback_builds_with_no_KERAS_BACKEND_chosen`, which
runs the original repro in a SUBPROCESS with the variable unset — the only place the ordering is
observable, since by the time an in-process test runs, keras is already imported.
"""

import os
import subprocess
import sys
import textwrap

import pytest

from matrainer._imports import require

keras = pytest.importorskip("recordstream.keras").keras

_BUILD_WITH_NO_BACKEND_CHOSEN = textwrap.dedent(
    """
    from matrainer.keras import keras_tracking_callback

    callback = keras_tracking_callback([])
    import keras

    print(f"OK:{type(callback).__name__}:{keras.backend.backend()}")
    """
)


def test_the_callback_builds_with_no_KERAS_BACKEND_chosen() -> None:
    """The regression: this exact invocation used to raise a misleading ImportError.

    A fresh interpreter with `KERAS_BACKEND` REMOVED from the environment. Skipped if the machine
    has TensorFlow, where Keras's own default happens to work and this would pass for the wrong
    reason.
    """
    import importlib.util

    if importlib.util.find_spec("tensorflow") is not None:
        pytest.skip("TensorFlow present — Keras's own default works, so the ordering is unobservable")

    env = {k: v for k, v in os.environ.items() if k != "KERAS_BACKEND"}
    result = subprocess.run(
        [sys.executable, "-c", _BUILD_WITH_NO_BACKEND_CHOSEN],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, f"building the callback without KERAS_BACKEND failed:\n{result.stderr}"
    assert "OK:" in result.stdout, result.stdout
    assert ":tensorflow" not in result.stdout, "an engine this extra does not install was selected"


def test_a_missing_keras_still_names_keras_not_the_shim() -> None:
    """`via=` must not leak into the message: an operator missing the extra has to be told to
    install `keras`, not `recordstream.keras` — which is already installed, so chasing it is
    a dead end."""
    with pytest.raises(ImportError) as excinfo:
        require("keras", "keras", feature="keras_tracking_callback", via="recordstream.keras_does_not_exist")

    message = str(excinfo.value)
    assert "'keras'" in message and "matrainer[keras]" in message
    assert "recordstream" not in message


def test_via_returns_the_shim_module_so_its_handle_can_be_read() -> None:
    """The contract that makes `.keras` legal at the call site."""
    module = require("keras", "keras", via="recordstream.keras")

    assert module.__name__ == "recordstream.keras"
    assert module.keras is keras


def test_require_without_via_is_unchanged() -> None:
    """Every other caller (mlflow / wandb / lightning) must be untouched by the new parameter."""
    assert require("os", "keras").__name__ == "os"

    with pytest.raises(ImportError, match=r"matrainer\[mlflow\]"):
        require("no_such_module_xyz", "mlflow")


def test_the_callback_is_a_keras_callback() -> None:
    from matrainer.keras import keras_tracking_callback

    assert isinstance(keras_tracking_callback([]), keras.callbacks.Callback)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

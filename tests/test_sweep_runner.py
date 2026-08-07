"""The sweep DRIVER (`matrainer.sweep.spec` + `matrainer.sweep.runner`).

The report half keeps its own suite (`test_sweep.py`); what this file pins is the machinery
lifted from the per-project bash drivers — spec loading with loud validation, the three-mode
matrix, the default `matrainer train` cell command, per-row env scoping, failure isolation with
the report rendered BEFORE the non-zero exit, and the dry-run surface.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from matrainer.sweep import main as sweep_main
from matrainer.sweep.runner import Cell, build_matrix, cell_command, run_sweep
from matrainer.sweep.spec import load_spec

SPEC_YAML = """
config: all_backends.yaml
epochs: 5
frameworks:
  - name: lightning
    requires: [json]            # stdlib names, so the probe RESOLVES in any test venv
    models: [zoo, spec]
  - name: keras
    requires: [json, os]
    env: {KERAS_BACKEND: tensorflow}
    models: [spec]
  - name: ghost
    requires: [no_such_module_anywhere]
    models: [zoo]
  - plainname
models:
  - name: zoo
  - name: spec
    flags: [--model, spec]
trackers: [csv, mlflow]
tracker_setup:
  mlflow: ["true"]
"""


@pytest.fixture()
def spec_path(tmp_path: Path) -> Path:
    (tmp_path / "all_backends.yaml").write_text("runnable: !class:whatever\n")
    path = tmp_path / "sweep.yaml"
    path.write_text(SPEC_YAML)
    return path


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------
class TestTheSpec:
    def test_it_loads_and_resolves_the_config_against_its_own_directory(self, spec_path: Path) -> None:
        spec = load_spec(spec_path)
        assert spec.config == (spec_path.parent / "all_backends.yaml").resolve()
        assert spec.epochs == 5
        assert spec.trackers == ["csv", "mlflow"]

    def test_a_bare_string_framework_probes_its_own_name(self, spec_path: Path) -> None:
        row = load_spec(spec_path).framework("plainname")
        assert row.requires == ("plainname",)

    def test_an_unknown_top_level_key_is_named_not_ignored(self, tmp_path: Path) -> None:
        """A misspelt `modles:` would quietly halve the sweep and the report would look complete."""
        (tmp_path / "cfg.yaml").write_text("x: 1\n")
        bad = tmp_path / "sweep.yaml"
        bad.write_text("config: cfg.yaml\nframeworks: [a]\ntrackers: [csv]\nmodles: []\n")
        with pytest.raises(ValueError, match="modles"):
            load_spec(bad)

    def test_a_framework_naming_an_undeclared_variant_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "cfg.yaml").write_text("x: 1\n")
        bad = tmp_path / "sweep.yaml"
        bad.write_text(
            "config: cfg.yaml\ntrackers: [csv]\nmodels: [{name: zoo}]\n" "frameworks: [{name: a, models: [nope]}]\n"
        )
        with pytest.raises(ValueError, match="nope"):
            load_spec(bad)

    def test_a_missing_config_key_is_a_clear_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "sweep.yaml"
        bad.write_text("frameworks: [a]\ntrackers: [csv]\n")
        with pytest.raises(ValueError, match="config"):
            load_spec(bad)


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------
class TestTheMatrix:
    def test_backends_mode_walks_frameworks_times_models_and_skips_the_uninstalled(self, spec_path: Path) -> None:
        cells, skipped = build_matrix(load_spec(spec_path), "backends")
        assert [c.run_name for c in cells] == ["lightning_zoo", "lightning_spec", "keras_spec"]
        assert skipped == ["ghost (no no_such_module_anywhere)", "plainname (no plainname)"]

    def test_only_filters_the_framework_axis(self, spec_path: Path) -> None:
        cells, _ = build_matrix(load_spec(spec_path), "backends", only=["keras"])
        assert [c.run_name for c in cells] == ["keras_spec"]

    def test_trackers_mode_runs_one_framework_at_its_first_variant(self, spec_path: Path) -> None:
        """The dimension under study is where the record GOES, not what trains."""
        cells, _ = build_matrix(load_spec(spec_path), "trackers")
        assert [c.run_name for c in cells] == ["lightning_zoo"]

    def test_trackers_mode_honours_an_explicit_framework(self, spec_path: Path) -> None:
        cells, _ = build_matrix(load_spec(spec_path), "trackers", framework="keras")
        assert [c.run_name for c in cells] == ["keras_spec"]

    def test_a_spec_without_a_model_dimension_names_cells_by_backend_alone(self, tmp_path: Path) -> None:
        (tmp_path / "cfg.yaml").write_text("x: 1\n")
        path = tmp_path / "sweep.yaml"
        path.write_text("config: cfg.yaml\ntrackers: [csv]\nframeworks: [{name: torch, requires: [json]}]\n")
        cells, _ = build_matrix(load_spec(path), "backends")
        assert [c.run_name for c in cells] == ["torch"]

    def test_an_unknown_mode_is_refused(self, spec_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown sweep mode"):
            build_matrix(load_spec(spec_path), "everything")


# ---------------------------------------------------------------------------
# The cell command
# ---------------------------------------------------------------------------
class TestTheCellCommand:
    def test_the_default_command_is_the_generic_matrainer_train(self, spec_path: Path) -> None:
        """`python -m matrainer.console` rather than the bare script name, so the sweep works
        whether or not the console script is on PATH."""
        spec = load_spec(spec_path)
        cell = Cell(framework="lightning", model="spec", env={}, flags=("--model", "spec"))
        command = cell_command(spec, cell, "csv", "Exp_1", epochs=2)
        assert command == [
            sys.executable, "-m", "matrainer.console", "train", str(spec.config),
            "--framework", "lightning", "--tracker", "csv",
            "--experiment_name", "Exp_1", "--run_name", "lightning_spec",
            "--model", "spec", "--max_epochs", "2",
        ]  # fmt: skip

    def test_a_spec_command_overrides_the_prefix(self, tmp_path: Path) -> None:
        (tmp_path / "cfg.yaml").write_text("x: 1\n")
        path = tmp_path / "sweep.yaml"
        path.write_text(
            "config: cfg.yaml\ntrackers: [csv]\ncommand: [distilaide, train]\n"
            "frameworks: [{name: torch, requires: [json]}]\n"
        )
        spec = load_spec(path)
        command = cell_command(spec, Cell("torch", None, {}, ()), "csv", "E_1", epochs=None)
        assert command[:3] == ["distilaide", "train", str(spec.config)]
        assert "--max_epochs" not in command


# ---------------------------------------------------------------------------
# Running (cells stubbed with `python -c`)
# ---------------------------------------------------------------------------
def _stub_spec(tmp_path: Path, body: str, frameworks: str = "[{name: a, requires: [json]}]") -> Path:
    """A spec whose `command:` is a `python -c` stub, so a run exercises the real subprocess
    loop with no training framework anywhere near it."""
    (tmp_path / "cfg.yaml").write_text("x: 1\n")
    path = tmp_path / "sweep.yaml"
    path.write_text(
        f"config: cfg.yaml\ntrackers: [csv]\nframeworks: {frameworks}\n"
        f"command: [{sys.executable}, -c, '{body}', --]\n"
    )
    return path


class TestRunning:
    def test_failure_is_isolated_and_the_exit_code_says_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        """One backend dying must not abort the rest — a missing row is itself a result."""
        monkeypatch.chdir(tmp_path)
        path = _stub_spec(
            tmp_path,
            'import sys; sys.exit(1 if "--framework b" in " ".join(sys.argv) else 0)',
            frameworks="[{name: a, requires: [json]}, {name: b, requires: [json]}, {name: c, requires: [json]}]",
        )
        code = run_sweep(load_spec(path), "backends", save_dir=tmp_path / "logs")
        out = capsys.readouterr().out
        assert code == 1
        assert "failed: b" in out
        # The stub cells write nothing to the tracker, so the render is the empty-experiment
        # line — what matters is that it appears AFTER the failure summary (the report is the
        # deliverable, rendered before the non-zero exit).
        assert out.index("failed: b") < out.index("no runs recorded")
        assert (tmp_path / "sweep_logs").is_dir(), "per-cell logs land under sweep_logs/<experiment>/"

    def test_a_clean_sweep_exits_zero(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        path = _stub_spec(tmp_path, "pass")
        assert run_sweep(load_spec(path), "backends", save_dir=tmp_path / "logs") == 0

    def test_the_row_env_reaches_its_one_subprocess(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The engine pin stays scoped to the cell — never exported to the sweep process."""
        monkeypatch.chdir(tmp_path)
        probe = tmp_path / "probe.txt"
        path = _stub_spec(
            tmp_path,
            f'import os, pathlib; pathlib.Path(r"{probe}").write_text(os.environ.get("SWEEP_PIN", ""))',
            frameworks="[{name: a, requires: [json], env: {SWEEP_PIN: pinned}}]",
        )
        assert run_sweep(load_spec(path), "backends", save_dir=tmp_path / "logs") == 0
        assert probe.read_text() == "pinned"
        assert "SWEEP_PIN" not in __import__("os").environ

    def test_each_tracker_gets_its_own_experiment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "cfg.yaml").write_text("x: 1\n")
        path = tmp_path / "sweep.yaml"
        path.write_text(
            f"config: cfg.yaml\ntrackers: [csv, tensorboard]\n"
            f"frameworks: [{{name: a, requires: [json]}}]\ncommand: [{sys.executable}, -c, pass, --]\n"
        )
        assert run_sweep(load_spec(path), "trackers", save_dir=tmp_path / "logs") == 0
        out = capsys.readouterr().out
        assert out.count("experiment :") == 2
        assert "tracker    : csv" in out and "tracker    : tensorboard" in out

    def test_a_failing_tracker_setup_skips_that_tracker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "cfg.yaml").write_text("x: 1\n")
        path = tmp_path / "sweep.yaml"
        path.write_text(
            f"config: cfg.yaml\ntrackers: [csv]\nframeworks: [{{name: a, requires: [json]}}]\n"
            f'command: [{sys.executable}, -c, pass, --]\ntracker_setup: {{csv: ["false"]}}\n'
        )
        code = run_sweep(load_spec(path), "backends", save_dir=tmp_path / "logs")
        assert code == 1
        assert "tracker setup for 'csv' failed" in capsys.readouterr().err

    def test_dry_run_prints_the_commands_and_runs_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        monkeypatch.chdir(tmp_path)
        path = _stub_spec(tmp_path, "raise SystemExit(9)")  # would fail loudly if executed
        assert run_sweep(load_spec(path), "backends", dry_run=True, save_dir=tmp_path / "logs") == 0
        out = capsys.readouterr().out
        assert "-c" in out and "--framework a" in out.replace("'", "")
        assert not (tmp_path / "sweep_logs").exists()


# ---------------------------------------------------------------------------
# The dispatch (`matrainer sweep …` ≡ `python -m matrainer.sweep …`)
# ---------------------------------------------------------------------------
class TestTheDispatch:
    def test_a_sweep_mode_reaches_the_runner(self, spec_path: Path, capsys: pytest.CaptureFixture) -> None:
        assert sweep_main(["backends", str(spec_path), "--dry-run"]) == 0
        assert "matrainer.console" in capsys.readouterr().out

    def test_the_report_surface_is_untouched(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        """The pinned exit codes and commands of the original module main survive the package
        conversion — the whole point of re-exporting it verbatim."""
        assert sweep_main(["next-experiment", "csv", "Prefix", "--save-dir", str(tmp_path)]) == 0
        assert capsys.readouterr().out.strip() == "Prefix_1"
        assert sweep_main(["summarise", "csv", "x"]) == 2

    def test_a_bad_spec_is_a_named_error_not_a_traceback(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        missing = tmp_path / "nope.yaml"
        assert sweep_main(["backends", str(missing)]) == 2
        assert "sweep:" in capsys.readouterr().err

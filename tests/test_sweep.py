"""``matrainer.sweep`` — how a sweep is named, read back and rendered.

What is pinned here is the RENDERING and the NAMING, which are pure or filesystem-only. The MLflow
reader needs a live server and is exercised by running a real sweep; the CSV reader is covered here
against a written-out tree, because that is the tracker a sweep uses when it must not depend on one.

These cases were lifted from two task projects that each carried a byte-identical copy of this
module as an example script (an AST comparison with docstrings stripped showed no difference), and
they are matrainer's for the same reason the module is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from matrainer import sweep


def _rows() -> List[Dict[str, Any]]:
    """Two runs shaped like the real thing: one torch backend, one `transformers.Trainer`.

    The keys are what a real sweep recorded (`sweep_logs/csv/<experiment>/<run>/metrics.csv`) once
    its backends were reading from one vocabulary.
    """
    return [
        {
            "run": "huggingface_zoo",
            "status": "FINISHED",
            "seconds": 96.0,
            "metrics": {
                "train/loss": 0.93,
                "val/loss": 0.71,
                "val/psnr": 27.44,
                "epoch": 1.0,
                "train/runtime": 61.2,
                "train/samples_per_second": 4.18,
                "train/total_flos": 1.0e16,
                "val/runtime": 8.4,
            },
        },
        {
            "run": "lightning_spec",
            "status": "FINISHED",
            "seconds": 24.0,
            "metrics": {
                "train/loss": 0.81,
                # Lightning's per-BATCH series of the SAME quantity — a real result, and the one
                # column every other run would leave blank.
                "train/loss_step": 0.78,
                "val/loss": 0.77,
                "val/psnr": 20.38,
                "val/ssim": 0.62,
            },
        },
    ]


class TestFolding:
    """Which columns a comparison table hides by default, and why each one is hidden."""

    def test_bookkeeping_columns_are_folded_away_by_default(self) -> None:
        """A `transformers.Trainer` records its timings beside the loss; a comparison table is not
        where a reader wants them, and a dozen runs of mostly-blank columns hide the ones every
        backend fills."""
        header = sweep.render(_rows(), "exp", "csv").splitlines()[0]

        assert "val/psnr" in header and "val/loss" in header and "train/loss" in header
        assert "train/total_flos" not in header
        assert "train/samples_per_second" not in header

    def test_what_was_hidden_is_named_with_the_command_that_shows_it(self) -> None:
        """Never a silent trim: the footer says how many columns went, which, and how to get them."""
        table = sweep.render(_rows(), "Thu_Aug_6_2026_1", "csv")

        assert "columns hidden — bookkeeping" in table
        assert "train/total_flos" in table  # named in the footer even though its column is gone
        assert "python -m matrainer.sweep report csv Thu_Aug_6_2026_1 --all" in table

    def test_show_all_puts_every_recorded_value_back(self) -> None:
        table = sweep.render(_rows(), "exp", "csv", show_all=True)

        assert "train/total_flos" in table and "epoch" in table
        assert "hidden" not in table

    @pytest.mark.parametrize(
        "key,bookkeeping",
        [
            ("epoch", True),
            ("train/runtime", True),
            ("val/samples_per_second", True),
            ("train/total_flos", True),
            ("train/loss", False),
            ("val/psnr", False),
            ("val/ssim", False),
        ],
    )
    def test_the_rule_reads_the_BASE_name(self, key: str, bookkeeping: bool) -> None:
        """Split-agnostic: `train/runtime` and `val/runtime` are the same kind of column."""
        assert sweep.is_bookkeeping(key) is bookkeeping

    def test_a_within_epoch_series_is_folded_but_is_NOT_called_bookkeeping(self) -> None:
        """`train/loss_step` is a genuine result, so it gets its own reason and its own word.

        Lightning appends `_step` to a name it logs per batch as well as per epoch. That series is
        worth recording — nothing else shows the loss moving WITHIN an epoch, and it drives the
        progress bar — but a one-row-per-run table cannot show a finer axis, and only the backend
        whose engine emits the suffix fills it. Folding it under `BOOKKEEPING` would have been two
        fewer lines and would have made the footer call a metric something it is not.
        """
        assert sweep.is_within_epoch_series("train/loss_step")
        assert not sweep.is_bookkeeping("train/loss_step")
        assert sweep.is_folded("train/loss_step")

        table = sweep.render(_rows(), "exp", "csv")
        header = table.splitlines()[0]
        assert "train/loss" in header
        assert "train/loss_step" not in header
        assert "within-epoch series" in table and "train/loss_step" in table  # named in the footer

    def test_the_footer_names_only_the_reasons_that_apply(self) -> None:
        """A run with no timings should not be told its bookkeeping was hidden."""
        rows = [
            {"run": "a", "status": "FINISHED", "seconds": 1.0, "metrics": {"train/loss": 1.0, "train/loss_step": 1.1}}
        ]
        table = sweep.render(rows, "exp", "csv")

        assert "within-epoch series" in table
        assert "bookkeeping" not in table

    def test_a_bare_step_suffix_is_not_mistaken_for_a_series(self) -> None:
        """`step` is an AXIS key (already bookkeeping); the rule needs a base name in front of it."""
        assert not sweep.is_within_epoch_series("step")
        assert not sweep.is_within_epoch_series("train/_step")
        assert sweep.is_bookkeeping("step")

    def test_show_all_puts_the_within_epoch_series_back_too(self) -> None:
        assert "train/loss_step" in sweep.render(_rows(), "exp", "csv", show_all=True).splitlines()[0]


class TestRendering:
    """What the table shows, and in what order."""

    def test_a_metric_only_one_backend_computes_keeps_its_column(self) -> None:
        """The union, not a hand-picked list: `val/ssim` is the column the Keras family cannot fill."""
        assert "val/ssim" in sweep.render(_rows(), "exp", "csv")

    def test_validation_columns_come_first(self) -> None:
        """That is the column a sweep is read for — the training-split ones are context."""
        header = sweep.render(_rows(), "exp", "csv").splitlines()[0]

        assert header.index("val/psnr") < header.index("train/loss")

    def test_a_run_that_never_reported_is_still_a_row(self) -> None:
        """A backend that died mid-fit must appear with blanks rather than vanish.

        A sweep isolates failures on purpose — a missing row is itself a result — and a report that
        silently dropped the run would undo that, leaving rows that look like the whole matrix.
        """
        rows = [
            {"run": "mlx_spec", "status": "FAILED", "seconds": 3.0, "metrics": {}},
            {"run": "torch_spec", "status": "FINISHED", "seconds": 20.0, "metrics": {"val/psnr": 20.4}},
        ]
        table = sweep.render(rows, "exp", "mlflow")

        assert "mlx_spec" in table and "FAILED" in table
        assert "2 runs in mlflow" in table

    def test_an_empty_sweep_says_so_rather_than_printing_an_empty_table(self) -> None:
        assert "no runs recorded" in sweep.render([], "exp", "csv")


class TestUnreadableTrackers:
    """A tracker that is write-only from here says where to look instead of showing nothing."""

    def test_a_tracker_that_cannot_be_read_says_where_to_look_instead(self) -> None:
        """The reason `render` takes a `note` at all: an empty table from a write-only backend reads
        as "the sweep recorded nothing", which is a different and much worse claim."""
        rows, note = sweep.collect("wandb", "exp")

        assert rows == [] and note is not None
        assert sweep.render(rows, "exp", "wandb", note).strip() == note

    @pytest.mark.parametrize("tracker", ["tensorboard", "aim", "clearml", "wandb"])
    def test_every_unreadable_tracker_names_its_own_way_in(self, tracker: str) -> None:
        _, note = sweep.collect(tracker, "exp")

        assert note is not None and tracker in note

    def test_an_unknown_tracker_is_not_an_error(self, tmp_path: Path) -> None:
        """A project may wire a tracker this module has never heard of; that is not a failure here."""
        rows, note = sweep.collect("something_new", "exp", save_dir=tmp_path)

        assert rows == [] and note is not None and "own UI" in note

    def test_the_queryable_set_is_what_collect_actually_branches_on(self) -> None:
        """`QUERYABLE` is load-bearing rather than documentation: `collect` reads it."""
        for tracker in sweep.QUERYABLE:
            _, note = sweep.collect(tracker, "no_such_experiment", save_dir=Path("/nonexistent"))
            assert note is None, f"{tracker} is advertised as queryable but reported a note"


class TestNaming:
    """`<prefix>_<N>`, N counted from the TRACKER."""

    def test_the_first_sweep_of_a_day_is_1(self, tmp_path: Path) -> None:
        assert sweep.next_experiment_name("csv", "Thu_Aug_6_2026", save_dir=tmp_path) == "Thu_Aug_6_2026_1"

    def test_it_counts_past_what_the_tracker_already_holds(self, tmp_path: Path) -> None:
        for name in ("Thu_Aug_6_2026_1", "Thu_Aug_6_2026_2"):
            (tmp_path / "csv" / name).mkdir(parents=True)

        assert sweep.next_experiment_name("csv", "Thu_Aug_6_2026", save_dir=tmp_path) == "Thu_Aug_6_2026_3"

    def test_a_gap_does_not_reuse_a_number(self, tmp_path: Path) -> None:
        """One PAST THE HIGHEST, not the first free slot — a deleted run must not be overwritten."""
        (tmp_path / "csv" / "Thu_Aug_6_2026_5").mkdir(parents=True)

        assert sweep.next_experiment_name("csv", "Thu_Aug_6_2026", save_dir=tmp_path) == "Thu_Aug_6_2026_6"

    def test_another_day_is_counted_separately(self, tmp_path: Path) -> None:
        (tmp_path / "csv" / "Wed_Aug_5_2026_3").mkdir(parents=True)

        assert sweep.next_experiment_name("csv", "Thu_Aug_6_2026", save_dir=tmp_path) == "Thu_Aug_6_2026_1"

    def test_a_wiped_tracker_restarts_at_1(self, tmp_path: Path) -> None:
        """The whole reason the count comes from the TRACKER rather than a local counter file."""
        (tmp_path / "csv" / "Thu_Aug_6_2026_1").mkdir(parents=True)
        assert sweep.next_experiment_name("csv", "Thu_Aug_6_2026", save_dir=tmp_path) == "Thu_Aug_6_2026_2"

        (tmp_path / "csv" / "Thu_Aug_6_2026_1").rmdir()

        assert sweep.next_experiment_name("csv", "Thu_Aug_6_2026", save_dir=tmp_path) == "Thu_Aug_6_2026_1"

    def test_an_unqueryable_tracker_falls_back_to_a_local_counter(self, tmp_path: Path) -> None:
        """Per-checkout and never wrong about how many times the driver has run HERE."""
        first = sweep.next_experiment_name("wandb", "Thu_Aug_6_2026", save_dir=tmp_path)
        second = sweep.next_experiment_name("wandb", "Thu_Aug_6_2026", save_dir=tmp_path)

        assert (first, second) == ("Thu_Aug_6_2026_1", "Thu_Aug_6_2026_2")


class TestTheCsvReader:
    """The tracker a sweep falls back to when it must not depend on a server."""

    def _write(self, root: Path, run: str, rows: List[str]) -> None:
        run_dir = root / "csv" / "exp" / run
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.csv").write_text("\n".join(rows) + "\n")

    def test_it_reads_one_row_per_run_directory(self, tmp_path: Path) -> None:
        self._write(tmp_path, "torch_spec", ["step,val/psnr", "0,20.4"])
        self._write(tmp_path, "keras_spec", ["step,val/psnr", "0,20.38"])

        rows, note = sweep.collect("csv", "exp", save_dir=tmp_path)

        assert note is None
        assert [r["run"] for r in rows] == ["keras_spec", "torch_spec"]  # sorted
        assert rows[1]["metrics"]["val/psnr"] == 20.4

    def test_the_LAST_value_of_a_metric_wins(self, tmp_path: Path) -> None:
        """The CSV backend appends a row per `log_metrics` call; the report wants the final value,
        exactly like MLflow's `data.metrics`."""
        self._write(tmp_path, "torch_spec", ["step,val/psnr", "0,20.4", "1,21.9", "2,22.7"])

        rows, _ = sweep.collect("csv", "exp", save_dir=tmp_path)

        assert rows[0]["metrics"]["val/psnr"] == 22.7

    def test_a_blank_cell_does_not_overwrite_a_recorded_value(self, tmp_path: Path) -> None:
        """A tracker writes one column per metric and blanks the ones this call did not report."""
        self._write(tmp_path, "torch_spec", ["step,val/psnr,train/loss", "0,20.4,0.9", "1,,0.7"])

        rows, _ = sweep.collect("csv", "exp", save_dir=tmp_path)

        assert rows[0]["metrics"] == {"val/psnr": 20.4, "train/loss": 0.7}

    def test_the_step_column_is_an_axis_and_never_a_metric(self, tmp_path: Path) -> None:
        self._write(tmp_path, "torch_spec", ["step,val/psnr", "3,20.4"])

        rows, _ = sweep.collect("csv", "exp", save_dir=tmp_path)

        assert "step" not in rows[0]["metrics"]

    def test_a_run_directory_with_no_metrics_file_is_still_a_row(self, tmp_path: Path) -> None:
        """A backend that died before its first log must appear rather than vanish."""
        (tmp_path / "csv" / "exp" / "mlx_spec").mkdir(parents=True)

        rows, _ = sweep.collect("csv", "exp", save_dir=tmp_path)

        assert [(r["run"], r["metrics"]) for r in rows] == [("mlx_spec", {})]

    def test_an_experiment_that_was_never_run_reads_as_empty_rather_than_raising(self, tmp_path: Path) -> None:
        rows, note = sweep.collect("csv", "never_ran", save_dir=tmp_path)

        assert rows == [] and note is None


class TestTheModuleMain:
    """`python -m matrainer.sweep …` — matrainer ships no binary, so this is the entry point."""

    def test_report_prints_the_table(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        run_dir = tmp_path / "csv" / "exp" / "torch_spec"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.csv").write_text("step,val/psnr\n0,20.4\n")

        assert sweep.main(["report", "csv", "exp", "--save-dir", str(tmp_path)]) == 0

        assert "torch_spec" in capsys.readouterr().out

    def test_next_experiment_prints_only_the_name(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """The driver captures this into a shell variable, so anything else on stdout breaks it."""
        assert sweep.main(["next-experiment", "csv", "Thu_Aug_6_2026", "--save-dir", str(tmp_path)]) == 0

        assert capsys.readouterr().out.strip() == "Thu_Aug_6_2026_1"

    def test_json_gives_a_caller_the_rows_instead_of_the_table(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import json

        run_dir = tmp_path / "csv" / "exp" / "torch_spec"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.csv").write_text("step,val/psnr\n0,20.4\n")

        assert sweep.main(["json", "csv", "exp", "--save-dir", str(tmp_path)]) == 0

        assert json.loads(capsys.readouterr().out)[0]["run"] == "torch_spec"

    def test_an_unknown_command_fails_rather_than_doing_nothing(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert sweep.main(["summarise", "csv", "exp"]) == 2

        assert "unknown command" in capsys.readouterr().err

    def test_too_few_arguments_prints_the_usage(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert sweep.main(["report", "csv"]) == 2

        assert "python -m matrainer.sweep" in capsys.readouterr().out

    def test_a_flag_is_never_mistaken_for_a_positional(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """`--all` sits where an experiment name could, so the positional split must skip flags."""
        assert sweep.main(["next-experiment", "csv", "Thu_Aug_6_2026", "--all", "--save-dir", str(tmp_path)]) == 0

        assert capsys.readouterr().out.strip() == "Thu_Aug_6_2026_1"

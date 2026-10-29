"""Fault injection: bad files, bad dtypes, failing collaborators, size guard (#8), streaming (#9).

Every failure must surface as a clean one-line ``child-mortality: error: ...`` with exit code 2
and nothing on stdout; no traceback may escape ``main``.
"""

import io
import sys
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from child_mortality import cube
from child_mortality.cli import main
from child_mortality.cube import ratio_cube
from child_mortality.survey import join_women_households, load_households, load_women
from conftest import HOUSEHOLD_HEADER, WOMEN_HEADER, write_csv, write_indexed_csv
from mock_helpers import RunawayError, csv_text, observe, wide_table, write_text

RC_HEADER = ("a", "b", "num", "den")


def ratio_argv(path: Path, *extra: str) -> list[str]:
    return [
        "ratio-cube",
        str(path),
        "-d",
        "a",
        "-d",
        "b",
        "--numerator",
        "num",
        "--denominator",
        "den",
        *extra,
    ]


def assert_clean_failure(capsys: pytest.CaptureFixture[str], rc: int) -> str:
    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    lines = captured.err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("child-mortality: error: ")
    assert "Traceback" not in captured.err
    return lines[0]


# --- unreadable, missing and corrupt input ------------------------------------------------------


def test_missing_input_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert_clean_failure(capsys, main(ratio_argv(tmp_path / "absent.csv")))


def test_input_is_a_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert_clean_failure(capsys, main(ratio_argv(tmp_path)))


def test_zero_byte_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert_clean_failure(capsys, main(ratio_argv(write_text(tmp_path / "e.csv", ""))))


def test_non_utf8_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "bin.csv"
    path.write_bytes(b"a,b,num,den\n\xff\xfe,\x80,1,2\n")
    assert_clean_failure(capsys, main(ratio_argv(path)))


def test_unreadable_file_permissions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_text(tmp_path / "locked.csv", csv_text(RC_HEADER, [("x", "p", "1", "2")]))
    path.chmod(0)
    try:
        try:
            path.read_bytes()
        except PermissionError:
            pass
        else:  # running as root: permissions are not enforced
            pytest.skip("file is readable despite mode 000")
        assert_clean_failure(capsys, main(ratio_argv(path)))
    finally:
        path.chmod(0o600)


def test_header_only_csv_yields_an_empty_cube(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_text(tmp_path / "h.csv", csv_text(RC_HEADER, []))
    assert main(ratio_argv(path)) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == ["cut_size,cut,a,b,rows,numerator,denominator,ratio"]
    assert captured.err == ""


def test_ragged_rows_are_a_clean_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_text(tmp_path / "r.csv", "a,b,num,den\nx,p,1,2\nx,p,1,2,3,4,5\n")
    assert_clean_failure(capsys, main(ratio_argv(path)))


@pytest.mark.parametrize("bad", ["abc", "1,5", "0x10", "1e", "--3"])
def test_non_numeric_measure_strings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], bad: str
) -> None:
    path = write_text(tmp_path / "n.csv", csv_text(RC_HEADER, [("x", "p", bad, "2")]))
    assert_clean_failure(capsys, main(ratio_argv(path)))


def test_non_numeric_survey_count(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    women = write_indexed_csv(
        tmp_path / "w.csv", WOMEN_HEADER, [("1", "1", "20-24", "two", "0", "0")]
    )
    houses = write_indexed_csv(tmp_path / "h.csv", HOUSEHOLD_HEADER, [("1", "1", "Urban", "40")])
    argv = [
        "cd-ceb",
        "--women",
        str(women),
        "--households",
        str(houses),
        "-d",
        "Age in 5-year groups",
    ]
    assert_clean_failure(capsys, main(argv))


def test_survey_file_missing_a_required_column(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    women = write_csv(tmp_path / "w.csv", ("Cluster number", "Household number"), [("1", "1")])
    houses = write_indexed_csv(tmp_path / "h.csv", HOUSEHOLD_HEADER, [("1", "1", "Urban", "40")])
    argv = [
        "cd-ceb",
        "--women",
        str(women),
        "--households",
        str(houses),
        "-d",
        "Age in 5-year groups",
    ]
    assert_clean_failure(capsys, main(argv))


def test_mismatched_join_key_dtypes_do_not_silently_match() -> None:
    women = pl.LazyFrame(
        {
            "Cluster number": [1],
            "Household number": [1],
            "Total children ever born": [2],
            "Sons who have died": [0],
            "Daughters who have died": [0],
        }
    )
    houses = pl.LazyFrame({"Cluster number": ["1"], "Household number": ["1"], "x": ["u"]})
    with pytest.raises((ValueError, pl.exceptions.PolarsError)):
        join_women_households(women, houses).collect()


def test_survey_loaders_agree_on_key_dtype(tmp_path: Path) -> None:
    """Both loaders read keys as String, so a real pair of files can never mismatch."""
    women = write_indexed_csv(
        tmp_path / "w.csv", WOMEN_HEADER, [("1", "1", "20-24", "2", "0", "0")]
    )
    houses = write_indexed_csv(tmp_path / "h.csv", HOUSEHOLD_HEADER, [("1", "1", "Urban", "40")])
    for key in ("Cluster number", "Household number"):
        assert load_women(women).collect_schema()[key] == pl.String
        assert load_households(houses).collect_schema()[key] == pl.String


# --- failing collaborators ----------------------------------------------------------------------


def good_csv(tmp_path: Path) -> Path:
    rows = [("x", "p", "1", "2"), ("y", "q", "3", "4")]
    return write_text(tmp_path / "ok.csv", csv_text(RC_HEADER, rows))


@pytest.mark.parametrize(
    "error", [OSError("No space left on device"), pl.exceptions.ComputeError("boom")]
)
def test_writer_raising_is_a_clean_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    def failing_write(self: pl.DataFrame, *args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(pl.DataFrame, "write_csv", failing_write)
    out = tmp_path / "out.csv"
    line = assert_clean_failure(capsys, main([*ratio_argv(good_csv(tmp_path)), "-o", str(out)]))
    assert str(error) in line


def test_writer_raising_on_stdout_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_write(self: pl.DataFrame, *args: Any, **kwargs: Any) -> Any:
        raise OSError("closed")

    monkeypatch.setattr(pl.DataFrame, "write_csv", failing_write)
    assert_clean_failure(capsys, main(ratio_argv(good_csv(tmp_path))))


def test_polars_raising_inside_group_by_is_a_clean_cli_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_group_by(self: pl.DataFrame, *args: Any, **kwargs: Any) -> Any:
        raise pl.exceptions.ComputeError("group_by exploded")

    monkeypatch.setattr(pl.DataFrame, "group_by", failing_group_by)
    line = assert_clean_failure(capsys, main(ratio_argv(good_csv(tmp_path))))
    assert "group_by exploded" in line


def test_polars_error_inside_group_by_propagates_from_the_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_group_by(self: pl.DataFrame, *args: Any, **kwargs: Any) -> Any:
        raise pl.exceptions.ComputeError("group_by exploded")

    monkeypatch.setattr(pl.DataFrame, "group_by", failing_group_by)
    with pytest.raises(pl.exceptions.PolarsError, match="exploded"):
        ratio_cube(wide_table(8, ["a", "b"], 0), ["a", "b"], "num", "den")


def test_failure_mid_cube_returns_no_partial_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = pl.DataFrame.group_by

    def flaky(self: pl.DataFrame, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise pl.exceptions.ComputeError("third roll-up fails")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "group_by", flaky)
    result: list[pl.DataFrame] = []
    with pytest.raises(pl.exceptions.ComputeError):
        result.append(ratio_cube(wide_table(16, ["a", "b", "c"], 0), ["a", "b", "c"], "num", "den"))
    assert result == []


def test_stdout_failure_after_compute_is_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    class ClosedPipe(io.RawIOBase):
        def writable(self) -> bool:
            return True

        def write(self, data: Any, /) -> int:
            raise OSError("pipe closed")

    class Broken:
        """A stdout whose text and binary writers both fail."""

        buffer = io.BufferedWriter(ClosedPipe(), buffer_size=1)

        def write(self, text: str, /) -> int:
            raise OSError("pipe closed")

        def flush(self) -> None:
            return None

    monkeypatch.setattr(sys, "stdout", Broken())
    rc = main(ratio_argv(good_csv(tmp_path)))
    monkeypatch.undo()
    assert rc == 2
    assert capsys.readouterr().err.startswith("child-mortality: error: ")


# --- wide dimension lists and the size guard (#8) -----------------------------------------------

WIDE_DIMS = [f"d{k:02d}" for k in range(20)]


def wide_csv(tmp_path: Path, n_dims: int = 20) -> Path:
    header = [*WIDE_DIMS[:n_dims], "num", "den"]
    return write_text(tmp_path / "wide.csv", csv_text(header, [["x"] * n_dims + ["1", "2"]]))


def wide_argv(path: Path, n_dims: int, *extra: str) -> list[str]:
    flags = [arg for d in WIDE_DIMS[:n_dims] for arg in ("-d", d)]
    return ["ratio-cube", str(path), *flags, "--numerator", "num", "--denominator", "den", *extra]


@pytest.mark.xfail(strict=True, reason="#8: 20 dims at default max_dims must be refused up front")
def test_library_refuses_a_million_cuts_by_default() -> None:
    frame = wide_table(4, [], 0).with_columns(*[pl.lit("x").alias(d) for d in WIDE_DIMS])
    with observe(max_rollups=500) as work, pytest.raises(ValueError, match=r"max_dims|min_rows"):
        ratio_cube(frame, WIDE_DIMS, "num", "den")
    assert work.rollup_calls == 0


@pytest.mark.xfail(strict=True, reason="#8: ratio-cube with 20 dims must exit 2 quickly")
def test_cli_refuses_a_million_cuts_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with observe(max_rollups=500):
        rc = main(wide_argv(wide_csv(tmp_path), 20))
    line = assert_clean_failure(capsys, rc)
    assert "--max-dims" in line


@pytest.mark.xfail(strict=True, reason="#8: --max-cuts raises or lowers the limit")
def test_cli_max_cuts_controls_the_limit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = wide_csv(tmp_path, 4)
    assert_clean_failure(capsys, main(wide_argv(path, 4, "--max-cuts", "5")))
    assert main(wide_argv(path, 4, "--max-cuts", "15")) == 0
    assert len(capsys.readouterr().out.splitlines()) == 1 + 15


def test_wide_dimension_list_is_fine_when_max_dims_bounds_the_cuts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(wide_argv(wide_csv(tmp_path), 20, "--max-dims", "2")) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 + 20 + 190  # header, 20 singles, C(20, 2) pairs


@pytest.mark.xfail(strict=True, reason="#8: -v prints one progress line per level to stderr")
def test_verbose_prints_one_progress_line_per_level(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = wide_csv(tmp_path, 4)
    assert main(wide_argv(path, 4, "-v")) == 0
    captured = capsys.readouterr()
    assert len(captured.err.strip().splitlines()) == 4
    assert captured.out.splitlines()[0].startswith("cut_size,cut")


# --- streaming output (#9) ----------------------------------------------------------------------


class ByteSink(io.TextIOWrapper):
    """A stdout stand-in that records whether text or bytes were written."""

    def __init__(self) -> None:
        self.raw = io.BytesIO()
        super().__init__(self.raw, encoding="utf-8", newline="")
        self.text_writes = 0

    def write(self, text: str, /) -> int:
        self.text_writes += 1
        return super().write(text)


@pytest.mark.xfail(strict=True, reason="#9: CSV must go to stdout as bytes, not one giant str")
def test_stdout_csv_is_written_as_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sink = ByteSink()
    monkeypatch.setattr(sys, "stdout", sink)
    rc = main(ratio_argv(good_csv(tmp_path)))
    monkeypatch.undo()
    sink.flush()
    assert rc == 0
    assert sink.text_writes == 0
    assert sink.raw.getvalue().startswith(b"cut_size,cut,a,b,")


@pytest.mark.xfail(strict=True, reason="#9: iter_ratio_cube generator API not implemented")
def test_iter_ratio_cube_matches_ratio_cube() -> None:
    frame = wide_table(32, ["a", "b", "c"], 0)
    dims = ["a", "b", "c"]
    blocks = list(cube.iter_ratio_cube(frame, dims, "num", "den"))  # type: ignore[attr-defined]
    assert pl.concat(blocks).equals(ratio_cube(frame, dims, "num", "den"))
    assert [b["cut"][0] for b in blocks] == ["a", "b", "c", "a|b", "a|c", "b|c", "a|b|c"]


@pytest.mark.xfail(strict=True, reason="#9: iter_ratio_cube must be lazy (work happens per block)")
def test_iter_ratio_cube_is_lazy() -> None:
    frame = wide_table(64, ["a", "b", "c", "d"], 0)
    dims = ["a", "b", "c", "d"]
    with observe() as full:
        ratio_cube(frame, dims, "num", "den")
    with observe() as first:
        iterator = iter(cube.iter_ratio_cube(frame, dims, "num", "den"))  # type: ignore[attr-defined]
        next(iterator)
    assert first.rollup_calls < full.rollup_calls


def test_tripwire_helper_aborts_runaway_work() -> None:
    with observe(max_rollups=2), pytest.raises(RunawayError):
        ratio_cube(wide_table(16, ["a", "b", "c"], 0), ["a", "b", "c"], "num", "den")

"""Fault injection: bad files, bad dtypes, failing collaborators, size guard (#8), streaming (#9).

Every failure must surface as a clean one-line ``child-mortality: error: ...`` with exit code 2
and nothing on stdout; no traceback may escape ``main``.
"""

import hashlib
import io
import random
import re
import sys
import tracemalloc
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from child_mortality import cube
from child_mortality.cli import main, read_generic
from child_mortality.cube import ratio_cube
from child_mortality.survey import join_women_households, load_households, load_women
from conftest import HOUSEHOLD_HEADER, WOMEN_HEADER, write_csv, write_indexed_csv
from mock_helpers import (
    RunawayError,
    assert_rollups_observed,
    csv_text,
    observe,
    run_cli,
    run_with_flag,
    wide_table,
    write_text,
)

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
        else:  # root, or Windows (mode 000 only sets the read-only flag): nothing to test
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


def fail_every_group_by(monkeypatch: pytest.MonkeyPatch) -> None:
    """Eager and lazy alike, so the fault fires whichever one the implementation uses."""

    def failing_group_by(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise pl.exceptions.ComputeError("group_by exploded")

    monkeypatch.setattr(pl.DataFrame, "group_by", failing_group_by)
    monkeypatch.setattr(pl.LazyFrame, "group_by", failing_group_by)


def test_polars_raising_inside_group_by_is_a_clean_cli_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fail_every_group_by(monkeypatch)
    line = assert_clean_failure(capsys, main(ratio_argv(good_csv(tmp_path))))
    assert "group_by exploded" in line


def test_polars_error_inside_group_by_propagates_from_the_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = wide_table(8, ["a", "b"], 0)
    fail_every_group_by(monkeypatch)
    with pytest.raises(pl.exceptions.PolarsError, match="exploded"):
        ratio_cube(table, ["a", "b"], "num", "den")


def test_failure_mid_cube_propagates_instead_of_returning_part_of_the_cube(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    originals = {cls: cls.group_by for cls in (pl.DataFrame, pl.LazyFrame)}

    def flaky(cls: Any) -> Any:
        def group_by(self: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            if calls == 3:
                raise pl.exceptions.ComputeError("third group_by fails")
            return originals[cls](self, *args, **kwargs)

        return group_by

    table = wide_table(16, ["a", "b", "c"], 0)
    for cls in originals:
        monkeypatch.setattr(cls, "group_by", flaky(cls))
    with pytest.raises(pl.exceptions.ComputeError, match="third"):
        ratio_cube(table, ["a", "b", "c"], "num", "den")


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
def test_cli_refuses_a_million_cuts_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 2 with an actionable message before any roll-up (the issue: "in under 1 s").

    The tripwire stops today's code after 500 of its 1,048,575 roll-ups.
    """
    with observe(max_rollups=500) as work:
        rc = main(wide_argv(wide_csv(tmp_path), 20))
    line = assert_clean_failure(capsys, rc)
    assert re.search(r"--max-dims|--min-rows|--max-cuts", line), line
    assert work.rollup_calls == 0


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
def test_verbose_prints_progress_per_level_and_leaves_stdout_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = wide_argv(wide_csv(tmp_path, 4), 4)
    quiet = run_cli(argv, capsys)
    verbose = run_with_flag(argv, ["-v"], capsys)
    assert (quiet.rc, verbose.rc) == (0, 0)
    assert verbose.out == quiet.out
    # at least one line per level (4 levels); other diagnostics (#7's summary) may add more
    assert len(verbose.err.strip().splitlines()) >= 4


# --- streaming output (#9) ----------------------------------------------------------------------


class DigestSink(io.RawIOBase):
    """A binary stdout stand-in that hashes and counts what it receives, but keeps none of it."""

    def __init__(self) -> None:
        super().__init__()
        self.size = 0
        self.sha = hashlib.sha256()

    def writable(self) -> bool:
        return True

    def write(self, data: Any, /) -> int:
        chunk = bytes(data)
        self.size += len(chunk)
        self.sha.update(chunk)
        return len(chunk)


def many_cuts_csv(tmp_path: Path) -> tuple[Path, list[str]]:
    """30 dims of 10 values: at ``--max-dims 2`` about 44k cells in 465 cuts of at most 100."""
    rng = random.Random(9)
    dims = [f"d{k:02d}" for k in range(30)]
    rows = [[str(rng.randrange(10)) for _ in dims] + ["1", "2"] for _ in range(3000)]
    return write_text(tmp_path / "many.csv", csv_text([*dims, "num", "den"], rows)), dims


@pytest.mark.xfail(strict=True, reason="#9: stdout gets the whole CSV built as one Python str")
def test_stdout_csv_is_streamed_and_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Peak Python-heap allocation while writing to stdout stays under a quarter of the CSV.

    Stands in for "stdout peak RSS at most 1.2x the ``-o out.csv`` run": a file is written by
    polars without Python allocations, so stdout may add only a small fraction of the output.
    Today the str plus its encoded copy peak at about 2x the CSV size. Writing bytes in chunks
    (``write_csv(sys.stdout.buffer)``) or one block per cut both pass. The bytes must equal
    today's CSV exactly (no CRLF translation).
    """
    path, dims = many_cuts_csv(tmp_path)
    expected = ratio_cube(read_generic(path, "num", "den"), dims, "num", "den", max_dims=2)
    expected_bytes = expected.write_csv().encode()
    argv = ["ratio-cube", str(path), *[a for d in dims for a in ("-d", d)]]
    argv += ["--numerator", "num", "--denominator", "den", "--max-dims", "2"]

    sink = DigestSink()
    stdout = io.TextIOWrapper(io.BufferedWriter(sink), encoding="utf-8", newline="")
    monkeypatch.setattr(sys, "stdout", stdout)
    tracemalloc.start()
    try:
        rc = main(argv)
        stdout.flush()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        monkeypatch.undo()
    assert rc == 0
    assert (sink.size, sink.sha.hexdigest()) == (
        len(expected_bytes),
        hashlib.sha256(expected_bytes).hexdigest(),
    )
    assert peak < len(expected_bytes) / 4, (peak, len(expected_bytes))


@pytest.mark.xfail(strict=True, reason="#9: iter_ratio_cube generator API not implemented")
def test_iter_ratio_cube_yields_one_block_per_cut_in_output_order() -> None:
    frame = wide_table(32, ["a", "b", "c"], 0)
    dims = ["a", "b", "c"]
    blocks = list(cube.iter_ratio_cube(frame, dims, "num", "den"))  # type: ignore[attr-defined]
    expected = ratio_cube(frame, dims, "num", "den")
    assert pl.concat(blocks).equals(expected)
    assert all(block["cut"].n_unique() == 1 for block in blocks)
    assert [b["cut"][0] for b in blocks] == expected["cut"].unique(maintain_order=True).to_list()


@pytest.mark.xfail(strict=True, reason="#9: iter_ratio_cube must be lazy (work happens per block)")
def test_iter_ratio_cube_is_lazy() -> None:
    frame = wide_table(64, ["a", "b", "c", "d"], 0)
    dims = ["a", "b", "c", "d"]
    with observe() as full:
        whole = ratio_cube(frame, dims, "num", "den")
    assert_rollups_observed(full, whole)
    with observe() as first:
        iterator = iter(cube.iter_ratio_cube(frame, dims, "num", "den"))  # type: ignore[attr-defined]
        next(iterator)
    assert first.rollup_calls < full.rollup_calls


def test_tripwire_helper_aborts_runaway_work() -> None:
    table = wide_table(16, ["a", "b", "c"], 0)
    with observe(max_rollups=2), pytest.raises(RunawayError):
        ratio_cube(table, ["a", "b", "c"], "num", "den")


# --- new defects, to be filed (see the orchestrator's new-issues list) --------------------------


@pytest.mark.xfail(strict=True, reason="NEW: a repeated CSV header silently uses the first column")
def test_repeated_header_in_generic_csv_is_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_text(tmp_path / "dup.csv", "a,a,num,den\nx,p,1,2\ny,q,3,4\n")
    argv = ["ratio-cube", str(path), "-d", "a", "--numerator", "num", "--denominator", "den"]
    line = assert_clean_failure(capsys, main(argv))
    assert re.search(r"\ba\b", line.removeprefix("child-mortality: error: ")), line


@pytest.mark.xfail(
    strict=True, reason="NEW: a repeated survey header silently uses the first column"
)
def test_repeated_header_in_survey_file_is_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    born = "Total children ever born"
    women = write_indexed_csv(
        tmp_path / "w.csv", (*WOMEN_HEADER, born), [("1", "1", "20-24", "2", "0", "1", "9")]
    )
    houses = write_indexed_csv(tmp_path / "h.csv", HOUSEHOLD_HEADER, [("1", "1", "Urban", "40")])
    argv = ["cd-ceb", "--women", str(women), "--households", str(houses)]
    line = assert_clean_failure(capsys, main([*argv, "-d", "Age in 5-year groups"]))
    assert born in line


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_dims", 1.0),
        ("max_dims", True),
        ("max_dims", "2"),
        ("min_rows", 1.0),
        ("min_rows", True),
        ("min_rows", "2"),
        ("min_rows", None),
    ],
)
@pytest.mark.xfail(strict=True, reason="NEW: non-integer max_dims/min_rows are not a ValueError")
def test_non_integer_max_dims_and_min_rows_are_rejected(name: str, value: Any) -> None:
    table = wide_table(8, ["a", "b"], 0)
    kwargs: dict[str, Any] = {name: value}
    with pytest.raises(ValueError, match=name):
        ratio_cube(table, ["a", "b"], "num", "den", **kwargs)

"""Tests for ``child_mortality.cli`` and the package entry points."""

import errno
import importlib
import importlib.metadata
import runpy
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

import child_mortality
from child_mortality import __version__, cd_ceb_cube, ratio_cube
from child_mortality.cli import _detach_stdout, build_parser, fail, main, read_generic, write_cube
from conftest import write_csv

DIM = "Type of place of residence"


@pytest.fixture
def generic_csv(tmp_path: Path) -> Path:
    rows = [
        ("x", "p", "1", "2"),
        ("x", "q", "2", "0"),
        ("y", "p", "3", "6"),
        ("NA", "p", "4", "8"),
        ("y", "q", "", "NA"),
    ]
    return write_csv(tmp_path / "in.csv", ("a", "b", "num", "den"), rows)


def run_ok(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    assert main(argv) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    return captured.out


def assert_one_line_error(capsys: pytest.CaptureFixture[str]) -> str:
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("child-mortality: error: ")
    assert captured.err.count("\n") == 1
    assert "Traceback" not in captured.err
    return captured.err


def expected_generic(generic_csv: Path) -> pl.DataFrame:
    frame = pl.read_csv(generic_csv, infer_schema=False, null_values=["NA", ""])
    frame = frame.with_columns(pl.col("num", "den").cast(pl.Float64))
    return ratio_cube(frame, ["a", "b"], "num", "den")


# --- success paths -------------------------------------------------------------------------


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out == f"child-mortality {__version__}\n"


def test_ratio_cube_to_stdout(capsys: pytest.CaptureFixture[str], generic_csv: Path) -> None:
    out = run_ok(
        capsys, "ratio-cube", str(generic_csv), "-d", "a", "-d", "b", "--numerator", "num",
        "--denominator", "den",
    )  # fmt: skip
    expected = expected_generic(generic_csv)
    assert out == expected.write_csv()
    assert out.splitlines()[0] == "cut_size,cut,a,b,rows,numerator,denominator,ratio"
    # NA / empty cells became nulls, so the "y|q" row contributes 0 to both measures.
    assert pl.read_csv(out.encode(), schema=expected.schema).equals(expected)


def test_cd_ceb_to_stdout(
    capsys: pytest.CaptureFixture[str], women_csv: Path, households_csv: Path
) -> None:
    out = run_ok(
        capsys, "cd-ceb", "--women", str(women_csv), "--households", str(households_csv),
        "-d", DIM,
    )  # fmt: skip
    expected = cd_ceb_cube(women_csv, households_csv, dimensions=[DIM])
    assert out == expected.write_csv()


def test_repeatable_dimension_flag_keeps_order(
    capsys: pytest.CaptureFixture[str], women_csv: Path, households_csv: Path
) -> None:
    out = run_ok(
        capsys, "cd-ceb", "--women", str(women_csv), "--households", str(households_csv),
        "-d", DIM, "--dimension", "Age in 5-year groups", "--max-dims", "2", "--min-rows", "1",
    )  # fmt: skip
    expected = cd_ceb_cube(
        women_csv, households_csv, dimensions=[DIM, "Age in 5-year groups"], max_dims=2
    )
    assert out == expected.write_csv()
    assert out.splitlines()[0].startswith(f"cut_size,cut,{DIM},Age in 5-year groups,rows")


def test_dimension_names_may_contain_commas(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = write_csv(tmp_path / "c.csv", ("a, b", "n", "d"), [("x", "1", "2"), ("y", "3", "4")])
    out = run_ok(
        capsys, "ratio-cube", str(path), "-d", "a, b", "--numerator", "n", "--denominator", "d"
    )
    assert out.splitlines()[1] == '1,"a, b",x,1,1.0,2.0,0.5'


def test_cd_ceb_defaults_to_survey_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake(*args: object, **kwargs: object) -> pl.DataFrame:
        seen.update(kwargs)
        return pl.DataFrame({"x": [1]})

    monkeypatch.setattr("child_mortality.cli.cd_ceb_cube", fake)
    assert main(["cd-ceb", "--women", "w.csv", "--households", "h.csv"]) == 0
    from child_mortality.survey import DIMENSIONS

    assert seen == {"dimensions": DIMENSIONS, "max_dims": None, "min_rows": 1}


def test_min_rows_and_max_dims_options(
    capsys: pytest.CaptureFixture[str], generic_csv: Path
) -> None:
    out = run_ok(
        capsys, "ratio-cube", str(generic_csv), "-d", "a", "-d", "b", "--numerator", "num",
        "--denominator", "den", "--max-dims", "1", "--min-rows", "2",
    )  # fmt: skip
    expected = ratio_cube(
        pl.read_csv(generic_csv, infer_schema=False, null_values=["NA", ""]).with_columns(
            pl.col("num", "den").cast(pl.Float64)
        ),
        ["a", "b"],
        "num",
        "den",
        max_dims=1,
        min_rows=2,
    )
    assert out == expected.write_csv()
    assert set(expected["cut_size"]) == {1}


@pytest.mark.parametrize("suffix", [".csv", ".CSV"])
def test_csv_output_file(
    capsys: pytest.CaptureFixture[str], generic_csv: Path, tmp_path: Path, suffix: str
) -> None:
    out_path = tmp_path / f"out{suffix}"
    out = run_ok(
        capsys, "ratio-cube", str(generic_csv), "-d", "a", "-d", "b", "--numerator", "num",
        "--denominator", "den", "-o", str(out_path),
    )  # fmt: skip
    assert out == ""
    expected = expected_generic(generic_csv)
    assert pl.read_csv(out_path, schema=expected.schema).equals(expected)


@pytest.mark.parametrize("suffix", [".parquet", ".PARQUET"])
def test_parquet_output_file(
    capsys: pytest.CaptureFixture[str],
    women_csv: Path,
    households_csv: Path,
    tmp_path: Path,
    suffix: str,
) -> None:
    out_path = tmp_path / f"out{suffix}"
    out = run_ok(
        capsys, "cd-ceb", "--women", str(women_csv), "--households", str(households_csv),
        "-d", DIM, "-d", "Age in 5-year groups", "--output", str(out_path),
    )  # fmt: skip
    assert out == ""
    expected = cd_ceb_cube(women_csv, households_csv, dimensions=[DIM, "Age in 5-year groups"])
    assert pl.read_parquet(out_path).equals(expected)


def test_empty_result_is_still_written(
    capsys: pytest.CaptureFixture[str], generic_csv: Path
) -> None:
    out = run_ok(
        capsys, "ratio-cube", str(generic_csv), "-d", "a", "--numerator", "num",
        "--denominator", "den", "--min-rows", "99",
    )  # fmt: skip
    assert out == "cut_size,cut,a,rows,numerator,denominator,ratio\n"


def test_read_generic_types(generic_csv: Path) -> None:
    frame = read_generic(generic_csv, "num", "den").collect()
    assert frame.schema == pl.Schema(
        {"a": pl.String, "b": pl.String, "num": pl.Float64, "den": pl.Float64}
    )
    assert frame["a"].null_count() == 1
    assert frame["num"].to_list() == [1.0, 2.0, 3.0, 4.0, None]


def test_write_cube_stdout_and_files(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    cube = pl.DataFrame({"x": [1]})
    write_cube(cube, None)
    assert capsys.readouterr().out == "x\n1\n"
    write_cube(cube, tmp_path / "o.csv")
    assert (tmp_path / "o.csv").read_text() == "x\n1\n"
    write_cube(cube, tmp_path / "o.parquet")
    assert pl.read_parquet(tmp_path / "o.parquet").equals(cube)


def test_build_parser_prog() -> None:
    assert build_parser().prog == "child-mortality"


# --- error paths ---------------------------------------------------------------------------


def test_bad_output_suffix(
    capsys: pytest.CaptureFixture[str], generic_csv: Path, tmp_path: Path
) -> None:
    code = main(
        ["ratio-cube", str(generic_csv), "-d", "a", "--numerator", "num", "--denominator", "den",
         "-o", str(tmp_path / "out.txt")]
    )  # fmt: skip
    assert code == 2
    err = assert_one_line_error(capsys)
    assert "output must end in .csv or .parquet" in err
    assert "out.txt" in err
    assert not (tmp_path / "out.txt").exists()


def test_missing_input_file(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code = main(
        [
            "ratio-cube",
            str(tmp_path / "nope.csv"),
            "-d",
            "a",
            "--numerator",
            "n",
            "--denominator",
            "d",
        ]
    )
    assert code == 2
    assert_one_line_error(capsys)


def test_missing_survey_file(
    capsys: pytest.CaptureFixture[str], households_csv: Path, tmp_path: Path
) -> None:
    code = main(
        ["cd-ceb", "--women", str(tmp_path / "nope.csv"), "--households", str(households_csv)]
    )
    assert code == 2
    assert_one_line_error(capsys)


def test_unwritable_output_location(
    capsys: pytest.CaptureFixture[str], generic_csv: Path, tmp_path: Path
) -> None:
    code = main(
        ["ratio-cube", str(generic_csv), "-d", "a", "--numerator", "num", "--denominator", "den",
         "-o", str(tmp_path / "missing_dir" / "out.csv")]
    )  # fmt: skip
    assert code == 2
    assert_one_line_error(capsys)


def test_library_value_error_is_reported(
    capsys: pytest.CaptureFixture[str], generic_csv: Path
) -> None:
    code = main(
        ["ratio-cube", str(generic_csv), "-d", "a", "--numerator", "num", "--denominator", "den",
         "--max-dims", "5"]
    )  # fmt: skip
    assert code == 2
    err = assert_one_line_error(capsys)
    assert "max_dims must be between 1 and 1, got 5" in err


def test_unknown_column_is_a_one_line_error(
    capsys: pytest.CaptureFixture[str], generic_csv: Path
) -> None:
    code = main(
        ["ratio-cube", str(generic_csv), "-d", "zzz", "--numerator", "num", "--denominator", "den"]
    )
    assert code == 2
    assert "columns not found in data: ['zzz']" in assert_one_line_error(capsys)


def test_non_numeric_measure_column_is_a_one_line_error(
    capsys: pytest.CaptureFixture[str], generic_csv: Path
) -> None:
    code = main(
        ["ratio-cube", str(generic_csv), "-d", "b", "--numerator", "a", "--denominator", "den"]
    )
    assert code == 2
    assert_one_line_error(capsys)


def test_polars_error_message_is_trimmed_to_one_line(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = write_csv(tmp_path / "bad.csv", ("a", "n", "d"), [("x", "not-a-number", "1")])
    code = main(["ratio-cube", str(path), "-d", "a", "--numerator", "n", "--denominator", "d"])
    assert code == 2
    assert_one_line_error(capsys)


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["bogus"],
        ["--nope"],
        ["cd-ceb"],
        ["cd-ceb", "--women", "w.csv"],
        ["ratio-cube"],
        ["ratio-cube", "in.csv", "-d", "a", "--numerator", "n"],
        ["ratio-cube", "in.csv", "--numerator", "n", "--denominator", "d"],
        ["ratio-cube", "in.csv", "-d", "a", "--numerator", "n", "--denominator", "d",
         "--max-dims", "many"],
        ["ratio-cube", "in.csv", "-d", "a", "--numerator", "n", "--denominator", "d",
         "--min-rows", "1.5"],
    ],
)  # fmt: skip
def test_usage_errors_exit_2_with_single_line(
    capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert_one_line_error(capsys)


def test_fail_keeps_only_the_first_line(capsys: pytest.CaptureFixture[str]) -> None:
    assert fail("first\nsecond\nthird") == 2
    assert capsys.readouterr().err == "child-mortality: error: first\n"


def test_fail_with_an_empty_message(capsys: pytest.CaptureFixture[str]) -> None:
    assert fail("") == 2
    assert fail("  \n") == 2
    assert capsys.readouterr().err == "child-mortality: error: unknown error\n" * 2


def test_error_with_empty_message_is_still_a_clean_exit(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> pl.DataFrame:
        raise ValueError

    monkeypatch.setattr("child_mortality.cli.cd_ceb_cube", boom)
    assert main(["cd-ceb", "--women", "w.csv", "--households", "h.csv"]) == 2
    err = capsys.readouterr().err
    assert err == "child-mortality: error: ValueError\n"


# --- output files and pipes ----------------------------------------------------------------


@pytest.mark.parametrize("name", ["o.csv", "o.parquet"])
def test_write_cube_leaves_no_temporary_files(tmp_path: Path, name: str) -> None:
    write_cube(pl.DataFrame({"x": [1]}), tmp_path / name)
    assert [path.name for path in tmp_path.iterdir()] == [name]


@pytest.mark.parametrize("name", ["o.csv", "o.parquet"])
def test_write_cube_replaces_an_existing_file(tmp_path: Path, name: str) -> None:
    target = tmp_path / name
    write_cube(pl.DataFrame({"x": [1]}), target)
    new = pl.DataFrame({"x": [2, 3]})
    write_cube(new, target)
    read = pl.read_parquet if name.endswith(".parquet") else pl.read_csv
    assert read(target).equals(new)
    assert [path.name for path in tmp_path.iterdir()] == [name]


@pytest.mark.parametrize("name", ["o.csv", "o.parquet"])
def test_failed_write_keeps_the_previous_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str
) -> None:
    target = tmp_path / name
    target.write_bytes(b"previous")

    def fail_midway(self: pl.DataFrame, file: Path) -> None:
        file.write_bytes(b"trunc")
        raise OSError("disk full")

    monkeypatch.setattr(pl.DataFrame, "write_csv", fail_midway)
    monkeypatch.setattr(pl.DataFrame, "write_parquet", fail_midway)
    with pytest.raises(OSError, match="disk full"):
        write_cube(pl.DataFrame({"x": [1]}), target)
    assert target.read_bytes() == b"previous"
    assert [path.name for path in tmp_path.iterdir()] == [name]


def test_cli_reports_a_failed_write_and_keeps_the_previous_file(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    generic_csv: Path,
    tmp_path: Path,
) -> None:
    target = tmp_path / "out.csv"
    target.write_text("previous")

    def fail_midway(self: pl.DataFrame, file: Path) -> None:
        file.write_text("trunc")
        raise OSError("disk full")

    monkeypatch.setattr(pl.DataFrame, "write_csv", fail_midway)
    code = main(
        ["ratio-cube", str(generic_csv), "-d", "a", "--numerator", "num", "--denominator", "den",
         "-o", str(target)]
    )  # fmt: skip
    assert code == 2
    assert "disk full" in assert_one_line_error(capsys)
    assert target.read_text() == "previous"


class FailingStdout:
    """A stdout whose writes raise ``error``."""

    def __init__(self, error: OSError) -> None:
        self.error = error

    def write(self, text: str) -> int:
        raise self.error

    def flush(self) -> None:
        raise AssertionError("write fails first")

    def fileno(self) -> int:
        return 99


@pytest.mark.parametrize(
    "error",
    [
        BrokenPipeError(errno.EPIPE, "Broken pipe"),
        OSError(errno.EINVAL, "Invalid argument"),  # how Windows reports a closed pipe
    ],
    ids=["epipe", "windows-einval"],
)
def test_broken_pipe_is_a_quiet_exit(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    generic_csv: Path,
    error: OSError,
) -> None:
    redirected: list[int] = []
    monkeypatch.setattr(sys, "stdout", FailingStdout(error))
    monkeypatch.setattr("os.dup2", lambda source, target: redirected.append(target))
    code = main(
        ["ratio-cube", str(generic_csv), "-d", "a", "--numerator", "num", "--denominator", "den"]
    )
    assert code == 1
    assert capsys.readouterr().err == ""
    assert redirected == [99]  # stdout now points at the null device


def test_other_stdout_errors_are_reported(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    generic_csv: Path,
) -> None:
    monkeypatch.setattr(sys, "stdout", FailingStdout(OSError(errno.EIO, "I/O error")))
    code = main(
        ["ratio-cube", str(generic_csv), "-d", "a", "--numerator", "num", "--denominator", "den"]
    )
    assert code == 2
    assert "I/O error" in assert_one_line_error(capsys)


def test_detach_stdout_tolerates_a_stdout_without_a_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NoFileno:
        def fileno(self) -> int:
            raise ValueError("no descriptor")

    monkeypatch.setattr(sys, "stdout", NoFileno())
    _detach_stdout()


def test_closed_pipe_prints_nothing_to_stderr(generic_csv: Path) -> None:
    process = subprocess.Popen(
        [sys.executable, "-m", "child_mortality", "ratio-cube", str(generic_csv), "-d", "a",
         "--numerator", "num", "--denominator", "den"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )  # fmt: skip
    assert process.stdout is not None
    assert process.stderr is not None
    process.stdout.close()  # the reader exits before the CLI writes
    # Not communicate(): on Windows its reader thread fails on the stdout we just closed.
    stderr = process.stderr.read()
    process.stderr.close()
    assert process.wait(timeout=60) == 1
    assert stderr == b""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes and unprivileged symlinks")
def test_write_follows_a_symlink_and_keeps_permissions(tmp_path: Path) -> None:
    real = tmp_path / "real.csv"
    real.write_text("previous")
    real.chmod(0o600)
    link = tmp_path / "link.csv"
    link.symlink_to(real)
    write_cube(pl.DataFrame({"x": [1]}), link)
    assert link.is_symlink()
    assert pl.read_csv(real).equals(pl.DataFrame({"x": [1]}))
    assert real.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("name", ["missing_dir/out.csv", "a_dir.csv"])
def test_output_errors_name_the_output_not_the_temporary_file(
    capsys: pytest.CaptureFixture[str], generic_csv: Path, tmp_path: Path, name: str
) -> None:
    (tmp_path / "a_dir.csv").mkdir()
    code = main(
        ["ratio-cube", str(generic_csv), "-d", "a", "--numerator", "num", "--denominator", "den",
         "-o", str(tmp_path / name)]
    )  # fmt: skip
    assert code == 2
    message = assert_one_line_error(capsys)
    assert ".tmp" not in message
    # OSError shows the path's repr (backslashes doubled on Windows), so match the user's leaf name.
    assert Path(name).parts[0] in message


def test_a_read_only_output_is_not_replaced(tmp_path: Path) -> None:
    target = tmp_path / "ro.csv"
    target.write_text("previous")
    target.chmod(0o444)
    try:
        with pytest.raises(PermissionError):
            write_cube(pl.DataFrame({"x": [1]}), target)
        assert target.read_text() == "previous"
    finally:
        target.chmod(0o644)


def test_whitespace_only_error_message_names_the_exception(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, generic_csv: Path
) -> None:
    def raise_blank(*args: object, **kwargs: object) -> pl.DataFrame:
        raise ValueError("   ")

    monkeypatch.setattr("child_mortality.cli.ratio_cube", raise_blank)
    code = main(
        ["ratio-cube", str(generic_csv), "-d", "a", "--numerator", "num", "--denominator", "den"]
    )
    assert code == 2
    assert "ValueError" in assert_one_line_error(capsys)


# --- package entry points ------------------------------------------------------------------


def test_module_entry_point_in_process(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["child-mortality", "--version"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("child_mortality", run_name="__main__")
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.startswith("child-mortality ")


def test_python_dash_m_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "child_mortality", "--help"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0
    assert result.stdout.startswith("usage: child-mortality")
    assert "cd-ceb" in result.stdout
    assert "ratio-cube" in result.stdout
    assert result.stderr == ""


def test_python_dash_m_error_has_exit_code_2(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "child_mortality", "ratio-cube", str(tmp_path / "nope.csv"),
         "-d", "a", "--numerator", "n", "--denominator", "d"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )  # fmt: skip
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith("child-mortality: error: ")
    assert "Traceback" not in result.stderr


def test_public_exports() -> None:
    assert set(child_mortality.__all__) == {"__version__", "cd_ceb_cube", "ratio_cube"}
    assert child_mortality.ratio_cube is ratio_cube
    assert child_mortality.cd_ceb_cube is cd_ceb_cube


def test_version_matches_installed_metadata() -> None:
    assert __version__ == importlib.metadata.version("child-mortality")


def test_version_falls_back_when_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    try:
        reloaded = importlib.reload(child_mortality)
        assert reloaded.__version__ == "0+unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(child_mortality)
    assert child_mortality.__version__ == importlib.metadata.version("child-mortality")

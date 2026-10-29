"""Smoke tests; the full suite lives in later changes."""

from pathlib import Path

import polars as pl
import pytest

from child_mortality import __version__, cd_ceb_cube, ratio_cube
from child_mortality.cli import main
from child_mortality.survey import DIMENSIONS

DATA = Path(__file__).parent.parent / "data"
WOMEN = DATA / "women.csv"
HOUSEHOLDS = DATA / "household.csv"
OVERALL_CD_CEB = 0.20189301107073437


def test_tiny_cube_by_hand() -> None:
    df = pl.DataFrame(
        {
            "a": ["x", "x", "y", None],
            "b": ["p", "q", "p", "p"],
            "num": [1, 2, 3, 4],
            "den": [2, 0, 6, 8],
        }
    )
    cube = ratio_cube(df, ["a", "b"], "num", "den")
    rows = cube.rows()
    # cut_size, cut, a, b, rows, numerator, denominator, ratio
    assert rows == [
        (1, "a", "x", None, 2, 3.0, 2.0, 1.5),
        (1, "a", "y", None, 1, 3.0, 6.0, 0.5),
        (1, "a", None, None, 1, 4.0, 8.0, 0.5),
        (1, "b", None, "p", 3, 8.0, 16.0, 0.5),
        (1, "b", None, "q", 1, 2.0, 0.0, None),
        (2, "a|b", "x", "p", 1, 1.0, 2.0, 0.5),
        (2, "a|b", "x", "q", 1, 2.0, 0.0, None),
        (2, "a|b", "y", "p", 1, 3.0, 6.0, 0.5),
        (2, "a|b", None, "p", 1, 4.0, 8.0, 0.5),
    ]


def test_min_rows_matches_filtered_full_cube() -> None:
    full = cd_ceb_cube(WOMEN, HOUSEHOLDS, max_dims=2)
    pruned = cd_ceb_cube(WOMEN, HOUSEHOLDS, max_dims=2, min_rows=30)
    assert pruned.equals(full.filter(pl.col("rows") >= 30))
    assert 0 < pruned.height < full.height


def test_min_rows_above_table_size_is_empty() -> None:
    df = pl.DataFrame({"a": ["x"], "num": [1], "den": [1]})
    assert ratio_cube(df, ["a"], "num", "den", min_rows=2).is_empty()


def test_invalid_arguments() -> None:
    df = pl.DataFrame({"a": ["x"], "num": [1], "den": [1]})
    with pytest.raises(ValueError, match="not found"):
        ratio_cube(df, ["nope"], "num", "den")


def test_overall_ratio_on_real_data() -> None:
    cube = cd_ceb_cube(WOMEN, HOUSEHOLDS, max_dims=1)
    residence = cube.filter(pl.col("cut") == "Type of place of residence")
    assert residence["rows"].sum() == 8348
    overall = residence.select(pl.col("numerator").sum() / pl.col("denominator").sum()).item()
    assert overall == OVERALL_CD_CEB
    assert cube["cut"].n_unique() == len(DIMENSIONS)


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_cli_error_is_one_line(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["cd-ceb", "--women", "missing.csv", "--households", str(HOUSEHOLDS)])
    err = capsys.readouterr().err
    assert code == 2
    assert err.startswith("child-mortality: error:")
    assert err.count("\n") == 1

"""Dimension semantics: natural ordering (#15) and configurable null tokens (#16)."""

import random
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from child_mortality.cli import main, read_generic
from child_mortality.cube import ratio_cube
from child_mortality.survey import cd_ceb_cube, load_households
from conftest import DATA_DIR, HOUSEHOLD_HEADER, write_indexed_csv
from mock_helpers import csv_text, write_text


def one_dim(values: list[Any], dtype: Any) -> pl.DataFrame:
    return pl.DataFrame(
        {"g": pl.Series(values, dtype=dtype), "num": [1] * len(values), "den": [2] * len(values)}
    )


def labels(frame: pl.DataFrame) -> list[str | None]:
    return frame.filter(pl.col("cut") == "g")["g"].to_list()


# --- ordering (#15) -----------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, reason="#15: numeric dimensions must sort numerically")
def test_integer_dimension_sorts_numerically() -> None:
    out = ratio_cube(one_dim([10, 2, 1, 11, 9, 20], pl.Int64), ["g"], "num", "den")
    assert labels(out) == ["1", "2", "9", "10", "11", "20"]


@pytest.mark.xfail(strict=True, reason="#15: negative numbers must sort numerically")
def test_negative_integers_sort_numerically() -> None:
    out = ratio_cube(one_dim([-1, -10, 2, 0], pl.Int64), ["g"], "num", "den")
    assert labels(out) == ["-10", "-1", "0", "2"]


@pytest.mark.xfail(strict=True, reason="#15: float dimension sorts numerically, nulls last")
def test_float_dimension_sorts_numerically_nulls_last() -> None:
    out = ratio_cube(one_dim([10.5, 2.25, None, 1.0], pl.Float64), ["g"], "num", "den")
    assert labels(out) == ["1.0", "2.25", "10.5", None]


@pytest.mark.xfail(strict=True, reason="#15: -0.0 and 0.0 must be one cell")
def test_negative_zero_and_zero_are_one_cell() -> None:
    out = ratio_cube(one_dim([-0.0, 0.0, 1.5], pl.Float64), ["g"], "num", "den")
    assert out.filter(pl.col("cut") == "g").sort("rows")["rows"].to_list() == [1, 2]


@pytest.mark.xfail(strict=True, reason="#15: Enum dimension keeps its declared order")
def test_enum_dimension_keeps_declared_order() -> None:
    dtype = pl.Enum(["low", "med", "high"])
    out = ratio_cube(one_dim(["high", "low", "med", "low", None], dtype), ["g"], "num", "den")
    assert labels(out) == ["low", "med", "high", None]


@pytest.mark.xfail(strict=True, reason="#15: Enum order also applies inside multi-dimension cuts")
def test_enum_order_inside_a_two_dimension_cut() -> None:
    dtype = pl.Enum(["low", "med", "high"])
    frame = pl.DataFrame(
        {
            "g": pl.Series(["high", "low", "high", "low"], dtype=dtype),
            "h": ["y", "y", "x", "x"],
            "num": [1, 1, 1, 1],
            "den": [2, 2, 2, 2],
        }
    )
    out = ratio_cube(frame, ["g", "h"], "num", "den").filter(pl.col("cut") == "g|h")
    assert out.select("g", "h").rows() == [("low", "x"), ("low", "y"), ("high", "x"), ("high", "y")]


@pytest.mark.xfail(strict=True, reason="#15: Categorical dimension uses physical order")
def test_categorical_dimension_uses_physical_order() -> None:
    out = ratio_cube(one_dim(["b", "a", "c", "a"], pl.Categorical), ["g"], "num", "den")
    assert labels(out) == ["b", "a", "c"]


@pytest.mark.xfail(strict=True, reason="#15: household size lists 1..24 in numeric order")
def test_survey_household_size_is_in_numeric_order() -> None:
    out = cd_ceb_cube(
        DATA_DIR / "women.csv",
        DATA_DIR / "household.csv",
        dimensions=["Number of household members"],
    )
    sizes = [int(v) for v in out["Number of household members"].drop_nulls()]
    assert sizes == sorted(sizes)
    assert sizes[:3] == [1, 2, 3]


def test_string_dimension_order_is_lexicographic_with_nulls_last() -> None:
    """The documented contract today; stays true for String dimensions after #15."""
    out = ratio_cube(one_dim(["b", None, "B", "a", ""], pl.String), ["g"], "num", "den")
    assert labels(out) == ["", "B", "a", "b", None]


def test_output_order_does_not_depend_on_input_row_order() -> None:
    rng = random.Random(3)
    values = [rng.choice(["x", "y", "z", None]) for _ in range(200)]
    frame = pl.DataFrame({"g": values, "h": values[::-1], "num": [1] * 200, "den": [3] * 200})
    reference = ratio_cube(frame, ["g", "h"], "num", "den")
    for seed in range(3):
        shuffled = frame.sample(fraction=1.0, shuffle=True, seed=seed)
        assert ratio_cube(shuffled, ["g", "h"], "num", "den").equals(reference)


# --- null tokens (#16) --------------------------------------------------------------------------

REGION_CSV = "region,orders,refunds\nNA,10,1\nNA,10,1\nEU,10,5\n,10,9\n"


def region_argv(path: Path, *extra: str) -> list[str]:
    return [
        "ratio-cube",
        str(path),
        "-d",
        "region",
        "--numerator",
        "refunds",
        "--denominator",
        "orders",
        *extra,
    ]


def region_cells(out: str) -> dict[str, tuple[int, float]]:
    cells: dict[str, tuple[int, float]] = {}
    for line in out.splitlines()[1:]:
        _, _, region, rows, num, _, _ = line.split(",")
        cells[region] = (int(rows), float(num))
    return cells


def test_default_tokens_merge_na_and_empty_into_one_missing_cell(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_text(tmp_path / "na.csv", REGION_CSV)
    assert main(region_argv(path)) == 0
    assert region_cells(capsys.readouterr().out) == {"EU": (1, 5.0), "": (3, 11.0)}


@pytest.mark.xfail(strict=True, reason='#16: --null-value "" keeps a literal NA category')
def test_cli_null_value_empty_only_keeps_na_as_a_category(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_text(tmp_path / "na.csv", REGION_CSV)
    assert main(region_argv(path, "--null-value", "")) == 0
    assert region_cells(capsys.readouterr().out) == {
        "EU": (1, 5.0),
        "NA": (2, 2.0),
        "": (1, 9.0),
    }


@pytest.mark.xfail(strict=True, reason="#16: --null-value is repeatable; both tokens merge again")
def test_cli_null_value_is_repeatable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_text(tmp_path / "na.csv", REGION_CSV)
    assert main(region_argv(path, "--null-value", "NA", "--null-value", "")) == 0
    assert region_cells(capsys.readouterr().out) == {"EU": (1, 5.0), "": (3, 11.0)}


@pytest.mark.xfail(strict=True, reason="#16: custom token (e.g. '-') is read as missing")
def test_cli_custom_null_token(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    text = csv_text(("region", "orders", "refunds"), [("-", "10", "1"), ("NA", "10", "1")])
    path = write_text(tmp_path / "dash.csv", text)
    assert main(region_argv(path, "--null-value", "-")) == 0
    assert region_cells(capsys.readouterr().out) == {"NA": (1, 1.0), "": (1, 1.0)}


@pytest.mark.xfail(strict=True, reason="#16: read_generic(null_values=...) not implemented")
def test_read_generic_null_values_argument(tmp_path: Path) -> None:
    path = write_text(tmp_path / "na.csv", REGION_CSV)
    frame = read_generic(path, "refunds", "orders", null_values=[""]).collect()  # type: ignore[call-arg]
    assert frame["region"].to_list() == ["NA", "NA", "EU", None]


@pytest.mark.xfail(strict=True, reason="#16: load_households(null_values=...) not implemented")
def test_loader_null_values_argument(tmp_path: Path) -> None:
    path = write_indexed_csv(
        tmp_path / "h.csv", HOUSEHOLD_HEADER, [("1", "1", "NA", "40"), ("1", "2", "", "50")]
    )
    frame = load_households(path, null_values=[""]).collect()  # type: ignore[call-arg]
    assert frame.sort("Household number")["Type of place of residence"].to_list() == ["NA", None]


def test_default_loader_reads_na_and_empty_as_null(tmp_path: Path) -> None:
    path = write_indexed_csv(
        tmp_path / "h.csv", HOUSEHOLD_HEADER, [("1", "1", "NA", "40"), ("1", "2", "", "50")]
    )
    frame = load_households(path).collect()
    assert frame["Type of place of residence"].to_list() == [None, None]

"""Tests for ``child_mortality.survey`` on tiny temporary CSV files."""

from pathlib import Path

import polars as pl
import pytest

from child_mortality.survey import (
    DENOMINATOR,
    DIMENSIONS,
    JOIN_KEYS,
    NA_VALUES,
    NUMERATOR,
    cd_ceb_cube,
    join_women_households,
    load_households,
    load_women,
)
from conftest import HOUSEHOLD_HEADER, WOMEN_HEADER, write_csv, write_indexed_csv


def test_constants() -> None:
    assert JOIN_KEYS == ("Cluster number", "Household number")
    assert NUMERATOR == "Children died"
    assert DENOMINATOR == "Total children ever born"
    assert NA_VALUES == ("NA", "")
    assert len(DIMENSIONS) == len(set(DIMENSIONS)) == 9


def test_load_women_drops_index_and_types_columns(women_csv: Path) -> None:
    lazy = load_women(women_csv)
    assert isinstance(lazy, pl.LazyFrame)
    frame = lazy.collect()
    assert frame.columns == list(WOMEN_HEADER)  # the unnamed index column is gone
    assert frame.height == 4
    for column in ("Total children ever born", "Sons who have died", "Daughters who have died"):
        assert frame.schema[column] == pl.Int64
    for column in ("Cluster number", "Household number", "Age in 5-year groups"):
        assert frame.schema[column] == pl.String


def test_load_women_reads_na_and_empty_as_null(women_csv: Path) -> None:
    frame = load_women(women_csv).collect()
    assert frame["Total children ever born"].to_list() == [2, 3, None, 4]
    assert frame["Sons who have died"].to_list() == [1, 0, None, None]


def test_load_households_keeps_every_column_as_string(households_csv: Path) -> None:
    frame = load_households(households_csv).collect()
    assert frame.columns == list(HOUSEHOLD_HEADER)
    assert set(frame.schema.values()) == {pl.String}
    assert frame["Type of place of residence"].to_list() == ["Urban", None, "Rural"]
    assert frame["Age in 5-year groups"].to_list() == ["40", None, "50"]


def test_loaders_accept_str_paths(women_csv: Path, households_csv: Path) -> None:
    assert load_women(str(women_csv)).collect().height == 4
    assert load_households(str(households_csv)).collect().height == 3


def test_join_keeps_unmatched_women_with_null_household_columns(
    women_csv: Path, households_csv: Path
) -> None:
    joined = join_women_households(load_women(women_csv), load_households(households_csv))
    frame = joined.collect().sort("Cluster number", "Household number", "Age in 5-year groups")
    assert frame.height == 4
    assert frame["Type of place of residence"].to_list() == ["Urban", "Urban", None, None]
    unmatched = frame.filter(pl.col("Cluster number") == "9")
    assert unmatched.height == 1
    assert unmatched["Type of place of residence"].null_count() == 1


def test_join_suffixes_overlapping_columns(women_csv: Path, households_csv: Path) -> None:
    joined = join_women_households(load_women(women_csv), load_households(households_csv))
    columns = joined.collect_schema().names()
    assert "Age in 5-year groups" in columns
    assert "Age in 5-year groups_household" in columns
    assert columns.count("Cluster number") == 1
    assert columns.count("Household number") == 1
    frame = joined.collect().sort("Cluster number", "Household number", "Age in 5-year groups")
    assert frame["Age in 5-year groups_household"].to_list() == ["40", "40", None, None]


def test_children_died_is_sons_plus_daughters(women_csv: Path, households_csv: Path) -> None:
    joined = join_women_households(load_women(women_csv), load_households(households_csv))
    frame = joined.collect()
    assert frame.schema[NUMERATOR] == pl.Int64
    # (1, 0) -> 1; (0, 2) -> 2; (NA, NA) -> null; (NA, 1) -> null (an unknown count is unknown)
    by_key = {
        (row["Cluster number"], row["Household number"], row["Age in 5-year groups"]): row[
            NUMERATOR
        ]
        for row in frame.to_dicts()
    }
    assert by_key == {
        ("1", "1", "20-24"): 1,
        ("1", "1", "25-29"): 2,
        ("1", "2", "20-24"): None,
        ("9", "9", "30-34"): None,
    }


def test_unknown_deaths_are_zero_restores_the_lenient_sum(
    women_csv: Path, households_csv: Path
) -> None:
    joined = join_women_households(
        load_women(women_csv), load_households(households_csv), unknown_deaths_are_zero=True
    )
    died = joined.collect().sort("Cluster number", "Household number")[NUMERATOR]
    assert died.to_list() == [1, 2, 0, 1]


def test_cd_ceb_cube_errors_on_null_counts_by_default(
    women_csv: Path, households_csv: Path
) -> None:
    with pytest.raises(ValueError, match="Total children ever born"):
        cd_ceb_cube(women_csv, households_csv, dimensions=["Type of place of residence"])


def test_duplicate_household_keys_raise(tmp_path: Path, women_csv: Path) -> None:
    households = write_indexed_csv(
        tmp_path / "dup.csv",
        HOUSEHOLD_HEADER,
        [("1", "1", "Urban", "40"), ("1", "1", "Rural", "41")],
    )
    with pytest.raises(ValueError, match="not unique"):
        join_women_households(load_women(women_csv), load_households(households))


def test_cd_ceb_cube_with_custom_dimensions(women_csv: Path, households_csv: Path) -> None:
    cube = cd_ceb_cube(
        women_csv,
        households_csv,
        dimensions=["Type of place of residence"],
        null_measures="zero",
    )
    assert cube.columns == [
        "cut_size",
        "cut",
        "Type of place of residence",
        "rows",
        "numerator",
        "denominator",
        "ratio",
    ]
    # Urban: women 1 and 2 (died 1 + 2 = 3, born 2 + 3 = 5); missing: NA household (0 / 0) and
    # the unmatched woman (died 1, born 4).
    assert cube.rows() == [
        (1, "Type of place of residence", "Urban", 2, 3.0, 5.0, 0.6),
        (1, "Type of place of residence", None, 2, 1.0, 4.0, 0.25),
    ]


def test_cd_ceb_cube_passes_max_dims_and_min_rows(women_csv: Path, households_csv: Path) -> None:
    dims = ["Type of place of residence", "Age in 5-year groups"]
    full = cd_ceb_cube(women_csv, households_csv, dimensions=dims, null_measures="zero")
    assert full["cut_size"].max() == 2
    one = cd_ceb_cube(women_csv, households_csv, dimensions=dims, max_dims=1, null_measures="zero")
    assert one["cut_size"].max() == 1
    frequent = cd_ceb_cube(
        women_csv, households_csv, dimensions=dims, min_rows=2, null_measures="zero"
    )
    assert frequent.equals(full.filter(pl.col("rows") >= 2))


def test_cd_ceb_cube_defaults_to_survey_dimensions(tmp_path: Path) -> None:
    keys = list(JOIN_KEYS)
    women = write_indexed_csv(
        tmp_path / "w.csv",
        [*keys, DENOMINATOR, "Sons who have died", "Daughters who have died"],
        [("1", "1", "3", "1", "0")],
    )
    households = write_csv(
        tmp_path / "h.csv", [*keys, *DIMENSIONS[1:]], [("1", "1", *["x"] * (len(DIMENSIONS) - 1))]
    )
    with pytest.raises(ValueError, match="columns not found"):
        cd_ceb_cube(women, households)  # the women's file lacks the age dimension


def test_cd_ceb_cube_default_dimensions_when_all_present(tmp_path: Path) -> None:
    keys = list(JOIN_KEYS)
    women = write_csv(
        tmp_path / "w.csv",
        ["", *keys, DENOMINATOR, "Sons who have died", "Daughters who have died", DIMENSIONS[0]],
        [("1", "1", "1", "3", "1", "0", "20-24")],
    )
    households = write_csv(
        tmp_path / "h.csv", [*keys, *DIMENSIONS[1:]], [("1", "1", *["x"] * (len(DIMENSIONS) - 1))]
    )
    cube = cd_ceb_cube(women, households, max_dims=1)
    assert cube["cut"].to_list() == list(DIMENSIONS)
    assert set(cube["ratio"].to_list()) == {1 / 3}

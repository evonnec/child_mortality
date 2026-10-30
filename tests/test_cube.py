"""Unit tests for ``child_mortality.cube``: hand-computed cubes, validation, helpers."""

import math
from itertools import product
from typing import Any

import polars as pl
import pytest

from child_mortality.cube import (
    NullMeasures,
    empty_cube,
    finest_grain,
    format_cells,
    order_cells,
    ratio_cube,
    restrict_to_cells,
    roll_up,
    validate_inputs,
    walk_cuts,
)
from conftest import assert_matches_oracle, naive_cube, ratio_matches

POLICIES: tuple[NullMeasures, ...] = ("error", "drop", "zero")


def tiny() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": ["x", "x", "y", None],
            "b": ["p", "q", "p", "p"],
            "num": [1, 2, 3, 4],
            "den": [2, 0, 6, 8],
        }
    )


def grid() -> pl.DataFrame:
    """3 age groups x 2 residence types x 2 household sizes, every combination once."""
    combos = list(product(["a", "b", "c"], ["u", "r"], ["s", "l"]))
    return pl.DataFrame(
        {
            "age": [c[0] for c in combos],
            "res": [c[1] for c in combos],
            "size": [c[2] for c in combos],
            "num": list(range(12)),
            "den": [10] * 12,
        }
    )


GRID_DIMS = ["age", "res", "size"]

# --- hand-computed cubes -------------------------------------------------------------------


def test_two_dimension_cube_by_hand() -> None:
    cube = ratio_cube(tiny(), ["a", "b"], "num", "den")
    # cut_size, cut, a, b, rows, numerator, denominator, ratio
    assert cube.rows() == [
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


def test_single_dimension_cube() -> None:
    df = pl.DataFrame({"g": ["m", "f", "m"], "n": [1, 2, 3], "d": [4, 4, 4]})
    cube = ratio_cube(df, ["g"], "n", "d")
    assert cube.rows() == [
        (1, "g", "f", 1, 2.0, 4.0, 0.5),
        (1, "g", "m", 2, 4.0, 8.0, 0.5),
    ]


def test_grid_cell_counts_per_cut_size() -> None:
    cube = ratio_cube(grid(), GRID_DIMS, "num", "den")
    counts = cube.group_by("cut_size").len().sort("cut_size")
    assert counts.rows() == [(1, 3 + 2 + 2), (2, 3 * 2 + 3 * 2 + 2 * 2), (3, 12)]
    assert cube.height == 35
    assert cube.group_by("cut").len().height == 7


def test_grid_exact_cells() -> None:
    cube = ratio_cube(grid(), GRID_DIMS, "num", "den")

    def one(cut: str, **values: str) -> tuple[Any, ...]:
        frame = cube.filter(pl.col("cut") == cut)
        for column, value in values.items():
            frame = frame.filter(pl.col(column) == value)
        assert frame.height == 1
        row = frame.row(0, named=True)
        return row["rows"], row["numerator"], row["denominator"], row["ratio"]

    assert one("age", age="a") == (4, 6.0, 40.0, 0.15)  # i = 0..3
    assert one("res", res="u") == (6, 27.0, 60.0, 0.45)  # i = 0, 1, 4, 5, 8, 9
    assert one("size", size="s") == (6, 30.0, 60.0, 0.5)  # even i
    assert one("age|res", age="b", res="r") == (2, 13.0, 20.0, 0.65)  # i = 6, 7
    assert one("age|res|size", age="a", res="u", size="s") == (1, 0.0, 10.0, 0.0)
    assert one("age|res|size", age="c", res="r", size="l") == (1, 11.0, 10.0, 1.1)


def test_every_cut_partitions_the_input() -> None:
    cube = ratio_cube(grid(), GRID_DIMS, "num", "den")
    for cut, cells in cube.group_by("cut"):
        assert cells["rows"].sum() == 12, cut
        assert cells["numerator"].sum() == 66.0, cut
        assert cells["denominator"].sum() == 120.0, cut


def test_matches_naive_oracle() -> None:
    records = tiny().to_dicts()
    cube = ratio_cube(tiny(), ["a", "b"], "num", "den")
    assert_matches_oracle(cube.rows(), naive_cube(records, ["a", "b"], "num", "den"))


# --- output shape --------------------------------------------------------------------------


def test_output_schema_and_column_order() -> None:
    cube = ratio_cube(tiny(), ["b", "a"], "num", "den")
    assert cube.schema == pl.Schema(
        {
            "cut_size": pl.Int64,
            "cut": pl.String,
            "b": pl.String,
            "a": pl.String,
            "rows": pl.Int64,
            "numerator": pl.Float64,
            "denominator": pl.Float64,
            "ratio": pl.Float64,
        }
    )


@pytest.mark.parametrize("dtype", [pl.Int8, pl.UInt32, pl.Int64, pl.Float32, pl.Float64])
def test_measure_dtypes_become_float64(dtype: pl.DataType) -> None:
    df = tiny().with_columns(pl.col("num", "den").cast(dtype))
    cube = ratio_cube(df, ["a"], "num", "den")
    assert cube.schema["numerator"] == pl.Float64
    assert cube.schema["denominator"] == pl.Float64
    assert cube.row(0) == (1, "a", "x", 2, 3.0, 2.0, 1.5)


def test_non_string_dimensions_are_cast_to_string() -> None:
    df = pl.DataFrame({"k": [10, 9, 10], "flag": [True, False, True], "n": [1, 1, 1], "d": [2] * 3})
    cube = ratio_cube(df, ["k", "flag"], "n", "d", max_dims=1)
    assert cube.schema["k"] == pl.String
    assert cube.filter(cut="k")["k"].to_list() == ["10", "9"]  # string order, not numeric
    assert cube.filter(cut="flag")["flag"].to_list() == ["false", "true"]


def test_cut_follows_dimension_order_not_alphabetical() -> None:
    cube = ratio_cube(tiny(), ["b", "a"], "num", "den")
    assert cube["cut"].unique(maintain_order=True).to_list() == ["b", "a", "b|a"]


# --- deterministic ordering ----------------------------------------------------------------


def test_cut_order_is_itertools_combinations_order() -> None:
    df = pl.DataFrame({"a": ["1"], "b": ["1"], "c": ["1"], "n": [1], "d": [1]})
    cube = ratio_cube(df, ["a", "b", "c"], "n", "d")
    assert cube["cut"].to_list() == ["a", "b", "c", "a|b", "a|c", "b|c", "a|b|c"]
    assert cube["cut_size"].to_list() == [1, 1, 1, 2, 2, 2, 3]


def test_values_ascending_nulls_last_within_cut() -> None:
    df = pl.DataFrame({"a": ["b", "a", None, "c", "a"], "n": [1] * 5, "d": [1] * 5})
    cube = ratio_cube(df, ["a"], "n", "d")
    assert cube["a"].to_list() == ["a", "b", "c", None]
    assert cube["rows"].to_list() == [2, 1, 1, 1]


def test_result_is_independent_of_input_row_order() -> None:
    df = grid()
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=7)
    assert ratio_cube(df, GRID_DIMS, "num", "den").equals(
        ratio_cube(shuffled, GRID_DIMS, "num", "den")
    )


# --- max_dims / min_rows -------------------------------------------------------------------


def test_max_dims_defaults_to_all_dimensions() -> None:
    cube = ratio_cube(grid(), GRID_DIMS, "num", "den")
    assert cube["cut_size"].max() == 3
    explicit = ratio_cube(grid(), GRID_DIMS, "num", "den", max_dims=3)
    assert cube.equals(explicit)


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_max_dims_limits_cut_size(limit: int) -> None:
    full = ratio_cube(grid(), GRID_DIMS, "num", "den")
    limited = ratio_cube(grid(), GRID_DIMS, "num", "den", max_dims=limit)
    assert limited.equals(full.filter(pl.col("cut_size") <= limit))
    assert limited["cut_size"].max() == limit


def test_min_rows_drops_small_cells() -> None:
    cube = ratio_cube(grid(), GRID_DIMS, "num", "den", min_rows=4)
    assert cube.rows() == [
        (1, "age", "a", None, None, 4, 6.0, 40.0, 0.15),
        (1, "age", "b", None, None, 4, 22.0, 40.0, 0.55),
        (1, "age", "c", None, None, 4, 38.0, 40.0, 0.95),
        (1, "res", None, "r", None, 6, 39.0, 60.0, 0.65),
        (1, "res", None, "u", None, 6, 27.0, 60.0, 0.45),
        (1, "size", None, None, "l", 6, 36.0, 60.0, 0.6),
        (1, "size", None, None, "s", 6, 30.0, 60.0, 0.5),
    ]


def test_min_rows_equals_filtered_full_cube_including_order() -> None:
    records = [
        {"a": a, "b": b, "c": c, "n": i % 3, "d": 1 + i % 2}
        for i, (a, b, c) in enumerate(
            list(product(["x", "y", None], ["p", "q"], ["u", "v", "w"])) * 3
        )
    ]
    df = pl.DataFrame(records, schema_overrides={"a": pl.String})
    full = ratio_cube(df, ["a", "b", "c"], "n", "d")
    for threshold in (1, 2, 3, 6, 9, 18, 54, 55):
        pruned = ratio_cube(df, ["a", "b", "c"], "n", "d", min_rows=threshold)
        assert pruned.equals(full.filter(pl.col("rows") >= threshold)), threshold


def test_min_rows_above_total_rows_is_empty_with_schema() -> None:
    cube = ratio_cube(tiny(), ["a", "b"], "num", "den", min_rows=5)
    assert cube.is_empty()
    assert cube.schema == empty_cube(("a", "b")).schema


def test_min_rows_equal_to_total_rows_keeps_only_full_cells() -> None:
    df = pl.DataFrame({"a": ["x"] * 3, "b": ["p", "p", "q"], "n": [1, 1, 1], "d": [2, 2, 2]})
    cube = ratio_cube(df, ["a", "b"], "n", "d", min_rows=3)
    assert cube.rows() == [(1, "a", "x", None, 3, 3.0, 6.0, 0.5)]


def test_min_rows_skips_cut_whose_subcut_has_no_frequent_cell() -> None:
    # "a" is unique per row so it has no frequent cell: cuts containing it are skipped.
    df = pl.DataFrame(
        {"a": ["1", "2", "3", "4"], "b": ["p", "p", "q", "q"], "c": ["u", "u", "u", "u"]}
    ).with_columns(n=pl.lit(1), d=pl.lit(1))
    cube = ratio_cube(df, ["a", "b", "c"], "n", "d", min_rows=2)
    assert cube["cut"].unique(maintain_order=True).to_list() == ["b", "c", "b|c"]
    assert cube.filter(cut="b|c").rows() == [
        (2, "b|c", None, "p", "u", 2, 2.0, 2.0, 1.0),
        (2, "b|c", None, "q", "u", 2, 2.0, 2.0, 1.0),
    ]


def test_min_rows_stops_when_a_level_has_no_frequent_cell() -> None:
    df = pl.DataFrame({"a": ["1", "2", "3"], "b": ["x", "y", "z"], "n": [1, 1, 1], "d": [1] * 3})
    cube = ratio_cube(df, ["a", "b"], "n", "d", min_rows=2)
    assert cube.is_empty()
    assert cube.schema == empty_cube(("a", "b")).schema


def test_min_rows_prunes_cells_whose_subcell_is_infrequent() -> None:
    # Every (a, b) cell has one row, so no 2-cut cell is frequent although a and b are.
    df = pl.DataFrame(
        {"a": ["x", "x", "y", "y"], "b": ["p", "q", "p", "q"], "n": [1] * 4, "d": [1] * 4}
    )
    cube = ratio_cube(df, ["a", "b"], "n", "d", min_rows=2)
    assert set(cube["cut"]) == {"a", "b"}


def test_min_rows_keeps_null_cells_through_pruning() -> None:
    df = pl.DataFrame(
        {
            "a": pl.Series([None, None, "x"], dtype=pl.String),
            "b": ["p", "p", "p"],
            "n": [1, 1, 1],
            "d": [1, 1, 1],
        }
    )
    cube = ratio_cube(df, ["a", "b"], "n", "d", min_rows=2)
    assert cube.rows() == [
        (1, "a", None, None, 2, 2.0, 2.0, 1.0),
        (1, "b", None, "p", 3, 3.0, 3.0, 1.0),
        (2, "a|b", None, "p", 2, 2.0, 2.0, 1.0),
    ]


# --- zero denominators and nulls -----------------------------------------------------------


def test_zero_denominator_gives_null_ratio() -> None:
    df = pl.DataFrame({"a": ["x", "y"], "n": [3, 0], "d": [0, 0]})
    cube = ratio_cube(df, ["a"], "n", "d")
    assert cube.rows() == [
        (1, "a", "x", 1, 3.0, 0.0, None),
        (1, "a", "y", 1, 0.0, 0.0, None),
    ]


def test_zero_policy_counts_nulls_as_zero_but_rows_are_counted() -> None:
    df = pl.DataFrame(
        {
            "a": ["x", "x", "y"],
            "n": pl.Series([None, 2, None], dtype=pl.Int64),
            "d": pl.Series([None, 4, None], dtype=pl.Int64),
        }
    )
    cube = ratio_cube(df, ["a"], "n", "d", null_measures="zero")
    assert cube.rows() == [
        (1, "a", "x", 2, 2.0, 4.0, 0.5),
        (1, "a", "y", 1, 0.0, 0.0, None),
    ]


def test_error_policy_names_every_null_column_with_count_and_first_row() -> None:
    df = pl.DataFrame(
        {
            "a": ["x"] * 4,
            "n": pl.Series([1, None, 3, None], dtype=pl.Int64),
            "d": pl.Series([None, 2, 2, 2], dtype=pl.Int64),
        }
    )
    with pytest.raises(ValueError) as info:
        ratio_cube(df, ["a"], "n", "d")
    assert str(info.value).startswith("column 'n' is null in 2 rows (first at row 2); column 'd'")
    assert "is null in 1 row (first at row 1)" in str(info.value)


def test_drop_policy_excludes_incomplete_rows_and_logs_the_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    df = pl.DataFrame(
        {
            "a": ["x", "x", "y"],
            "n": pl.Series([None, 2, 5], dtype=pl.Int64),
            "d": pl.Series([3, None, 10], dtype=pl.Int64),
        }
    )
    cube = ratio_cube(df, ["a"], "n", "d", null_measures="drop")
    assert cube.rows() == [(1, "a", "y", 1, 5.0, 10.0, 0.5)]
    assert "2 rows have a null measure" in caplog.text


def test_complete_data_is_the_same_under_every_policy(caplog: pytest.LogCaptureFixture) -> None:
    cubes = [ratio_cube(tiny(), ["a", "b"], "num", "den", null_measures=p) for p in POLICIES]
    assert cubes[0].equals(cubes[1]) and cubes[0].equals(cubes[2])
    assert not caplog.records


@pytest.mark.parametrize("policy", POLICIES)
def test_null_policy_allows_one_column_as_both_measures_and_a_measure_named_row(
    policy: NullMeasures,
) -> None:
    df = pl.DataFrame({"a": ["x", "x", "y"], "row": [1, 3, 0]})
    cube = ratio_cube(df, ["a"], "row", "row", null_measures=policy)
    assert cube.rows() == [(1, "a", "x", 2, 4.0, 4.0, 1.0), (1, "a", "y", 1, 0.0, 0.0, None)]


def test_error_policy_reports_the_first_null_row_of_a_column_named_row() -> None:
    df = pl.DataFrame({"a": ["x"] * 3, "row": [1, None, 2], "d": [1, 1, 1]})
    with pytest.raises(ValueError, match=r"^column 'row' is null in 1 row \(first at row 2\);"):
        ratio_cube(df, ["a"], "row", "d")


def test_unknown_null_policy_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="null_measures"):
        ratio_cube(tiny(), ["a"], "num", "den", null_measures="ignore")  # type: ignore[arg-type]


def test_null_dimension_values_form_their_own_cell() -> None:
    cube = ratio_cube(tiny(), ["a", "b"], "num", "den")
    missing_a = cube.filter((pl.col("cut") == "a") & pl.col("a").is_null())
    assert missing_a.rows() == [(1, "a", None, None, 1, 4.0, 8.0, 0.5)]
    # In cut "b" the a column is null because a is not in the cut, not because it is missing.
    not_in_cut = cube.filter(cut="b", b="p")
    assert not_in_cut["a"].to_list() == [None]
    assert not_in_cut["rows"].to_list() == [3]
    # A null inside a multi-dimension cut is its own cell, told apart by the cut column.
    assert cube.filter((pl.col("cut") == "a|b") & pl.col("a").is_null()).rows() == [
        (2, "a|b", None, "p", 1, 4.0, 8.0, 0.5)
    ]


def test_all_null_dimension_column_is_a_single_missing_cell() -> None:
    df = pl.DataFrame({"a": pl.Series([None, None], dtype=pl.String), "n": [1, 2], "d": [2, 2]})
    cube = ratio_cube(df, ["a"], "n", "d")
    assert cube.rows() == [(1, "a", None, 2, 3.0, 4.0, 0.75)]


# --- input kinds and emptiness -------------------------------------------------------------


def test_lazyframe_and_dataframe_agree() -> None:
    eager = ratio_cube(tiny(), ["a", "b"], "num", "den")
    lazy = ratio_cube(tiny().lazy(), ["a", "b"], "num", "den")
    assert isinstance(lazy, pl.DataFrame)
    assert eager.equals(lazy)


def test_dimensions_may_be_a_tuple() -> None:
    assert ratio_cube(tiny(), ("a", "b"), "num", "den").equals(
        ratio_cube(tiny(), ["a", "b"], "num", "den")
    )


def test_empty_input_returns_empty_frame_with_schema() -> None:
    cube = ratio_cube(tiny().clear(), ["a", "b"], "num", "den")
    assert cube.is_empty()
    assert cube.schema == empty_cube(("a", "b")).schema
    assert cube.columns == [
        "cut_size",
        "cut",
        "a",
        "b",
        "rows",
        "numerator",
        "denominator",
        "ratio",
    ]


def test_empty_lazy_input_with_min_rows() -> None:
    cube = ratio_cube(tiny().clear().lazy(), ["a"], "num", "den", min_rows=3)
    assert cube.is_empty()
    assert cube.schema == empty_cube(("a",)).schema


# --- validation ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dimensions": []}, "at least one dimension"),
        ({"dimensions": ["a", "b", "a"]}, r"duplicate dimensions: \['a'\]"),
        ({"dimensions": ["a|b"]}, r"must not contain '\|': \['a\|b'\]"),
        ({"dimensions": ["a", "cut"]}, r"clash with output columns: \['cut'\]"),
        ({"dimensions": ["rows"]}, "clash with output columns"),
        ({"dimensions": ["ratio"]}, "clash with output columns"),
        ({"dimensions": ["cut_size"]}, "clash with output columns"),
        ({"dimensions": ["numerator"]}, "clash with output columns"),
        ({"dimensions": ["denominator"]}, "clash with output columns"),
        ({"dimensions": ["a", "zzz", "yyy"]}, r"columns not found in data: \['yyy', 'zzz'\]"),
        ({"numerator": "missing"}, r"columns not found in data: \['missing'\]"),
        ({"denominator": "missing"}, r"columns not found in data: \['missing'\]"),
        ({"numerator": "a"}, "column 'a' must be numeric, not String"),
        ({"denominator": "b"}, "column 'b' must be numeric, not String"),
        ({"numerator": "flag"}, "column 'flag' must be numeric, not Boolean"),
        ({"dimensions": ["a", "num"]}, "column 'num' cannot be both a measure and a dimension"),
        ({"dimensions": ["a", "den"]}, "column 'den' cannot be both a measure and a dimension"),
        ({"max_dims": 0}, "max_dims must be between 1 and 2, got 0"),
        ({"max_dims": 3}, "max_dims must be between 1 and 2, got 3"),
        ({"max_dims": -1}, "max_dims must be between 1 and 2, got -1"),
        ({"min_rows": 0}, "min_rows must be at least 1, got 0"),
        ({"min_rows": -5}, "min_rows must be at least 1, got -5"),
    ],
)
def test_validation_errors(kwargs: dict[str, Any], message: str) -> None:
    df = tiny().with_columns(flag=pl.lit(True))
    arguments: dict[str, Any] = {
        "dimensions": ["a", "b"],
        "numerator": "num",
        "denominator": "den",
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        ratio_cube(df, **arguments)


def test_validation_also_applies_to_empty_input() -> None:
    with pytest.raises(ValueError, match="not found"):
        ratio_cube(tiny().clear(), ["nope"], "num", "den")


# --- helpers -------------------------------------------------------------------------------

SCHEMA = pl.Schema({"a": pl.String, "b": pl.String, "n": pl.Int64, "d": pl.Float64})


def test_validate_inputs_returns_resolved_max_dims() -> None:
    assert validate_inputs(SCHEMA, ["a", "b"], "n", "d", None, 1) == 2
    assert validate_inputs(SCHEMA, ["a", "b"], "n", "d", 1, 5) == 1


def test_validate_inputs_rejects_bad_max_dims() -> None:
    with pytest.raises(ValueError, match="max_dims"):
        validate_inputs(SCHEMA, ["a", "b"], "n", "d", 3, 1)


def test_finest_grain_aggregates_distinct_combinations() -> None:
    df = pl.DataFrame(
        {
            "a": ["x", "x", "x", None],
            "b": ["p", "p", "q", "p"],
            "n": pl.Series([1, None, 3, 4], dtype=pl.Int64),
            "d": [2] * 4,
        }
    )
    result = finest_grain(df.lazy(), ("a", "b"), "n", "d", "zero").sort("a", "b", nulls_last=True)
    assert result.columns == ["a", "b", "rows", "numerator", "denominator"]
    assert result.rows() == [
        ("x", "p", 2, 1.0, 4.0),
        ("x", "q", 1, 3.0, 2.0),
        (None, "p", 1, 4.0, 2.0),
    ]
    assert result.schema["rows"] == pl.Int64
    assert result.schema["numerator"] == pl.Float64


def test_roll_up_sums_measures_over_cut_cells() -> None:
    source = pl.DataFrame(
        {
            "a": ["x", "x", "y"],
            "b": ["p", "q", "p"],
            "rows": [1, 2, 4],
            "numerator": [1.0, 2.0, 3.0],
            "denominator": [4.0, 5.0, 6.0],
        }
    )
    result = roll_up(source, ("a",)).sort("a")
    # the last column counts the source rows in each cell
    assert result.rows() == [("x", 3, 3.0, 9.0, 2), ("y", 4, 3.0, 6.0, 1)]
    assert roll_up(source, ("a", "b")).height == 3


def test_restrict_to_cells_keeps_rows_inside_the_frequent_cells() -> None:
    source = pl.DataFrame(
        {
            "a": pl.Series(["x", "x", "y", None], dtype=pl.String),
            "b": ["p", "q", "p", "p"],
            "rows": [1, 1, 1, 1],
            "numerator": [0.0] * 4,
            "denominator": [0.0] * 4,
        }
    )
    cells = pl.DataFrame({"a": pl.Series(["x", None], dtype=pl.String), "rows": [5, 5]})
    result = restrict_to_cells(source, ("a",), cells)
    # (y, p) is outside the frequent cells; the null a-cell matches the null key.
    kept = result.select("a", "b").sort("a", "b", nulls_last=True)
    assert kept.rows() == [("x", "p"), ("x", "q"), (None, "p")]


def abc_finest() -> pl.DataFrame:
    """a and c are frequent at min_rows=2 (4 rows each); b never repeats, so no cut with b is."""
    return pl.DataFrame(
        {
            "a": ["x"] * 4,
            "b": ["1", "2", "3", "4"],
            "c": ["z"] * 4,
            "rows": [1] * 4,
            "numerator": [1.0] * 4,
            "denominator": [2.0] * 4,
        }
    )


def test_walk_cuts_skips_a_cut_unless_every_subcut_is_frequent() -> None:
    assert walk_cuts(abc_finest(), ("a", "b", "c"), 3, 2).keys() == {(0,), (2,), (0, 2)}
    assert walk_cuts(abc_finest(), ("a", "b", "c"), 1, 2).keys() == {(0,), (2,)}


def test_a_dimension_may_share_a_name_with_the_internal_group_count() -> None:
    df = pl.DataFrame({"groups": ["x", "x", "y"], "n": [1, 0, 1], "d": [1, 1, 1]})
    cube = ratio_cube(df, ["groups"], "n", "d", min_rows=2)
    assert cube.columns[2:4] == ["groups", "rows"]
    assert cube.rows() == [(1, "groups", "x", 2, 1.0, 2.0, 0.5)]


def test_walk_cuts_hands_children_only_the_columns_their_subtree_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # c's only candidate child is (a, c): b is dropped from the source c hands down.
    read: dict[tuple[str, ...], list[str]] = {}

    def spy(source: pl.DataFrame, cut: tuple[str, ...]) -> pl.DataFrame:
        read[cut] = source.columns
        return roll_up(source, cut)

    monkeypatch.setattr("child_mortality.cube.roll_up", spy)
    walk_cuts(abc_finest(), ("a", "b", "c"), 3, 2)
    assert read[("a",)] == abc_finest().columns
    assert read[("a", "c")] == ["a", "c", "rows", "numerator", "denominator"]


def test_order_cells_sorts_ascending_with_nulls_last() -> None:
    cells = pl.DataFrame(
        {
            "a": pl.Series(["b", None, "a"], dtype=pl.String),
            "b": pl.Series(["z", "y", None], dtype=pl.String),
            "rows": [1, 2, 3],
        }
    )
    assert order_cells(cells, ("a",))["a"].to_list() == ["a", "b", None]
    assert order_cells(cells, ("b", "a"))["b"].to_list() == ["y", "z", None]


def test_format_cells_shapes_output_columns() -> None:
    cells = pl.DataFrame(
        {"b": ["p", "q"], "rows": [2, 1], "numerator": [3.0, 1.0], "denominator": [6.0, 0.0]}
    )
    result = format_cells(cells, ("b",), ("a", "b"))
    assert result.schema == empty_cube(("a", "b")).schema
    assert result.rows() == [
        (1, "b", None, "p", 2, 3.0, 6.0, 0.5),
        (1, "b", None, "q", 1, 1.0, 0.0, None),
    ]


def test_empty_cube_schema() -> None:
    result = empty_cube(("x", "y"))
    assert result.is_empty()
    assert result.schema == pl.Schema(
        {
            "cut_size": pl.Int64,
            "cut": pl.String,
            "x": pl.String,
            "y": pl.String,
            "rows": pl.Int64,
            "numerator": pl.Float64,
            "denominator": pl.Float64,
            "ratio": pl.Float64,
        }
    )


def test_oracle_ratio_tolerance_is_one_ulp_not_a_real_difference() -> None:
    exact = 5 / 3
    one_ulp_low = math.nextafter(exact, 0.0)  # what a reciprocal-multiply division can return
    assert ratio_matches(one_ulp_low, exact)
    assert not ratio_matches(exact * (1 + 1e-9), exact)
    assert ratio_matches(None, None)
    assert not ratio_matches(None, 0.0)
    assert not ratio_matches(0.0, None)

"""Property-based tests: ``ratio_cube`` against a naive oracle and its structural invariants."""

from math import comb
from typing import Any

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from child_mortality.cube import NullMeasures, ratio_cube
from conftest import assert_matches_oracle, naive_cube

PROFILE = settings(max_examples=60, deadline=None, derandomize=False)

DIM_NAMES = ["d0", "d1", "d2", "d3"]
VALUES = st.one_of(st.none(), st.sampled_from(["a", "b", "c", "10", "9"]))
MEASURE = st.one_of(st.none(), st.integers(min_value=0, max_value=4))


@st.composite
def frames(draw: st.DrawFn) -> tuple[pl.DataFrame, list[str]]:
    """A small frame (0-25 rows) with 1-4 dimensions, nullable String dims and Int measures."""
    n_dims = draw(st.integers(min_value=1, max_value=4))
    dims = DIM_NAMES[:n_dims]
    n_rows = draw(st.integers(min_value=0, max_value=25))
    columns: dict[str, Any] = {
        d: pl.Series(d, draw(st.lists(VALUES, min_size=n_rows, max_size=n_rows)), dtype=pl.String)
        for d in dims
    }
    for measure in ("num", "den"):
        values = draw(st.lists(MEASURE, min_size=n_rows, max_size=n_rows))
        columns[measure] = pl.Series(measure, values, dtype=pl.Int64)
    return pl.DataFrame(columns), dims


@PROFILE
@given(frames(), st.data())
def test_matches_naive_oracle(case: tuple[pl.DataFrame, list[str]], data: st.DataObject) -> None:
    frame, dims = case
    max_dims = data.draw(st.integers(min_value=1, max_value=len(dims)))
    min_rows = data.draw(st.integers(min_value=1, max_value=4))
    policy: NullMeasures = data.draw(st.sampled_from(["drop", "zero"]))
    actual = ratio_cube(
        frame, dims, "num", "den", max_dims=max_dims, min_rows=min_rows, null_measures=policy
    )
    expected = naive_cube(
        frame.to_dicts(),
        dims,
        "num",
        "den",
        max_dims=max_dims,
        min_rows=min_rows,
        null_measures=policy,
    )
    assert_matches_oracle(actual.rows(), expected)


@PROFILE
@given(frames())
def test_default_policy_raises_exactly_when_a_measure_is_null(
    case: tuple[pl.DataFrame, list[str]],
) -> None:
    frame, dims = case
    if frame["num"].has_nulls() or frame["den"].has_nulls():
        with pytest.raises(ValueError, match="null"):
            ratio_cube(frame, dims, "num", "den")
    else:
        actual = ratio_cube(frame, dims, "num", "den")
        assert_matches_oracle(actual.rows(), naive_cube(frame.to_dicts(), dims, "num", "den"))


@PROFILE
@given(frames(), st.integers(min_value=1, max_value=30))
def test_min_rows_is_a_filter_of_the_full_cube(
    case: tuple[pl.DataFrame, list[str]], min_rows: int
) -> None:
    frame, dims = case
    full = ratio_cube(frame, dims, "num", "den", null_measures="zero")
    pruned = ratio_cube(frame, dims, "num", "den", min_rows=min_rows, null_measures="zero")
    expected = full.filter(pl.col("rows") >= min_rows)
    assert pruned.schema == expected.schema
    # Same rows and order, every column exact except ``ratio``: polars 2.0.0 can divide one ulp
    # differently on a filtered frame than on a fresh one (the pruned path rolls up subsets).
    assert_matches_oracle(pruned.rows(), expected.rows())


@PROFILE
@given(frames())
def test_every_one_cut_sums_to_the_grand_totals(case: tuple[pl.DataFrame, list[str]]) -> None:
    frame, dims = case
    cube = ratio_cube(frame, dims, "num", "den", null_measures="zero")
    for dim in dims:  # nulls are a cell of their own, so null-free dimensions are covered too
        cells = cube.filter(pl.col("cut") == dim)
        assert cells["rows"].sum() == frame.height
        assert cells["numerator"].sum() == float(frame["num"].fill_null(0).sum())
        assert cells["denominator"].sum() == float(frame["den"].fill_null(0).sum())


@PROFILE
@given(frames(), st.data())
def test_number_of_cuts_is_sum_of_binomials(
    case: tuple[pl.DataFrame, list[str]], data: st.DataObject
) -> None:
    frame, dims = case
    max_dims = data.draw(st.integers(min_value=1, max_value=len(dims)))
    cube = ratio_cube(frame, dims, "num", "den", max_dims=max_dims, null_measures="zero")
    expected = sum(comb(len(dims), k) for k in range(1, max_dims + 1)) if frame.height else 0
    assert cube["cut"].n_unique() == expected
    assert cube["cut_size"].is_sorted()

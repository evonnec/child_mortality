"""End-to-end checks on the bundled survey data in ``data/``."""

import csv
from collections import defaultdict
from pathlib import Path

import polars as pl
import pytest

from child_mortality.cli import main
from child_mortality.survey import (
    DENOMINATOR,
    DIMENSIONS,
    JOIN_KEYS,
    NUMERATOR,
    cd_ceb_cube,
    join_women_households,
    load_households,
    load_women,
)
from conftest import DATA_DIR, naive_cube

WOMEN = DATA_DIR / "women.csv"
HOUSEHOLDS = DATA_DIR / "household.csv"
OVERALL_CD_CEB = 0.20189301107073437
AGE = "Age in 5-year groups"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def count(value: str) -> int:
    return 0 if value in ("NA", "") else int(value)


@pytest.fixture(scope="module")
def joined() -> pl.DataFrame:
    return join_women_households(load_women(WOMEN), load_households(HOUSEHOLDS)).collect()


@pytest.fixture(scope="module")
def cube3() -> pl.DataFrame:
    return cd_ceb_cube(WOMEN, HOUSEHOLDS, max_dims=3)


def test_row_counts() -> None:
    assert load_women(WOMEN).collect().height == 8348
    assert load_households(HOUSEHOLDS).collect().height == 6290
    assert len(read_rows(WOMEN)) == 8348
    assert len(read_rows(HOUSEHOLDS)) == 6290


def test_index_column_is_dropped() -> None:
    assert "" not in load_women(WOMEN).collect_schema().names()
    assert "" not in load_households(HOUSEHOLDS).collect_schema().names()


def test_household_keys_are_unique() -> None:
    keys = {tuple(row[k] for k in JOIN_KEYS) for row in read_rows(HOUSEHOLDS)}
    assert len(keys) == 6290


def test_every_woman_matches_a_household(joined: pl.DataFrame) -> None:
    household_keys = {tuple(row[k] for k in JOIN_KEYS) for row in read_rows(HOUSEHOLDS)}
    women_keys = {tuple(row[k] for k in JOIN_KEYS) for row in read_rows(WOMEN)}
    assert women_keys <= household_keys
    assert joined.height == 8348
    assert joined["Type of place of residence"].null_count() == 0


def test_documented_missing_value_counts() -> None:
    women = read_rows(WOMEN)
    assert sum(1 for r in women if r[DENOMINATOR] == "0") == 2187
    households = read_rows(HOUSEHOLDS)
    water = "Time to get to water source (minutes)"
    assert sum(1 for r in households if r[water] == "") == 121
    assert sum(1 for r in households if r[water] == "NA") == 9
    assert load_households(HOUSEHOLDS).collect()[water].null_count() == 130


def test_overall_cd_ceb(cube3: pl.DataFrame) -> None:
    for dim in DIMENSIONS:
        cells = cube3.filter(pl.col("cut") == dim)
        assert cells["rows"].sum() == 8348
        overall = cells.select(pl.col("numerator").sum() / pl.col("denominator").sum()).item()
        assert overall == OVERALL_CD_CEB


def test_overall_cd_ceb_by_independent_computation() -> None:
    women = read_rows(WOMEN)
    died = sum(count(r["Sons who have died"]) + count(r["Daughters who have died"]) for r in women)
    born = sum(count(r[DENOMINATOR]) for r in women)
    assert died / born == OVERALL_CD_CEB


def test_cd_ceb_by_age_group_matches_stdlib_csv(cube3: pl.DataFrame) -> None:
    totals: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for row in read_rows(WOMEN):
        acc = totals[row[AGE]]
        acc[0] += 1
        acc[1] += count(row["Sons who have died"]) + count(row["Daughters who have died"])
        acc[2] += count(row[DENOMINATOR])
    cells = cube3.filter(pl.col("cut") == AGE)
    assert cells.height == len(totals)
    for cell in cells.iter_rows(named=True):
        rows, died, born = totals[cell[AGE]]
        assert cell["rows"] == rows
        assert cell["numerator"] == died
        assert cell["denominator"] == born
        assert cell["ratio"] == (died / born if born else None)


def test_max_dims_3_equals_oracle(joined: pl.DataFrame, cube3: pl.DataFrame) -> None:
    expected = naive_cube(joined.to_dicts(), DIMENSIONS, NUMERATOR, DENOMINATOR, max_dims=3)
    assert cube3.rows() == expected


def test_min_rows_invariant_on_real_data(cube3: pl.DataFrame) -> None:
    for threshold in (2, 30, 200):
        pruned = cd_ceb_cube(WOMEN, HOUSEHOLDS, max_dims=3, min_rows=threshold)
        assert pruned.equals(cube3.filter(pl.col("rows") >= threshold)), threshold
    assert cd_ceb_cube(WOMEN, HOUSEHOLDS, max_dims=3, min_rows=8349).is_empty()


def test_na_household_values_are_missing_cells(cube3: pl.DataFrame) -> None:
    toilet = cube3.filter(pl.col("cut") == "Type of toilet facility")
    assert toilet["Type of toilet facility"].null_count() <= 1
    assert toilet["rows"].sum() == 8348


def test_problem_1_is_every_subset_of_the_first_k_dimensions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Problem 1 for k = 3: 2**3 - 1 = 7 cuts and 555 cells (not ``--max-dims 3``)."""
    k = 3
    cube = cd_ceb_cube(WOMEN, HOUSEHOLDS, dimensions=DIMENSIONS[:k])
    assert cube["cut"].n_unique() == 2**k - 1 == 7
    assert cube.height == 555
    assert cube.columns[2 : 2 + k] == list(DIMENSIONS[:k])

    # The same computation from the command line, as documented in the README.
    argv = ["cd-ceb", "--women", str(WOMEN), "--households", str(HOUSEHOLDS)]
    for dimension in DIMENSIONS[:k]:
        argv += ["-d", dimension]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert out == cube.write_csv()


def test_max_dims_caps_cut_size_over_all_listed_dimensions(cube3: pl.DataFrame) -> None:
    """``--max-dims 3`` over all nine dimensions is a different, much larger computation."""
    assert cube3["cut"].n_unique() == 129
    assert cube3.height == 75_626

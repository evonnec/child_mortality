# child-mortality

Ratio metrics over every dimensional cut of a table, with pruning for the thresholded case.
Includes a CLI and a survey application: under-5 mortality (CD/CEB) from household survey data.

## What it is

Input: a table with several categorical dimension columns and two numeric measure columns. For
any group of rows the metric is

    ratio = sum(numerator) / sum(denominator)

A *cut* is a subset of the dimensions; a *cell* is one combination of values within a cut. Two
problems are solved:

1. **Full cube.** For the first `k` dimensions, compute the ratio for every cell of every cut of
   size 1..`k` (every non-empty subset of those `k` dimensions, every value combination that
   occurs in the data). There is no grand-total row. Pass only those `k` dimensions and leave
   `--max-dims` unset. On the survey data with k = 3 (`Age in 5-year groups`,
   `Type of place of residence`, `Number of household members`) this gives 7 cuts and 555 cells:

   ```sh
   uv run child-mortality cd-ceb --women data/women.csv --households data/household.csv \
       -d "Age in 5-year groups" -d "Type of place of residence" \
       -d "Number of household members" -o problem1.csv
   ```

   ```python
   cube = ratio_cube(df, dims[:3], numerator, denominator)
   # survey: cd_ceb_cube(women, households, dimensions=DIMENSIONS[:3])
   ```

   `--max-dims` is not the same thing: it caps the cut size over *all* listed dimensions.
   `cd-ceb --max-dims 3` with the default nine dimensions is 129 cuts and 75,626 cells.
2. **Thresholded ("iceberg") cube.** With many dimensions the number of cuts and cells explodes,
   so return only cells that contain at least `min_rows` input rows, and avoid computing the rest.
   A finer cell never has more rows than a coarser cell that contains it, so whole branches can
   be skipped.

Application: where birth and death registries are incomplete, household surveys of women give a
proxy for under-5 mortality, CD/CEB = children died / children ever born, usually reported by the
mother's 5-year age group. `cd_ceb_cube` joins the women's and household files and cuts the ratio
by nine dimensions:

`Age in 5-year groups`, `Type of place of residence`, `Number of household members`,
`Source of drinking water`, `Time to get to water source (minutes)`, `Type of toilet facility`,
`Has electricity`, `Has radio`, `Age of head of household`.

    CD/CEB = sum(Sons who have died + Daughters who have died) / sum(Total children ever born)

## Install and quickstart

Requires Python 3.15 and [uv](https://docs.astral.sh/uv/). The only runtime dependency is polars (2.0 or newer; CI tests that floor as well as the locked version).

```sh
uv sync
uv run child-mortality cd-ceb \
    --women data/women.csv --households data/household.csv --max-dims 2 -o out.csv
```

Any CSV works with `ratio-cube`. `-d` is repeatable; `--numerator` and `--denominator` name the
measure columns; every other column is read as a string, and `NA` or empty values are null:

```sh
uv run child-mortality ratio-cube sales.csv -d region -d channel -d tier \
    --numerator refunds --denominator orders --max-dims 2 --min-rows 50 -o cube.parquet
```

CLI summary (`child-mortality --help` for all flags):

| Option | Meaning |
| --- | --- |
| `-d/--dimension` | Dimension column, repeatable. Required for `ratio-cube`; `cd-ceb` defaults to the nine survey dimensions. |
| `--max-dims N` | Largest cut size, over all listed dimensions (default: all of them). Not "the first N dimensions"; to cut by the first N, list only those with `-d`. |
| `--min-rows N` | Keep only cells with at least N rows (default 1). |
| `-o/--output` | `.csv` or `.parquet` by suffix; omitted: CSV on stdout. |

Errors (bad arguments, missing files, invalid data) print one line, `child-mortality: error: ...`,
to stderr and exit with code 2. `python -m child_mortality` is equivalent to the console script.

### Library

```python
from child_mortality import cd_ceb_cube, ratio_cube

# Survey application (polars DataFrame)
cube = cd_ceb_cube("data/women.csv", "data/household.csv", max_dims=2, min_rows=30)

# Any polars DataFrame or LazyFrame
cube = ratio_cube(
    df, ["region", "channel"], numerator="refunds", denominator="orders", max_dims=2, min_rows=50
)
```

`ratio_cube(data, dimensions, numerator, denominator, *, max_dims=None, min_rows=1)` raises
`ValueError` for an empty or duplicated dimension list, a dimension name containing `|` or equal
to an output column name, missing columns, a non-numeric measure, a measure that is also a
dimension, `max_dims` outside `1..len(dimensions)`, or `min_rows < 1`.

## Output schema

One row per cell. Columns, in this order:

| Column | Type | Meaning |
| --- | --- | --- |
| `cut_size` | Int64 | Number of dimensions in the cut. |
| `cut` | String | The cut's dimension names joined with `\|`, in the order given in `dimensions`. |
| one column per dimension | String | The cell's value (cast to String); null when the dimension is not in the cut. A null for a dimension that *is* in the cut means a missing value; `cut` disambiguates. |
| `rows` | Int64 | Input rows in the cell. |
| `numerator` | Float64 | Sum of the numerator (null inputs count as 0). |
| `denominator` | Float64 | Sum of the denominator (null inputs count as 0). |
| `ratio` | Float64 | `numerator / denominator`; null when the denominator is 0. |

Ordering is deterministic: by `cut_size`, then cut in `itertools.combinations(dimensions, k)`
order, then the cell's values ascending as strings with nulls last. An empty input, or
`min_rows` larger than the table, returns an empty frame with this schema.
`ratio_cube(df, ..., min_rows=m)` equals `ratio_cube(df, ..., min_rows=1)` filtered to
`rows >= m`, including row order.

## How it works

1. **Finest-grain pre-aggregation.** One pass over the data groups by all dimensions and keeps
   `rows`, `sum(numerator)`, `sum(denominator)`. Every cut is then rolled up from this table, not
   from the raw rows. Sums are additive, so roll-ups are exact.
2. **Roll-ups.** For each cut, group the finest-grain table by the cut's columns and sum the
   three measures.
3. **Anti-monotone pruning (`min_rows > 1`).** Cuts are computed level by level (size 1, 2, ...).
   For a k-cut, the finest-grain table is semi-joined against the frequent cells of each of its
   (k-1)-sub-cuts (nulls compare equal, so a missing value is a cell of its own), so only
   finest-grain groups all of whose sub-cells reached `min_rows` are rolled up. Further shortcuts:
   a cut is skipped when any of its (k-1)-sub-cuts has no frequent cell; the level loop stops as
   soon as a level has no frequent cell; and if `min_rows` exceeds the total row count nothing is
   computed. Size-1 cuts are never restricted. With `min_rows == 1` no pruning is done.

Complexity notes: with `n` dimensions and `max_dims = m` there are `sum(C(n, k) for k in 1..m)`
cuts (511 for n = 9, m = 9). The number of cells in a cut is bounded by the number of distinct
value combinations of its dimensions, and the total work is bounded by cuts times the size of the
finest-grain table, which is at most the number of input rows and in practice much smaller for
low-cardinality dimensions. Pruning shrinks the table each cut reads as `min_rows` grows.

## Results on the survey data

CD/CEB by `Age in 5-year groups` (`--max-dims 1`, unweighted):

| Age group | Women | Children died | Children ever born | CD/CEB |
| --- | ---: | ---: | ---: | ---: |
| 15-19 | 2,041 | 73 | 537 | 0.1359 |
| 20-24 | 1,371 | 243 | 1,725 | 0.1409 |
| 25-29 | 1,357 | 567 | 3,517 | 0.1612 |
| 30-34 | 1,110 | 801 | 4,389 | 0.1825 |
| 35-39 | 1,108 | 1,177 | 5,490 | 0.2144 |
| 40-44 | 644 | 794 | 3,589 | 0.2212 |
| 45-49 | 717 | 1,123 | 4,419 | 0.2541 |

Overall CD/CEB over all 8,348 women: 4,778 / 23,666 = **0.2019**. Older mothers have higher
CD/CEB because their children have had longer exposure to risk.

Cost of the full nine-dimension cube (511 cuts), measured in-process with `cd_ceb_cube` (CSV
reading, join and cube; excludes writing output), polars on a laptop, three runs each:

| Run | Output rows | Time | Peak RSS |
| --- | ---: | ---: | ---: |
| all 9 dims, `min_rows=1` | 1,546,271 | 0.6 s | about 415 MB |
| all 9 dims, `--min-rows 30` | 11,706 | 0.5 s | about 135 MB |

With `--min-rows 30` the largest surviving cut has 6 dimensions (cells per cut size 1..6: 135,
1,517, 4,212, 4,192, 1,526, 124). Absolute numbers depend on the machine, core count and polars
version; read the ratios (output size, memory) rather than the seconds.

## Data

| File | Rows | Notes |
| --- | ---: | --- |
| `data/women.csv` | 8,348 | One row per woman (23 columns). |
| `data/household.csv` | 6,290 | One row per household. |
| `data/subset_women.csv` | 4 | Small sample of the women's file. |

These are the DHS Program model dataset (a public teaching dataset with a synthetic country code,
"ZZ"). Women join to households on `Cluster number` + `Household number`; the key is unique in
the household file (checked at load; duplicates raise `ValueError`), and every woman matches. The
join is a left join, so unmatched women would be kept with null household columns.

Both files have an unnamed leading pandas index column, which is dropped on load. The literal
`NA` and empty fields are read as null. A null dimension value is its own cell (for example,
`Time to get to water source (minutes)` is null for 161 women: 148 empty and 13 `NA`). Null
counts contribute 0 to the sums. 2,187 women have zero children ever born; they
count in `rows` and add 0 to both sums.

Assumptions and caveats:

- Sums are unweighted; the women's sample weight column is not applied, so results describe the
  sample, not the population.
- CD/CEB is a crude proxy: it mixes children of different ages and exposure, and it is not a
  direct under-5 mortality rate.

## Development

```sh
uv sync
uv run pytest
uv run ruff check
uv run ruff format --check
uv run mypy src tests
```

The code targets Python 3.15 and uses: PEP 810 explicit lazy imports (`lazy import polars as
pl`), PEP 695 `type` aliases, PEP 758 bracketless `except A, B:`, and PEP 649 deferred annotation
evaluation (no `from __future__ import annotations`).

```
src/child_mortality/
    cube.py       ratio_cube: validation, pre-aggregation, roll-ups, pruning
    survey.py     CSV loaders, women-household join, cd_ceb_cube, survey constants
    cli.py        argparse CLI (cd-ceb, ratio-cube)
    __main__.py   python -m child_mortality
tests/            pytest suite
data/             DHS model dataset CSVs
docs/             RELEASING.md
.github/workflows/  ci.yml, release.yml
```

## CI and releases

CI (`.github/workflows/ci.yml`) runs lint, format, type checks, tests and a package build on
pushes and pull requests. Releases are cut from semver tags (`vX.Y.Z`): `release.yml` verifies
the tag matches the project version, re-runs checks and tests, builds, and creates a GitHub
Release. Publishing to PyPI is opt-in (repository variable `PUBLISH_TO_PYPI`). See
[docs/RELEASING.md](docs/RELEASING.md).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).

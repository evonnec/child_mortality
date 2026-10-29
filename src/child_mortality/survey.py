"""Under-5 mortality proxy from household surveys: CD/CEB = children died / children ever born."""

import os
from collections.abc import Sequence

lazy import polars as pl

from child_mortality.cube import NullMeasures, ratio_cube

type StrPath = str | os.PathLike[str]

JOIN_KEYS: tuple[str, str] = ("Cluster number", "Household number")
NUMERATOR = "Children died"  # derived: Sons who have died + Daughters who have died
DENOMINATOR = "Total children ever born"
DIMENSIONS: tuple[str, ...] = (
    "Age in 5-year groups",
    "Type of place of residence",
    "Number of household members",
    "Source of drinking water",
    "Time to get to water source (minutes)",
    "Type of toilet facility",
    "Has electricity",
    "Has radio",
    "Age of head of household",
)
NA_VALUES: tuple[str, ...] = ("NA", "")

_DEATH_COLUMNS = ("Sons who have died", "Daughters who have died")
_WOMEN_COUNTS = (DENOMINATOR, *_DEATH_COLUMNS)


def _scan(path: StrPath, counts: Sequence[str]) -> pl.LazyFrame:
    """Read a survey CSV: String columns, ``counts`` as Int64, NA as null, index column dropped."""
    frame = pl.scan_csv(os.fspath(path), infer_schema=False, null_values=list(NA_VALUES))
    frame = frame.select(pl.exclude(""))  # unnamed leading pandas index
    return frame.with_columns(pl.col(counts).cast(pl.Int64))


def load_women(path: StrPath) -> pl.LazyFrame:
    """Lazily read the women's file; the three count columns are Int64, the rest String."""
    return _scan(path, _WOMEN_COUNTS)


def load_households(path: StrPath) -> pl.LazyFrame:
    """Lazily read the household file; every column is String."""
    return _scan(path, ())


def join_women_households(
    women: pl.LazyFrame, households: pl.LazyFrame, *, unknown_deaths_are_zero: bool = False
) -> pl.LazyFrame:
    """Left-join household columns onto each woman and add the ``Children died`` column.

    ``Children died`` is sons plus daughters who died, and null if either count is null. With
    ``unknown_deaths_are_zero`` each null death count is 0 instead (the ``"zero"`` policy).

    Raises:
        ValueError: if the household keys are not unique.
    """
    duplicated = households.group_by(JOIN_KEYS).len().filter(pl.col("len") > 1).head(1).collect()
    if not duplicated.is_empty():
        raise ValueError(f"household keys {JOIN_KEYS} are not unique")
    joined = women.join(households, on=list(JOIN_KEYS), how="left", suffix="_household")
    sons, daughters = _DEATH_COLUMNS
    died = (
        pl.sum_horizontal(sons, daughters)
        if unknown_deaths_are_zero
        else pl.col(sons) + pl.col(daughters)  # one unknown count makes the total unknown
    )
    return joined.with_columns(died.alias(NUMERATOR))


def cd_ceb_cube(
    women_path: StrPath,
    households_path: StrPath,
    *,
    dimensions: Sequence[str] = DIMENSIONS,
    max_dims: int | None = None,
    min_rows: int = 1,
    null_measures: NullMeasures = "error",
) -> pl.DataFrame:
    """Children died / children ever born for every cut of the survey dimensions.

    ``null_measures`` is the policy for null counts, as in ``ratio_cube``; under ``"zero"`` a null
    death count is 0 on its own, so one unknown count does not null ``Children died``.
    """
    joined = join_women_households(
        load_women(women_path),
        load_households(households_path),
        unknown_deaths_are_zero=null_measures == "zero",
    )
    return ratio_cube(
        joined,
        dimensions,
        NUMERATOR,
        DENOMINATOR,
        max_dims=max_dims,
        min_rows=min_rows,
        null_measures=null_measures,
    )

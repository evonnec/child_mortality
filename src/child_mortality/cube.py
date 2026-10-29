"""Ratio metrics over every dimensional cut of a table, with iceberg pruning."""

from collections.abc import Sequence
from itertools import combinations

lazy import polars as pl

# A cut is a subset of the dimensions, kept in the caller's dimension order.
type Cut = tuple[str, ...]

CUT_SEPARATOR = "|"
_MEASURES = ("rows", "numerator", "denominator")
# Output column names a dimension may not take.
_RESERVED = ("cut_size", "cut", *_MEASURES, "ratio")


def ratio_cube(
    data: pl.LazyFrame | pl.DataFrame,
    dimensions: Sequence[str],
    numerator: str,
    denominator: str,
    *,
    max_dims: int | None = None,
    min_rows: int = 1,
) -> pl.DataFrame:
    """Compute ``sum(numerator) / sum(denominator)`` for every cell of every cut.

    A cut is a subset of 1..``max_dims`` of ``dimensions``; its cells are the value combinations
    that occur in ``data``. Only cells with at least ``min_rows`` input rows are returned.

    The result has columns ``cut_size``, ``cut`` (names joined with ``"|"``), one String column
    per dimension (null when the dimension is not in the cut; a null inside the cut means a
    missing value), ``rows``, ``numerator``, ``denominator`` and ``ratio`` (null when the
    denominator is 0). Rows are ordered by cut size, then cut, then cell values (nulls last).

    Raises:
        ValueError: if the arguments or the columns of ``data`` are invalid.
    """
    lazy = data.lazy()
    depth = validate_inputs(
        lazy.collect_schema(), dimensions, numerator, denominator, max_dims, min_rows
    )
    dims = tuple(dimensions)
    finest = finest_grain(lazy, dims, numerator, denominator)
    # min_rows above the table size can never be met: skip every cut.
    if finest.is_empty() or min_rows > finest["rows"].sum():
        return empty_cube(dims)

    parts: list[pl.DataFrame] = []
    frequent: dict[Cut, pl.DataFrame] = {}  # previous level: cut -> keys of its frequent cells
    for size in range(1, depth + 1):
        current: dict[Cut, pl.DataFrame] = {}
        for cut in combinations(dims, size):
            source = finest if min_rows == 1 else restrict_to_frequent(finest, cut, frequent)
            if source is None:
                continue
            cells = roll_up(source, cut).filter(pl.col("rows") >= min_rows)
            if cells.is_empty():
                continue
            parts.append(format_cells(order_cells(cells, cut), cut, dims))
            current[cut] = cells.select(cut)
        frequent = current
        if min_rows > 1 and not frequent:
            break  # no frequent cell at this level, so none at any deeper level
    return pl.concat(parts) if parts else empty_cube(dims)


def validate_inputs(
    schema: pl.Schema,
    dimensions: Sequence[str],
    numerator: str,
    denominator: str,
    max_dims: int | None,
    min_rows: int,
) -> int:
    """Check the arguments against ``schema``; return the resolved ``max_dims``."""
    if not dimensions:
        raise ValueError("at least one dimension is required")
    if len(set(dimensions)) != len(dimensions):
        duplicated = sorted({d for d in dimensions if list(dimensions).count(d) > 1})
        raise ValueError(f"duplicate dimensions: {duplicated}")
    if bad := [d for d in dimensions if CUT_SEPARATOR in d]:
        raise ValueError(f"dimension names must not contain {CUT_SEPARATOR!r}: {bad}")
    if clash := [d for d in dimensions if d in _RESERVED]:
        raise ValueError(f"dimension names clash with output columns: {clash}")
    if missing := [c for c in (*dimensions, numerator, denominator) if c not in schema]:
        raise ValueError(f"columns not found in data: {sorted(set(missing))}")
    for name in (numerator, denominator):
        if not schema[name].is_numeric():
            raise ValueError(f"column {name!r} must be numeric, not {schema[name]}")
        if name in dimensions:
            raise ValueError(f"column {name!r} cannot be both a measure and a dimension")
    depth = len(dimensions) if max_dims is None else max_dims
    if not 1 <= depth <= len(dimensions):
        raise ValueError(f"max_dims must be between 1 and {len(dimensions)}, got {max_dims}")
    if min_rows < 1:
        raise ValueError(f"min_rows must be at least 1, got {min_rows}")
    return depth


def finest_grain(data: pl.LazyFrame, dims: Cut, numerator: str, denominator: str) -> pl.DataFrame:
    """Aggregate once to every distinct dimension combination: rows, numerator, denominator."""
    return (
        data.select(
            pl.col(dims).cast(pl.String),
            pl.col(numerator).fill_null(0).cast(pl.Float64).alias("numerator"),
            pl.col(denominator).fill_null(0).cast(pl.Float64).alias("denominator"),
        )
        .group_by(dims)
        .agg(
            pl.len().cast(pl.Int64).alias("rows"),
            pl.col("numerator").sum(),
            pl.col("denominator").sum(),
        )
        .collect()
    )


def roll_up(source: pl.DataFrame, cut: Cut) -> pl.DataFrame:
    """Sum the measures of ``source`` (finest grain or a subset of it) over the cut's cells."""
    return source.group_by(cut).agg(pl.col(*_MEASURES).sum())


def restrict_to_frequent(
    finest: pl.DataFrame, cut: Cut, frequent: dict[Cut, pl.DataFrame]
) -> pl.DataFrame | None:
    """Keep finest-grain groups whose every (k-1)-sub-cell is frequent (iceberg pruning).

    Row counts only shrink as a cut gets finer, so a cell of ``cut`` can reach ``min_rows`` only
    if each of its sub-cells did. Returns None when some sub-cut has no frequent cell at all,
    which rules out the whole cut. Size-1 cuts have no sub-cuts and are not restricted.
    """
    if len(cut) == 1:
        return finest
    source = finest
    for sub_cut in combinations(cut, len(cut) - 1):
        keys = frequent.get(sub_cut)
        if keys is None:
            return None
        # nulls_equal: a missing value is a cell of its own, not an unmatched key
        source = source.join(keys, on=sub_cut, how="semi", nulls_equal=True)
    return source


def order_cells(cells: pl.DataFrame, cut: Cut) -> pl.DataFrame:
    """Sort a cut's cells by their values ascending, nulls last."""
    return cells.sort(cut, nulls_last=True)


def format_cells(cells: pl.DataFrame, cut: Cut, dims: Cut) -> pl.DataFrame:
    """Shape one cut's cells into the output schema."""
    return cells.select(
        pl.lit(len(cut), dtype=pl.Int64).alias("cut_size"),
        pl.lit(CUT_SEPARATOR.join(cut)).alias("cut"),
        *[pl.col(d) if d in cut else pl.lit(None, dtype=pl.String).alias(d) for d in dims],
        pl.col("rows"),
        pl.col("numerator"),
        pl.col("denominator"),
        pl.when(pl.col("denominator") != 0)
        .then(pl.col("numerator") / pl.col("denominator"))
        .otherwise(None)
        .alias("ratio"),
    )


def empty_cube(dims: Cut) -> pl.DataFrame:
    """An empty frame with the output schema."""
    return pl.DataFrame(
        schema={
            "cut_size": pl.Int64,
            "cut": pl.String,
            **dict.fromkeys(dims, pl.String),
            "rows": pl.Int64,
            "numerator": pl.Float64,
            "denominator": pl.Float64,
            "ratio": pl.Float64,
        }
    )

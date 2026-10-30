"""Ratio metrics over every dimensional cut of a table, with iceberg pruning."""

import logging
from collections.abc import Sequence
from typing import Literal, get_args

lazy import polars as pl

# A cut is a subset of the dimensions, kept in the caller's dimension order.
type Cut = tuple[str, ...]
# What a null numerator or denominator means; see ``ratio_cube``.
type NullMeasures = Literal["error", "drop", "zero"]

_NULL_POLICIES = get_args(NullMeasures.__value__)

logger = logging.getLogger(__name__)

# A cut as ascending indices into the dimension list.
type Node = tuple[int, ...]

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
    null_measures: NullMeasures = "error",
) -> pl.DataFrame:
    """Compute ``sum(numerator) / sum(denominator)`` for every cell of every cut.

    A cut is a subset of 1..``max_dims`` of ``dimensions``; its cells are the value combinations
    that occur in ``data``. Only cells with at least ``min_rows`` input rows are returned.

    The result has columns ``cut_size``, ``cut`` (names joined with ``"|"``), one String column
    per dimension (null when the dimension is not in the cut; a null inside the cut means a
    missing value), ``rows``, ``numerator``, ``denominator`` and ``ratio`` (null when the
    denominator is 0). Rows are ordered by cut size, then cut, then cell values (nulls last).

    ``null_measures`` says what a null numerator or denominator means. ``"error"`` (default)
    raises. ``"drop"`` leaves the whole row out of ``rows``, ``numerator`` and ``denominator``
    (complete-case). ``"zero"`` counts each null as 0 on its own side. For ``"drop"`` and
    ``"zero"`` the number of affected rows is logged at WARNING.

    Raises:
        ValueError: if the arguments or the columns of ``data`` are invalid.
    """
    lazy = data.lazy()
    depth = validate_inputs(
        lazy.collect_schema(), dimensions, numerator, denominator, max_dims, min_rows
    )
    dims = tuple(dimensions)
    finest = finest_grain(lazy, dims, numerator, denominator, null_measures)
    # min_rows above the table size can never be met: skip every cut.
    if finest.is_empty() or min_rows > finest["rows"].sum():
        return empty_cube(dims)

    parts = walk_cuts(finest, dims, depth, min_rows)
    if not parts:
        return empty_cube(dims)
    # Documented order: cut size, then itertools.combinations(dims, k) order (ascending indices).
    return pl.concat([parts[node] for node in sorted(parts, key=lambda n: (len(n), n))])


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


def apply_null_policy(
    data: pl.LazyFrame, measures: Cut, null_measures: NullMeasures
) -> pl.LazyFrame:
    """Apply ``null_measures`` to the measure columns: raise, drop incomplete rows or fill 0.

    The error names each measure with nulls, its null count and its first null row, a 1-based
    position in ``data`` (for a CSV, the data row, not counting the header).

    Raises:
        ValueError: if the policy is unknown, or it is ``"error"`` and a measure has nulls.
    """
    if null_measures not in _NULL_POLICIES:
        raise ValueError(f"null_measures must be one of {_NULL_POLICIES}, got {null_measures!r}")
    measures = tuple(dict.fromkeys(measures))  # the numerator may also be the denominator
    # No row-index column, so no clash with a measure of that name; arg_true is 0-based.
    stats = (
        data.select(
            *[pl.col(m).is_null().sum().alias(f"{m}.count") for m in measures],
            *[(pl.col(m).is_null().arg_true().min() + 1).alias(f"{m}.first") for m in measures],
            pl.any_horizontal(pl.col(*measures).is_null()).sum().alias("affected"),
        )
        .collect()
        .row(0, named=True)
    )
    if not stats["affected"]:
        return data
    if null_measures == "error":
        found = "; ".join(
            f"column {m!r} is null in {_rows(stats[f'{m}.count'])}"
            f" (first at row {stats[f'{m}.first']})"
            for m in measures
            if stats[f"{m}.count"]
        )
        raise ValueError(
            f"{found}; set null_measures (CLI: --null-measures) to 'drop' or 'zero' to proceed"
        )
    logger.warning(
        "null_measures=%r: %d rows have a null measure", null_measures, stats["affected"]
    )
    if null_measures == "drop":
        return data.drop_nulls(list(measures))
    return data.with_columns(pl.col(*measures).fill_null(0))


def _rows(count: int) -> str:
    """``"1 row"`` or ``"N rows"``."""
    return f"{count} row" if count == 1 else f"{count} rows"


def finest_grain(
    data: pl.LazyFrame, dims: Cut, numerator: str, denominator: str, null_measures: NullMeasures
) -> pl.DataFrame:
    """Aggregate once to every distinct dimension combination: rows, numerator, denominator.

    Nulls in the two measures are handled first, by the ``null_measures`` policy.
    """
    data = apply_null_policy(data, (numerator, denominator), null_measures)
    return (
        data.select(
            pl.col(dims).cast(pl.String),
            pl.col(numerator).cast(pl.Float64).alias("numerator"),
            pl.col(denominator).cast(pl.Float64).alias("denominator"),
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


def restrict_to_cells(source: pl.DataFrame, cut: Cut, cells: pl.DataFrame) -> pl.DataFrame:
    """Keep the rows of ``source`` inside the cut's frequent ``cells`` (iceberg pruning).

    A finer cell can reach ``min_rows`` only inside a frequent cell of each of its sub-cuts, so
    everything outside them is dead weight for every cut below this one in the walk.
    """
    # nulls_equal: a missing value is a cell of its own, not an unmatched key
    return source.join(cells.select(cut), on=cut, how="semi", nulls_equal=True)


def walk_cuts(
    finest: pl.DataFrame, dims: Cut, depth: int, min_rows: int
) -> dict[Node, pl.DataFrame]:
    """Format the frequent cells of every cut, keyed by the cut's dimension indices.

    Depth-first walk of the set-enumeration tree: a node's children add one dimension with a
    smaller index than the node's smallest. Its preorder is colexicographic, so every subset of a
    cut is visited before the cut, and a cut is skipped unless all its (k-1)-sub-cuts have a
    frequent cell (row counts only shrink as a cut gets finer). Candidates are only the children
    of frequent cuts, never all C(n, k) combinations.

    A frequent cut with at least one candidate child hands its children a source restricted to
    its own frequent cells and to the columns its subtree can read, so roll-ups read less data
    with depth. At most ``depth`` restricted sources are alive at once. With ``min_rows == 1``
    every cell is frequent and no rows are dropped.
    """
    parts: dict[Node, pl.DataFrame] = {}

    def visit(source: pl.DataFrame, parent: Node, extensions: Sequence[int]) -> None:
        for j in extensions:
            node = (j, *parent)
            cut = tuple(dims[i] for i in node)
            cells = roll_up(source, cut).filter(pl.col("rows") >= min_rows)
            if cells.is_empty():
                continue
            parts[node] = format_cells(order_cells(cells, cut), cut, dims)
            if len(node) == depth:
                continue
            # A child adds i < j. Its sub-cuts other than this node all precede the node in
            # colexicographic order, so they are decided already: keep the children whose every
            # sub-cut is frequent, and skip the others with their whole subtrees.
            children = [
                i
                for i in range(j)
                if all((i, *node[:m], *node[m + 1 :]) in parts for m in range(len(node)))
            ]
            if children:  # no restriction for a cut whose children are all ruled out
                # Every cut below this one adds only dims of candidate children (its sub-cut
                # "this cut + that dim" must be frequent), so the other columns are dropped.
                child_source = source.select(*(dims[i] for i in children), *cut, *_MEASURES)
                if min_rows > 1:
                    child_source = restrict_to_cells(child_source, cut, cells)
                visit(child_source, node, children)

    visit(finest, (), range(len(dims)))
    return parts


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

"""Shared helpers: tiny CSV writers and a naive pure-Python reference implementation."""

import csv
import math
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest

type Record = Mapping[str, Any]
type Row = tuple[Any, ...]

DATA_DIR = Path(__file__).parent.parent / "data"

WOMEN_HEADER = (
    "Cluster number",
    "Household number",
    "Age in 5-year groups",
    "Total children ever born",
    "Sons who have died",
    "Daughters who have died",
)
HOUSEHOLD_HEADER = (
    "Cluster number",
    "Household number",
    "Type of place of residence",
    "Age in 5-year groups",
)


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> Path:
    """Write a plain CSV file."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def write_indexed_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> Path:
    """Write a CSV with an unnamed leading pandas-style index column."""
    return write_csv(path, ("", *header), [(str(i + 1), *row) for i, row in enumerate(rows)])


def _sort_key(values: Sequence[str | None]) -> tuple[tuple[bool, str], ...]:
    return tuple((v is None, v or "") for v in values)


def naive_cube(
    records: Sequence[Record],
    dims: Sequence[str],
    numerator: str,
    denominator: str,
    *,
    max_dims: int | None = None,
    min_rows: int = 1,
    null_measures: str = "error",
) -> list[Row]:
    """Reference implementation: group every cut straight from the raw records.

    ``null_measures`` is the policy for a null numerator or denominator: "error" raises
    ValueError, "drop" skips the whole record, "zero" counts the null as 0 on its own side.

    Returns tuples shaped like ``ratio_cube(...).rows()``.
    """
    if null_measures not in ("error", "drop", "zero"):
        raise ValueError(null_measures)

    def incomplete(rec: Record) -> bool:
        return rec[numerator] is None or rec[denominator] is None

    if null_measures == "error" and any(incomplete(rec) for rec in records):
        raise ValueError("null measure")
    if null_measures == "drop":
        records = [rec for rec in records if not incomplete(rec)]
    if null_measures == "zero":
        records = [
            {**rec, **{m: 0 for m in (numerator, denominator) if rec[m] is None}} for rec in records
        ]
    # From here no measure is None: a None would fail the sums below rather than count as 0.
    depth = len(dims) if max_dims is None else max_dims
    out: list[Row] = []
    for size in range(1, depth + 1):
        for cut in combinations(dims, size):
            cells: dict[tuple[str | None, ...], list[float]] = {}
            for rec in records:
                key = tuple(None if rec[d] is None else str(rec[d]) for d in cut)
                acc = cells.setdefault(key, [0, 0, 0])
                acc[0] += 1
                acc[1] += rec[numerator]
                acc[2] += rec[denominator]
            for key in sorted(cells, key=_sort_key):
                rows, num, den = cells[key]
                if rows < min_rows:
                    continue
                values = dict(zip(cut, key, strict=True))
                out.append(
                    (
                        size,
                        "|".join(cut),
                        *[values.get(d) for d in dims],
                        int(rows),
                        float(num),
                        float(den),
                        num / den if den else None,
                    )
                )
    return out


# Some polars builds (seen: 2.0.0 on Linux x86-64) divide via a reciprocal, one ulp off the
# correctly rounded quotient Python computes, so ratios are compared with a relative tolerance.
RATIO_REL_TOL = 1e-12


def ratio_matches(actual: float | None, expected: float | None) -> bool:
    """Both null, or equal within ``RATIO_REL_TOL``."""
    if actual is None or expected is None:
        return actual is expected
    return math.isclose(actual, expected, rel_tol=RATIO_REL_TOL, abs_tol=0.0)


def assert_matches_oracle(actual: Sequence[Row], expected: Sequence[Row]) -> None:
    """Every column equal, the last one (``ratio``) within ``RATIO_REL_TOL``."""
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert got[:-1] == want[:-1], (got, want)
        assert ratio_matches(got[-1], want[-1]), (got, want)


@pytest.fixture
def women_csv(tmp_path: Path) -> Path:
    """Four women in three households; one has NA counts, one has no household."""
    rows = [
        ("1", "1", "20-24", "2", "1", "0"),
        ("1", "1", "25-29", "3", "0", "2"),
        ("1", "2", "20-24", "NA", "NA", "NA"),
        ("9", "9", "30-34", "4", "NA", "1"),
    ]
    return write_indexed_csv(tmp_path / "women.csv", WOMEN_HEADER, rows)


@pytest.fixture
def households_csv(tmp_path: Path) -> Path:
    rows = [
        ("1", "1", "Urban", "40"),
        ("1", "2", "NA", ""),
        ("2", "1", "Rural", "50"),
    ]
    return write_indexed_csv(tmp_path / "household.csv", HOUSEHOLD_HEADER, rows)

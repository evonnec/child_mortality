"""Shared helpers: tiny CSV writers and a naive pure-Python reference implementation."""

import csv
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
) -> list[Row]:
    """Reference implementation: group every cut straight from the raw records.

    Returns tuples shaped like ``ratio_cube(...).rows()``.
    """
    depth = len(dims) if max_dims is None else max_dims
    out: list[Row] = []
    for size in range(1, depth + 1):
        for cut in combinations(dims, size):
            cells: dict[tuple[str | None, ...], list[float]] = {}
            for rec in records:
                key = tuple(None if rec[d] is None else str(rec[d]) for d in cut)
                acc = cells.setdefault(key, [0, 0, 0])
                acc[0] += 1
                acc[1] += rec[numerator] or 0
                acc[2] += rec[denominator] or 0
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

"""Observation seams for the mock-test layer.

Everything here hooks at the polars level (``DataFrame.group_by`` / ``DataFrame.join``) or at one
optional module attribute (``cube.combinations``), so a rewrite of the cube algorithm does not have
to keep any helper name from ``child_mortality.cube``.

Counting conventions (shared by every work test):
* a *roll-up* is a ``DataFrame.group_by``; its cost is the height of the frame it reads;
* a *restriction* is a ``DataFrame.join``; its cost is the height of its left input;
* the finest-grain aggregation runs on a ``LazyFrame`` and is deliberately not counted: it is the
  same fixed cost for every algorithm.
"""

import csv
import io
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import combinations as _combinations
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from child_mortality import cube


class RunawayError(Exception):
    """Raised by a tripwire when an algorithm does far more work than any sane one would."""


@dataclass
class Work:
    """Counters filled in while a computation runs under :func:`observe`."""

    rollup_calls: int = 0
    rollup_rows: int = 0
    join_calls: int = 0
    join_rows: int = 0
    combinations_yielded: int = 0
    rolled_up_cuts: list[tuple[str, ...]] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        """Rows scanned by roll-ups plus restriction joins."""
        return self.rollup_rows + self.join_rows

    def cuts_of_size(self, size: int) -> set[tuple[str, ...]]:
        return {c for c in self.rolled_up_cuts if len(c) == size}


def _flatten_keys(by: tuple[Any, ...]) -> tuple[str, ...]:
    keys: list[str] = []
    for item in by:
        if isinstance(item, str):
            keys.append(item)
        elif isinstance(item, list | tuple):
            keys.extend(k for k in item if isinstance(k, str))
    return tuple(keys)


@contextmanager
def observe(*, max_rollups: int | None = None) -> Iterator[Work]:
    """Count roll-ups, restriction joins and ``itertools.combinations`` use inside the block.

    ``max_rollups`` installs a tripwire that raises :class:`RunawayError`, so a test of a missing
    size guard can fail fast instead of running a combinatorial explosion to completion.
    """
    work = Work()
    original_group_by = pl.DataFrame.group_by
    original_join = pl.DataFrame.join

    def group_by(self: pl.DataFrame, *by: Any, **kwargs: Any) -> Any:
        work.rollup_calls += 1
        work.rollup_rows += self.height
        work.rolled_up_cuts.append(_flatten_keys(by))
        if max_rollups is not None and work.rollup_calls > max_rollups:
            raise RunawayError(f"more than {max_rollups} roll-ups")
        return original_group_by(self, *by, **kwargs)

    def join(self: pl.DataFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
        work.join_calls += 1
        work.join_rows += self.height
        return original_join(self, *args, **kwargs)

    def combinations(iterable: Any, r: int) -> Iterator[tuple[Any, ...]]:
        for combo in _combinations(iterable, r):
            work.combinations_yielded += 1
            yield combo

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pl.DataFrame, "group_by", group_by)
        patch.setattr(pl.DataFrame, "join", join)
        # raising=False: a rewrite need not use itertools.combinations at all.
        patch.setattr(cube, "combinations", combinations, raising=False)
        yield work


def work_of(fn: Callable[[], object], *, max_rollups: int | None = None) -> Work:
    """Run ``fn`` under :func:`observe` and return its counters."""
    with observe(max_rollups=max_rollups) as work:
        fn()
    return work


def wide_table(n_rows: int, frequent: Sequence[str], n_other: int) -> pl.DataFrame:
    """``n_rows`` rows; binary ``frequent`` dims (every cell is frequent) plus ``n_other`` dims
    taking a distinct value on every row (no cell is ever frequent); measures ``num``/``den``."""
    columns: dict[str, list[Any]] = {}
    for bit, name in enumerate(frequent):
        columns[name] = [str((i >> bit) & 1) for i in range(n_rows)]
    for k in range(n_other):
        columns[f"u{k:02d}"] = [f"v{i}" for i in range(n_rows)]
    columns["num"] = [i % 3 for i in range(n_rows)]
    columns["den"] = [1 + i % 5 for i in range(n_rows)]
    return pl.DataFrame(columns)


def csv_text(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def cut_rows(frame: pl.DataFrame, cut: str, *columns: str) -> list[tuple[Any, ...]]:
    """The rows of one cut, restricted to ``columns``, in output order."""
    return frame.filter(pl.col("cut") == cut).select(columns).rows()

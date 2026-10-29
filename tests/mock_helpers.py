"""Observation seams for the mock-test layer.

Work is observed where every implementation of the cube has to pass: polars. No helper name from
``child_mortality.cube`` is needed, so the level loop and ``restrict_to_frequent`` can be replaced
(issue #4 proposes a depth-first walk) without losing any of these counters.

Counting conventions (shared by every work test):

* a *roll-up* is a ``DataFrame.group_by``; its cost is the height of the frame it reads;
* a *restriction* is a ``DataFrame.join``; its cost is the height of its left input;
* only calls made directly from code in the ``child_mortality`` package are counted. Polars calls
  that a test makes inside an ``observe()`` block, and calls polars makes internally, are ignored;
* the finest-grain aggregation runs on a ``LazyFrame`` and is not counted: it is the same fixed
  cost for every algorithm.

Blind spot: an implementation that rolls up through ``LazyFrame`` would be invisible to the
counters. :func:`assert_rollups_observed` turns that into a loud failure (each cut in the output
needs at least one observed roll-up), so a work test cannot pass by observing nothing.

``cube.combinations`` (today's cut enumeration) is wrapped when the module has it. That gives a
fast in-process signal and a tripwire on today's code. A rewrite that drops it loses only that
signal: the subprocess time budget in ``test_mock_work`` is the implementation-agnostic guard.
"""

import csv
import io
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import combinations as _combinations
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from child_mortality import cube
from child_mortality.cli import main

_PACKAGE = "child_mortality"


class RunawayError(Exception):
    """Raised by a tripwire when an algorithm does far more work than any sane one would."""


@dataclass
class Work:
    """Counters filled in while a computation runs under :func:`observe`."""

    rollup_calls: int = 0
    rollup_rows: int = 0
    join_calls: int = 0
    join_rows: int = 0
    rolled_up_cuts: list[tuple[str, ...]] = field(default_factory=list)
    # yields of ``cube.combinations``, keyed by the size of the pool being combined
    combinations_by_pool: Counter[int] = field(default_factory=Counter)

    @property
    def total_rows(self) -> int:
        """Rows scanned by roll-ups plus restriction joins."""
        return self.rollup_rows + self.join_rows

    @property
    def combinations_yielded(self) -> int:
        return self.combinations_by_pool.total()

    def cuts_of_size(self, size: int) -> set[tuple[str, ...]]:
        return {c for c in self.rolled_up_cuts if len(c) == size}


def _called_from_package() -> bool:
    """True if the caller of the patched polars method is library code, not a test or polars."""
    caller = sys._getframe(2).f_globals.get("__name__", "")
    return bool(caller == _PACKAGE or caller.startswith(f"{_PACKAGE}."))


def _key_names(item: Any) -> list[str]:
    if isinstance(item, str):
        return [item]
    if isinstance(item, pl.Expr):
        return list(item.meta.root_names())
    if isinstance(item, Iterable):
        return [name for sub in item for name in _key_names(sub)]
    return []


def _group_keys(by: tuple[Any, ...], named: dict[str, Any]) -> tuple[str, ...]:
    """The column names a ``group_by(*by, **named)`` groups on (strings, expressions, lists)."""
    return (*_key_names(by), *(k for k in named if k != "maintain_order"))


@contextmanager
def observe(
    *,
    max_rollups: int | None = None,
    max_combinations: int | None = 1_000_000,
    disable_restriction: bool = False,
) -> Iterator[Work]:
    """Count roll-ups, restriction joins and ``cube.combinations`` use inside the block.

    ``max_rollups`` and ``max_combinations`` install tripwires that raise :class:`RunawayError`,
    so a test can fail fast instead of running a combinatorial explosion to completion. The
    default combination budget keeps every observed run bounded on today's code even under a
    mutation (no stop at an empty level enumerates 2**n candidates on a wide table).

    ``disable_restriction`` is a mutation, not an observation: every semi-join the library makes
    returns its left input unchanged, so pruning keeps its cut-level decisions but every roll-up
    reads the unrestricted source. It needs no helper name, so it applies to any implementation
    that restricts with a semi-join.
    """
    work = Work()
    original_group_by = pl.DataFrame.group_by
    original_join = pl.DataFrame.join

    def group_by(self: pl.DataFrame, *by: Any, **named: Any) -> Any:
        if _called_from_package():
            work.rollup_calls += 1
            work.rollup_rows += self.height
            work.rolled_up_cuts.append(_group_keys(by, named))
            if max_rollups is not None and work.rollup_calls > max_rollups:
                raise RunawayError(f"more than {max_rollups} roll-ups")
        return original_group_by(self, *by, **named)

    def join(self: pl.DataFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
        if _called_from_package():
            how = kwargs.get("how", args[2] if len(args) > 2 else "inner")
            if disable_restriction and how == "semi":
                return self
            work.join_calls += 1
            work.join_rows += self.height
        return original_join(self, *args, **kwargs)

    def combinations(iterable: Iterable[Any], r: int) -> Iterator[tuple[Any, ...]]:
        pool = tuple(iterable)
        for combo in _combinations(pool, r):
            work.combinations_by_pool[len(pool)] += 1
            if max_combinations is not None and work.combinations_yielded > max_combinations:
                raise RunawayError(f"more than {max_combinations} combinations")
            yield combo

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pl.DataFrame, "group_by", group_by)
        patch.setattr(pl.DataFrame, "join", join)
        # raising=False: a rewrite need not import itertools.combinations at all.
        patch.setattr(cube, "combinations", combinations, raising=False)
        yield work


def work_of(fn: Callable[[], object], **tripwires: Any) -> Work:
    """Run ``fn`` under :func:`observe` and return its counters."""
    with observe(**tripwires) as work:
        fn()
    return work


def assert_rollups_observed(work: Work, result: pl.DataFrame) -> None:
    """Fail loudly if the counters are blind to how ``result`` was computed."""
    cuts = result["cut"].n_unique()
    assert work.rollup_calls >= cuts, (
        f"{cuts} cuts in the output but only {work.rollup_calls} roll-ups observed: the "
        "implementation no longer rolls up through DataFrame.group_by; update mock_helpers.observe"
    )


def wide_table(n_rows: int, frequent: Sequence[str], n_other: int) -> pl.DataFrame:
    """``n_rows`` rows; binary ``frequent`` dims (every cell is frequent) plus ``n_other`` dims
    ``u00``, ``u01``, ... taking a distinct value on every row (no cell is ever frequent);
    measures ``num``/``den``."""
    columns: dict[str, list[Any]] = {}
    for bit, name in enumerate(frequent):
        columns[name] = [str((i >> bit) & 1) for i in range(n_rows)]
    for k in range(n_other):
        columns[f"u{k:02d}"] = [f"v{i}" for i in range(n_rows)]
    columns["num"] = [i % 3 for i in range(n_rows)]
    columns["den"] = [1 + i % 5 for i in range(n_rows)]
    return pl.DataFrame(columns)


def skewed_table(n_rows: int, n_dims: int) -> pl.DataFrame:
    """Dims ``s0``, ``s1``, ...: dim k is ``"h"`` where bit k of the row number is set, else a value
    unique to the row. Half the rows share the heavy value of each dim, the rest are singletons, so
    restricting to frequent cells halves the source at every level of a cut."""
    columns: dict[str, list[Any]] = {
        f"s{k}": ["h" if (i >> k) & 1 else f"r{i}" for i in range(n_rows)] for k in range(n_dims)
    }
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
    path.write_text(text, encoding="utf-8", newline="")
    return path


def last_cell(stdout: str) -> tuple[int, float, float, float | None]:
    """``rows, numerator, denominator, ratio`` of the last CSV line, parsed as numbers.

    Comparing numbers rather than text keeps CLI tests valid when integer sums print as ``4``
    instead of ``4.0`` (#14).
    """
    *_, rows, num, den, ratio = next(csv.reader([stdout.splitlines()[-1]]))
    return int(rows), float(num), float(den), float(ratio) if ratio else None


@dataclass(frozen=True)
class CliRun:
    rc: int
    out: str
    err: str


def run_cli(argv: Sequence[str], capsys: pytest.CaptureFixture[str]) -> CliRun:
    """Run ``main``; an argparse rejection (``SystemExit``) counts as its exit code."""
    try:
        rc = main(list(argv))
    except SystemExit as exit_:
        rc = exit_.code if isinstance(exit_.code, int) else 2
    captured = capsys.readouterr()
    return CliRun(rc, captured.out, captured.err)


def run_with_flag(
    argv: Sequence[str], flag: Sequence[str], capsys: pytest.CaptureFixture[str]
) -> CliRun:
    """Run with ``flag`` after the subcommand's arguments or, if the parser only accepts it
    there, before the subcommand: the issues leave the placement of ``-v``/``--report`` open."""
    after = run_cli([*argv, *flag], capsys)
    if after.rc == 2 and "unrecognized arguments" in after.err:
        return run_cli([*flag, *argv], capsys)
    return after

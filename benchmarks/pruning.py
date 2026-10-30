"""Benchmark iceberg pruning (issue #4): time and peak RSS per case, each in a fresh process.

Run from the repository root (not part of the default test run)::

    uv run python benchmarks/pruning.py            # survey, 100k x 12, 50 dimensions
    uv run python benchmarks/pruning.py --large    # also the 1M x 12 case
    uv run python benchmarks/pruning.py --only survey

``ratio_cube`` with ``min_rows`` is compared with the baseline "compute everything
(``min_rows=1``), then filter". The baseline is skipped where it cannot finish (all depths on 12
dimensions, 50 dimensions). The last section checks the issue's targets.

Each case runs in a fresh process. Its time is the best of ``REPEAT`` runs on a frame built once,
outside the timer, so a target with a thin margin is not decided by one noisy sample. Its peak RSS
is the whole process's after the first run, so it includes polars and building the input frame.
"""

import argparse
import json
import random
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import polars as pl

from child_mortality.cube import ratio_cube
from child_mortality.survey import (
    DENOMINATOR,
    DIMENSIONS,
    NUMERATOR,
    join_women_households,
    load_households,
    load_women,
)

DATA = Path(__file__).parent.parent / "data"
MB = 1024 * 1024  # ru_maxrss is in bytes on macOS (kilobytes on Linux, handled below)
REPEAT = 3


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (MB if sys.platform == "darwin" else 1024)


def survey_frame() -> pl.DataFrame:
    return join_women_households(
        load_women(DATA / "women.csv"), load_households(DATA / "household.csv")
    ).collect()


def skewed_frame(rows: int, n_dims: int = 12) -> pl.DataFrame:
    """``n_dims`` dims with cardinalities 2..1000 and skewed values; the finest grain does not
    compress (about one group per row)."""
    rng = random.Random(7)
    cards = [round(2 * 500 ** (k / (n_dims - 1))) for k in range(n_dims)]
    cols: dict[str, list[int]] = {
        f"d{k:02d}": [int(card * rng.random() ** 2) for _ in range(rows)]
        for k, card in enumerate(cards)
    }
    cols["num"] = [rng.randrange(3) for _ in range(rows)]
    cols["den"] = [1 + rng.randrange(5) for _ in range(rows)]
    return pl.DataFrame(cols)


def wide_frame(rows: int, binary: int, total: int = 50) -> pl.DataFrame:
    """``binary`` binary dims plus ``total - binary`` dims with 5,000 values each."""
    cols: dict[str, list[str]] = {
        f"u{k:02d}": [f"v{(i * 7 + k) % 5000}" for i in range(rows)] for k in range(total - binary)
    }
    cols.update({f"b{k}": [str((i >> k) & 1) for i in range(rows)] for k in range(binary)})
    return pl.DataFrame(
        {**cols, "num": [i % 3 for i in range(rows)], "den": [1 + i % 5 for i in range(rows)]}
    )


type Builder = Callable[[], pl.DataFrame]
type Row = tuple[str, *tuple[dict[str, float] | None, ...]]


def run_case(name: str, mode: str, min_rows: int, max_dims: int | None) -> dict[str, float]:
    """Best of ``REPEAT`` timed runs (the frame is built first, outside the timer); child entry."""
    builders: dict[str, Builder] = {
        "survey": survey_frame,
        "100k": lambda: skewed_frame(100_000),
        "1m": lambda: skewed_frame(1_000_000),
        "w20k": lambda: wide_frame(20_000, 10),
        "w80k": lambda: wide_frame(80_000, 10),
        "w5k": lambda: wide_frame(5_000, 8),
    }
    frame = builders[name]()
    if name == "survey":
        dims, num, den = list(DIMENSIONS), NUMERATOR, DENOMINATOR
    else:
        dims, num, den = [c for c in frame.columns if c not in ("num", "den")], "num", "den"
    seconds, cells, rss_mb = float("inf"), 0, 0.0
    for run in range(REPEAT):
        start = time.perf_counter()
        if mode == "iceberg":
            out = ratio_cube(frame, dims, num, den, max_dims=max_dims, min_rows=min_rows)
        else:
            out = ratio_cube(frame, dims, num, den, max_dims=max_dims).filter(
                pl.col("rows") >= min_rows
            )
        seconds = min(seconds, time.perf_counter() - start)
        cells = out.height
        if run == 0:  # later runs reuse a fragmented heap, which inflates the high-water mark
            rss_mb = peak_rss_mb()
        del out
    return {"cells": cells, "seconds": seconds, "rss_mb": rss_mb}


def measure(name: str, mode: str, min_rows: int, max_dims: int | None) -> dict[str, float]:
    """Run :func:`run_case` in a fresh process so peak RSS is per case."""
    argv = [sys.executable, __file__, "--child", name, mode, str(min_rows), str(max_dims or 0)]
    done = subprocess.run(argv, capture_output=True, encoding="utf-8", check=True)
    result: dict[str, float] = json.loads(done.stdout.splitlines()[-1])
    return result


def table(title: str, rows: list[Row]) -> None:
    print(f"\n{title}")
    for label, *results in rows:
        cells = "  ".join(
            "{:>26}".format(f"{r['seconds']:7.2f} s {r['rss_mb']:6.0f} MB") if r else " " * 26
            for r in results
        )
        first = next((r for r in results if r), None)
        print(f"  {label:<24}{int(first['cells']) if first else 0:>11,}  {cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--large", action="store_true", help="also run 1M rows x 12 dimensions")
    parser.add_argument("--only", choices=["survey", "synthetic", "wide"])
    parser.add_argument("--child", nargs=4, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        name, mode, min_rows, max_dims = args.child[0], args.child[1], *map(int, args.child[2:4])
        print(json.dumps(run_case(name, mode, min_rows, max_dims or None)))
        return

    verdicts: list[tuple[str, bool]] = []
    print("columns: cells out | iceberg | compute all, then filter  (seconds, peak RSS)")

    if args.only in (None, "survey"):
        rows: list[Row] = []
        for min_rows in (1, 2, 5, 30, 100):
            ice = measure("survey", "iceberg", min_rows, None)
            base = measure("survey", "filter", min_rows, None)
            rows.append((f"min_rows={min_rows}", ice, base))
            if min_rows == 1:
                one = ice
            if min_rows == 2:
                verdicts.append(
                    ("survey min_rows=2 <= min_rows=1", ice["seconds"] <= one["seconds"])
                )
            if min_rows in (5, 30, 100):
                verdicts.append(
                    (
                        f"survey min_rows={min_rows} >= 1.3x faster",
                        base["seconds"] >= 1.3 * ice["seconds"],
                    )
                )
        table("Survey (9 dims, all depths)", rows)

    if args.only in (None, "synthetic"):
        rows = []
        for min_rows in (10, 100, 1000):
            ice = measure("100k", "iceberg", min_rows, 3)
            base = measure("100k", "filter", min_rows, 3)
            rows.append((f"min_rows={min_rows}", ice, base))
            verdicts.append(
                (
                    f"100k x 12 max_dims=3 min_rows={min_rows} >= 1.5x faster",
                    base["seconds"] >= 1.5 * ice["seconds"],
                )
            )
        table("Synthetic 100k rows x 12 skewed dims, max_dims=3", rows)
        rows = [(f"min_rows={m}", measure("100k", "iceberg", m, None), None) for m in (100, 1000)]
        if args.large:
            rows.append(("1M, min_rows=10000", measure("1m", "iceberg", 10_000, None), None))
        table("Synthetic, all depths (compute-all infeasible)", rows)

    if args.only in (None, "wide"):
        rows = []
        for label, case, min_rows in (
            ("20k rows, min_rows=200", "w20k", 200),
            ("80k rows, min_rows=100", "w80k", 100),
            ("5k rows, 8 binary, 150", "w5k", 150),
        ):
            result = measure(case, "iceberg", min_rows, None)
            rows.append((label, result, None))
            if case == "w80k":
                verdicts.append(
                    (
                        "50 dims x 80k rows min_rows=100 < 30 s and < 2 GB",
                        result["seconds"] < 30 and result["rss_mb"] < 2048,
                    )
                )
        table("50 dimensions (10 or 8 binary, the rest 5,000 values)", rows)

    print("\nTargets")
    for label, ok in verdicts:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    sys.exit(0 if all(ok for _, ok in verdicts) else 1)


if __name__ == "__main__":
    main()

"""Work-avoidance tests (#4, #6): observe what the algorithm does, not just what it returns.

Roll-ups and restriction joins are counted at the polars level (see ``mock_helpers``), so any
algorithm that computes the same cube is measured the same way, whatever its helpers are called.
The tests that were strict xfails tagged ``#4`` (pruning added work instead of avoiding it) pass
since the depth-first walk replaced the level loop. Together with the safety net #6 asks for, they
hold for the walk, and each mutation below fails at least one:

* no restriction (every roll-up reads the whole finest grain):
  ``test_pruning_shrinks_rollup_input_on_skewed_data``;
* restricting the finest grain afresh instead of the carried source, or restricting for a cut
  whose children are all ruled out: ``test_total_rows_scanned_at_min_rows_30_at_most_75_percent``
  and ``test_no_restriction_for_a_cut_whose_children_are_all_ruled_out``;
* no cut-level check: ``test_dimension_without_frequent_cell_is_never_in_a_deeper_rollup``.
  Infrequent dimensions sit before, between and after the frequent ones, so this holds for a
  level-wise walk and for a colexicographic one (whose children only add earlier dimensions);
* a cut-level check that looks up only some of the (k-1)-sub-cuts:
  ``test_cut_is_skipped_when_any_one_subcut_is_infrequent``;
* no stop once no cut of a size is frequent: ``test_all_distinct_dims_stop_after_level_one``;
* no shortcut for ``min_rows`` above the table size:
  ``test_min_rows_beyond_table_size_does_no_rollup``.
"""

import itertools
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest

from child_mortality import cube
from child_mortality.cube import ratio_cube
from child_mortality.survey import (
    DENOMINATOR,
    DIMENSIONS,
    NUMERATOR,
    join_women_households,
    load_households,
    load_women,
)
from conftest import DATA_DIR, assert_matches_oracle, naive_cube
from mock_helpers import (
    RunawayError,
    Work,
    _group_keys,
    assert_rollups_observed,
    observe,
    skewed_table,
    wide_table,
    work_of,
)

SRC = Path(__file__).parent.parent / "src"


# --- survey data: rows scanned -----------------------------------------------------------------


@dataclass(frozen=True)
class SurveyRun:
    work: Work
    cuts: int
    cells_per_size: list[int]


@pytest.fixture(scope="module")
def survey_runs() -> dict[int, SurveyRun]:
    """Work done by the full survey cube at several thresholds (computed once).

    The survey is joined in memory first, outside the observed block, so only the cube is measured.
    """
    joined = join_women_households(
        load_women(DATA_DIR / "women.csv"), load_households(DATA_DIR / "household.csv")
    ).collect()
    runs: dict[int, SurveyRun] = {}
    for threshold in (1, 30, 100):
        with observe() as work:
            out = ratio_cube(joined, DIMENSIONS, NUMERATOR, DENOMINATOR, min_rows=threshold)
        assert_rollups_observed(work, out)
        per_size = out.group_by("cut_size").len().sort("cut_size")["len"].to_list()
        runs[threshold] = SurveyRun(work, out["cut"].n_unique(), per_size)
    return runs


def test_exhaustive_run_rolls_up_each_cut_once_without_joins(
    survey_runs: dict[int, SurveyRun],
) -> None:
    exhaustive = survey_runs[1].work
    assert exhaustive.join_calls == 0
    assert exhaustive.rollup_calls == 2**9 - 1


def test_survey_cell_counts_match_the_readme(survey_runs: dict[int, SurveyRun]) -> None:
    """#6: the README figures for the survey cube, so a rewrite cannot change the output."""
    assert sum(survey_runs[1].cells_per_size) == 1_546_271
    assert survey_runs[30].cells_per_size == [135, 1_517, 4_212, 4_192, 1_526, 124]
    assert sum(survey_runs[30].cells_per_size) == 11_706


def test_pruning_shrinks_rollup_input_on_survey_data(survey_runs: dict[int, SurveyRun]) -> None:
    """Implied by #4's 75% bound. Today 30%, #4's prototype 42%, no restriction (today) 98%."""
    exhaustive, at30, at100 = (survey_runs[t].work for t in (1, 30, 100))
    assert at30.rollup_rows <= 0.75 * exhaustive.total_rows
    assert at100.rollup_rows <= at30.rollup_rows
    assert at30.rollup_calls < exhaustive.rollup_calls


def test_total_rows_scanned_at_min_rows_30_at_most_75_percent(
    survey_runs: dict[int, SurveyRun],
) -> None:
    assert survey_runs[30].work.total_rows <= 0.75 * survey_runs[1].work.total_rows


# --- skewed data: the restriction must pay off -------------------------------------------------

SKEWED_DIMS = [f"s{k}" for k in range(6)]


def test_pruning_shrinks_rollup_input_on_skewed_data() -> None:
    """Every one of the 63 cuts has a frequent cell, so only the restriction can shrink the input.

    Roll-up rows relative to the exhaustive run: today 20%, issue #4's prototype 32%, no
    restriction 100%. On the survey data the prototype without restriction still reads only 62%,
    which is why this case exists.
    """
    table = skewed_table(4000, 6)
    exhaustive = work_of(lambda: ratio_cube(table, SKEWED_DIMS, "num", "den"))
    pruned = work_of(lambda: ratio_cube(table, SKEWED_DIMS, "num", "den", min_rows=40))
    assert pruned.rollup_rows <= 0.5 * exhaustive.rollup_rows


def test_disabling_the_restriction_is_caught_but_invisible_in_the_output() -> None:
    """The mutation "no restriction", applied at the polars level so it needs no helper name."""
    table = skewed_table(4000, 6)
    exhaustive = work_of(lambda: ratio_cube(table, SKEWED_DIMS, "num", "den"))
    honest = ratio_cube(table, SKEWED_DIMS, "num", "den", min_rows=40)
    with observe(disable_restriction=True) as mutant:
        mutated = ratio_cube(table, SKEWED_DIMS, "num", "den", min_rows=40)
    assert mutated.equals(honest), "black-box tests cannot see a removed restriction"
    assert mutant.rollup_rows > 0.5 * exhaustive.rollup_rows, (
        "disabling semi-joins did not grow the roll-up input: the restriction no longer uses "
        "DataFrame.join(how='semi'); update mock_helpers.observe(disable_restriction=...)"
    )


# --- synthetic wide table: cuts examined -------------------------------------------------------

US = [f"u{k:02d}" for k in range(48)]
# Only a and b have frequent cells; infrequent dims come before, between and after them.
DIMS50 = [*US[:24], "a", *US[24:], "b"]


# Today's code: 51 roll-ups and about 42,000 combinations. The budgets bound a mutant (for example
# one that rolls up every candidate, or never stops) to a fraction of a second.
WIDE_ROLLUPS, WIDE_COMBINATIONS = 500, 100_000


def wide50() -> pl.DataFrame:
    """50 dims of which only ``a`` and ``b`` have any frequent cell (200 rows, min_rows=40)."""
    return wide_table(200, ["a", "b"], 48)


def test_wide_table_result_is_just_the_frequent_cuts() -> None:
    with observe(max_rollups=WIDE_ROLLUPS, max_combinations=WIDE_COMBINATIONS):
        result = ratio_cube(wide50(), DIMS50, "num", "den", min_rows=40)
    assert result["cut"].unique(maintain_order=True).to_list() == ["a", "b", "a|b"]
    expected = naive_cube(wide50().to_dicts(), DIMS50, "num", "den", min_rows=40, max_dims=2)
    assert_matches_oracle(result.rows(), expected)


def test_dimension_without_frequent_cell_is_never_in_a_deeper_rollup() -> None:
    """#4: exactly one cut of size 2, ``(a, b)``, is rolled up, and none of size 3."""
    work = work_of(
        lambda: ratio_cube(wide50(), DIMS50, "num", "den", min_rows=40),
        max_rollups=WIDE_ROLLUPS,
        max_combinations=WIDE_COMBINATIONS,
    )
    assert [cut for cut in work.rolled_up_cuts if len(cut) >= 2] == [("a", "b")]


def test_no_restriction_for_a_cut_whose_children_are_all_ruled_out() -> None:
    """Only ``b`` has a child that can be frequent, ``(a, b)``: a restriction made for ``a`` (every
    dimension before it is infrequent) or for ``(a, b)`` would be read by no roll-up."""
    work = work_of(
        lambda: ratio_cube(wide50(), DIMS50, "num", "den", min_rows=40),
        max_rollups=WIDE_ROLLUPS,
        max_combinations=WIDE_COMBINATIONS,
    )
    assert work.join_calls <= 1


@pytest.mark.parametrize("dims", list(itertools.permutations("abc")), ids="".join)
def test_cut_is_skipped_when_any_one_subcut_is_infrequent(dims: tuple[str, ...]) -> None:
    """Every dimension and the pairs ``(a, b)`` and ``(b, c)`` have a frequent cell, ``(a, c)`` has
    none, so ``(a, b, c)`` must not be rolled up. Each dimension order puts the infrequent pair in
    a different position among the 3-cut's sub-cuts."""
    table = pl.DataFrame(
        {
            "a": ["x", "x", "y", "z"],
            "b": ["p", "p", "q", "q"],
            "c": ["1", "2", "3", "3"],
            "num": [1, 0, 1, 0],
            "den": [1, 1, 1, 1],
        }
    )
    work = work_of(lambda: ratio_cube(table, list(dims), "num", "den", min_rows=2))
    assert work.cuts_of_size(3) == set()
    assert len(work.cuts_of_size(2)) == 3


def test_cuts_examined_is_linear_in_frequent_cuts_not_combinatorial() -> None:
    """Fast in-process signal on today's loop. A rewrite without ``itertools.combinations`` reads 0
    here; the 50-dimension time budget below is the guard that holds for any implementation.
    (#4 also bounds the count reported by ``-v``; that needs #8's output format first.)"""
    work = work_of(
        lambda: ratio_cube(wide50(), DIMS50, "num", "den", min_rows=40),
        max_rollups=WIDE_ROLLUPS,
        max_combinations=WIDE_COMBINATIONS,
    )
    n, frequent_cuts = len(DIMS50), 3  # a, b, a|b
    # Issue #4: at most n * (number of frequent cuts) + n candidate cuts are examined. Today's loop
    # draws its candidates as combinations of the whole dimension list (pool size n); combinations
    # of one cut's own dimensions (sub-cut checks) are not candidates and are not counted.
    assert work.combinations_by_pool[n] <= n * frequent_cuts + n


def test_all_distinct_dims_stop_after_level_one() -> None:
    dims = [f"u{k:02d}" for k in range(50)]
    table = wide_table(60, [], 50)
    work = work_of(
        lambda: ratio_cube(table, dims, "num", "den", min_rows=2), max_combinations=10_000
    )
    assert work.rolled_up_cuts == [(d,) for d in dims]
    # No candidate of size 2 is even generated once level 1 has no frequent cell.
    assert work.combinations_by_pool[len(dims)] <= len(dims)


def test_min_rows_beyond_table_size_does_no_rollup() -> None:
    table = wide_table(40, ["a", "b"], 3)
    work = work_of(lambda: ratio_cube(table, ["a", "b", "u00"], "num", "den", min_rows=41))
    assert work.rollup_calls == 0
    assert work.join_calls == 0


def test_max_dims_bounds_the_cut_size_rolled_up() -> None:
    table = wide_table(64, ["a", "b", "c", "d"], 0)
    dims = ["a", "b", "c", "d"]
    work = work_of(lambda: ratio_cube(table, dims, "num", "den", max_dims=2, min_rows=2))
    assert max(len(c) for c in work.rolled_up_cuts) == 2
    assert len(work.rolled_up_cuts) == 4 + 6


# --- end-to-end time budget: issue #4's 50-dimension case --------------------------------------

_FIFTY_DIMS_SCRIPT = """
import itertools, sys, time
import polars as pl
from child_mortality import cube

# Tripwires, so a broken run fails in seconds instead of minutes (the wall clock is the spec):
# - 500,000 candidate cuts, about 12x the issue's bound on cuts examined (n * frequent cuts + n =
#   42,400); applies only while cube uses itertools.combinations (today's C(n, k) loop);
# - 5,000 roll-ups through DataFrame.group_by, about 5x what issue #4's prototype makes (1,007).
BUDGET, ROLLUP_BUDGET = 500_000, 5_000
yielded = rollups = 0
if hasattr(cube, "combinations"):
    def combinations(iterable, r):
        global yielded
        for combo in itertools.combinations(iterable, r):
            yielded += 1
            if yielded > BUDGET:
                sys.exit(f"tripwire: more than {BUDGET:,} candidate cuts enumerated")
            yield combo
    cube.combinations = combinations
group_by = pl.DataFrame.group_by
def counted_group_by(self, *args, **kwargs):
    global rollups
    rollups += 1
    if rollups > ROLLUP_BUDGET:
        sys.exit(f"tripwire: more than {ROLLUP_BUDGET:,} roll-ups")
    return group_by(self, *args, **kwargs)
pl.DataFrame.group_by = counted_group_by

n = 20_000
cols = {f"u{k:02d}": [f"v{(i * 7 + k) % 5000}" for i in range(n)] for k in range(40)}
cols.update({f"b{k}": [str((i >> k) & 1) for i in range(n)] for k in range(10)})
cols["num"] = [i % 3 for i in range(n)]
cols["den"] = [1 + i % 5 for i in range(n)]
frame = pl.DataFrame(cols)
dims = [c for c in cols if c not in ("num", "den")]
start = time.perf_counter()
out = cube.ratio_cube(frame, dims, "num", "den", min_rows=200)
print(out.height, out["cut"].n_unique(), out["cut_size"].max(), time.perf_counter() - start)
print("roll-ups", rollups, file=sys.stderr)
"""


def test_50_dims_20k_rows_finishes_within_15_s() -> None:
    """Issue #4's case: 10 binary dims plus 40 dims of 5,000 values each, ``min_rows=200``.

    The prototype takes 1.7 s on a laptop, 1.8 s with one polars thread. The 15 s budget is the
    issue's and applies to ``ratio_cube`` alone; the 60 s hard timeout only bounds a broken run.
    """
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [str(SRC), os.environ.get("PYTHONPATH")])),
        "PYTHONIOENCODING": "utf-8",
    }
    try:
        done = subprocess.run(
            [sys.executable, "-c", _FIFTY_DIMS_SCRIPT],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("ratio_cube did not finish within 60 s")
    assert done.returncode == 0, done.stderr[-2000:]
    cells, cuts, depth, seconds = done.stdout.split()
    # every subset of the 10 binary dims up to size 6 (20,000 / 2**6 >= 200 > 20,000 / 2**7)
    assert (int(cells), int(cuts), int(depth)) == (
        sum(math.comb(10, k) * 2**k for k in range(1, 7)),
        sum(math.comb(10, k) for k in range(1, 7)),
        6,
    )
    assert float(seconds) < 15


# --- the counters themselves -------------------------------------------------------------------


def test_observe_counts_nothing_outside_the_block() -> None:
    with observe() as work:
        pass
    ratio_cube(wide_table(8, ["a"], 0), ["a"], "num", "den")
    assert work.rollup_calls == 0


def test_observe_ignores_polars_calls_made_by_the_test() -> None:
    frame = wide_table(8, ["a", "b"], 0)
    with observe() as work:
        frame.group_by("a").agg(pl.len())
        frame.join(frame.select("a").unique(), on="a", how="semi")
    assert (work.rollup_calls, work.join_calls) == (0, 0)


def test_observe_reads_string_list_and_expression_keys() -> None:
    assert _group_keys(("a", ["b", pl.col("c")], (pl.col("d"),)), {"e": pl.col("x")}) == (
        "a",
        "b",
        "c",
        "d",
        "e",
    )
    assert _group_keys(("a",), {"maintain_order": True}) == ("a",)


def test_combination_tripwire_aborts_runaway_enumeration() -> None:
    with observe(max_combinations=5) as work, pytest.raises(RunawayError):
        list(vars(cube)["combinations"](range(10), 2))
    assert work.combinations_by_pool[10] == 6

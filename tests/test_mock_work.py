"""Work-avoidance tests (#4, #6): observe what the algorithm does, not just what it returns.

Roll-ups and restriction joins are counted at the polars level (see ``mock_helpers``), so any
algorithm that computes the same cube can be measured the same way. Tests tagged ``#4`` pin the
current defect (pruning does more work than computing everything) and flip once it is fixed.
"""

import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from child_mortality import cube
from child_mortality.cube import ratio_cube
from child_mortality.survey import cd_ceb_cube
from conftest import DATA_DIR, naive_cube
from mock_helpers import Work, observe, wide_table, work_of

WOMEN = DATA_DIR / "women.csv"
HOUSEHOLDS = DATA_DIR / "household.csv"
SRC = Path(__file__).parent.parent / "src"


@pytest.fixture(scope="module")
def survey_work() -> dict[int, Work]:
    """Work done by the full survey cube at several thresholds (computed once)."""
    out: dict[int, Work] = {}
    for threshold in (1, 30, 100):
        with observe() as work:
            cd_ceb_cube(WOMEN, HOUSEHOLDS, min_rows=threshold)
        out[threshold] = work
    return out


# --- survey data: rows scanned -----------------------------------------------------------------


def test_exhaustive_run_does_no_restriction_joins(survey_work: dict[int, Work]) -> None:
    assert survey_work[1].join_calls == 0
    # 2^9 - 1 cuts, each rolled up once from the finest grain.
    assert survey_work[1].rollup_calls == 2**9 - 1


def test_pruning_shrinks_rollup_input(survey_work: dict[int, Work]) -> None:
    assert survey_work[30].rollup_rows < survey_work[1].rollup_rows
    assert survey_work[100].rollup_rows < survey_work[30].rollup_rows
    assert survey_work[30].rollup_calls < survey_work[1].rollup_calls


@pytest.mark.xfail(
    strict=True, reason="#4: roll-up + semi-join rows at min_rows=30 are ~198% of min_rows=1"
)
def test_total_rows_scanned_at_min_rows_30_at_most_75_percent(
    survey_work: dict[int, Work],
) -> None:
    assert survey_work[30].total_rows <= 0.75 * survey_work[1].total_rows


@pytest.mark.xfail(
    strict=True, reason="#4: restriction joins add more input rows than the roll-ups save"
)
def test_total_rows_scanned_at_min_rows_100_below_exhaustive(survey_work: dict[int, Work]) -> None:
    assert survey_work[100].total_rows < survey_work[1].total_rows


@pytest.mark.xfail(
    strict=True, reason="#4: every restricted cut makes k semi-joins against the full finest grain"
)
def test_restriction_joins_at_most_one_per_rolled_up_cut(survey_work: dict[int, Work]) -> None:
    work = survey_work[30]
    assert work.join_calls <= work.rollup_calls


# --- synthetic wide table: cuts examined -------------------------------------------------------


def wide50() -> pl.DataFrame:
    """50 dims of which only ``a`` and ``b`` have any frequent cell (200 rows, min_rows=40)."""
    return wide_table(200, ["a", "b"], 48)


DIMS50 = ["a", "b", *[f"u{k:02d}" for k in range(48)]]


def test_wide_table_result_is_just_the_frequent_cuts() -> None:
    result = ratio_cube(wide50(), DIMS50, "num", "den", min_rows=40)
    assert set(result["cut"]) == {"a", "b", "a|b"}
    assert result["rows"].sum() == 200 + 200 + 200
    assert result.equals(
        pl.DataFrame(
            naive_cube(wide50().to_dicts(), DIMS50, "num", "den", min_rows=40, max_dims=3),
            schema=result.schema,
            orient="row",
        )
    )


def test_dimension_without_frequent_cell_is_never_in_a_deeper_rollup() -> None:
    work = work_of(lambda: ratio_cube(wide50(), DIMS50, "num", "den", min_rows=40))
    deeper = [cut for cut in work.rolled_up_cuts if len(cut) >= 2]
    assert deeper == [("a", "b")]
    assert not any(c.startswith("u") for cut in deeper for c in cut)


def test_exactly_one_size2_cut_and_no_size3_cut_is_rolled_up() -> None:
    work = work_of(lambda: ratio_cube(wide50(), DIMS50, "num", "den", min_rows=40))
    assert work.cuts_of_size(2) == {("a", "b")}
    assert work.cuts_of_size(3) == set()


@pytest.mark.xfail(
    strict=True, reason="#4: C(n, k) cuts are enumerated at every level (19,600 at size 3)"
)
def test_cuts_enumerated_is_linear_in_frequent_cuts_not_combinatorial() -> None:
    work = work_of(lambda: ratio_cube(wide50(), DIMS50, "num", "den", min_rows=40))
    frequent_cuts = 3  # a, b, a|b
    n = len(DIMS50)
    # Issue #4: at most n * (number of frequent cuts) + n candidate cuts are ever examined.
    assert work.combinations_yielded <= n * frequent_cuts + n


def test_all_distinct_dims_stop_after_level_one() -> None:
    table = wide_table(60, [], 6)
    dims = [f"u{k:02d}" for k in range(6)]
    work = work_of(lambda: ratio_cube(table, dims, "num", "den", min_rows=2))
    assert work.rolled_up_cuts == [(d,) for d in dims]


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


# --- end-to-end time budget on a wide table ----------------------------------------------------

_WIDE_SCRIPT = """
import polars as pl
from child_mortality.cube import ratio_cube

n = 4000
cols = {f"b{k}": [str((i >> k) & 1) for i in range(n)] for k in range(6)}
cols.update({f"u{k:02d}": [f"v{i}" for i in range(n)] for k in range(44)})
cols["num"] = [i % 3 for i in range(n)]
cols["den"] = [1 + i % 5 for i in range(n)]
dims = [c for c in cols if c not in ("num", "den")]
out = ratio_cube(pl.DataFrame(cols), dims, "num", "den", min_rows=50)
assert out["cut_size"].max() == 6, out["cut_size"].max()
print(out.height)
"""


@pytest.mark.xfail(
    strict=True, reason="#4: 50 dims with 6 binary ones does not finish (C(50, k) enumeration)"
)
def test_50_dims_with_few_frequent_ones_finishes_quickly() -> None:
    env = {"PYTHONPATH": str(SRC), "PATH": ""}
    try:
        done = subprocess.run(
            [sys.executable, "-c", _WIDE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=6,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("did not finish within 6 s")
    assert done.returncode == 0, done.stderr
    # sum over sizes 1..6 of C(6, k) * 2^k cells (plus the 44 size-1 cuts, none frequent)
    assert int(done.stdout) == sum(math.comb(6, k) * 2**k for k in range(1, 7))


# --- mutation-style: the counters must notice a removed restriction (#6) -----------------------


def test_harness_detects_removed_restriction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the restriction step with "no restriction": the roll-up input must grow."""
    if not hasattr(cube, "restrict_to_frequent"):
        pytest.skip("restriction step lives under another name in this implementation")
    honest = work_of(lambda: cd_ceb_cube(WOMEN, HOUSEHOLDS, max_dims=3, min_rows=30))

    def no_restriction(finest: pl.DataFrame, cut: tuple[str, ...], frequent: Any) -> Any:
        return finest

    monkeypatch.setattr(cube, "restrict_to_frequent", no_restriction)
    mutant = work_of(lambda: cd_ceb_cube(WOMEN, HOUSEHOLDS, max_dims=3, min_rows=30))
    assert mutant.rollup_rows > honest.rollup_rows
    assert mutant.join_calls == 0 < honest.join_calls


def test_output_is_unchanged_by_a_removed_restriction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Why black-box tests cannot catch it: the mutant returns the very same cube."""
    if not hasattr(cube, "restrict_to_frequent"):
        pytest.skip("restriction step lives under another name in this implementation")
    honest = cd_ceb_cube(WOMEN, HOUSEHOLDS, max_dims=3, min_rows=30)
    monkeypatch.setattr(cube, "restrict_to_frequent", lambda finest, cut, frequent: finest)
    assert cd_ceb_cube(WOMEN, HOUSEHOLDS, max_dims=3, min_rows=30).equals(honest)


def test_observe_counts_nothing_outside_the_block() -> None:
    with observe() as work:
        pass
    ratio_cube(wide_table(8, ["a"], 0), ["a"], "num", "den")
    assert work.rollup_calls == 0

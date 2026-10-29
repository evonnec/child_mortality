"""Measure semantics: nulls (#5), validation (#7) and exact sums / float tolerance (#14).

Tests tagged with an issue number are strict xfails that encode the issue's acceptance criteria
with the names the issue proposes (``null_measures=``, ``--null-measures``, ``--on-invalid``,
``--require-match``); they must pass once the fix lands. CLI output is compared as numbers, so the
tests stay valid when integer sums print as ``4`` rather than ``4.0`` (#14).
"""

import random
import re
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from child_mortality.cli import main
from child_mortality.cube import ratio_cube
from child_mortality.survey import (
    NUMERATOR,
    cd_ceb_cube,
    join_women_households,
    load_households,
    load_women,
)
from conftest import DATA_DIR, HOUSEHOLD_HEADER, WOMEN_HEADER, write_indexed_csv
from mock_helpers import csv_text, last_cell, write_text

AGE = "Age in 5-year groups"
RES = "Type of place of residence"

# --- nulls (#5): the two cases from the issue ---------------------------------------------------

# Case A: the first woman's children-ever-born is unknown but 3 deaths are recorded.
CASE_A = [("30-34", "NA", "2", "1"), ("30-34", "2", "0", "1")]
# Case B: the first woman has one unknown death count.
CASE_B = [("30-34", "3", "NA", "1"), ("30-34", "3", "1", "1")]
type Case = list[tuple[str, str, str, str]]


def survey_files(tmp_path: Path, women: Case) -> tuple[Path, Path]:
    """One woman per household; ``women`` rows are (age, born, sons died, daughters died)."""
    women_rows = [(str(i), "1", age, b, s, d) for i, (age, b, s, d) in enumerate(women, 1)]
    house_rows = [(str(i), "1", "Urban" if i == 1 else "Rural", "40") for i in range(1, 3)]
    return (
        write_indexed_csv(tmp_path / "w.csv", WOMEN_HEADER, women_rows),
        write_indexed_csv(tmp_path / "h.csv", HOUSEHOLD_HEADER, house_rows),
    )


def cd_ceb_argv(women: Path, households: Path, *extra: str) -> list[str]:
    argv = ["cd-ceb", "--women", str(women), "--households", str(households)]
    return [*argv, "-d", AGE, "--max-dims", "1", *extra]


def generic_frame(case: Case) -> pl.DataFrame:
    """The case as a generic table: ``num`` = sons + daughters (null if either is), ``den``."""

    def number(text: str) -> int | None:
        return None if text == "NA" else int(text)

    return pl.DataFrame(
        {
            "g": [age for age, *_ in case],
            "den": [number(b) for _, b, _, _ in case],
            "num": [
                None if s is None or d is None else s + d
                for s, d in ((number(s), number(d)) for _, _, s, d in case)
            ],
        },
        schema={"g": pl.String, "den": pl.Int64, "num": pl.Int64},
    )


def only_row(frame: pl.DataFrame) -> dict[str, Any]:
    (row,) = frame.filter(pl.col("cut") == "g").to_dicts()
    return row


@pytest.mark.parametrize(("case", "column"), [(CASE_A, "den"), (CASE_B, "num")], ids=["A", "B"])
@pytest.mark.xfail(strict=True, reason="#5: default policy must raise on null measures")
def test_default_policy_errors_on_null_measures(case: Case, column: str) -> None:
    with pytest.raises(ValueError, match=rf"\b{column}\b"):
        ratio_cube(generic_frame(case), ["g"], "num", "den")


@pytest.mark.xfail(strict=True, reason="#5: error names the column and the affected row count")
def test_default_policy_message_names_column_and_count() -> None:
    frame = pl.DataFrame(
        {"g": ["x"] * 9, "num": [1] * 9, "den": [None] * 7 + [2, 2]},
        schema={"g": pl.String, "num": pl.Int64, "den": pl.Int64},
    )
    with pytest.raises(ValueError, match=r"\bden\b") as info:
        ratio_cube(frame, ["g"], "num", "den")
    assert re.search(r"\b7\b", str(info.value)), str(info.value)


@pytest.mark.parametrize(
    ("case", "expected"),
    [(CASE_A, (1, 1, 2, 0.5)), (CASE_B, (1, 2, 3, pytest.approx(2 / 3)))],
    ids=["A", "B"],
)
@pytest.mark.xfail(strict=True, reason="#5: null_measures='drop' (complete-case) not implemented")
def test_drop_is_complete_case(case: Case, expected: tuple[Any, ...]) -> None:
    row = only_row(ratio_cube(generic_frame(case), ["g"], "num", "den", null_measures="drop"))  # type: ignore[call-arg]
    assert (row["rows"], row["numerator"], row["denominator"], row["ratio"]) == expected


@pytest.mark.parametrize(
    ("case", "expected"),
    # today's ratio_cube on these frames: a null counts as 0 on its own side
    [(CASE_A, (2, 4, 2, 2.0)), (CASE_B, (2, 2, 6, pytest.approx(1 / 3)))],
    ids=["A", "B"],
)
@pytest.mark.xfail(strict=True, reason="#5: null_measures='zero' (opt-in legacy) not implemented")
def test_zero_reproduces_todays_output(case: Case, expected: tuple[Any, ...]) -> None:
    row = only_row(ratio_cube(generic_frame(case), ["g"], "num", "den", null_measures="zero"))  # type: ignore[call-arg]
    assert (row["rows"], row["numerator"], row["denominator"], row["ratio"]) == expected


@pytest.mark.xfail(strict=True, reason="#5: an unknown null_measures value must be rejected")
def test_unknown_null_policy_is_rejected() -> None:
    """Every other argument of ``ratio_cube`` is validated with a ValueError; so is this one."""
    complete = pl.DataFrame({"g": ["x"], "num": [1], "den": [2]})
    with pytest.raises(ValueError, match=r"null_measures|ignore"):
        ratio_cube(complete, ["g"], "num", "den", null_measures="ignore")  # type: ignore[call-arg]


@pytest.mark.xfail(strict=True, reason="#5: cd-ceb must exit 2 on null counts by default")
def test_cli_default_exits_2_naming_the_column_and_count_case_a(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(cd_ceb_argv(*survey_files(tmp_path, CASE_A))) == 2
    (line,) = capsys.readouterr().err.strip().splitlines()
    assert "Total children ever born" in line
    assert re.search(r"\b1\b", line), line


@pytest.mark.xfail(strict=True, reason="#5: cd-ceb must exit 2 on null counts by default")
def test_cli_default_exits_2_on_a_null_death_count_case_b(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(cd_ceb_argv(*survey_files(tmp_path, CASE_B))) == 2
    (line,) = capsys.readouterr().err.strip().splitlines()
    # the derived column or the raw one, depending on where the policy is applied
    assert re.search(r"Children died|Sons who have died", line), line


@pytest.mark.parametrize(
    ("case", "ratio"), [(CASE_A, 0.5), (CASE_B, pytest.approx(2 / 3))], ids=["A", "B"]
)
@pytest.mark.xfail(strict=True, reason="#5: --null-measures drop not implemented")
def test_cli_drop_is_complete_case(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], case: Case, ratio: Any
) -> None:
    argv = cd_ceb_argv(*survey_files(tmp_path, case), "--null-measures", "drop")
    assert main(argv) == 0
    rows, *_, got = last_cell(capsys.readouterr().out)
    assert (rows, got) == (1, ratio)


@pytest.mark.parametrize(
    ("case", "expected"),
    # the issue's evidence: today's cd-ceb output for each case
    [(CASE_A, (2, 4.0, 2.0, 2.0)), (CASE_B, (2, 3.0, 6.0, 0.5))],
    ids=["A", "B"],
)
@pytest.mark.xfail(strict=True, reason="#5: --null-measures zero not implemented")
def test_cli_zero_reproduces_todays_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], case: Case, expected: tuple[Any, ...]
) -> None:
    argv = cd_ceb_argv(*survey_files(tmp_path, case), "--null-measures", "zero")
    assert main(argv) == 0
    assert last_cell(capsys.readouterr().out) == expected


@pytest.mark.xfail(strict=True, reason="#5: Children died is null when either death count is null")
def test_children_died_is_null_if_one_death_count_is_null(tmp_path: Path) -> None:
    women, households = survey_files(tmp_path, CASE_B)
    joined = join_women_households(load_women(women), load_households(households)).collect()
    assert joined.sort("Cluster number")[NUMERATOR].to_list() == [None, 2]


@pytest.mark.xfail(strict=True, reason="#5: cd_ceb_cube(null_measures='drop') not implemented")
def test_survey_library_drop_case_b(tmp_path: Path) -> None:
    women, households = survey_files(tmp_path, CASE_B)
    cube = cd_ceb_cube(women, households, dimensions=[AGE], null_measures="drop")  # type: ignore[call-arg]
    assert cube["rows"].to_list() == [1]
    assert cube["ratio"].to_list() == pytest.approx([2 / 3])


def test_shipped_survey_has_no_null_counts_so_its_numbers_do_not_move() -> None:
    """#5 must not change the published numbers: the three count columns have no nulls."""
    women = load_women(DATA_DIR / "women.csv").collect()
    counts = ("Total children ever born", "Sons who have died", "Daughters who have died")
    assert [women[c].null_count() for c in counts] == [0, 0, 0]


# --- validation (#7) ----------------------------------------------------------------------------


def frame_with(num: list[float], den: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {"g": ["x"] * len(num), "num": num, "den": den},
        schema={"g": pl.String, "num": pl.Float64, "den": pl.Float64},
    )


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(frame_with([1.0, float("nan")], [2, 2]), id="nan-numerator"),
        pytest.param(frame_with([1.0, 1.0], [2, float("nan")]), id="nan-denominator"),
        pytest.param(frame_with([1.0, float("inf")], [2, 2]), id="inf"),
        pytest.param(frame_with([1.0, 1.0], [2, float("-inf")]), id="-inf"),
    ],
)
@pytest.mark.xfail(strict=True, reason="#7: ratio-cube measures must be finite")
def test_non_finite_measures_are_rejected(bad: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match=r"\b(num|den)\b"):
        ratio_cube(bad, ["g"], "num", "den")


def test_ratio_cube_accepts_negative_and_fractional_measures() -> None:
    """#7 asks only for finite values in ``ratio-cube`` (non-negative integers are a count rule)."""
    row = only_row(ratio_cube(frame_with([-1.5, 0.5], [2.0, -4.0]), ["g"], "num", "den"))
    assert (row["rows"], row["numerator"], row["denominator"], row["ratio"]) == (2, -1, -2, 0.5)


@pytest.mark.parametrize("token", ["nan", "inf", "-inf"])
@pytest.mark.xfail(strict=True, reason="#7: ratio-cube must reject non-finite values by default")
def test_cli_rejects_non_finite_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], token: str
) -> None:
    path = write_text(tmp_path / "f.csv", csv_text(("g", "num", "den"), [("x", token, "2")]))
    argv = ["ratio-cube", str(path), "-d", "g", "--numerator", "num", "--denominator", "den"]
    assert main(argv) == 2
    assert len(capsys.readouterr().err.strip().splitlines()) == 1


def death_rows(born: str, sons: str, daughters: str) -> Case:
    return [("30-34", born, sons, daughters), ("30-34", "2", "0", "0")]


@pytest.mark.parametrize(
    "rows",
    [death_rows("1", "1", "1"), death_rows("-1", "0", "0"), death_rows("2", "-1", "0")],
    ids=["died>born", "negative-born", "negative-deaths"],
)
@pytest.mark.xfail(strict=True, reason="#7: cd-ceb must reject out-of-range counts by default")
def test_cli_rejects_out_of_range_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], rows: Case
) -> None:
    assert main(cd_ceb_argv(*survey_files(tmp_path, rows))) == 2
    assert len(capsys.readouterr().err.strip().splitlines()) == 1


@pytest.mark.xfail(strict=True, reason="#7: --on-invalid drop not implemented")
def test_cli_on_invalid_drop_removes_the_woman(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = cd_ceb_argv(*survey_files(tmp_path, death_rows("1", "1", "1")), "--on-invalid", "drop")
    assert main(argv) == 0
    assert last_cell(capsys.readouterr().out) == (1, 0.0, 2.0, 0.0)


def run_cli(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    """Exit code and stderr of ``main``; an argparse rejection (SystemExit) counts as an exit."""
    try:
        rc = main(argv)
    except SystemExit as exit_:
        rc = exit_.code if isinstance(exit_.code, int) else 2
    return rc, capsys.readouterr().err


@pytest.mark.xfail(strict=True, reason="#7: an unmatched woman must fail with --require-match")
def test_cli_unmatched_woman_fails_when_matching_is_required(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The issue allows ``--require-match`` or making it the default; either passes."""
    women_rows = [("1", "1", "30-34", "2", "0", "1"), ("7", "7", "30-34", "1", "0", "0")]
    women = write_indexed_csv(tmp_path / "w.csv", WOMEN_HEADER, women_rows)
    households = write_indexed_csv(
        tmp_path / "h.csv", HOUSEHOLD_HEADER, [("1", "1", "Urban", "40")]
    )
    argv = ["cd-ceb", "--women", str(women), "--households", str(households), "-d", RES]
    rc, err = run_cli(argv, capsys)
    if rc == 0:  # matching is opt-in
        rc, err = run_cli([*argv, "--require-match"], capsys)
    assert rc == 2
    assert "unrecognized arguments" not in err
    assert re.search(r"match|household", err, re.IGNORECASE), err


@pytest.mark.xfail(strict=True, reason="#7: data-quality summary on stderr not implemented")
def test_cli_summary_on_the_shipped_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """8,348 women, 6,290 households, 8,348 joined, 0 unmatched, 0 invalid (with -v or --report)."""
    argv = [
        "cd-ceb",
        "--women",
        str(DATA_DIR / "women.csv"),
        "--households",
        str(DATA_DIR / "household.csv"),
        "--max-dims",
        "1",
        "-o",
        str(tmp_path / "out.parquet"),
    ]
    rc, err = run_cli([*argv, "--report"], capsys)
    if rc != 0:  # the issue allows either flag
        rc, err = run_cli([*argv, "-v"], capsys)
    assert rc == 0
    for pattern in (r"8,?348", r"6,?290", r"unmatched", r"invalid"):
        assert re.search(pattern, err, re.IGNORECASE), (pattern, err)


# --- exact sums and float tolerance (#14) -------------------------------------------------------

BIG = 2**53 + 1


@pytest.mark.xfail(strict=True, reason="#14: integer measures must sum to Int64")
def test_integer_measures_give_int64_sums() -> None:
    frame = pl.DataFrame({"g": ["x", "x"], "num": [1, 2], "den": [3, 4]})
    out = ratio_cube(frame, ["g"], "num", "den")
    assert out.schema["numerator"] == pl.Int64
    assert out.schema["denominator"] == pl.Int64


@pytest.mark.xfail(strict=True, reason="#14: Int64 sums beyond 2**53 must be exact")
def test_int64_sums_beyond_2_pow_53_are_exact() -> None:
    frame = pl.DataFrame({"g": ["x", "x"], "num": [BIG, 1], "den": [BIG, 1]})
    row = ratio_cube(frame, ["g"], "num", "den").row(0, named=True)
    assert row["numerator"] == BIG + 1
    assert row["denominator"] == BIG + 1


def test_int_sums_below_2_pow_53_are_exact_and_order_independent() -> None:
    rng = random.Random(7)
    n = 3000
    base = pl.DataFrame(
        {
            "a": [rng.choice("xyz") for _ in range(n)],
            "b": [rng.choice("pq") for _ in range(n)],
            "num": [rng.randrange(0, 10**9) for _ in range(n)],
            "den": [rng.randrange(1, 10**9) for _ in range(n)],
        }
    )
    first = ratio_cube(base, ["a", "b"], "num", "den")
    for seed in range(3):
        again = ratio_cube(
            base.sample(fraction=1.0, shuffle=True, seed=seed), ["a", "b"], "num", "den"
        )
        assert again.equals(first)
    expected = sum(base.filter(pl.col("a") == "x")["num"].to_list())
    assert first.filter(pl.col("cut") == "a", pl.col("a") == "x")["numerator"].item() == expected


@pytest.fixture(scope="module")
def float_frame() -> pl.DataFrame:
    rng = random.Random(11)
    n = 20_000
    return pl.DataFrame(
        {
            "a": [rng.choice("wxyz") for _ in range(n)],
            "b": [str(rng.randrange(30)) for _ in range(n)],
            "c": [str(rng.randrange(7)) for _ in range(n)],
            "num": [rng.random() * 1e3 for _ in range(n)],
            "den": [rng.random() * 1e3 + 1 for _ in range(n)],
        }
    )


def test_float_runs_agree_within_tolerance(float_frame: pl.DataFrame) -> None:
    dims = ["a", "b", "c"]
    first = ratio_cube(float_frame, dims, "num", "den")
    second = ratio_cube(float_frame, dims, "num", "den")
    assert_frame_equal(first, second, check_exact=False, rel_tol=1e-9, abs_tol=0.0)


def test_float_row_order_does_not_change_the_cube_beyond_tolerance(
    float_frame: pl.DataFrame,
) -> None:
    dims = ["a", "b", "c"]
    reference = ratio_cube(float_frame, dims, "num", "den")
    shuffled = float_frame.sample(fraction=1.0, shuffle=True, seed=3)
    assert_frame_equal(
        ratio_cube(shuffled, dims, "num", "den"),
        reference,
        check_exact=False,
        rel_tol=1e-9,
        abs_tol=0.0,
    )


@pytest.mark.parametrize("threshold", [2, 40, 300])
def test_float_pruned_equals_filtered_full_within_tolerance(
    float_frame: pl.DataFrame, threshold: int
) -> None:
    dims = ["a", "b", "c"]
    full = ratio_cube(float_frame, dims, "num", "den")
    pruned = ratio_cube(float_frame, dims, "num", "den", min_rows=threshold)
    assert_frame_equal(
        pruned,
        full.filter(pl.col("rows") >= threshold),
        check_exact=False,
        rel_tol=1e-9,
        abs_tol=0.0,
    )


def test_integer_pruned_equals_filtered_full_exactly() -> None:
    rng = random.Random(5)
    n = 4000
    frame = pl.DataFrame(
        {
            "a": [rng.choice("xyz") for _ in range(n)],
            "b": [str(rng.randrange(12)) for _ in range(n)],
            "num": [rng.randrange(100) for _ in range(n)],
            "den": [rng.randrange(1, 100) for _ in range(n)],
        }
    )
    full = ratio_cube(frame, ["a", "b"], "num", "den")
    for threshold in (2, 50, 400):
        pruned = ratio_cube(frame, ["a", "b"], "num", "den", min_rows=threshold)
        assert pruned.equals(full.filter(pl.col("rows") >= threshold)), threshold

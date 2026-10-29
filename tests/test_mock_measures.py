"""Measure semantics: nulls (#5), validation (#7) and exact sums / float tolerance (#14).

Tests tagged with an issue number are strict xfails that pin the open defect; they name the
parameters and messages the issue proposes (``null_measures=``, ``--null-measures``,
``--on-invalid``, ``--require-match``) and must pass once the fix lands.
"""

import random
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
from conftest import HOUSEHOLD_HEADER, WOMEN_HEADER, write_indexed_csv

AGE = "Age in 5-year groups"
RES = "Type of place of residence"
CEB = "Total children ever born"

# --- nulls (#5): the two cases from the issue ---------------------------------------------------

# Case A: the first woman's children-ever-born is unknown but 3 deaths are recorded.
CASE_A = [("30-34", "NA", "2", "1"), ("30-34", "2", "0", "1")]
# Case B: the first woman has one unknown death count.
CASE_B = [("30-34", "3", "NA", "1"), ("30-34", "3", "1", "1")]


def survey_files(tmp_path: Path, women: list[tuple[str, str, str, str]]) -> tuple[Path, Path]:
    """One woman per household; ``women`` rows are (age, born, sons died, daughters died)."""
    women_rows = [(str(i), "1", age, b, s, d) for i, (age, b, s, d) in enumerate(women, 1)]
    house_rows = [(str(i), "1", "Urban" if i == 1 else "Rural", "40") for i in range(1, 3)]
    return (
        write_indexed_csv(tmp_path / "w.csv", WOMEN_HEADER, women_rows),
        write_indexed_csv(tmp_path / "h.csv", HOUSEHOLD_HEADER, house_rows),
    )


def generic_frame(case: list[tuple[str, str, str, str]]) -> pl.DataFrame:
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


def test_today_null_denominator_still_adds_its_numerator() -> None:
    """Pins the defect's mechanism (not an endorsement): numerator 4 over denominator 2."""
    row = only_row(ratio_cube(generic_frame(CASE_A), ["g"], "num", "den"))
    assert (row["rows"], row["numerator"], row["denominator"]) == (2, 4.0, 2.0)


@pytest.mark.parametrize("case", [CASE_A, CASE_B], ids=["A", "B"])
@pytest.mark.xfail(strict=True, reason="#5: default policy must raise on null measures")
def test_default_policy_errors_on_null_measures(case: list[tuple[str, str, str, str]]) -> None:
    with pytest.raises(ValueError, match=r"den|num"):
        ratio_cube(generic_frame(case), ["g"], "num", "den")


@pytest.mark.xfail(strict=True, reason="#5: error names the column and the affected row count")
def test_default_policy_message_names_column_and_count() -> None:
    with pytest.raises(ValueError, match=r"den") as info:
        ratio_cube(generic_frame(CASE_A), ["g"], "num", "den")
    assert "1" in str(info.value)


@pytest.mark.xfail(strict=True, reason="#5: null_measures='drop' (complete-case) not implemented")
def test_drop_case_a_is_complete_case() -> None:
    frame = generic_frame(CASE_A)
    row = only_row(ratio_cube(frame, ["g"], "num", "den", null_measures="drop"))  # type: ignore[call-arg]
    assert (row["rows"], row["numerator"], row["denominator"], row["ratio"]) == (1, 1.0, 2.0, 0.5)


@pytest.mark.xfail(strict=True, reason="#5: null_measures='drop' (complete-case) not implemented")
def test_drop_case_b_is_complete_case() -> None:
    frame = generic_frame(CASE_B)
    row = only_row(ratio_cube(frame, ["g"], "num", "den", null_measures="drop"))  # type: ignore[call-arg]
    assert (row["rows"], row["numerator"], row["denominator"]) == (1, 2.0, 3.0)
    assert row["ratio"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("case", "expected"),
    [(CASE_A, (2, 4.0, 2.0, 2.0)), (CASE_B, (2, 3.0, 6.0, 0.5))],
    ids=["A", "B"],
)
@pytest.mark.xfail(strict=True, reason="#5: null_measures='zero' (opt-in legacy) not implemented")
def test_zero_reproduces_todays_output(
    case: list[tuple[str, str, str, str]], expected: tuple[float, ...]
) -> None:
    frame = generic_frame(case)
    row = only_row(ratio_cube(frame, ["g"], "num", "den", null_measures="zero"))  # type: ignore[call-arg]
    assert (row["rows"], row["numerator"], row["denominator"], row["ratio"]) == expected


@pytest.mark.xfail(strict=True, reason="#5: invalid null_measures value must be rejected")
def test_unknown_null_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="null_measures"):
        ratio_cube(generic_frame(CASE_A), ["g"], "num", "den", null_measures="ignore")  # type: ignore[call-arg]


@pytest.mark.parametrize("case", [CASE_A, CASE_B], ids=["A", "B"])
@pytest.mark.xfail(strict=True, reason="#5: cd-ceb must exit 2 on null counts by default")
def test_cli_default_exits_2_on_null_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], case: list[tuple[str, str, str, str]]
) -> None:
    women, households = survey_files(tmp_path, case)
    argv = ["cd-ceb", "--women", str(women), "--households", str(households)]
    assert main([*argv, "-d", AGE, "--max-dims", "1"]) == 2
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1
    assert "children" in err.lower()


@pytest.mark.xfail(strict=True, reason="#5: --null-measures drop not implemented")
def test_cli_drop_case_a_gives_half(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    women, households = survey_files(tmp_path, CASE_A)
    argv = ["cd-ceb", "--women", str(women), "--households", str(households)]
    assert main([*argv, "-d", AGE, "--max-dims", "1", "--null-measures", "drop"]) == 0
    assert capsys.readouterr().out.splitlines()[-1].endswith(",1,1.0,2.0,0.5")


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


# --- validation (#7) ----------------------------------------------------------------------------


def frame_with(num: list[float], den: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {"g": ["x"] * len(num), "num": num, "den": den},
        schema={"g": pl.String, "num": pl.Float64, "den": pl.Float64},
    )


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(frame_with([1, -1], [2, 2]), id="negative-numerator"),
        pytest.param(frame_with([1, 1], [2, -2]), id="negative-denominator"),
        pytest.param(frame_with([1.0, float("nan")], [2, 2]), id="nan"),
        pytest.param(frame_with([1.0, float("inf")], [2, 2]), id="inf"),
        pytest.param(frame_with([1.0, 1.0], [2, float("-inf")]), id="-inf"),
    ],
)
@pytest.mark.xfail(strict=True, reason="#7: invalid measure values must raise ValueError")
def test_invalid_measures_are_rejected(bad: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match=r"num|den"):
        ratio_cube(bad, ["g"], "num", "den")


def test_today_nan_measure_poisons_the_ratio_silently() -> None:
    """Pins the mechanism behind #7: NaN flows through to a NaN ratio, rc 0, no error."""
    out = ratio_cube(frame_with([1.0, float("nan")], [2, 2]), ["g"], "num", "den")
    assert out["ratio"].is_nan().any()


def death_rows(born: str, sons: str, daughters: str) -> list[tuple[str, str, str, str]]:
    return [("30-34", born, sons, daughters), ("30-34", "2", "0", "0")]


@pytest.mark.parametrize(
    "rows",
    [death_rows("1", "1", "1"), death_rows("-1", "0", "0"), death_rows("2", "-1", "0")],
    ids=["died>born", "negative-born", "negative-deaths"],
)
@pytest.mark.xfail(strict=True, reason="#7: cd-ceb must reject out-of-range counts by default")
def test_cli_rejects_out_of_range_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], rows: list[tuple[str, str, str, str]]
) -> None:
    women, households = survey_files(tmp_path, rows)
    argv = ["cd-ceb", "--women", str(women), "--households", str(households), "-d", AGE]
    assert main([*argv, "--max-dims", "1"]) == 2
    assert len(capsys.readouterr().err.strip().splitlines()) == 1


@pytest.mark.xfail(strict=True, reason="#7: --on-invalid drop not implemented")
def test_cli_on_invalid_drop_removes_the_woman(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    women, households = survey_files(tmp_path, death_rows("1", "1", "1"))
    argv = ["cd-ceb", "--women", str(women), "--households", str(households), "-d", AGE]
    assert main([*argv, "--max-dims", "1", "--on-invalid", "drop"]) == 0
    assert capsys.readouterr().out.splitlines()[-1].endswith(",1,0.0,2.0,0.0")


@pytest.mark.xfail(strict=True, reason="#7: unmatched women must fail with --require-match")
def test_cli_require_match_fails_on_unmatched_woman(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    women_rows = [("1", "1", "30-34", "2", "0", "1"), ("7", "7", "30-34", "1", "0", "0")]
    women = write_indexed_csv(tmp_path / "w.csv", WOMEN_HEADER, women_rows)
    households = write_indexed_csv(
        tmp_path / "h.csv", HOUSEHOLD_HEADER, [("1", "1", "Urban", "40")]
    )
    argv = ["cd-ceb", "--women", str(women), "--households", str(households), "-d", RES]
    assert main([*argv, "--require-match"]) == 2
    assert "match" in capsys.readouterr().err


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
    assert int(row["numerator"]) == BIG + 1
    assert int(row["denominator"]) == BIG + 1


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

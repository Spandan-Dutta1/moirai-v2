"""The replacement-share estimator's logic, testable without the panel.

The data is licensed and lives outside the repository, so the estimate itself
cannot run here. What can, and what took three attempts to get right, is the
month arithmetic the windows are built from and the refusal to run without a
data path. The estimate was wrong twice because the windows pointed at the
months after the interview rather than the months it asks about, and nothing
in the code objected either time.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import check_cphs_replacement as share  # noqa: E402


def test_a_month_label_becomes_an_ordinal_that_supports_arithmetic() -> None:
    assert share.to_ordinal("Mar 2022") == 2022 * 12 + 3
    assert share.to_ordinal("Jan 2022") + 1 == share.to_ordinal("Feb 2022")
    assert share.to_ordinal("Dec 2021") + 1 == share.to_ordinal("Jan 2022")


def test_twelve_months_apart_is_twelve() -> None:
    assert share.to_ordinal("Jun 2026") - share.to_ordinal("Jun 2025") == 12


@pytest.mark.parametrize("label", ["Jan 2021", "Aug 2025", "Dec 2026"])
def test_every_month_name_is_recognised(label: str) -> None:
    assert share.to_ordinal(label) > 0


def test_both_windows_sit_before_the_interview_month() -> None:
    """The error that produced 1.011 twice: measuring the recovery instead.

    CPHS asks each interview about the preceding four months, so every offset
    used must be negative. A window reaching month zero or beyond is measuring
    what happened after the spell was recorded.
    """
    for before_offsets, during_offsets in share.WINDOWS.values():
        assert all(offset < 0 for offset in before_offsets)
        assert all(offset < 0 for offset in during_offsets)


def test_the_before_window_precedes_the_during_window() -> None:
    for before_offsets, during_offsets in share.WINDOWS.values():
        assert max(before_offsets) < min(during_offsets)


def test_the_windows_are_the_same_length_on_both_sides() -> None:
    """Unequal windows would compare a mean of four months against two."""
    for before_offsets, during_offsets in share.WINDOWS.values():
        assert len(list(before_offsets)) == len(list(during_offsets))


def test_both_income_measures_are_reported() -> None:
    """Wages validate the method; household income is what the model uses."""
    assert set(share.MEASURES.values()) == {
        "TOT_INC",
        "INC_OF_ALL_MEMS_FRM_WAGES",
    }


def test_the_script_refuses_to_run_without_a_data_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MOIRAI_DATA", raising=False)
    with pytest.raises(SystemExit, match="MOIRAI_DATA"):
        share.data_root()


def test_a_data_path_missing_either_half_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOIRAI_DATA", str(tmp_path))
    (tmp_path / "panel").mkdir()
    (tmp_path / "income").mkdir()
    with pytest.raises(SystemExit, match="no poi_"):
        share.data_root()


def test_the_model_uses_the_household_measure_not_the_wage_one() -> None:
    """ADR 038 took the wage figure where the model needed the household one."""
    from moirai.engine.economy.households import (
        CPHS_HOUSEHOLD_INCOME_SHARE,
        CPHS_WAGE_SHARE,
        NonEmploymentIncomeParameters,
    )

    assert CPHS_WAGE_SHARE < CPHS_HOUSEHOLD_INCOME_SHARE
    assert (
        pytest.approx(CPHS_HOUSEHOLD_INCOME_SHARE)
        == NonEmploymentIncomeParameters().unemployed
    )


def test_the_wage_estimate_reproduces_the_published_figure() -> None:
    """0.57 published, 0.598 measured here, never fitted to it. ADR 040."""
    from moirai.engine.economy.households import CPHS_WAGE_SHARE

    published = 1.0 - 0.43
    assert abs(CPHS_WAGE_SHARE - published) < 0.05

"""The CPHS labour script's logic, which is testable without the data.

The panel is licensed and lives outside the repository, so the script's
output cannot be tested here. What can be, and what has already broken twice,
is the part that maps a survey's categories onto the model's: CPHS renamed its
employment-arrangement values in May 2022 and dropped its weight columns in
May 2023, and both changes produced plausible wrong numbers rather than
errors. These tests cover the classifier, the rate conversion, and the refusal
to run without a data path.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import check_cphs_labour as labour  # noqa: E402

#: Every value CPHS used across the nineteen waves from 2019 to 2025.
REAL_CATEGORIES = {
    "Salaried - Permanent": "salaried",
    "Salaried - Temporary": "salaried",
    "Salaried - Temporary, not on digital platform": "salaried",
    "Digital platform based salaried temporary": "salaried",
    "Self-employed": "self",
    "Self-employed not on digital platform": "self",
    "Digital platform based self employed": "self",
    "Daily Wage worker/ Casual labour": "casual",
    "Not Applicable": "none",
    "Data Not Available": "none",
}


@pytest.mark.parametrize(("value", "expected"), sorted(REAL_CATEGORIES.items()))
def test_every_category_cphs_has_used_is_classified(value: str, expected: str) -> None:
    assert labour.arrangement_group(value) == expected


def test_the_renamed_categories_land_with_the_ones_they_replaced() -> None:
    """The May 2022 rename: matching the old names dropped a third of them."""
    before = labour.arrangement_group("Salaried - Temporary")
    after = labour.arrangement_group("Salaried - Temporary, not on digital platform")
    platform = labour.arrangement_group("Digital platform based salaried temporary")
    assert before == after == platform == "salaried"


def test_an_unrecognised_category_stops_the_run() -> None:
    """A silent drop is what made the last rename look like a real shift."""
    with pytest.raises(ValueError, match="unrecognised employment arrangement"):
        labour.arrangement_group("Apprentice on a stipend")


def test_a_four_month_rate_becomes_a_smaller_monthly_one() -> None:
    assert labour.monthly(0.10) < 0.10
    assert labour.monthly(0.0) == pytest.approx(0.0)
    assert labour.monthly(1.0) == pytest.approx(1.0)


def test_four_months_of_the_monthly_rate_rebuild_the_original() -> None:
    for four_month in (0.05, 0.10, 0.45):
        assert pytest.approx(four_month) == 1.0 - (1.0 - labour.monthly(four_month)) ** 4


def test_the_script_refuses_to_run_without_a_data_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Licensed data lives outside the repository, so the path is required."""
    monkeypatch.delenv("MOIRAI_DATA", raising=False)
    with pytest.raises(SystemExit, match="MOIRAI_DATA"):
        labour.panel_directory()


def test_a_data_path_with_no_waves_in_it_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOIRAI_DATA", str(tmp_path))
    (tmp_path / "panel").mkdir()
    with pytest.raises(SystemExit, match="no poi_"):
        labour.panel_directory()


def test_the_model_default_is_the_rate_this_script_measured() -> None:
    """If one moves without the other, the comparison it prints is a lie."""
    from moirai.engine.economy.behaviour import (
        CPHS_JOB_FINDING_RATE,
        BehaviourParameters,
    )

    assert pytest.approx(CPHS_JOB_FINDING_RATE) == BehaviourParameters().job_finding_rate
    assert pytest.approx(0.18) == labour.PRIOR_JOB_FINDING_RATE

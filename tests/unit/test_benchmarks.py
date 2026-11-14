"""External benchmarking of the model against published estimates (Stage 6).

The script's figures are the result, so they are not what these tests assert.
What is asserted is the shape the comparison has to have for the figures to
mean anything: that a verdict reports position against the published range
honestly in both signs, that pass-through rises with the horizon rather than
arriving at once, that floating-rate borrowers are the ones who cut, and that
the chain multiplies in the order the model claims.

The sizes are small on purpose. The script runs 200,000 households and 100,000
loans; these run a few thousand, which is enough to fix every sign and
ordering and fast enough to belong in the suite.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import check_benchmarks as bench  # noqa: E402

LOANS = 3_000
HOUSEHOLDS = 4_000


@pytest.fixture(scope="module")
def slope() -> float:
    return bench.rbi_response_to_the_fed()


@pytest.fixture(scope="module")
def rates() -> dict[str, float]:
    return bench.lending_rate_pass_through(n=LOANS)


@pytest.fixture(scope="module")
def households() -> dict[str, float]:
    return bench.household_response(n=HOUSEHOLDS)


def _benchmark(model: float, low: float, high: float, **kwargs: object) -> bench.Benchmark:
    defaults: dict[str, object] = {
        "link": "link",
        "published": "published",
        "unit": "x",
        "source": "source",
    }
    defaults.update(kwargs)
    return bench.Benchmark(model=model, low=low, high=high, **defaults)  # type: ignore[arg-type]


def test_a_model_value_within_the_published_range_is_inside() -> None:
    assert _benchmark(0.78, 0.70, 0.90).verdict == "inside the range"


def test_the_range_includes_its_own_endpoints() -> None:
    assert _benchmark(0.70, 0.70, 0.90).verdict == "inside the range"
    assert _benchmark(0.90, 0.70, 0.90).verdict == "inside the range"


def test_a_value_outside_the_range_names_the_side_it_is_on() -> None:
    assert _benchmark(0.40, 0.70, 0.90).verdict == "below the range"
    assert _benchmark(1.20, 0.70, 0.90).verdict == "above the range"


def test_a_negative_range_reports_position_not_magnitude() -> None:
    """A smaller cut than published is above the range, not below it."""
    assert _benchmark(-1.20, -2.40, -1.50).verdict == "above the range"
    assert _benchmark(-3.00, -2.40, -1.50).verdict == "below the range"
    assert _benchmark(-1.95, -2.40, -1.50).verdict == "inside the range"


def test_an_incomparable_measure_is_never_given_a_verdict() -> None:
    """The IMF row measures a bond yield, so no range can pass or fail it."""
    item = _benchmark(12.8, 40.0, 60.0, comparable=False)
    assert item.verdict == "not like-for-like"


def test_the_reserve_bank_follows_the_fed_without_matching_it(slope: float) -> None:
    assert 0.0 < slope < 1.0


def test_the_reserve_bank_response_is_deterministic(slope: float) -> None:
    assert bench.rbi_response_to_the_fed() == pytest.approx(slope)


def test_pass_through_is_a_fraction_of_the_shock(rates: dict[str, float]) -> None:
    assert all(0.0 <= value <= 1.0 for value in rates.values())


def test_pass_through_rises_with_the_horizon(rates: dict[str, float]) -> None:
    """A benchmark reset takes a quarter, so a year moves more than three months."""
    assert rates["book_12m"] > rates["book_3m"]
    assert rates["floating_12m"] > rates["floating_3m"]


def test_the_floating_part_of_the_book_moves_more_than_the_whole(
    rates: dict[str, float],
) -> None:
    assert rates["floating_12m"] > rates["book_12m"]
    assert rates["floating_3m"] > rates["book_3m"]


def test_the_lending_rate_rises_by_more_than_the_deposit_rate(
    households: dict[str, float],
) -> None:
    """Borrowers absorb more of a tightening than savers receive."""
    assert households["lending"] > households["deposit"] > 0.0
    assert households["lending"] < 1.0


def test_floating_borrowers_cut_relative_to_households_with_no_debt(
    households: dict[str, float],
) -> None:
    assert households["gap_impact"] < 0.0
    assert households["gap_12m"] < 0.0


def test_the_consumption_gap_is_the_size_the_micro_evidence_reports(
    households: dict[str, float],
) -> None:
    """Within a factor of two of Floden et al. (2017), in the right direction."""
    assert -4.0 < households["gap_12m"] < -0.5


def test_the_marginal_propensity_is_a_fraction_of_the_extra_instalment(
    households: dict[str, float],
) -> None:
    assert 0.0 < households["mpc_impact"] < 1.0
    assert 0.0 < households["mpc_12m"] < 1.0


def test_the_response_is_front_loaded_which_is_the_open_finding(
    households: dict[str, float],
) -> None:
    """The defect the comparison exposed, asserted so a fix has to change it."""
    assert households["mpc_impact"] > households["mpc_12m"]
    assert abs(households["gap_impact"]) > abs(households["gap_12m"])


def test_every_link_is_reported_with_a_verdict_and_a_source(
    slope: float, rates: dict[str, float], households: dict[str, float]
) -> None:
    table = bench.benchmarks(slope, rates, households)
    assert len(table) == 5
    assert all(item.link and item.published and item.source for item in table)
    assert all(item.low <= item.high for item in table)


def test_exactly_one_link_is_declared_not_comparable(
    slope: float, rates: dict[str, float], households: dict[str, float]
) -> None:
    table = bench.benchmarks(slope, rates, households)
    assert sum(not item.comparable for item in table) == 1


def test_the_chain_is_smaller_than_either_link_that_makes_it(
    slope: float, households: dict[str, float]
) -> None:
    """A Fed shock reaches an Indian borrower only through the Reserve Bank."""
    implied = slope * abs(households["gap_12m"])
    assert 0.0 < implied < abs(households["gap_12m"])


def test_the_chain_reproduces_the_headline_this_project_published(
    slope: float, households: dict[str, float]
) -> None:
    low, high = bench.HEADLINE
    assert low <= slope * abs(households["gap_12m"]) <= high


def test_the_household_shock_is_a_deviation_from_the_stated_baseline() -> None:
    """Every variable is held at baseline except the policy rate."""
    assert set(bench.BASELINE) == {
        bench.MacroVariable.POLICY_RATE,
        bench.MacroVariable.INFLATION,
        bench.MacroVariable.INCOME_GROWTH,
        bench.MacroVariable.UNEMPLOYMENT,
    }
    assert pytest.approx(0.01) == bench.SHOCK

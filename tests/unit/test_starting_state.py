"""The RBI's starting inflation read from data, reported beside the declared one (ADR 029).

Written before the module existed. The declared starting inflation (4.4%) had
gone stale against the warehouse's latest CPI Combined reading (3.93%). The
fix reads the observed value and keeps a record of both. Two tests pin the
economics: with no condition on the RBI, the caused move does not depend on
the starting level; with one (twin and global tightening declare Indian
inflation as a level), the caused move absorbs the change in the level
exactly, because the condition fixes the shocked solve and only the
reference solve moves. The comparison run found the second case.
"""

from datetime import UTC, date, datetime

import pytest

from moirai.core.exceptions import EngineError
from moirai.engine.data_fabric.series.models import (
    Frequency,
    Observation,
    SeasonalAdjustment,
    SeriesMetadata,
    Source,
    TimeSeries,
    Unit,
)
from moirai.engine.data_fabric.warehouse.duckdb_store import Warehouse
from moirai.engine.financial.central_banks import MAJOR_CENTRAL_BANKS, RBI
from moirai.engine.financial.network import DEFAULT_TIERS, SpilloverMatrix, network_nash
from moirai.engine.financial.starting_state import (
    RBI_INFLATION_SERIES,
    latest_inflation,
    observed_starting_inflation,
    with_observed_inflation,
)
from moirai.engine.scenarios import DEFAULT_SCENARIOS, HEADLINE_SCENARIO, _solve

KNOWN = datetime(2026, 6, 12, tzinfo=UTC)
MONTHS = [date(2026, m, 1) for m in range(1, 6)]


def cpi(values: list[float | None], unit: Unit = Unit.PERCENT_CHANGE) -> TimeSeries:
    """CPI Combined year on year, in percent, January to May 2026."""
    metadata = SeriesMetadata(
        series_id=RBI_INFLATION_SERIES,
        title="CPI Combined, year on year",
        source=Source.RBI,
        source_series_id="CPI_COMBINED_YOY",
        frequency=Frequency.MONTHLY,
        unit=unit,
        unit_label="Percent, year on year",
        seasonal_adjustment=SeasonalAdjustment.NOT_ADJUSTED,
        geography="IN",
        start=MONTHS[0],
        end=MONTHS[-1],
    )
    observations = [
        Observation(period=p, value=v, known_at=KNOWN) for p, v in zip(MONTHS, values, strict=True)
    ]
    return TimeSeries.from_observations(metadata, observations)


MAY_2026 = cpi([4.26, 3.61, 3.34, 3.16, 3.93])


# ---- reading the observed value ------------------------------------------------


def test_the_observed_value_replaces_the_declared_one() -> None:
    bank, record = with_observed_inflation(RBI, MAY_2026)

    assert bank.current_inflation == pytest.approx(0.0393)
    assert record.declared == pytest.approx(0.044)
    assert record.observed == pytest.approx(0.0393)
    assert record.period == date(2026, 5, 1)
    assert record.used == "observed"


def test_the_declared_bank_itself_is_unchanged() -> None:
    with_observed_inflation(RBI, MAY_2026)
    assert RBI.current_inflation == pytest.approx(0.044)


def test_trailing_missing_months_are_skipped() -> None:
    period, value = latest_inflation(cpi([4.26, 3.61, 3.34, 3.16, None]))
    assert period == date(2026, 4, 1)
    assert value == pytest.approx(0.0316)


def test_the_summary_reports_both_values() -> None:
    _, record = with_observed_inflation(RBI, MAY_2026)
    text = record.summary()
    assert "4.40%" in text and "3.93%" in text and "May 2026" in text


# ---- refusals --------------------------------------------------------------------


def test_an_index_level_is_not_an_inflation_rate() -> None:
    with pytest.raises(EngineError):
        latest_inflation(cpi([190.1, 190.8, 191.2, 191.9, 192.4], unit=Unit.INDEX))


def test_a_series_with_no_values_is_refused() -> None:
    with pytest.raises(EngineError):
        latest_inflation(cpi([None, None, None, None, None]))


def test_an_implausible_value_is_refused() -> None:
    with pytest.raises(EngineError):
        latest_inflation(cpi([4.26, 3.61, 3.34, 3.16, 393.0]))


def test_a_series_with_several_vintages_is_refused() -> None:
    revised = MAY_2026.observations[-1].model_copy(
        update={"value": 3.90, "known_at": datetime(2026, 7, 12, tzinfo=UTC), "revision": 1}
    )
    series = TimeSeries.from_observations(MAY_2026.metadata, [*MAY_2026.observations, revised])
    with pytest.raises(EngineError):
        latest_inflation(series)


# ---- the warehouse ---------------------------------------------------------------


def test_read_from_the_warehouse() -> None:
    with Warehouse(":memory:") as warehouse:
        warehouse.upsert_series(MAY_2026)
        bank, record = observed_starting_inflation(RBI, warehouse)

    assert bank.current_inflation == pytest.approx(0.0393)
    assert record.used == "observed"


def test_without_the_series_the_declared_value_is_kept_and_said() -> None:
    with Warehouse(":memory:") as warehouse:
        bank, record = observed_starting_inflation(RBI, warehouse)

    assert bank.current_inflation == pytest.approx(0.044)
    assert record.used == "declared"
    assert "declared value is used" in record.summary()


# ---- the economic claim ----------------------------------------------------------


def caused_rbi_move(rbi_inflation: float) -> float:
    banks = tuple(
        b.model_copy(update={"current_inflation": rbi_inflation}) if b.name == RBI.name else b
        for b in MAJOR_CENTRAL_BANKS
    )
    spillovers = SpilloverMatrix.from_literature(DEFAULT_TIERS)
    shocked = network_nash(HEADLINE_SCENARIO.apply_to(banks), spillovers)
    reference = network_nash(banks, spillovers)
    return shocked.rates[RBI.name] - reference.rates[RBI.name]


def test_the_caused_move_does_not_depend_on_the_starting_level() -> None:
    """Linear while no bank leaves its band: the level cancels from the difference."""
    assert caused_rbi_move(0.0393) == pytest.approx(caused_rbi_move(0.044), abs=1e-12)


def test_the_unprompted_move_does() -> None:
    """The reference solve is measured from the level, so it moves."""
    spillovers = SpilloverMatrix.from_literature(DEFAULT_TIERS)

    def reference_rbi(rbi_inflation: float) -> float:
        banks = tuple(
            b.model_copy(update={"current_inflation": rbi_inflation}) if b.name == RBI.name else b
            for b in MAJOR_CENTRAL_BANKS
        )
        return network_nash(banks, spillovers).rates[RBI.name]

    assert reference_rbi(0.0393) < reference_rbi(0.044)


def test_with_a_condition_on_the_rbi_the_caused_move_absorbs_the_level() -> None:
    """Twin tightening declares Indian inflation as a level, so the shock is
    condition minus starting value: a lower start makes the same condition a
    bigger shock, by exactly the fall in the reference rate."""
    twin = next(s for s in DEFAULT_SCENARIOS if s.name == "twin_tightening")
    assert any(c.bank == RBI.name and c.inflation is not None for c in twin.conditions)
    spillovers = SpilloverMatrix.from_literature(DEFAULT_TIERS)

    def solves(rbi_inflation: float) -> tuple[float, float]:
        banks = tuple(
            b.model_copy(update={"current_inflation": rbi_inflation}) if b.name == RBI.name else b
            for b in MAJOR_CENTRAL_BANKS
        )
        shocked = _solve(twin, twin.apply_to(banks), spillovers)
        reference = _solve(twin, banks, spillovers)
        return shocked.rates[RBI.name], reference.rates[RBI.name]

    shocked_old, reference_old = solves(0.044)
    shocked_new, reference_new = solves(0.0393)

    assert shocked_new == pytest.approx(shocked_old, abs=1e-12)  # the condition fixes it
    caused_change = (shocked_new - reference_new) - (shocked_old - reference_old)
    assert caused_change == pytest.approx(reference_old - reference_new, abs=1e-12)
    assert caused_change > 0

"""
A central bank's starting state, read from data rather than declared (ADR 029).

The banks in central_banks.py carry declared starting values. They are
written down once, so they go stale: the RBI's declared starting inflation
of 4.4% was June 2026's CPI Combined, while the warehouse's latest reading
is 3.93% (May 2026). This module reads the observed value and returns the
bank at it, together with a record of both numbers, so a script can report
the declared state beside the observed one instead of replacing it silently.

What the starting inflation does and does not move:

  * The game's caused move does not change. The network is linear while no
    bank leaves its band, and the starting inflation enters the solve with
    the conditions and the reference solve without them identically, so it
    cancels from their difference. A test pins this.
  * The unprompted move does change, because it is measured from the level.
  * Households see the level. The seam holds inflation at the bank's
    starting value, and behaviour erodes wealth and debt by it each month,
    so household results move, by a second-order amount.

The warehouse is passed in rather than opened here, so importing the engine
never touches a database.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from moirai.core.exceptions import EngineError
from moirai.engine.data_fabric.series.models import TimeSeries, Unit
from moirai.engine.financial.central_banks import CentralBank

if TYPE_CHECKING:
    from moirai.engine.data_fabric.warehouse.duckdb_store import Warehouse

#: The RBI's target measure as the warehouse holds it: CPI Combined, year on year.
RBI_INFLATION_SERIES = "in_cpi_inflation"

#: Above this an observed inflation rate is treated as a data error, not a reading.
MAX_PLAUSIBLE_INFLATION = 0.50

#: Units that can carry an inflation rate. An index level cannot.
RATE_UNITS = frozenset({Unit.PERCENT_CHANGE, Unit.PERCENT})


class StartingInflation(BaseModel):
    """The declared and observed starting inflation of one bank."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bank: str
    declared: float
    observed: float | None = Field(description="None when no observed series was available.")
    period: date | None = None
    series_id: str
    used: str = Field(description='"observed" or "declared".')

    def summary(self) -> str:
        if self.observed is None or self.period is None:
            return (
                f"{self.bank}: starting inflation {self.declared:.2%} declared; "
                f"no {self.series_id} in the warehouse, so the declared value is used"
            )
        return (
            f"{self.bank}: starting inflation declared {self.declared:.2%}, "
            f"observed {self.observed:.2%} ({self.period:%B %Y}, {self.series_id}); "
            f"using observed"
        )


def latest_inflation(series: TimeSeries) -> tuple[date, float]:
    """The most recent non-missing value of a published inflation rate, as a decimal.

    Raises
    ------
    EngineError
        If the series is not a rate, still holds several vintages, has no
        values, or its latest value is not a plausible inflation rate.
    """
    if series.metadata.unit not in RATE_UNITS:
        raise EngineError(
            f"{series.series_id} has unit {series.metadata.unit.value}; a starting "
            f"inflation needs a published rate, not an index level"
        )
    if series.has_revisions:
        raise EngineError(
            f"{series.series_id} holds several vintages; pass latest() or as_of() "
            f"so the vintage the starting state comes from is explicit"
        )
    present = [o for o in series.observations if o.value is not None]
    if not present:
        raise EngineError(f"{series.series_id} has no values")
    last = max(present, key=lambda o: o.period)
    assert last.value is not None
    value = last.value / 100.0
    if abs(value) > MAX_PLAUSIBLE_INFLATION:
        raise EngineError(
            f"{series.series_id} latest value {last.value} ({last.period}) is not a "
            f"plausible inflation rate in percent"
        )
    return last.period, value


def with_observed_inflation(
    bank: CentralBank, series: TimeSeries
) -> tuple[CentralBank, StartingInflation]:
    """The bank at its observed starting inflation, and the record of both values."""
    period, value = latest_inflation(series)
    record = StartingInflation(
        bank=bank.name,
        declared=bank.current_inflation,
        observed=value,
        period=period,
        series_id=series.series_id,
        used="observed",
    )
    return bank.model_copy(update={"current_inflation": value}), record


def observed_starting_inflation(
    bank: CentralBank, warehouse: Warehouse, series_id: str = RBI_INFLATION_SERIES
) -> tuple[CentralBank, StartingInflation]:
    """Read the bank's observed starting inflation from the warehouse.

    When the warehouse does not hold the series, as on a fresh clone with no
    data, the declared value is kept and the record says so, rather than
    failing a run that can still proceed on a declared state.
    """
    if not warehouse.has_series(series_id):
        return bank, StartingInflation(
            bank=bank.name,
            declared=bank.current_inflation,
            observed=None,
            series_id=series_id,
            used="declared",
        )
    return with_observed_inflation(bank, warehouse.latest(series_id))

"""
Canonical representation of an economic time series.

Every source (FRED, World Bank, RBI) normalises into these types, so the
rest of Moirai never sees a source-specific shape. Adding a source means
writing a translator, not touching any layer above.

The central idea is bitemporality. Each observation carries two independent
timestamps:

    period      when the fact was true in the world
    known_at    when Moirai learned it

Indian Q1 GDP is published two months late and revised twice. "Q1 2025 GDP"
therefore has several values depending on when you ask. Storing both axes is
what lets a backtest ask the only honest question: what did we know on the
day the forecast was made?
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Frequency(StrEnum):
    """Observation frequency. Ordered coarsest to finest by `rank`."""

    ANNUAL = "annual"
    SEMIANNUAL = "semiannual"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"
    WEEKLY = "weekly"
    DAILY = "daily"

    @property
    def rank(self) -> int:
        """Higher means finer. Used to forbid upsampling."""
        return _FREQUENCY_RANK[self]

    @property
    def periods_per_year(self) -> float:
        return _PERIODS_PER_YEAR[self]


_FREQUENCY_RANK: dict[Frequency, int] = {
    Frequency.ANNUAL: 0,
    Frequency.SEMIANNUAL: 1,
    Frequency.QUARTERLY: 2,
    Frequency.MONTHLY: 3,
    Frequency.WEEKLY: 4,
    Frequency.DAILY: 5,
}

_PERIODS_PER_YEAR: dict[Frequency, float] = {
    Frequency.ANNUAL: 1.0,
    Frequency.SEMIANNUAL: 2.0,
    Frequency.QUARTERLY: 4.0,
    Frequency.MONTHLY: 12.0,
    Frequency.WEEKLY: 52.0,
    Frequency.DAILY: 365.0,
}


class SeasonalAdjustment(StrEnum):
    NOT_ADJUSTED = "nsa"
    SEASONALLY_ADJUSTED = "sa"
    SEASONALLY_ADJUSTED_ANNUAL_RATE = "saar"
    UNKNOWN = "unknown"


class Source(StrEnum):
    """Where a series came from. Extend as adapters are added."""

    FRED = "fred"
    WORLD_BANK = "world_bank"
    RBI = "rbi"
    DATA_GOV_IN = "data_gov_in"
    MANUAL = "manual"
    DERIVED = "derived"


class Unit(StrEnum):
    """Coarse unit class. The precise string lives in `unit_label`."""

    PERCENT = "percent"
    PERCENT_CHANGE = "percent_change"
    INDEX = "index"
    CURRENCY = "currency"
    COUNT = "count"
    RATIO = "ratio"
    UNKNOWN = "unknown"


class Observation(BaseModel):
    """A single bitemporal data point.

    `value` is optional because a missing observation is itself information:
    a series can be published with a gap, and erasing that gap would be a
    silent lie about what was known.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    period: date = Field(description="Start of the period the value describes (valid time).")
    value: float | None = Field(default=None, description="None means published-but-missing.")
    known_at: datetime = Field(description="When Moirai learned this value (transaction time).")
    revision: int = Field(default=0, ge=0, description="0 is first release; higher is a revision.")

    @field_validator("value")
    @classmethod
    def _reject_non_finite(cls, value: float | None) -> float | None:
        """NaN and infinity must never enter the fabric.

        A NaN that reaches the warehouse propagates silently through every
        downstream computation and surfaces as an empty chart, not an error.
        """
        if value is None:
            return None
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"value must be finite, got {value!r}")
        return value

    @property
    def is_missing(self) -> bool:
        return self.value is None


class SeriesMetadata(BaseModel):
    """What a series *is*, independent of its values.

    Rich metadata is not documentation, it is a precondition for correct
    computation. Differencing a series that is already a percent change,
    or comparing SA against NSA data, produces plausible nonsense. These
    fields let such errors be caught mechanically.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    series_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=512)
    source: Source
    source_series_id: str = Field(min_length=1, description="Identifier used by the source.")

    frequency: Frequency
    unit: Unit = Unit.UNKNOWN
    unit_label: str = Field(default="", max_length=128, description="Verbatim source unit string.")
    seasonal_adjustment: SeasonalAdjustment = SeasonalAdjustment.UNKNOWN

    geography: str = Field(default="", max_length=64, description="ISO code or region name.")
    start: date | None = None
    end: date | None = None

    notes: str = Field(default="", max_length=4096)
    license: str = Field(default="", max_length=256)
    retrieved_at: datetime | None = None

    @field_validator("series_id")
    @classmethod
    def _series_id_is_a_safe_token(cls, value: str) -> str:
        """series_id becomes a filename and a table name, so restrict it now."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("series_id must not be empty")
        if not all(ch.isalnum() or ch in "_-." for ch in cleaned):
            raise ValueError(
                f"series_id may contain only letters, digits, underscore, hyphen "
                f"and dot; got {value!r}"
            )
        return cleaned

    @model_validator(mode="after")
    def _coverage_is_ordered(self) -> Self:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")
        return self


class TimeSeries(BaseModel):
    """Metadata plus observations, sorted and internally consistent.

    Invariants enforced at construction:
      * observations are ordered by (period, known_at, revision)
      * no duplicate (period, known_at, revision) triple
      * metadata coverage matches the observations actually present
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metadata: SeriesMetadata
    observations: tuple[Observation, ...] = ()

    @model_validator(mode="after")
    def _sorted_and_unique(self) -> Self:
        keys = [(o.period, o.known_at, o.revision) for o in self.observations]
        if len(set(keys)) != len(keys):
            duplicate = next(k for k in keys if keys.count(k) > 1)
            raise ValueError(f"duplicate observation for {duplicate}")
        if keys != sorted(keys):
            raise ValueError("observations must be sorted by (period, known_at, revision)")
        return self

    # ---- convenience ----

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def series_id(self) -> str:
        return self.metadata.series_id

    @property
    def periods(self) -> tuple[date, ...]:
        """Distinct periods covered, in order."""
        return tuple(dict.fromkeys(o.period for o in self.observations))

    @property
    def is_empty(self) -> bool:
        return not self.observations

    @property
    def has_revisions(self) -> bool:
        """True if any period carries more than one vintage."""
        return len(self.periods) != len(self.observations)

    def to_records(self) -> list[dict[str, Any]]:
        """Flat rows for the warehouse writer or a DataFrame constructor."""
        return [
            {
                "series_id": self.metadata.series_id,
                "period": o.period,
                "value": o.value,
                "known_at": o.known_at,
                "revision": o.revision,
            }
            for o in self.observations
        ]

    @classmethod
    def from_observations(
        cls, metadata: SeriesMetadata, observations: list[Observation]
    ) -> TimeSeries:
        """Build from unsorted observations, sorting on the way in.

        Adapters use this: sources return data in arbitrary order, and
        sorting at the boundary keeps that concern out of every adapter.
        """
        ordered = sorted(observations, key=lambda o: (o.period, o.known_at, o.revision))
        return cls(metadata=metadata, observations=tuple(ordered))
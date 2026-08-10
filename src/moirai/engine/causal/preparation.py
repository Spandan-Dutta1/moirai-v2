"""
Data preparation for causal models.

Two series that both trend upward will appear strongly related even when
they are independent: the regression picks up the shared trend, not a
relationship. Granger and Newbold (1974) showed this produces significant
t-statistics from pure noise. Every model in Layer 1 assumes stationary
inputs, so this module is not a convenience step, it is a correctness
precondition.

Two things make this Moirai-specific rather than a generic wrapper:

  * It reads SeriesMetadata. A series already published as a percent
    change is not differenced again, because the unit says so.
  * Every transformation is recorded. The output carries the exact
    sequence applied, which goes into the run ledger. "What was done to
    the data" is then answerable from the record, not from memory.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from statsmodels.tsa.stattools import adfuller, kpss

from moirai.core.exceptions import EngineError
from moirai.core.logging import get_logger
from moirai.engine.data_fabric.series.models import TimeSeries, Unit

log = get_logger(__name__)

#: Minimum usable observations. Below this, unit root tests are noise.
MIN_OBSERVATIONS = 20


class Transformation(StrEnum):
    """A single operation applied to a series."""

    NONE = "none"
    DIFFERENCE = "difference"
    LOG = "log"
    LOG_DIFFERENCE = "log_difference"
    PERCENT_CHANGE = "percent_change"
    DEMEAN = "demean"
    STANDARDISE = "standardise"


class StationarityVerdict(StrEnum):
    """Combined reading of the ADF and KPSS tests."""

    STATIONARY = "stationary"
    NON_STATIONARY = "non_stationary"
    #: The two tests disagree. Usually means near-unit-root or short samples.
    INCONCLUSIVE = "inconclusive"


class StationarityResult(BaseModel):
    """Outcome of testing one series for a unit root.

    ADF and KPSS have opposite null hypotheses, which is why both are run:

        ADF   H0: a unit root is present      -> low p-value means stationary
        KPSS  H0: the series is stationary    -> low p-value means non-stationary

    Agreement is evidence. Disagreement is information, and is reported as
    INCONCLUSIVE rather than resolved by picking the convenient test.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    series_id: str
    n_observations: int
    adf_statistic: float
    adf_pvalue: float
    kpss_statistic: float
    kpss_pvalue: float
    significance: float = Field(default=0.05, gt=0, lt=1)

    @property
    def adf_says_stationary(self) -> bool:
        return self.adf_pvalue < self.significance

    @property
    def kpss_says_stationary(self) -> bool:
        return self.kpss_pvalue > self.significance

    @property
    def verdict(self) -> StationarityVerdict:
        if self.adf_says_stationary and self.kpss_says_stationary:
            return StationarityVerdict.STATIONARY
        if not self.adf_says_stationary and not self.kpss_says_stationary:
            return StationarityVerdict.NON_STATIONARY
        return StationarityVerdict.INCONCLUSIVE

    @property
    def is_stationary(self) -> bool:
        """Conservative: only an unambiguous verdict counts as stationary."""
        return self.verdict is StationarityVerdict.STATIONARY

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "n_observations": self.n_observations,
            "adf_statistic": self.adf_statistic,
            "adf_pvalue": self.adf_pvalue,
            "kpss_statistic": self.kpss_statistic,
            "kpss_pvalue": self.kpss_pvalue,
            "significance": self.significance,
            "verdict": self.verdict.value,
        }


class TransformationRecord(BaseModel):
    """What was done to a series, and why.

    This is the provenance of a transformation. Without it, a published
    impulse response cannot be reproduced: the same model on the same raw
    data gives different answers depending on whether it was differenced.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    series_id: str
    steps: tuple[Transformation, ...]
    reason: str
    observations_before: int
    observations_after: int
    original_unit: Unit

    @property
    def observations_lost(self) -> int:
        return self.observations_before - self.observations_after

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "series_id": self.series_id,
            "steps": [step.value for step in self.steps],
            "reason": self.reason,
            "observations_before": self.observations_before,
            "observations_after": self.observations_after,
            "original_unit": self.original_unit.value,
        }


class PreparedSeries(BaseModel):
    """A stationary array plus the record of how it got that way."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    series_id: str
    periods: tuple[date, ...]
    values: Any = Field(description="1-D numpy array, aligned with periods.")
    record: TransformationRecord
    stationarity: StationarityResult | None = None

    def __len__(self) -> int:
        return len(self.periods)


# ---- extraction -----------------------------------------------------------


def to_array(series: TimeSeries) -> tuple[tuple[date, ...], np.ndarray]:
    """Extract a dense (periods, values) pair from the latest vintage.

    Raises
    ------
    EngineError
        If the series still holds several vintages per period. Collapsing
        silently would hide which vintage a result was computed from.
    """
    if series.has_revisions:
        raise EngineError(
            f"series {series.series_id!r} holds multiple vintages. Apply "
            f"as_of() or latest() before estimation so the vintage is explicit."
        )

    periods = tuple(o.period for o in series.observations)
    values = np.array(
        [np.nan if o.value is None else o.value for o in series.observations],
        dtype=float,
    )
    return periods, values


def drop_leading_and_trailing_nans(
    periods: Sequence[date], values: np.ndarray
) -> tuple[tuple[date, ...], np.ndarray]:
    """Trim missing values at the edges, which are usually coverage gaps."""
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return (), np.array([], dtype=float)
    start, end = int(finite[0]), int(finite[-1]) + 1
    return tuple(periods[start:end]), values[start:end]


# ---- stationarity ---------------------------------------------------------


def check_stationarity(
    series_id: str,
    values: np.ndarray,
    *,
    significance: float = 0.05,
    regression: str = "c",
) -> StationarityResult:
    """Run ADF and KPSS on one array.

    Parameters
    ----------
    regression
        Deterministic terms: "c" for a constant, "ct" to add a trend.
        Use "ct" when the series plausibly trends around a linear path.
    """
    finite = values[np.isfinite(values)]
    if finite.size < MIN_OBSERVATIONS:
        raise EngineError(
            f"series {series_id!r} has {finite.size} usable observations; "
            f"unit root tests need at least {MIN_OBSERVATIONS}"
        )
    if np.allclose(finite, finite[0]):
        raise EngineError(f"series {series_id!r} is constant; stationarity is undefined")

    adf_statistic, adf_pvalue, *_ = adfuller(finite, regression=regression, autolag="AIC")

    # KPSS p-values are interpolated from a small table, so statsmodels warns
    # at the edges. The warning is expected and the clipped value is correct.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_regression = "ct" if regression == "ct" else "c"
        kpss_statistic, kpss_pvalue, *_ = kpss(finite, regression=kpss_regression, nlags="auto")

    result = StationarityResult(
        series_id=series_id,
        n_observations=int(finite.size),
        adf_statistic=float(adf_statistic),
        adf_pvalue=float(adf_pvalue),
        kpss_statistic=float(kpss_statistic),
        kpss_pvalue=float(kpss_pvalue),
        significance=significance,
    )

    log.debug(
        "stationarity_tested",
        series=series_id,
        verdict=result.verdict.value,
        adf_pvalue=round(result.adf_pvalue, 4),
        kpss_pvalue=round(result.kpss_pvalue, 4),
    )
    return result


# ---- transformations ------------------------------------------------------


def apply_transformation(
    values: np.ndarray, transformation: Transformation
) -> tuple[np.ndarray, int]:
    """Apply one transformation. Returns the result and periods consumed.

    Differencing consumes the first observation, so callers must trim the
    matching periods to keep arrays aligned.
    """
    if transformation is Transformation.NONE:
        return values, 0

    if transformation is Transformation.DIFFERENCE:
        return np.diff(values), 1

    if transformation is Transformation.LOG:
        if np.any(values <= 0):
            raise EngineError("log transformation requires strictly positive values")
        return np.log(values), 0

    if transformation is Transformation.LOG_DIFFERENCE:
        if np.any(values <= 0):
            raise EngineError("log difference requires strictly positive values")
        return np.diff(np.log(values)), 1

    if transformation is Transformation.PERCENT_CHANGE:
        if np.any(values[:-1] == 0):
            raise EngineError("percent change requires non-zero denominators")
        return np.diff(values) / values[:-1] * 100.0, 1

    if transformation is Transformation.DEMEAN:
        return values - np.mean(values), 0

    if transformation is Transformation.STANDARDISE:
        deviation = np.std(values, ddof=1)
        if deviation == 0:
            raise EngineError("cannot standardise a constant series")
        return (values - np.mean(values)) / deviation, 0

    raise EngineError(f"unhandled transformation: {transformation}")


def suggest_transformation(series: TimeSeries, result: StationarityResult) -> Transformation:
    """Choose a transformation from the metadata and the test outcome.

    Metadata is consulted first because it is authoritative. A series
    published as a percent change is already a rate; differencing it again
    would produce the change in the change, which is rarely intended and
    never obvious from the numbers alone.
    """
    unit = series.metadata.unit

    if unit is Unit.PERCENT_CHANGE:
        return Transformation.NONE

    if result.is_stationary:
        return Transformation.NONE

    if unit in (Unit.INDEX, Unit.CURRENCY, Unit.COUNT):
        # Levels that grow multiplicatively: log differences give a growth rate.
        return Transformation.LOG_DIFFERENCE

    # Rates and ratios are already scale-free, so a plain difference is right.
    return Transformation.DIFFERENCE


def prepare(
    series: TimeSeries,
    *,
    transformation: Transformation | None = None,
    significance: float = 0.05,
    verify: bool = True,
) -> PreparedSeries:
    """Make a series ready for estimation and record what was done.

    Parameters
    ----------
    transformation
        Force a specific transformation. When omitted, one is chosen from
        the metadata and a stationarity test.
    verify
        Re-test after transforming and warn if the result is still not
        stationary. A warning rather than an error: sometimes the right
        answer is a second difference, and sometimes it is a different
        model, and this module should not silently decide which.
    """
    periods, values = to_array(series)
    periods, values = drop_leading_and_trailing_nans(periods, values)

    if not np.all(np.isfinite(values)):
        raise EngineError(
            f"series {series.series_id!r} has interior gaps. Fill or drop them "
            f"explicitly before estimation."
        )

    before = len(values)
    initial = check_stationarity(series.series_id, values, significance=significance)

    if transformation is None:
        chosen = suggest_transformation(series, initial)
        reason = (
            f"auto: unit={series.metadata.unit.value}, "
            f"verdict={initial.verdict.value}"
        )
    else:
        chosen = transformation
        reason = "explicit: requested by caller"

    transformed, consumed = apply_transformation(values, chosen)
    remaining_periods = periods[consumed:]

    final: StationarityResult | None = None
    if verify and transformed.size >= MIN_OBSERVATIONS:
        final = check_stationarity(series.series_id, transformed, significance=significance)
        if not final.is_stationary:
            log.warning(
                "still_not_stationary",
                series=series.series_id,
                transformation=chosen.value,
                verdict=final.verdict.value,
                adf_pvalue=round(final.adf_pvalue, 4),
            )

    record = TransformationRecord(
        series_id=series.series_id,
        steps=(chosen,),
        reason=reason,
        observations_before=before,
        observations_after=int(transformed.size),
        original_unit=series.metadata.unit,
    )

    log.info(
        "series_prepared",
        series=series.series_id,
        transformation=chosen.value,
        observations=int(transformed.size),
        lost=record.observations_lost,
    )

    return PreparedSeries(
        series_id=series.series_id,
        periods=remaining_periods,
        values=transformed,
        record=record,
        stationarity=final,
    )


def align(prepared: Sequence[PreparedSeries]) -> tuple[tuple[date, ...], np.ndarray]:
    """Align several prepared series onto their common periods.

    Returns the shared periods and a (T, k) matrix, one column per series
    in the order given. A VAR needs every variable observed at every date,
    so the intersection is the only defensible sample.
    """
    if not prepared:
        raise EngineError("align requires at least one series")

    common = set(prepared[0].periods)
    for item in prepared[1:]:
        common &= set(item.periods)

    if not common:
        raise EngineError(
            "prepared series share no common periods: "
            + ", ".join(f"{p.series_id}[{len(p)}]" for p in prepared)
        )

    ordered = tuple(sorted(common))
    columns = []
    for item in prepared:
        index = {period: position for position, period in enumerate(item.periods)}
        columns.append(np.array([item.values[index[period]] for period in ordered]))

    matrix = np.column_stack(columns)

    log.info(
        "series_aligned",
        variables=[item.series_id for item in prepared],
        observations=len(ordered),
        start=str(ordered[0]),
        end=str(ordered[-1]),
    )
    return ordered, matrix
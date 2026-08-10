"""
FRED (Federal Reserve Economic Data) adapter.

FRED is the first adapter deliberately: it is the only one of the three
that exposes genuine vintages. The realtime_start/realtime_end parameters
return data as it was known on a given date, which maps directly onto the
`known_at` axis of the canonical model. A source without vintages can only
ever fill one of the two time axes.

API reference: https://fred.stlouisfed.org/docs/api/fred/

Quirks handled here:
  * missing observations arrive as the string "." rather than null
  * every value is a string, including numbers
  * frequency is a short code ("M", "Q", "A", ...)
  * the API returns HTTP 400 with a JSON error body, not a plain status
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import Any

import httpx

from moirai.core.config import get_settings
from moirai.core.exceptions import ConfigError, IngestionError
from moirai.core.logging import get_logger
from moirai.engine.data_fabric.ingestion.base import (
    FetchResult,
    SourceAdapter,
    utc_now,
)
from moirai.engine.data_fabric.series.models import (
    Frequency,
    Observation,
    SeasonalAdjustment,
    SeriesMetadata,
    Source,
    TimeSeries,
    Unit,
)

log = get_logger(__name__)

BASE_URL = "https://api.stlouisfed.org/fred"
MISSING_MARKER = "."

FREQUENCY_CODES: dict[str, Frequency] = {
    "A": Frequency.ANNUAL,
    "SA": Frequency.SEMIANNUAL,
    "Q": Frequency.QUARTERLY,
    "M": Frequency.MONTHLY,
    "BW": Frequency.WEEKLY,
    "W": Frequency.WEEKLY,
    "D": Frequency.DAILY,
}

SEASONAL_CODES: dict[str, SeasonalAdjustment] = {
    "SA": SeasonalAdjustment.SEASONALLY_ADJUSTED,
    "NSA": SeasonalAdjustment.NOT_ADJUSTED,
    "SAAR": SeasonalAdjustment.SEASONALLY_ADJUSTED_ANNUAL_RATE,
}


def classify_unit(units: str) -> Unit:
    """Map FRED's free-text units string onto a coarse Unit class.

    FRED units are prose ("Index 2015=100", "Percent Change from Year Ago"),
    so this is a heuristic. The verbatim string is preserved in unit_label,
    which stays authoritative.
    """
    lowered = units.lower()
    if "percent change" in lowered or "% change" in lowered:
        return Unit.PERCENT_CHANGE
    if "percent" in lowered or lowered.startswith("%"):
        return Unit.PERCENT
    if "index" in lowered:
        return Unit.INDEX
    if any(token in lowered for token in ("dollar", "rupee", "euro", "yen", "currency")):
        return Unit.CURRENCY
    if "ratio" in lowered:
        return Unit.RATIO
    if any(token in lowered for token in ("persons", "number of", "count", "units")):
        return Unit.COUNT
    return Unit.UNKNOWN


class FredAdapter(SourceAdapter):
    """Fetches and parses series from the FRED API.

    Parameters
    ----------
    api_key
        Overrides the configured key. Mainly for tests.
    client
        An httpx.Client to reuse. Injecting one keeps tests off the network
        and lets callers share a connection pool across many fetches.
    """

    supports_revisions = True

    def __init__(
        self,
        api_key: str | None = None,
        *,
        client: httpx.Client | None = None,
        base_url: str = BASE_URL,
    ) -> None:
        settings = get_settings()
        self._api_key = api_key if api_key is not None else settings.fred_api_key
        if not self._api_key:
            raise ConfigError(
                "FRED API key is not configured. Set MOIRAI_FRED_API_KEY in .env "
                "or pass api_key explicitly."
            )
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=settings.http_timeout_seconds,
            headers={"User-Agent": "moirai/0.1 (research platform)"},
        )

    @property
    def source(self) -> Source:
        return Source.FRED

    # ---- HTTP ----

    def _redacted(self, path: str, params: dict[str, Any]) -> str:
        """Full URL with the API key replaced, safe to store in provenance."""
        safe = {**params, "api_key": "REDACTED"}
        query = "&".join(f"{k}={v}" for k, v in sorted(safe.items()))
        return f"{self._base_url}{path}?{query}"

    def _get(self, path: str, params: dict[str, Any]) -> httpx.Response:
        request_params = {**params, "api_key": self._api_key, "file_type": "json"}
        url = f"{self._base_url}{path}"
        try:
            response = self._client.get(url, params=request_params)
        except httpx.HTTPError as err:
            raise IngestionError(f"FRED request failed for {path}: {err}") from err

        if response.status_code != 200:
            detail = response.text[:300]
            raise IngestionError(
                f"FRED returned HTTP {response.status_code} for {path}: {detail}"
            )
        return response

    # ---- metadata ----

    def fetch_metadata(self, source_series_id: str) -> SeriesMetadata:
        """Retrieve the descriptive metadata for a series."""
        params = {"series_id": source_series_id}
        response = self._get("/series", params)

        try:
            payload = response.json()
            record = payload["seriess"][0]
        except (json.JSONDecodeError, KeyError, IndexError) as err:
            raise IngestionError(
                f"unexpected FRED metadata shape for {source_series_id!r}"
            ) from err

        units = record.get("units", "")
        return SeriesMetadata(
            series_id=source_series_id.lower(),
            title=record.get("title", source_series_id),
            source=Source.FRED,
            source_series_id=source_series_id,
            frequency=FREQUENCY_CODES.get(
                record.get("frequency_short", ""), Frequency.MONTHLY
            ),
            unit=classify_unit(units),
            unit_label=units[:128],
            seasonal_adjustment=SEASONAL_CODES.get(
                record.get("seasonal_adjustment_short", ""), SeasonalAdjustment.UNKNOWN
            ),
            start=_parse_date(record.get("observation_start")),
            end=_parse_date(record.get("observation_end")),
            notes=(record.get("notes") or "")[:4096],
            license="FRED Terms of Use; underlying source terms may apply",
            retrieved_at=utc_now(),
        )

    # ---- SourceAdapter contract ----

    def fetch_raw(self, source_series_id: str, **options: Any) -> FetchResult:
        """Fetch observations, optionally as of a historical vintage.

        Options
        -------
        realtime_start, realtime_end
            ISO dates. Together they request the vintage known during that
            window. Omitting both returns the current vintage only.
        observation_start, observation_end
            ISO dates bounding the periods returned.
        """
        params: dict[str, Any] = {"series_id": source_series_id}
        for key in (
            "realtime_start",
            "realtime_end",
            "observation_start",
            "observation_end",
        ):
            if key in options and options[key] is not None:
                params[key] = str(options[key])

        response = self._get("/series/observations", params)

        return FetchResult(
            source=Source.FRED,
            source_series_id=source_series_id,
            url=self._redacted("/series/observations", params),
            content=response.content,
            fetched_at=utc_now(),
            status_code=response.status_code,
            request_params=params,
        )

    def parse(self, result: FetchResult) -> TimeSeries:
        """Turn a FRED observations response into a canonical TimeSeries.

        Pure: no network and no clock. `known_at` comes from FRED's
        realtime_start when present, falling back to the fetch timestamp,
        so an archived response always reparses to identical values.
        """
        try:
            payload = json.loads(result.content)
        except json.JSONDecodeError as err:
            raise IngestionError(
                f"FRED response for {result.source_series_id!r} is not valid JSON"
            ) from err

        rows = payload.get("observations")
        if rows is None:
            raise IngestionError(
                f"FRED response for {result.source_series_id!r} has no observations key"
            )

        observations: list[Observation] = []
        seen: set[tuple[date, datetime]] = set()

        for row in rows:
            period = _parse_date(row.get("date"))
            if period is None:
                raise IngestionError(f"FRED observation has no usable date: {row!r}")

            known_at = _known_at(row.get("realtime_start"), result.fetched_at)

            key = (period, known_at)
            if key in seen:
                continue  # identical vintage repeated; FRED does this at boundaries
            seen.add(key)

            observations.append(
                Observation(
                    period=period,
                    value=_parse_value(row.get("value")),
                    known_at=known_at,
                )
            )

        metadata = SeriesMetadata(
            series_id=result.source_series_id.lower(),
            title=result.source_series_id,
            source=Source.FRED,
            source_series_id=result.source_series_id,
            frequency=Frequency.MONTHLY,
            license="FRED Terms of Use; underlying source terms may apply",
            retrieved_at=result.fetched_at,
        )
        return TimeSeries.from_observations(metadata, observations)

    def fetch_series(self, source_series_id: str, **options: Any) -> TimeSeries:
        """Fetch observations and real metadata, combined.

        `parse` cannot call the metadata endpoint because it must stay pure,
        so this method makes the second request and merges the result.
        """
        series = self.fetch(source_series_id, **options)
        metadata = self.fetch_metadata(source_series_id)
        return TimeSeries(metadata=metadata, observations=series.observations)

    # ---- lifecycle ----

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> FredAdapter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# ---- parsing helpers ----


def _parse_date(value: str | None) -> date | None:
    """FRED uses YYYY-MM-DD, with sentinel dates at the edges of coverage."""
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _parse_value(raw: str | None) -> float | None:
    """FRED encodes a missing observation as the string '.'."""
    if raw is None or raw == MISSING_MARKER or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError as err:
        raise IngestionError(f"FRED value is not numeric: {raw!r}") from err


def _known_at(realtime_start: str | None, fallback: datetime) -> datetime:
    """When FRED first published this value.

    ALFRED responses carry realtime_start per observation. Plain FRED
    responses do not, in which case the fetch time is the honest answer:
    it is when Moirai learned the value.
    """
    parsed = _parse_date(realtime_start)
    if parsed is None:
        return fallback
    return datetime(parsed.year, parsed.month, parsed.day, tzinfo=UTC)
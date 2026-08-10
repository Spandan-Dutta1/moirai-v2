"""
World Bank Open Data adapter.

The World Bank API differs from FRED in three ways that matter:

  * No authentication. Anyone can query it.
  * The response is a two-element array: [pagination, [observations]].
    Element 0 carries page/total; element 1 carries the data. A parser
    that assumes a dict will fail on the first call.
  * No vintages. The API returns the current snapshot only, with no way
    to ask what a figure looked like last year.

That last point is a real limitation, not an implementation gap. Where
FRED can fill both time axes, the World Bank fills one. `known_at` is set
to the fetch timestamp because that is the honest answer: it is when
Moirai learned the value, and nothing is claimed about earlier vintages.
The consequence is that real-time backtesting against World Bank data can
only start from the date collection began.

Identifiers are compound: an indicator code plus a country code, for
example NY.GDP.MKTP.KD.ZG for India is "IND/NY.GDP.MKTP.KD.ZG".

API reference: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import Any

import httpx

from moirai.core.config import get_settings
from moirai.core.exceptions import IngestionError
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

BASE_URL = "https://api.worldbank.org/v2"
DEFAULT_PER_PAGE = 20_000  # one page covers any realistic annual series
DEFAULT_COUNTRY = "IND"


def parse_series_id(source_series_id: str) -> tuple[str, str]:
    """Split 'IND/NY.GDP.MKTP.KD.ZG' into ('IND', 'NY.GDP.MKTP.KD.ZG').

    A bare indicator defaults to India, since that is Moirai's primary
    geography. Passing the country explicitly is always preferred.
    """
    if "/" in source_series_id:
        country, _, indicator = source_series_id.partition("/")
        country, indicator = country.strip(), indicator.strip()
        if not country or not indicator:
            raise IngestionError(
                f"malformed World Bank identifier {source_series_id!r}; "
                f"expected 'COUNTRY/INDICATOR'"
            )
        return country.upper(), indicator.upper()
    return DEFAULT_COUNTRY, source_series_id.strip().upper()


def to_moirai_id(country: str, indicator: str) -> str:
    """Build an internal series_id that survives the series_id validator."""
    return f"wb_{country}_{indicator}".lower().replace(".", "_")


def classify_unit(indicator: str, name: str) -> Unit:
    """Infer a coarse unit from the indicator code and its name.

    World Bank encodes meaning in the code suffix: ZG is growth, ZS is a
    share, KD/CD are constant/current dollars. The code is more reliable
    than the prose name, so it is checked first.
    """
    code = indicator.upper()
    if code.endswith(".ZG"):
        return Unit.PERCENT_CHANGE
    if code.endswith(".ZS"):
        return Unit.PERCENT
    if code.endswith((".KD", ".CD", ".CN", ".KN", ".PP.CD", ".PP.KD")):
        return Unit.CURRENCY

    lowered = name.lower()
    if "% of" in lowered or "percent" in lowered:
        return Unit.PERCENT
    if "growth" in lowered or "annual %" in lowered:
        return Unit.PERCENT_CHANGE
    if "index" in lowered:
        return Unit.INDEX
    if any(token in lowered for token in ("us$", "dollar", "lcu", "current")):
        return Unit.CURRENCY
    if any(token in lowered for token in ("total", "number", "population", "people")):
        return Unit.COUNT
    return Unit.UNKNOWN


class WorldBankAdapter(SourceAdapter):
    """Fetches and parses indicators from the World Bank Open Data API.

    Parameters
    ----------
    client
        An httpx.Client to reuse. Injecting one keeps tests off the
        network and lets callers share a connection pool.
    """

    #: The API exposes a current snapshot only; there are no vintages.
    supports_revisions = False

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        base_url: str = BASE_URL,
    ) -> None:
        settings = get_settings()
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=settings.http_timeout_seconds,
            headers={"User-Agent": "moirai/0.1 (research platform)"},
        )

    @property
    def source(self) -> Source:
        return Source.WORLD_BANK

    # ---- HTTP ----

    def _get(self, path: str, params: dict[str, Any]) -> httpx.Response:
        request_params = {**params, "format": "json"}
        url = f"{self._base_url}{path}"
        try:
            response = self._client.get(url, params=request_params)
        except httpx.HTTPError as err:
            raise IngestionError(f"World Bank request failed for {path}: {err}") from err

        if response.status_code != 200:
            raise IngestionError(
                f"World Bank returned HTTP {response.status_code} for {path}: "
                f"{response.text[:300]}"
            )
        return response

    @staticmethod
    def _unwrap(content: bytes, context: str) -> tuple[dict[str, Any], list[Any]]:
        """Split the [pagination, payload] envelope, or explain why it failed.

        The API signals errors inside a 200 response by returning a single
        element containing a message, so status codes alone are not enough.
        """
        try:
            document = json.loads(content)
        except json.JSONDecodeError as err:
            raise IngestionError(f"World Bank response for {context} is not valid JSON") from err

        if not isinstance(document, list) or not document:
            raise IngestionError(f"unexpected World Bank envelope for {context}: {document!r}")

        header = document[0]
        if isinstance(header, dict) and "message" in header:
            message = header["message"]
            detail = message[0].get("value") if isinstance(message, list) and message else message
            raise IngestionError(f"World Bank error for {context}: {detail}")

        if len(document) < 2 or document[1] is None:
            raise IngestionError(f"World Bank returned no data for {context}")

        payload = document[1]
        if not isinstance(payload, list):
            raise IngestionError(f"unexpected World Bank payload for {context}: {payload!r}")

        return (header if isinstance(header, dict) else {}), payload

    # ---- metadata ----

    def fetch_metadata(self, source_series_id: str) -> SeriesMetadata:
        """Retrieve indicator metadata: name, source note, licence."""
        country, indicator = parse_series_id(source_series_id)
        response = self._get(f"/indicator/{indicator}", {})
        _, payload = self._unwrap(response.content, source_series_id)

        if not payload:
            raise IngestionError(f"World Bank has no indicator {indicator!r}")

        record = payload[0]
        name = record.get("name", indicator)

        return SeriesMetadata(
            series_id=to_moirai_id(country, indicator),
            title=f"{name} ({country})"[:512],
            source=Source.WORLD_BANK,
            source_series_id=f"{country}/{indicator}",
            frequency=Frequency.ANNUAL,
            unit=classify_unit(indicator, name),
            unit_label=(record.get("unit") or name)[:128],
            seasonal_adjustment=SeasonalAdjustment.NOT_ADJUSTED,
            geography=country,
            notes=(record.get("sourceNote") or "")[:4096],
            license="CC BY-4.0, World Bank Open Data",
            retrieved_at=utc_now(),
        )

    # ---- SourceAdapter contract ----

    def fetch_raw(self, source_series_id: str, **options: Any) -> FetchResult:
        """Fetch observations for one country-indicator pair.

        Options
        -------
        date_range
            A string like "2000:2024" restricting the years returned.
        per_page
            Page size. The default is large enough that annual series
            arrive in a single response.
        """
        country, indicator = parse_series_id(source_series_id)

        params: dict[str, Any] = {"per_page": options.get("per_page", DEFAULT_PER_PAGE)}
        if options.get("date_range"):
            params["date"] = str(options["date_range"])

        path = f"/country/{country}/indicator/{indicator}"
        response = self._get(path, params)

        query = "&".join(f"{k}={v}" for k, v in sorted({**params, "format": "json"}.items()))
        return FetchResult(
            source=Source.WORLD_BANK,
            source_series_id=f"{country}/{indicator}",
            url=f"{self._base_url}{path}?{query}",
            content=response.content,
            fetched_at=utc_now(),
            status_code=response.status_code,
            request_params={**params, "country": country, "indicator": indicator},
        )

    def parse(self, result: FetchResult) -> TimeSeries:
        """Turn a World Bank response into a canonical TimeSeries.

        Pure: `known_at` comes from `result.fetched_at`, which is recorded
        in the archive, so replaying archived bytes reproduces the same
        observations rather than stamping them with today's date.
        """
        country, indicator = parse_series_id(result.source_series_id)
        _, payload = self._unwrap(result.content, result.source_series_id)

        observations: list[Observation] = []
        seen: set[date] = set()

        for row in payload:
            if not isinstance(row, dict):
                raise IngestionError(f"unexpected World Bank row: {row!r}")

            period = _parse_year(row.get("date"))
            if period is None:
                continue  # the API emits aggregate rows without a usable year

            if period in seen:
                continue
            seen.add(period)

            observations.append(
                Observation(
                    period=period,
                    value=_parse_value(row.get("value")),
                    known_at=result.fetched_at,
                )
            )

        metadata = SeriesMetadata(
            series_id=to_moirai_id(country, indicator),
            title=f"{indicator} ({country})",
            source=Source.WORLD_BANK,
            source_series_id=f"{country}/{indicator}",
            frequency=Frequency.ANNUAL,
            unit=classify_unit(indicator, ""),
            geography=country,
            license="CC BY-4.0, World Bank Open Data",
            retrieved_at=result.fetched_at,
        )
        return TimeSeries.from_observations(metadata, observations)

    def fetch_series(self, source_series_id: str, **options: Any) -> TimeSeries:
        """Fetch observations and real metadata, combined."""
        archive = options.pop("archive", True)
        series = self.fetch(source_series_id, archive=archive, **options)
        metadata = self.fetch_metadata(source_series_id)
        return TimeSeries(metadata=metadata, observations=series.observations)

    # ---- lifecycle ----

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> WorldBankAdapter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# ---- parsing helpers ----


def _parse_year(raw: Any) -> date | None:
    """World Bank dates are bare years for annual data.

    Quarterly and monthly indicators exist ('2024Q1', '2024M03') but are
    rare and not yet supported; those rows are skipped rather than guessed at.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if len(text) == 4 and text.isdigit():
        return date(int(text), 1, 1)
    return None


def _parse_value(raw: Any) -> float | None:
    """A missing World Bank observation is JSON null, not a sentinel string."""
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError) as err:
        raise IngestionError(f"World Bank value is not numeric: {raw!r}") from err
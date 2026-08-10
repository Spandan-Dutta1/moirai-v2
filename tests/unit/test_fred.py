"""Tests for the FRED adapter.

No network. A stubbed httpx transport returns recorded response shapes, so
these tests are fast, deterministic, and still fail if FRED's actual quirks
stop being handled.
"""

import json
from datetime import UTC, date, datetime

import httpx
import pytest

from moirai.core.exceptions import ConfigError, IngestionError
from moirai.engine.data_fabric.ingestion.base import FetchResult
from moirai.engine.data_fabric.ingestion.fred import (
    FredAdapter,
    _known_at,
    _parse_date,
    _parse_value,
    classify_unit,
)
from moirai.engine.data_fabric.series.models import (
    Frequency,
    SeasonalAdjustment,
    Source,
    Unit,
)

API_KEY = "0" * 32
FETCHED_AT = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)

OBSERVATIONS_BODY = {
    "realtime_start": "2026-08-10",
    "realtime_end": "2026-08-10",
    "observations": [
        {
            "realtime_start": "2024-02-15",
            "realtime_end": "9999-12-31",
            "date": "2024-01-01",
            "value": "153.0345",
        },
        {
            "realtime_start": "2024-03-15",
            "realtime_end": "9999-12-31",
            "date": "2024-02-01",
            "value": "153.365",
        },
        {
            "realtime_start": "2024-04-15",
            "realtime_end": "9999-12-31",
            "date": "2024-03-01",
            "value": ".",
        },
    ],
}

METADATA_BODY = {
    "seriess": [
        {
            "id": "INDCPIALLMINMEI",
            "title": "Consumer Price Index: Total for India",
            "observation_start": "1957-01-01",
            "observation_end": "2025-06-01",
            "frequency_short": "M",
            "units": "Index 2015=100",
            "seasonal_adjustment_short": "NSA",
            "notes": "OECD source.",
        }
    ]
}


def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def json_handler(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/series/observations"):
        return httpx.Response(200, json=OBSERVATIONS_BODY)
    if request.url.path.endswith("/series"):
        return httpx.Response(200, json=METADATA_BODY)
    return httpx.Response(404, text="not found")


@pytest.fixture
def adapter() -> FredAdapter:
    return FredAdapter(api_key=API_KEY, client=make_client(json_handler))


def make_result(content: bytes | None = None, **overrides) -> FetchResult:
    defaults = dict(
        source=Source.FRED,
        source_series_id="INDCPIALLMINMEI",
        url="https://api.stlouisfed.org/fred/series/observations?api_key=REDACTED",
        content=content if content is not None else json.dumps(OBSERVATIONS_BODY).encode(),
        fetched_at=FETCHED_AT,
    )
    return FetchResult(**{**defaults, **overrides})


# --- construction ----------------------------------------------------------

def test_adapter_reports_its_source(adapter):
    assert adapter.source is Source.FRED


def test_missing_api_key_raises_config_error(monkeypatch):
    from moirai.core.config import reset_settings_cache

    monkeypatch.setenv("MOIRAI_FRED_API_KEY", "")
    reset_settings_cache()
    with pytest.raises(ConfigError, match="FRED API key"):
        FredAdapter(api_key=None)
    reset_settings_cache()


def test_adapter_declares_revision_support(adapter):
    assert adapter.supports_revisions is True


# --- value parsing ---------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [("153.0345", 153.0345), ("0", 0.0), ("-2.5", -2.5), ("1e3", 1000.0)],
)
def test_numeric_values_are_parsed(raw, expected):
    assert _parse_value(raw) == expected


@pytest.mark.parametrize("raw", [".", "", "   ", None])
def test_missing_markers_become_none(raw):
    """FRED encodes a gap as '.', which must not become 0.0."""
    assert _parse_value(raw) is None


def test_non_numeric_value_raises():
    with pytest.raises(IngestionError, match="not numeric"):
        _parse_value("n/a")


# --- date parsing ----------------------------------------------------------

def test_iso_date_is_parsed():
    assert _parse_date("2024-01-01") == date(2024, 1, 1)


@pytest.mark.parametrize("raw", [None, "", "not-a-date", "9999-99-99"])
def test_unparseable_dates_become_none(raw):
    assert _parse_date(raw) is None


def test_known_at_uses_realtime_start():
    assert _known_at("2024-02-15", FETCHED_AT) == datetime(2024, 2, 15, tzinfo=UTC)


def test_known_at_falls_back_to_fetch_time():
    """A plain FRED response has no realtime_start; fetch time is honest."""
    assert _known_at(None, FETCHED_AT) == FETCHED_AT


def test_known_at_is_always_timezone_aware():
    assert _known_at("2024-02-15", FETCHED_AT).tzinfo is not None


# --- unit classification ---------------------------------------------------

@pytest.mark.parametrize(
    ("units", "expected"),
    [
        ("Index 2015=100", Unit.INDEX),
        ("Percent", Unit.PERCENT),
        ("Percent Change from Year Ago", Unit.PERCENT_CHANGE),
        ("Billions of Chained 2017 Dollars", Unit.CURRENCY),
        ("Ratio", Unit.RATIO),
        ("Thousands of Persons", Unit.COUNT),
        ("Something Unrecognised", Unit.UNKNOWN),
    ],
)
def test_units_are_classified(units, expected):
    assert classify_unit(units) is expected


def test_percent_change_wins_over_percent():
    """Ordering matters: 'Percent Change' must not classify as PERCENT."""
    assert classify_unit("Percent Change from Preceding Period") is Unit.PERCENT_CHANGE


# --- parse -----------------------------------------------------------------

def test_parse_produces_observations(adapter):
    series = adapter.parse(make_result())
    assert len(series) == 3


def test_parse_preserves_missing_values(adapter):
    series = adapter.parse(make_result())
    assert series.observations[2].value is None


def test_parse_uses_realtime_start_as_known_at(adapter):
    series = adapter.parse(make_result())
    assert series.observations[0].known_at == datetime(2024, 2, 15, tzinfo=UTC)


def test_parse_sorts_observations(adapter):
    scrambled = {
        "observations": [
            {"date": "2024-03-01", "value": "3", "realtime_start": "2024-04-15"},
            {"date": "2024-01-01", "value": "1", "realtime_start": "2024-02-15"},
        ]
    }
    series = adapter.parse(make_result(json.dumps(scrambled).encode()))
    assert [o.period.month for o in series.observations] == [1, 3]


def test_parse_deduplicates_identical_vintages(adapter):
    """FRED repeats rows at realtime boundaries."""
    duplicated = {
        "observations": [
            {"date": "2024-01-01", "value": "1", "realtime_start": "2024-02-15"},
            {"date": "2024-01-01", "value": "1", "realtime_start": "2024-02-15"},
        ]
    }
    series = adapter.parse(make_result(json.dumps(duplicated).encode()))
    assert len(series) == 1


def test_parse_keeps_distinct_vintages_of_one_period(adapter):
    revised = {
        "observations": [
            {"date": "2024-01-01", "value": "1.0", "realtime_start": "2024-02-15"},
            {"date": "2024-01-01", "value": "1.2", "realtime_start": "2024-05-15"},
        ]
    }
    series = adapter.parse(make_result(json.dumps(revised).encode()))
    assert len(series) == 2
    assert series.has_revisions is True


def test_parse_rejects_invalid_json(adapter):
    with pytest.raises(IngestionError, match="not valid JSON"):
        adapter.parse(make_result(b"<html>error</html>"))


def test_parse_rejects_a_response_without_observations(adapter):
    with pytest.raises(IngestionError, match="no observations"):
        adapter.parse(make_result(json.dumps({"error_code": 400}).encode()))


def test_parse_rejects_an_observation_without_a_date(adapter):
    body = {"observations": [{"value": "1.0"}]}
    with pytest.raises(IngestionError, match="no usable date"):
        adapter.parse(make_result(json.dumps(body).encode()))


def test_parse_is_pure(adapter):
    """Same bytes must always parse to the same values: replay depends on it."""
    result = make_result()
    first = adapter.parse(result)
    second = adapter.parse(result)
    assert first.observations == second.observations


# --- fetch_raw -------------------------------------------------------------

def test_fetch_raw_returns_content(adapter):
    result = adapter.fetch_raw("INDCPIALLMINMEI")
    assert result.status_code == 200
    assert b"observations" in result.content


def test_fetch_raw_redacts_the_api_key(adapter):
    result = adapter.fetch_raw("INDCPIALLMINMEI")
    assert API_KEY not in result.url
    assert "REDACTED" in result.url


def test_api_key_is_absent_from_provenance(adapter):
    provenance = adapter.fetch_raw("INDCPIALLMINMEI").provenance()
    assert API_KEY not in json.dumps(provenance)


def test_vintage_options_are_forwarded():
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json=OBSERVATIONS_BODY)

    adapter = FredAdapter(api_key=API_KEY, client=make_client(handler))
    adapter.fetch_raw("GDPC1", realtime_start="2020-01-01", realtime_end="2021-01-01")

    assert captured["realtime_start"] == "2020-01-01"
    assert captured["realtime_end"] == "2021-01-01"


def test_none_options_are_not_sent():
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json=OBSERVATIONS_BODY)

    adapter = FredAdapter(api_key=API_KEY, client=make_client(handler))
    adapter.fetch_raw("GDPC1", realtime_start=None)
    assert "realtime_start" not in captured


# --- error handling --------------------------------------------------------

def test_http_error_status_raises_ingestion_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text='{"error_message": "Bad Request"}')

    adapter = FredAdapter(api_key=API_KEY, client=make_client(handler))
    with pytest.raises(IngestionError, match="HTTP 400"):
        adapter.fetch_raw("NOPE")


def test_transport_failure_raises_ingestion_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    adapter = FredAdapter(api_key=API_KEY, client=make_client(handler))
    with pytest.raises(IngestionError, match="request failed"):
        adapter.fetch_raw("INDCPIALLMINMEI")


# --- metadata --------------------------------------------------------------

def test_metadata_is_mapped(adapter):
    metadata = adapter.fetch_metadata("INDCPIALLMINMEI")
    assert metadata.frequency is Frequency.MONTHLY
    assert metadata.unit is Unit.INDEX
    assert metadata.unit_label == "Index 2015=100"
    assert metadata.seasonal_adjustment is SeasonalAdjustment.NOT_ADJUSTED
    assert metadata.source is Source.FRED


def test_metadata_records_coverage(adapter):
    metadata = adapter.fetch_metadata("INDCPIALLMINMEI")
    assert metadata.start == date(1957, 1, 1)
    assert metadata.end == date(2025, 6, 1)


def test_metadata_series_id_is_lowercased(adapter):
    assert adapter.fetch_metadata("INDCPIALLMINMEI").series_id == "indcpiallminmei"


def test_unexpected_metadata_shape_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"seriess": []})

    adapter = FredAdapter(api_key=API_KEY, client=make_client(handler))
    with pytest.raises(IngestionError, match="unexpected FRED metadata"):
        adapter.fetch_metadata("NOPE")


# --- integration through the base class ------------------------------------

def test_fetch_series_combines_observations_and_metadata(adapter, tmp_path, monkeypatch):
    monkeypatch.setenv("MOIRAI_HOME", str(tmp_path))
    from moirai.core.paths import reset_paths_cache

    reset_paths_cache()
    series = adapter.fetch_series("INDCPIALLMINMEI", archive=False)
    reset_paths_cache()

    assert len(series) == 3
    assert series.metadata.unit_label == "Index 2015=100"


def test_context_manager_closes_the_client():
    adapter = FredAdapter(api_key=API_KEY)
    with adapter as entered:
        assert entered is adapter
    assert adapter._client.is_closed


def test_injected_client_is_not_closed(adapter):
    """The caller owns a client it passed in."""
    with adapter:
        pass
    assert adapter._client.is_closed is False
"""Tests for the World Bank adapter.

No network: a stubbed transport returns recorded response shapes. The
envelope handling gets the most attention because the API signals errors
inside a 200 response, so status codes alone cannot be trusted.
"""

import json
from datetime import UTC, date, datetime

import httpx
import pytest

from moirai.core.exceptions import IngestionError
from moirai.engine.data_fabric.ingestion.base import FetchResult
from moirai.engine.data_fabric.ingestion.worldbank import (
    WorldBankAdapter,
    _parse_value,
    _parse_year,
    classify_unit,
    parse_series_id,
    to_moirai_id,
)
from moirai.engine.data_fabric.series.models import Frequency, Source, Unit

FETCHED_AT = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)

OBSERVATIONS_BODY = [
    {"page": 1, "pages": 1, "per_page": 20000, "total": 3},
    [
        {
            "indicator": {"id": "NY.GDP.MKTP.KD.ZG", "value": "GDP growth (annual %)"},
            "country": {"id": "IN", "value": "India"},
            "countryiso3code": "IND",
            "date": "2022",
            "value": 7.6093649776932,
        },
        {
            "countryiso3code": "IND",
            "date": "2021",
            "value": 9.68959249191741,
        },
        {
            "countryiso3code": "IND",
            "date": "2020",
            "value": None,
        },
    ],
]

INDICATOR_BODY = [
    {"page": 1, "pages": 1, "total": 1},
    [
        {
            "id": "NY.GDP.MKTP.KD.ZG",
            "name": "GDP growth (annual %)",
            "unit": "",
            "sourceNote": "Annual percentage growth rate of GDP at market prices.",
        }
    ],
]

ERROR_BODY = [
    {
        "message": [
            {"id": "120", "key": "Invalid value", "value": "The provided parameter value is not valid"}
        ]
    }
]

def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def default_handler(request: httpx.Request) -> httpx.Response:
    if "/indicator/" in request.url.path and "/country/" not in request.url.path:
        return httpx.Response(200, json=INDICATOR_BODY)
    return httpx.Response(200, json=OBSERVATIONS_BODY)


@pytest.fixture
def adapter() -> WorldBankAdapter:
    return WorldBankAdapter(client=make_client(default_handler))


def make_result(content=None, **overrides) -> FetchResult:
    defaults = dict(
        source=Source.WORLD_BANK,
        source_series_id="IND/NY.GDP.MKTP.KD.ZG",
        url="https://api.worldbank.org/v2/country/IND/indicator/NY.GDP.MKTP.KD.ZG",
        content=content if content is not None else json.dumps(OBSERVATIONS_BODY).encode(),
        fetched_at=FETCHED_AT,
    )
    return FetchResult(**{**defaults, **overrides})


# --- identifiers -----------------------------------------------------------

def test_compound_identifier_is_split():
    assert parse_series_id("IND/NY.GDP.MKTP.KD.ZG") == ("IND", "NY.GDP.MKTP.KD.ZG")


def test_bare_indicator_defaults_to_india():
    assert parse_series_id("NY.GDP.MKTP.KD.ZG") == ("IND", "NY.GDP.MKTP.KD.ZG")


def test_identifiers_are_uppercased():
    assert parse_series_id("ind/ny.gdp.mktp.kd.zg") == ("IND", "NY.GDP.MKTP.KD.ZG")


@pytest.mark.parametrize("bad", ["/NY.GDP", "IND/", "/"])
def test_malformed_identifiers_are_rejected(bad):
    with pytest.raises(IngestionError, match="malformed"):
        parse_series_id(bad)


def test_moirai_id_survives_the_series_id_validator():
    """Dots are legal in series_id, but underscores read better as a table name."""
    assert to_moirai_id("IND", "NY.GDP.MKTP.KD.ZG") == "wb_ind_ny_gdp_mktp_kd_zg"


# --- value and year parsing ------------------------------------------------

def test_year_is_parsed_to_january_first():
    assert _parse_year("2022") == date(2022, 1, 1)


@pytest.mark.parametrize("raw", [None, "", "2024Q1", "2024M03", "not-a-year", "20"])
def test_unsupported_period_formats_are_skipped(raw):
    """Sub-annual World Bank formats exist but are not yet supported."""
    assert _parse_year(raw) is None


@pytest.mark.parametrize(("raw", "expected"), [(7.6, 7.6), ("7.6", 7.6), (0, 0.0)])
def test_numeric_values_are_parsed(raw, expected):
    assert _parse_value(raw) == expected


@pytest.mark.parametrize("raw", [None, ""])
def test_missing_values_become_none(raw):
    """The World Bank uses JSON null, not a sentinel string."""
    assert _parse_value(raw) is None


def test_non_numeric_value_raises():
    with pytest.raises(IngestionError, match="not numeric"):
        _parse_value("n/a")


# --- unit classification ---------------------------------------------------

@pytest.mark.parametrize(
    ("indicator", "expected"),
    [
        ("NY.GDP.MKTP.KD.ZG", Unit.PERCENT_CHANGE),
        ("SL.UEM.TOTL.ZS", Unit.PERCENT),
        ("NY.GDP.MKTP.CD", Unit.CURRENCY),
        ("NY.GDP.MKTP.KD", Unit.CURRENCY),
    ],
)
def test_units_are_inferred_from_the_code_suffix(indicator, expected):
    assert classify_unit(indicator, "") is expected


def test_name_is_used_when_the_code_is_uninformative():
    assert classify_unit("SP.POP.TOTL", "Population, total") is Unit.COUNT


def test_unknown_indicator_falls_back_to_unknown():
    assert classify_unit("XX.YY.ZZ", "Mystery measure") is Unit.UNKNOWN


# --- envelope handling -----------------------------------------------------

def test_parse_reads_the_two_element_envelope(adapter):
    series = adapter.parse(make_result())
    assert len(series) == 3


def test_parse_sorts_observations(adapter):
    """The API returns newest first; the model requires ascending order."""
    series = adapter.parse(make_result())
    assert [o.period.year for o in series.observations] == [2020, 2021, 2022]


def test_parse_preserves_missing_values(adapter):
    series = adapter.parse(make_result())
    assert series.observations[0].value is None


def test_parse_skips_rows_with_unsupported_periods(adapter):
    body = [{"page": 1}, [{"date": "2024Q1", "value": 1.0}, {"date": "2023", "value": 2.0}]]
    series = adapter.parse(make_result(json.dumps(body).encode()))
    assert len(series) == 1


def test_parse_deduplicates_repeated_years(adapter):
    body = [{"page": 1}, [{"date": "2023", "value": 1.0}, {"date": "2023", "value": 1.0}]]
    series = adapter.parse(make_result(json.dumps(body).encode()))
    assert len(series) == 1


def test_parse_rejects_invalid_json(adapter):
    with pytest.raises(IngestionError, match="not valid JSON"):
        adapter.parse(make_result(b"<html>gateway error</html>"))


def test_parse_rejects_a_non_list_envelope(adapter):
    with pytest.raises(IngestionError, match="unexpected World Bank envelope"):
        adapter.parse(make_result(json.dumps({"observations": []}).encode()))


def test_parse_surfaces_an_api_error_inside_a_200(adapter):
    """The API reports errors in the body, so status alone is insufficient."""
    with pytest.raises(IngestionError, match="not valid"):
        adapter.parse(make_result(json.dumps(ERROR_BODY).encode()))


def test_parse_rejects_a_null_payload(adapter):
    body = [{"page": 1, "total": 0}, None]
    with pytest.raises(IngestionError, match="no data"):
        adapter.parse(make_result(json.dumps(body).encode()))


# --- bitemporal honesty ----------------------------------------------------

def test_known_at_is_the_fetch_time(adapter):
    """Without vintages, fetch time is the only honest transaction time."""
    series = adapter.parse(make_result())
    assert all(o.known_at == FETCHED_AT for o in series.observations)


def test_adapter_declares_no_revision_support(adapter):
    assert adapter.supports_revisions is False


def test_parsed_series_has_no_revisions(adapter):
    series = adapter.parse(make_result())
    assert series.has_revisions is False


def test_parse_is_pure(adapter):
    """Replay depends on this: same bytes, same observations, every time."""
    result = make_result()
    assert adapter.parse(result).observations == adapter.parse(result).observations


# --- fetch_raw -------------------------------------------------------------

def test_fetch_raw_returns_content(adapter):
    result = adapter.fetch_raw("IND/NY.GDP.MKTP.KD.ZG")
    assert result.status_code == 200
    assert result.source is Source.WORLD_BANK


def test_fetch_raw_normalises_the_series_id(adapter):
    result = adapter.fetch_raw("ind/ny.gdp.mktp.kd.zg")
    assert result.source_series_id == "IND/NY.GDP.MKTP.KD.ZG"


def test_date_range_is_forwarded():
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json=OBSERVATIONS_BODY)

    adapter = WorldBankAdapter(client=make_client(handler))
    adapter.fetch_raw("IND/NY.GDP.MKTP.KD.ZG", date_range="2000:2024")
    assert captured["date"] == "2000:2024"


def test_no_api_key_is_sent():
    """The World Bank API is unauthenticated; sending a key would be a bug."""
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json=OBSERVATIONS_BODY)

    adapter = WorldBankAdapter(client=make_client(handler))
    adapter.fetch_raw("IND/NY.GDP.MKTP.KD.ZG")
    assert "api_key" not in captured


def test_http_error_status_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    adapter = WorldBankAdapter(client=make_client(handler))
    with pytest.raises(IngestionError, match="HTTP 500"):
        adapter.fetch_raw("IND/NY.GDP.MKTP.KD.ZG")


def test_transport_failure_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    adapter = WorldBankAdapter(client=make_client(handler))
    with pytest.raises(IngestionError, match="request failed"):
        adapter.fetch_raw("IND/NY.GDP.MKTP.KD.ZG")


# --- metadata --------------------------------------------------------------

def test_metadata_is_mapped(adapter):
    metadata = adapter.fetch_metadata("IND/NY.GDP.MKTP.KD.ZG")
    assert metadata.frequency is Frequency.ANNUAL
    assert metadata.unit is Unit.PERCENT_CHANGE
    assert metadata.geography == "IND"
    assert metadata.source is Source.WORLD_BANK
    assert "GDP growth" in metadata.title


def test_metadata_records_the_licence(adapter):
    assert "CC BY-4.0" in adapter.fetch_metadata("IND/NY.GDP.MKTP.KD.ZG").license


def test_unknown_indicator_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"page": 1, "total": 0}, []])

    adapter = WorldBankAdapter(client=make_client(handler))
    with pytest.raises(IngestionError, match="no indicator"):
        adapter.fetch_metadata("IND/XX.YY.ZZ")


# --- integration -----------------------------------------------------------

def test_fetch_series_combines_data_and_metadata(adapter, tmp_path, monkeypatch):
    monkeypatch.setenv("MOIRAI_HOME", str(tmp_path))
    from moirai.core.paths import reset_paths_cache

    reset_paths_cache()
    series = adapter.fetch_series("IND/NY.GDP.MKTP.KD.ZG", archive=False)
    reset_paths_cache()

    assert len(series) == 3
    assert "GDP growth" in series.metadata.title


def test_archiving_handles_the_slash_in_the_identifier(adapter, tmp_path, monkeypatch):
    """Regression: 'IND/NY.GDP...' must not become a directory separator."""
    monkeypatch.setenv("MOIRAI_HOME", str(tmp_path))
    from moirai.core.paths import get_paths, reset_paths_cache

    reset_paths_cache()
    adapter.fetch("IND/NY.GDP.MKTP.KD.ZG")
    archived = list((get_paths().raw / "world_bank").glob("*.raw"))
    reset_paths_cache()

    assert len(archived) == 1
    assert "/" not in archived[0].name


def test_context_manager_closes_an_owned_client():
    adapter = WorldBankAdapter()
    with adapter:
        pass
    assert adapter._client.is_closed


def test_injected_client_is_left_open(adapter):
    with adapter:
        pass
    assert adapter._client.is_closed is False
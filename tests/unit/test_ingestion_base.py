"""Tests for the source adapter contract.

A fake adapter stands in for FRED and the World Bank throughout. That is
deliberate: this module's job is the contract, and testing it against a
real network call would be testing the network instead.
"""

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from moirai.core.exceptions import IngestionError
from moirai.engine.data_fabric.ingestion.base import (
    FetchResult,
    SourceAdapter,
    utc_now,
)
from moirai.engine.data_fabric.series.models import (
    Frequency,
    Observation,
    SeriesMetadata,
    Source,
    TimeSeries,
)

FETCHED_AT = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
PAYLOAD = b'{"observations": [{"date": "2025-03-01", "value": "4.2"}]}'


def make_result(**overrides) -> FetchResult:
    defaults: dict[str, Any] = dict(
        source=Source.MANUAL,
        source_series_id="demo",
        url="https://example.test/series?id=demo",
        content=PAYLOAD,
        fetched_at=FETCHED_AT,
    )
    return FetchResult(**{**defaults, **overrides})


class FakeAdapter(SourceAdapter):
    """Minimal concrete adapter used to exercise the base class."""

    supports_revisions = True

    def __init__(self, content: bytes = PAYLOAD, fail_parse: bool = False) -> None:
        self._content = content
        self._fail_parse = fail_parse
        self.fetch_calls: list[str] = []

    @property
    def source(self) -> Source:
        return Source.MANUAL

    def fetch_raw(self, source_series_id: str, **options: Any) -> FetchResult:
        self.fetch_calls.append(source_series_id)
        return make_result(source_series_id=source_series_id, content=self._content)

    def parse(self, result: FetchResult) -> TimeSeries:
        if self._fail_parse:
            raise ValueError("simulated parser bug")
        payload = json.loads(result.content)
        metadata = SeriesMetadata(
            series_id=result.source_series_id,
            title="Demo Series",
            source=self.source,
            source_series_id=result.source_series_id,
            frequency=Frequency.MONTHLY,
        )
        observations = [
            Observation(
                period=date.fromisoformat(row["date"]),
                value=float(row["value"]),
                known_at=result.fetched_at,
            )
            for row in payload["observations"]
        ]
        return TimeSeries.from_observations(metadata, observations)


@pytest.fixture(autouse=True)
def _sandbox_paths(tmp_path: Path, monkeypatch):
    """Archival must never touch the real data/raw directory."""
    monkeypatch.setenv("MOIRAI_HOME", str(tmp_path))
    from moirai.core.paths import reset_paths_cache

    reset_paths_cache()
    yield
    reset_paths_cache()


# --- FetchResult: hashing --------------------------------------------------

def test_hash_is_computed_on_construction():
    assert make_result().content_hash.startswith("sha256:")


def test_identical_content_produces_identical_hashes():
    assert make_result().content_hash == make_result().content_hash


def test_different_content_produces_different_hashes():
    assert make_result().content_hash != make_result(content=b"other").content_hash


def test_verify_accepts_untampered_content():
    assert make_result().verify() is True


def test_verify_rejects_tampered_content():
    original = make_result()
    tampered = original.model_copy(update={"content": b"modified"})
    assert tampered.verify() is False


def test_an_explicit_hash_is_preserved():
    """Reconstructing an archived result must keep its recorded hash."""
    result = make_result(content_hash="sha256:deadbeef")
    assert result.content_hash == "sha256:deadbeef"


# --- FetchResult: validation -----------------------------------------------

def test_naive_fetched_at_is_rejected():
    with pytest.raises(Exception, match="timezone-aware"):
        make_result(fetched_at=datetime(2026, 8, 8, 12, 0))


def test_result_is_frozen():
    with pytest.raises(Exception):
        make_result().url = "https://elsewhere.test"  # type: ignore[misc]


def test_unknown_field_is_rejected():
    with pytest.raises(Exception):
        make_result(typo=True)


def test_empty_content_is_flagged():
    assert make_result(content=b"").is_empty is True


def test_size_reports_byte_count():
    assert make_result(content=b"12345").size_bytes == 5


# --- FetchResult: provenance -----------------------------------------------

def test_provenance_is_json_serialisable():
    json.dumps(make_result().provenance())  # must not raise


def test_provenance_excludes_the_payload():
    provenance = make_result().provenance()
    assert "content" not in provenance
    assert PAYLOAD.decode() not in json.dumps(provenance)


def test_provenance_records_the_identifying_fields():
    provenance = make_result().provenance()
    assert provenance["source"] == "manual"
    assert provenance["source_series_id"] == "demo"
    assert provenance["size_bytes"] == len(PAYLOAD)
    assert provenance["fetched_at"].endswith("+00:00")


# --- FetchResult: archival -------------------------------------------------

def test_archive_writes_payload_and_sidecar(tmp_path: Path):
    path = make_result().archive()
    assert path.read_bytes() == PAYLOAD
    sidecar = path.with_name(path.stem + ".provenance.json")
    assert json.loads(sidecar.read_text(encoding="utf-8"))["source"] == "manual"


def test_archive_filename_carries_the_content_hash():
    path = make_result().archive()
    digest = make_result().content_hash.partition(":")[2][:16]
    assert digest in path.name


def test_archiving_the_same_content_twice_is_idempotent():
    first = make_result().archive()
    second = make_result().archive()
    assert first == second


def test_different_content_archives_to_a_different_file():
    first = make_result().archive()
    second = make_result(content=b"different bytes").archive()
    assert first != second
    assert first.exists() and second.exists()


def test_archive_accepts_an_explicit_directory(tmp_path: Path):
    target = tmp_path / "custom"
    path = make_result().archive(target)
    assert path.parent == target


# --- SourceAdapter: the contract -------------------------------------------

def test_incomplete_subclass_cannot_be_instantiated():
    class Incomplete(SourceAdapter):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete()  # type: ignore[abstract]


def test_base_class_cannot_be_instantiated():
    with pytest.raises(TypeError, match="abstract"):
        SourceAdapter()  # type: ignore[abstract]


def test_concrete_subclass_works():
    assert FakeAdapter().source is Source.MANUAL


def test_repr_names_the_source():
    assert "manual" in repr(FakeAdapter())


# --- SourceAdapter: fetch --------------------------------------------------

def test_fetch_returns_a_parsed_series():
    series = FakeAdapter().fetch("demo")
    assert len(series) == 1
    assert series.observations[0].value == 4.2


def test_fetch_archives_by_default():
    from moirai.core.paths import get_paths

    FakeAdapter().fetch("demo")
    archived = list((get_paths().raw / "manual").glob("*.raw"))
    assert len(archived) == 1


def test_archiving_can_be_disabled():
    from moirai.core.paths import get_paths

    FakeAdapter().fetch("demo", archive=False)
    assert not (get_paths().raw / "manual").exists()


def test_empty_response_raises_ingestion_error():
    with pytest.raises(IngestionError, match="no content"):
        FakeAdapter(content=b"").fetch("demo")


def test_parser_failure_is_wrapped_with_the_content_hash():
    with pytest.raises(IngestionError, match="content_hash") as info:
        FakeAdapter(fail_parse=True).fetch("demo")
    assert isinstance(info.value.__cause__, ValueError)


def test_ingestion_errors_from_parse_are_not_double_wrapped():
    class RaisesIngestionError(FakeAdapter):
        def parse(self, result: FetchResult) -> TimeSeries:
            raise IngestionError("specific parser message")

    with pytest.raises(IngestionError, match="specific parser message"):
        RaisesIngestionError().fetch("demo")


def test_fetch_passes_the_series_id_through():
    adapter = FakeAdapter()
    adapter.fetch("some_series", archive=False)
    assert adapter.fetch_calls == ["some_series"]


# --- SourceAdapter: replay -------------------------------------------------

def test_replay_reparses_archived_bytes():
    adapter = FakeAdapter()
    result = adapter.fetch_raw("demo")
    path = result.archive()

    replayed = adapter.replay(path, result)
    assert replayed.observations[0].value == 4.2


def test_replay_detects_a_corrupted_archive(tmp_path: Path):
    adapter = FakeAdapter()
    result = adapter.fetch_raw("demo")
    path = result.archive()
    path.write_bytes(b'{"observations": [{"date": "2025-03-01", "value": "9.9"}]}')

    with pytest.raises(IngestionError, match="does not match"):
        adapter.replay(path, result)


# --- helpers ---------------------------------------------------------------

def test_utc_now_is_timezone_aware():
    assert utc_now().tzinfo is not None
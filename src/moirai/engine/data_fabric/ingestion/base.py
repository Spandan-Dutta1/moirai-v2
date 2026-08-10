"""
The contract every data source adapter must satisfy.

Layers above Layer 0 never import an adapter directly. They receive a
TimeSeries and cannot tell whether it came from FRED, the World Bank, or a
spreadsheet an analyst dropped in data/raw. Adding a source means writing a
translator; nothing above changes.

Fetch and parse are deliberately separate:

    fetch_raw()  bytes exactly as the source returned them
    parse()      bytes -> TimeSeries

Splitting them is what makes the fabric replayable. The raw bytes are
archived with a content hash, so when a source changes its schema in two
years, the historical data is re-derivable: fix the parser, replay the
archive. If fetching and parsing were one step, the original response
would be gone and the data would only be as good as the parser that
happened to run that day.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from moirai.core.exceptions import IngestionError
from moirai.core.logging import get_logger
from moirai.core.paths import get_paths
from moirai.engine.data_fabric.series.models import Source, TimeSeries

log = get_logger(__name__)

HASH_ALGORITHM = "sha256"


def compute_hash(content: bytes, algorithm: str = HASH_ALGORITHM) -> str:
    """Return a prefixed digest, e.g. 'sha256:ab12...'.

    The prefix records which algorithm was used, so an archive written
    today stays verifiable if the default changes later.
    """
    return f"{algorithm}:{hashlib.new(algorithm, content).hexdigest()}"


def split_hash(prefixed: str) -> tuple[str, str]:
    """Split 'sha256:ab12...' into ('sha256', 'ab12...').

    Raises
    ------
    IngestionError
        If the string is not a well-formed prefixed digest.
    """
    algorithm, separator, digest = prefixed.partition(":")
    if not separator or not algorithm or not digest:
        raise IngestionError(f"malformed content hash: {prefixed!r}")
    return algorithm, digest


class FetchResult(BaseModel):
    """Raw bytes from a source plus everything needed to prove where they came from.

    This is the provenance atom of the whole platform. Every number that
    eventually appears in a report traces back to one of these.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    source: Source
    source_series_id: str = Field(min_length=1)
    url: str = Field(min_length=1, description="Full request URL, with secrets redacted.")
    content: bytes = Field(repr=False)
    fetched_at: datetime
    content_hash: str = Field(default="", description="Computed on construction if omitted.")
    status_code: int | None = None
    request_params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _compute_hash_and_check_time(self) -> Self:
        if self.fetched_at.tzinfo is None:
            raise ValueError("fetched_at must be timezone-aware")
        if not self.content_hash:
            object.__setattr__(self, "content_hash", compute_hash(self.content))
        return self

    @property
    def size_bytes(self) -> int:
        return len(self.content)

    @property
    def is_empty(self) -> bool:
        return not self.content

    def verify(self) -> bool:
        """Recompute the hash of `content` and compare against the recorded one."""
        algorithm, expected = split_hash(self.content_hash)
        return hashlib.new(algorithm, self.content).hexdigest() == expected

    def provenance(self) -> dict[str, Any]:
        """Serialisable record for the catalog. Excludes the payload itself."""
        return {
            "source": self.source.value,
            "source_series_id": self.source_series_id,
            "url": self.url,
            "fetched_at": self.fetched_at.isoformat(),
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "status_code": self.status_code,
            "request_params": self.request_params,
        }

    def archive(self, directory: Path | None = None) -> Path:
        """Write the raw bytes and a sidecar provenance file to data/raw.

        The payload filename carries the content hash, so an identical
        response fetched twice occupies one file, and a changed response
        can never overwrite an earlier one.
        """
        root = directory if directory is not None else get_paths().raw / self.source.value
        root.mkdir(parents=True, exist_ok=True)

        _, digest = split_hash(self.content_hash)
        stem = f"{self.source_series_id}__{self.fetched_at:%Y%m%dT%H%M%SZ}__{digest[:16]}"

        payload_path = root / f"{stem}.raw"
        payload_path.write_bytes(self.content)
        (root / f"{stem}.provenance.json").write_text(
            json.dumps(self.provenance(), indent=2, sort_keys=True), encoding="utf-8"
        )

        log.debug(
            "raw_archived",
            source=self.source.value,
            series=self.source_series_id,
            path=str(payload_path),
            size_bytes=self.size_bytes,
        )
        return payload_path


class SourceAdapter(ABC):
    """Base class for every data source.

    Subclasses implement three things: which Source they are, how to fetch
    raw bytes, and how to parse those bytes. Everything else - archival,
    logging, error wrapping - is handled here so no adapter reimplements it.
    """

    #: Whether this source publishes revisions. False means one vintage only.
    supports_revisions: bool = False

    @property
    @abstractmethod
    def source(self) -> Source:
        """Which Source this adapter serves."""

    @abstractmethod
    def fetch_raw(self, source_series_id: str, **options: Any) -> FetchResult:
        """Retrieve untouched bytes for one series.

        Implementations must not parse, clean, or interpret. Any failure
        should be raised as IngestionError with the cause chained.
        """

    @abstractmethod
    def parse(self, result: FetchResult) -> TimeSeries:
        """Turn raw bytes into a canonical TimeSeries.

        Must be a pure function of `result`: no network, no clock, no
        global state. That is what makes archived responses replayable.
        """

    # ---- provided behaviour ----

    def fetch(
        self, source_series_id: str, *, archive: bool = True, **options: Any
    ) -> TimeSeries:
        """Fetch, optionally archive, then parse. The normal entry point."""
        result = self.fetch_raw(source_series_id, **options)

        if result.is_empty:
            raise IngestionError(
                f"{self.source.value} returned no content for {source_series_id!r}"
            )

        if archive:
            result.archive()

        try:
            series = self.parse(result)
        except IngestionError:
            raise
        except Exception as err:
            raise IngestionError(
                f"failed to parse {self.source.value} response for "
                f"{source_series_id!r} (content_hash={result.content_hash})"
            ) from err

        log.info(
            "series_ingested",
            source=self.source.value,
            series=series.metadata.series_id,
            observations=len(series),
            has_revisions=series.has_revisions,
            content_hash=result.content_hash,
        )
        return series

    def replay(self, payload_path: Path, result: FetchResult) -> TimeSeries:
        """Re-parse an archived response.

        Used when a parser is fixed or a source changes format: historical
        data is re-derived from bytes rather than re-fetched, which would
        return today's vintage instead of the original one.

        The bytes on disk are hashed and compared against the hash recorded
        at fetch time. A mismatch means the archive was altered, so the
        replay is refused rather than silently producing different numbers.

        Note this deliberately does not recompute `content_hash` from the
        file: a hash compared against itself always matches, which would
        make the integrity check incapable of failing.
        """
        content = payload_path.read_bytes()
        algorithm, expected = split_hash(result.content_hash)
        actual = hashlib.new(algorithm, content).hexdigest()

        if actual != expected:
            raise IngestionError(
                f"archived content at {payload_path} does not match its recorded hash "
                f"(expected {algorithm}:{expected[:16]}, got {algorithm}:{actual[:16]})"
            )

        return self.parse(result.model_copy(update={"content": content}))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(source={self.source.value!r})"


def utc_now() -> datetime:
    """Current UTC time.

    Centralised so tests can monkeypatch one function instead of hunting
    datetime.now() calls across every adapter.
    """
    return datetime.now(UTC)
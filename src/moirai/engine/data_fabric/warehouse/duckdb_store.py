"""
Bitemporal warehouse backed by DuckDB.

DuckDB is embedded and columnar: one file, no server, scans optimised for
analytical queries over long columns. That matches both the local-first
requirement and the shape of time series work.

Two tables:

    observations    the bitemporal facts
    series_catalog  metadata and provenance, one row per series

The composite primary key (series_id, period, known_at, revision) is
bitemporality expressed in SQL. It permits many vintages of one period
while forbidding two values for the same vintage, which is the same
invariant TimeSeries enforces in memory.

as_of queries are pushed into SQL rather than filtered in Python, because
the warehouse will eventually hold far more rows than fit comfortably in
memory, and the filter is what makes a query cheap.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import duckdb

from moirai.core.config import get_settings
from moirai.core.exceptions import DataFabricError, VintageError
from moirai.core.logging import get_logger
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

SCHEMA_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key    VARCHAR PRIMARY KEY,
    value  VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS series_catalog (
    series_id            VARCHAR PRIMARY KEY,
    title                VARCHAR NOT NULL,
    source               VARCHAR NOT NULL,
    source_series_id     VARCHAR NOT NULL,
    frequency            VARCHAR NOT NULL,
    unit                 VARCHAR NOT NULL,
    unit_label           VARCHAR NOT NULL DEFAULT '',
    seasonal_adjustment  VARCHAR NOT NULL,
    geography            VARCHAR NOT NULL DEFAULT '',
    coverage_start       DATE,
    coverage_end         DATE,
    notes                VARCHAR NOT NULL DEFAULT '',
    license              VARCHAR NOT NULL DEFAULT '',
    retrieved_at         TIMESTAMPTZ,
    provenance           VARCHAR NOT NULL DEFAULT '{}',
    updated_at           TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS observations (
    series_id  VARCHAR    NOT NULL,
    period     DATE       NOT NULL,
    known_at   TIMESTAMPTZ NOT NULL,
    revision   INTEGER    NOT NULL DEFAULT 0,
    value      DOUBLE,
    PRIMARY KEY (series_id, period, known_at, revision)
);

CREATE INDEX IF NOT EXISTS idx_observations_series_period
    ON observations (series_id, period);
"""


class Warehouse:
    """Persistent bitemporal store.

    Parameters
    ----------
    path
        Database file. Defaults to the configured warehouse path.
        Pass ":memory:" for an ephemeral store, which is what tests use.
    read_only
        Open without write access. Useful when several processes read
        the same file during a parallel simulation.
    """

    def __init__(self, path: Path | str | None = None, *, read_only: bool = False) -> None:
        if path is None:
            resolved: Path | str = get_settings().warehouse_path
        else:
            resolved = path

        if resolved != ":memory:":
            resolved = Path(resolved)
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved = str(resolved)

        self.path = resolved
        self._read_only = read_only
        try:
            self._connection = duckdb.connect(resolved, read_only=read_only)
        except duckdb.Error as err:
            raise DataFabricError(f"could not open warehouse at {resolved}: {err}") from err

        if not read_only:
            self._initialise_schema()

    # ---- lifecycle ----

    def _initialise_schema(self) -> None:
        self._connection.execute(SCHEMA_SQL)
        self._connection.execute(
            "INSERT INTO schema_meta (key, value) VALUES ('version', ?) "
            "ON CONFLICT (key) DO NOTHING",
            [str(SCHEMA_VERSION)],
        )

    @property
    def schema_version(self) -> int:
        row = self._connection.execute(
            "SELECT value FROM schema_meta WHERE key = 'version'"
        ).fetchone()
        if row is None:
            raise DataFabricError("warehouse has no schema version recorded")
        return int(row[0])

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Group writes so a failure leaves no partial data behind."""
        self._connection.execute("BEGIN TRANSACTION")
        try:
            yield
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def _require_writable(self) -> None:
        if self._read_only:
            raise DataFabricError("warehouse is open read-only")

    # ---- writing ----

    def upsert_series(self, series: TimeSeries, provenance: dict[str, Any] | None = None) -> int:
        """Store a series and its observations. Idempotent.

        Re-ingesting identical data is a no-op rather than a duplicate,
        because the primary key already identifies each vintage. Returns
        the number of observation rows written.
        """
        self._require_writable()
        metadata = series.metadata

        with self.transaction():
            self._connection.execute(
                """
                INSERT INTO series_catalog (
                    series_id, title, source, source_series_id, frequency, unit,
                    unit_label, seasonal_adjustment, geography, coverage_start,
                    coverage_end, notes, license, retrieved_at, provenance, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, now())
                ON CONFLICT (series_id) DO UPDATE SET
                    title = excluded.title,
                    source = excluded.source,
                    source_series_id = excluded.source_series_id,
                    frequency = excluded.frequency,
                    unit = excluded.unit,
                    unit_label = excluded.unit_label,
                    seasonal_adjustment = excluded.seasonal_adjustment,
                    geography = excluded.geography,
                    coverage_start = excluded.coverage_start,
                    coverage_end = excluded.coverage_end,
                    notes = excluded.notes,
                    license = excluded.license,
                    retrieved_at = excluded.retrieved_at,
                    provenance = excluded.provenance,
                    updated_at = now()
                """,
                [
                    metadata.series_id,
                    metadata.title,
                    metadata.source.value,
                    metadata.source_series_id,
                    metadata.frequency.value,
                    metadata.unit.value,
                    metadata.unit_label,
                    metadata.seasonal_adjustment.value,
                    metadata.geography,
                    metadata.start,
                    metadata.end,
                    metadata.notes,
                    metadata.license,
                    metadata.retrieved_at,
                    json.dumps(provenance or {}, sort_keys=True),
                ],
            )

            rows = [
                (metadata.series_id, o.period, o.known_at, o.revision, o.value)
                for o in series.observations
            ]
            if rows:
                self._connection.executemany(
                    """
                    INSERT INTO observations (series_id, period, known_at, revision, value)
                    VALUES (?,?,?,?,?)
                    ON CONFLICT (series_id, period, known_at, revision) DO UPDATE SET
                        value = excluded.value
                    """,
                    rows,
                )

        log.info(
            "series_stored",
            series=metadata.series_id,
            source=metadata.source.value,
            observations=len(rows),
        )
        return len(rows)

    def delete_series(self, series_id: str) -> None:
        """Remove a series and all its observations."""
        self._require_writable()
        with self.transaction():
            self._connection.execute(
                "DELETE FROM observations WHERE series_id = ?", [series_id]
            )
            self._connection.execute(
                "DELETE FROM series_catalog WHERE series_id = ?", [series_id]
            )
        log.warning("series_deleted", series=series_id)

    # ---- reading ----

    def list_series(self) -> tuple[str, ...]:
        rows = self._connection.execute(
            "SELECT series_id FROM series_catalog ORDER BY series_id"
        ).fetchall()
        return tuple(row[0] for row in rows)

    def has_series(self, series_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM series_catalog WHERE series_id = ?", [series_id]
        ).fetchone()
        return row is not None

    def get_metadata(self, series_id: str) -> SeriesMetadata:
        row = self._connection.execute(
            """
            SELECT series_id, title, source, source_series_id, frequency, unit,
                   unit_label, seasonal_adjustment, geography, coverage_start,
                   coverage_end, notes, license, retrieved_at
            FROM series_catalog WHERE series_id = ?
            """,
            [series_id],
        ).fetchone()
        if row is None:
            raise DataFabricError(f"series {series_id!r} is not in the warehouse")

        return SeriesMetadata(
            series_id=row[0],
            title=row[1],
            source=Source(row[2]),
            source_series_id=row[3],
            frequency=Frequency(row[4]),
            unit=Unit(row[5]),
            unit_label=row[6],
            seasonal_adjustment=SeasonalAdjustment(row[7]),
            geography=row[8],
            start=row[9],
            end=row[10],
            notes=row[11],
            license=row[12],
            retrieved_at=row[13],
        )

    def get_provenance(self, series_id: str) -> dict[str, Any]:
        row = self._connection.execute(
            "SELECT provenance FROM series_catalog WHERE series_id = ?", [series_id]
        ).fetchone()
        if row is None:
            raise DataFabricError(f"series {series_id!r} is not in the warehouse")
        return json.loads(row[0])

    def _build(self, series_id: str, rows: Sequence[tuple]) -> TimeSeries:
        observations = [
            Observation(period=row[0], known_at=row[1], revision=row[2], value=row[3])
            for row in rows
        ]
        return TimeSeries.from_observations(self.get_metadata(series_id), observations)

    def get_series(
        self,
        series_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> TimeSeries:
        """Every vintage of a series, optionally within a period window."""
        clauses = ["series_id = ?"]
        params: list[Any] = [series_id]
        if start is not None:
            clauses.append("period >= ?")
            params.append(start)
        if end is not None:
            clauses.append("period <= ?")
            params.append(end)

        rows = self._connection.execute(
            f"""
            SELECT period, known_at, revision, value
            FROM observations
            WHERE {" AND ".join(clauses)}
            ORDER BY period, known_at, revision
            """,
            params,
        ).fetchall()
        return self._build(series_id, rows)

    def as_of(
        self,
        series_id: str,
        when: datetime,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> TimeSeries:
        """The series as it was known at `when`, one observation per period.

        The vintage filter runs in SQL. Fetching everything and filtering in
        Python would work on a toy dataset and fall over on a real one.
        """
        if when.tzinfo is None:
            raise VintageError(f"when must be timezone-aware, got {when!r}")

        clauses = ["series_id = ?", "known_at <= ?"]
        params: list[Any] = [series_id, when]
        if start is not None:
            clauses.append("period >= ?")
            params.append(start)
        if end is not None:
            clauses.append("period <= ?")
            params.append(end)

        rows = self._connection.execute(
            f"""
            SELECT period, known_at, revision, value FROM (
                SELECT period, known_at, revision, value,
                       ROW_NUMBER() OVER (
                           PARTITION BY period
                           ORDER BY known_at DESC, revision DESC
                       ) AS rank
                FROM observations
                WHERE {" AND ".join(clauses)}
            )
            WHERE rank = 1
            ORDER BY period
            """,
            params,
        ).fetchall()
        return self._build(series_id, rows)

    def latest(
        self,
        series_id: str,
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> TimeSeries:
        """The most recent vintage of each period: today's best estimate.

        Implemented as an as-of query at a far-future timestamp, so there is
        one ranking path rather than two that could drift apart.
        """
        far_future = datetime(9999, 12, 31, tzinfo=UTC)
        return self.as_of(series_id, far_future, start=start, end=end)

    def vintage_dates(self, series_id: str) -> tuple[datetime, ...]:
        rows = self._connection.execute(
            "SELECT DISTINCT known_at FROM observations WHERE series_id = ? ORDER BY known_at",
            [series_id],
        ).fetchall()
        return tuple(row[0] for row in rows)

    def count_observations(self, series_id: str | None = None) -> int:
        if series_id is None:
            row = self._connection.execute("SELECT count(*) FROM observations").fetchone()
        else:
            row = self._connection.execute(
                "SELECT count(*) FROM observations WHERE series_id = ?", [series_id]
            ).fetchone()
        return int(row[0]) if row else 0

    def query(self, sql: str, params: Sequence[Any] | None = None) -> list[tuple]:
        """Escape hatch for ad-hoc analytical SQL."""
        return self._connection.execute(sql, list(params or [])).fetchall()

    def __repr__(self) -> str:
        return f"Warehouse(path={self.path!r}, read_only={self._read_only})"



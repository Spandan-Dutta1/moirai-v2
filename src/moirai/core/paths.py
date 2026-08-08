"""
Canonical filesystem layout for Moirai.

Design rules
------------
1. Nothing in Moirai builds paths by string concatenation.
2. The project root is discovered once, deterministically.
3. MOIRAI_HOME always wins, which makes tests hermetic and lets a
   packaged desktop build relocate its data directory.
4. The Paths object is frozen: a path cannot be mutated mid-run,
   which would silently invalidate a run's provenance record.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from moirai.core.exceptions import PathError

ENV_HOME = "MOIRAI_HOME"
ROOT_MARKERS: tuple[str, ...] = ("pyproject.toml", ".git")


def find_project_root(start: Path | None = None) -> Path:
    """Walk upward from `start` until a directory holding a root marker is found.

    Raises
    ------
    PathError
        If no marker is found before the filesystem root.
    """
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in ROOT_MARKERS):
            return candidate

    raise PathError(
        f"Could not locate the Moirai project root above {current}. "
        f"Expected one of {ROOT_MARKERS}, or set {ENV_HOME}."
    )


@dataclass(frozen=True, slots=True)
class Paths:
    """Immutable view of the Moirai directory layout."""

    root: Path

    # ---- top level ----
    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def docs(self) -> Path:
        return self.root / "docs"

    @property
    def research(self) -> Path:
        return self.root / "research"

    # ---- Layer 0: Data Fabric ----
    @property
    def raw(self) -> Path:
        """Immutable bytes exactly as downloaded. Never edited."""
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        """Intermediate artifacts. Safe to delete and regenerate."""
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        """Cleaned, schema-validated, analysis-ready datasets."""
        return self.data / "processed"

    @property
    def warehouse(self) -> Path:
        """DuckDB database files."""
        return self.data / "warehouse"

    @property
    def catalog(self) -> Path:
        """Dataset metadata and provenance records."""
        return self.data / "catalog"

    # ---- Reproducibility ----
    @property
    def runs(self) -> Path:
        """One directory per experiment run: the Run Ledger."""
        return self.root / "runs"

    @property
    def artifacts(self) -> Path:
        """Class B replay artifacts: prompts, LLM outputs, debate transcripts."""
        return self.root / "artifacts"

    @property
    def cache(self) -> Path:
        return self.root / ".moirai_cache"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    # ---- helpers ----
    def all_dirs(self) -> tuple[Path, ...]:
        return (
            self.data,
            self.raw,
            self.interim,
            self.processed,
            self.warehouse,
            self.catalog,
            self.runs,
            self.artifacts,
            self.cache,
            self.logs,
        )

    def ensure(self) -> Paths:
        """Create every managed directory. Idempotent. Returns self."""
        for directory in self.all_dirs():
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def run_dir(self, run_id: str) -> Path:
        """Path for one run's ledger entry. Guards against traversal."""
        if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
            raise PathError(f"Invalid run_id: {run_id!r}")
        return self.runs / run_id


@lru_cache(maxsize=1)
def get_paths() -> Paths:
    """Resolve the project layout once per process.

    Resolution order: $MOIRAI_HOME, then upward search for a root marker.
    """
    override = os.environ.get(ENV_HOME)
    root = Path(override).expanduser().resolve() if override else find_project_root()
    return Paths(root=root)


def reset_paths_cache() -> None:
    """Clear the memoised layout. Used by tests and config reloads."""
    get_paths.cache_clear()
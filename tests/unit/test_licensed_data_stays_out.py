"""Licensed data must not reach the public repository (ADR 039).

`moirai-v2` is public. CMIE's Consumer Pyramids data is licensed to an
institution and a named user, and committing any of it - a raw wave, a derived
extract, a sample, a few rows in a fixture - breaches that licence and cannot
be undone, because git keeps the history and GitHub keeps the forks.

A line in a decision record is advice, and so is `.gitignore`: `git add -f`
walks past it, and so does a file added before the pattern existed. This is
the control that fails.

A file is licensed data when its **name** looks like a CMIE download or
extract **and** its **extension** is one that can carry data. Both halves are
required. The first version checked only the name, and promptly failed on
`scripts/check_cphs_labour.py` and on this project's own decision records -
source code named after the survey is not the survey.

What may be committed is what a decision record can state: an estimated
parameter, a transition rate, a sample size, a method. Numbers are not the
licensed thing; records are.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Filenames CMIE's downloads and this project's extracts of them produce.
#: `poi_` and `inc_` are the compact per-wave and per-month extracts; a
#: derived file is as licensed as the download it came from.
LICENSED_PATTERNS = (
    "*people_of_india*",
    "*household_income*",
    "*income_pyramids*",
    "*consumption_pyramids*",
    "*aspirational_india*",
    "*_R_csv*",
    "*_MS_rev*",
    "poi_*",
    "inc_*",
)

#: Extensions that can carry data out of the repository. A `.py` or `.md`
#: whose name mentions CPHS is code or prose about the data, not the data.
DATA_SUFFIXES = {
    ".csv", ".tsv", ".parquet", ".feather", ".dta", ".sav", ".zip", ".gz",
    ".7z", ".rar", ".txt", ".xlsx", ".xls", ".json", ".db", ".sqlite",
}

#: Where the project's own small, publishable fixtures live.
ALLOWED = ("tests/fixtures/", "docs/")

#: Anything this size tracked by git is data, whatever it is called.
MAX_TRACKED_BYTES = 5_000_000

IGNORED_PARTS = {".git", "__pycache__", ".venv", ".mypy_cache", ".pytest_cache"}


def _is_data_file(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in DATA_SUFFIXES
        and not IGNORED_PARTS & set(path.parts)
    )


def _tracked() -> list[Path]:
    """Files git knows about, which is what would actually be published."""
    listed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [ROOT / name for name in listed.stdout.split("\0") if name]


def _allowed(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    return any(relative.startswith(prefix) for prefix in ALLOWED)


def looks_licensed(name: str) -> bool:
    """Name and extension both have to say data for this to be true."""
    path = Path(name)
    return path.suffix.lower() in DATA_SUFFIXES and any(
        path.match(pattern) for pattern in LICENSED_PATTERNS
    )


def test_no_licensed_data_file_sits_anywhere_in_the_repository() -> None:
    """The working tree, because a file exists before it is committed."""
    found = sorted(
        path.relative_to(ROOT).as_posix()
        for pattern in LICENSED_PATTERNS
        for path in ROOT.rglob(pattern)
        if _is_data_file(path)
    )
    assert not found, (
        f"licensed CMIE data inside the repository: {found}. Move it outside "
        "the repository entirely - a .gitignore entry is not sufficient, "
        "because git add -f ignores it."
    )


def test_git_is_not_tracking_a_bulk_data_file() -> None:
    """Catches an extract renamed to something innocuous."""
    offenders = sorted(
        path.relative_to(ROOT).as_posix()
        for path in _tracked()
        if path.is_file()
        and path.suffix.lower() in DATA_SUFFIXES
        and not _allowed(path)
        and path.stat().st_size > MAX_TRACKED_BYTES
    )
    assert not offenders, (
        f"large data files tracked by git: {offenders}. If one is legitimately "
        "part of the project, add its directory to ALLOWED and say why in a "
        "decision record."
    )


def test_the_patterns_catch_every_real_cmie_filename() -> None:
    """The guard is only worth having if it matches what CMIE actually ships."""
    real = [
        "people_of_india_20220101_20220430_R.csv",
        "people_of_india_20220101_20220430_R_csv.zip",
        "household_income_20220131_MS_rev.csv",
        "household_income_20220131_MS_rev_csv.zip",
        "consumption_pyramids_20230501_20230831_R_csv.zip",
        "income_pyramids_20240101_20240430_R.csv",
        "aspirational_india_20230101_20230430_R.csv",
        "poi_20220101_20220430.parquet",
        "inc_20220131.parquet",
    ]
    missed = [name for name in real if not looks_licensed(name)]
    assert not missed, f"real CMIE filenames the guard would miss: {missed}"


def test_the_guard_does_not_flag_the_code_that_reads_the_data() -> None:
    """Its first version failed on exactly these, which is why they are here."""
    innocent = [
        "scripts/check_cphs_labour.py",
        "scripts/check_cphs_replacement.py",
        "tests/unit/test_cphs_labour.py",
        "tests/unit/test_cphs_replacement.py",
        "docs/decisions/0040-the-replacement-share-measured-on-cphs.md",
        "docs/decisions/0039-labour-rates-measured-on-the-cphs-panel.md",
    ]
    flagged = [name for name in innocent if looks_licensed(name)]
    assert not flagged, f"the guard flagged source or prose, not data: {flagged}"


def test_a_data_extension_alone_is_not_enough() -> None:
    """The project's own outputs must stay publishable."""
    assert not looks_licensed("docs/model_documentation.md")
    assert not looks_licensed("tests/fixtures/small_tape.csv")
    assert not looks_licensed("data/fred_indpro.csv")

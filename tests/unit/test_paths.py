"""Tests for the canonical filesystem layout.

Every test runs against pytest's tmp_path sandbox, never the real project.
That is the point of the MOIRAI_HOME override: tests that touch the real
data/ or runs/ directories are not tests, they are accidents waiting.
"""

from pathlib import Path

import pytest

from moirai.core.exceptions import PathError
from moirai.core.paths import (
    ENV_HOME,
    Paths,
    find_project_root,
    get_paths,
    reset_paths_cache,
)


@pytest.fixture(autouse=True)
def _cold_cache():
    """Every test starts and ends with an empty path cache."""
    reset_paths_cache()
    yield
    reset_paths_cache()


# --- find_project_root -----------------------------------------------------

def test_finds_marker_in_same_directory(tmp_path: Path):
    (tmp_path / "pyproject.toml").touch()
    assert find_project_root(tmp_path) == tmp_path.resolve()


def test_walks_upward_to_find_marker(tmp_path: Path):
    (tmp_path / "pyproject.toml").touch()
    deep = tmp_path / "src" / "moirai" / "engine" / "causal"
    deep.mkdir(parents=True)
    assert find_project_root(deep) == tmp_path.resolve()


def test_accepts_a_file_and_uses_its_parent(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    module = tmp_path / "src" / "mod.py"
    module.parent.mkdir(parents=True)
    module.touch()
    assert find_project_root(module) == tmp_path.resolve()


def test_git_directory_also_counts_as_a_marker(tmp_path: Path):
    (tmp_path / ".git").mkdir()
    assert find_project_root(tmp_path) == tmp_path.resolve()


def test_raises_when_no_marker_exists(tmp_path: Path):
    orphan = tmp_path / "nowhere"
    orphan.mkdir()
    with pytest.raises(PathError, match="project root"):
        find_project_root(orphan)


# --- layout ----------------------------------------------------------------

@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        ("data", ("data",)),
        ("raw", ("data", "raw")),
        ("interim", ("data", "interim")),
        ("processed", ("data", "processed")),
        ("warehouse", ("data", "warehouse")),
        ("catalog", ("data", "catalog")),
        ("runs", ("runs",)),
        ("artifacts", ("artifacts",)),
        ("logs", ("logs",)),
        ("cache", (".moirai_cache",)),
    ],
)
def test_every_path_is_derived_from_root(tmp_path: Path, attribute, expected):
    paths = Paths(root=tmp_path)
    assert getattr(paths, attribute) == tmp_path.joinpath(*expected)


def test_paths_is_frozen(tmp_path: Path):
    paths = Paths(root=tmp_path)
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        paths.root = tmp_path / "elsewhere"  # type: ignore[misc]


def test_slots_prevent_attribute_injection(tmp_path: Path):
    """Cannot bolt new attributes onto a Paths instance.

    Note: `frozen=True` + `slots=True` rebuilds the class, so the rejection
    surfaces as TypeError rather than the AttributeError plain slots give.
    What matters is that the assignment is refused.
    """
    paths = Paths(root=tmp_path)
    with pytest.raises((AttributeError, TypeError)):
        paths.sneaky = "value"  # type: ignore[attr-defined]

def test_two_paths_with_same_root_are_equal(tmp_path: Path):
    assert Paths(root=tmp_path) == Paths(root=tmp_path)


# --- ensure ----------------------------------------------------------------

def test_ensure_creates_every_directory(tmp_path: Path):
    paths = Paths(root=tmp_path).ensure()
    missing = [d for d in paths.all_dirs() if not d.is_dir()]
    assert missing == []


def test_ensure_is_idempotent(tmp_path: Path):
    paths = Paths(root=tmp_path)
    paths.ensure()
    paths.ensure()  # must not raise on pre-existing directories
    assert all(d.is_dir() for d in paths.all_dirs())


def test_ensure_returns_self_for_chaining(tmp_path: Path):
    paths = Paths(root=tmp_path)
    assert paths.ensure() is paths


def test_ensure_does_not_touch_unmanaged_directories(tmp_path: Path):
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / "keep.ipynb").touch()
    Paths(root=tmp_path).ensure()
    assert (tmp_path / "notebooks" / "keep.ipynb").exists()


# --- run_dir traversal guard -----------------------------------------------

@pytest.mark.parametrize(
    "unsafe",
    ["", "../escape", "../../etc/passwd", "a/b", "a\\b", ".hidden", "./x"],
)
def test_run_dir_rejects_unsafe_identifiers(tmp_path: Path, unsafe: str):
    with pytest.raises(PathError):
        Paths(root=tmp_path).run_dir(unsafe)


@pytest.mark.parametrize(
    "safe",
    ["2026-08-08T101500Z-a1b2c3", "run_001", "experiment-42"],
)
def test_run_dir_accepts_valid_identifiers(tmp_path: Path, safe: str):
    paths = Paths(root=tmp_path)
    assert paths.run_dir(safe) == paths.runs / safe


def test_run_dir_stays_inside_the_runs_directory(tmp_path: Path):
    paths = Paths(root=tmp_path)
    resolved = paths.run_dir("valid-run").resolve()
    assert paths.runs.resolve() in resolved.parents


# --- get_paths and its cache -----------------------------------------------

def test_env_var_overrides_discovery(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(ENV_HOME, str(tmp_path))
    reset_paths_cache()
    assert get_paths().root == tmp_path.resolve()


def test_get_paths_is_cached_within_a_process(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(ENV_HOME, str(tmp_path))
    reset_paths_cache()
    first = get_paths()
    monkeypatch.setenv(ENV_HOME, str(tmp_path / "other"))
    assert get_paths() is first  # same object: the cache held


def test_resetting_the_cache_picks_up_a_new_root(tmp_path: Path, monkeypatch):
    first_root = tmp_path / "a"
    second_root = tmp_path / "b"
    first_root.mkdir()
    second_root.mkdir()

    monkeypatch.setenv(ENV_HOME, str(first_root))
    reset_paths_cache()
    assert get_paths().root == first_root.resolve()

    monkeypatch.setenv(ENV_HOME, str(second_root))
    reset_paths_cache()
    assert get_paths().root == second_root.resolve()


def test_falls_back_to_discovery_without_the_env_var(monkeypatch):
    monkeypatch.delenv(ENV_HOME, raising=False)
    reset_paths_cache()
    assert (get_paths().root / "pyproject.toml").exists()
"""Tests for the Moirai exception hierarchy.

These tests assert *structure*, not behaviour. That sounds odd until you
realise the hierarchy IS the contract: if someone later makes
SchemaValidationError inherit from Exception directly, every
`except DataFabricError` block in the Data Fabric silently stops
catching it. These tests make that regression impossible.
"""

import inspect

import pytest

from moirai.core import exceptions as exc
from moirai.core.exceptions import (
    ConfigError,
    DataFabricError,
    EngineError,
    ExperimentError,
    IngestionError,
    MoiraiError,
    PathError,
    ProvenanceError,
    ReproducibilityError,
    SchemaValidationError,
    VintageError,
)

ALL_ERRORS = [
    ConfigError,
    PathError,
    DataFabricError,
    IngestionError,
    SchemaValidationError,
    VintageError,
    ProvenanceError,
    ReproducibilityError,
    EngineError,
    ExperimentError,
]


# --- Root of the hierarchy -------------------------------------------------

def test_moirai_error_is_an_exception():
    assert issubclass(MoiraiError, Exception)


def test_moirai_error_is_not_base_exception_directly():
    """Never inherit from BaseException: it would dodge `except Exception`
    and swallow KeyboardInterrupt / SystemExit semantics."""
    assert MoiraiError.__mro__[1] is Exception


@pytest.mark.parametrize("error_cls", ALL_ERRORS)
def test_every_error_descends_from_moirai_error(error_cls):
    assert issubclass(error_cls, MoiraiError)


# --- Layer 0 sub-hierarchy -------------------------------------------------

@pytest.mark.parametrize(
    "error_cls", [IngestionError, SchemaValidationError, VintageError]
)
def test_data_fabric_errors_share_a_common_parent(error_cls):
    assert issubclass(error_cls, DataFabricError)


def test_catching_data_fabric_error_catches_children():
    with pytest.raises(DataFabricError):
        raise SchemaValidationError("bad column type")


def test_catching_data_fabric_error_does_not_catch_siblings():
    """A config problem must not be mistaken for a data problem."""
    with pytest.raises(ConfigError):
        try:
            raise ConfigError("missing setting")
        except DataFabricError:  # pragma: no cover - must not trigger
            pytest.fail("ConfigError was wrongly caught as DataFabricError")


# --- Behaviour as ordinary exceptions --------------------------------------

@pytest.mark.parametrize("error_cls", ALL_ERRORS)
def test_error_preserves_its_message(error_cls):
    err = error_cls("something specific went wrong")
    assert str(err) == "something specific went wrong"


@pytest.mark.parametrize("error_cls", ALL_ERRORS)
def test_error_can_be_raised_and_caught_as_moirai_error(error_cls):
    with pytest.raises(MoiraiError):
        raise error_cls("boom")


def test_exception_chaining_is_preserved():
    """`raise ... from ...` must keep the original cause for debugging."""
    original = ValueError("underlying parser failure")
    with pytest.raises(IngestionError) as info:
        try:
            raise original
        except ValueError as e:
            raise IngestionError("could not ingest source") from e
    assert info.value.__cause__ is original


# --- Hygiene ---------------------------------------------------------------

def test_no_error_class_was_forgotten_by_the_tests():
    """If someone adds a new exception, this test fails until it is
    registered in ALL_ERRORS - a cheap guard against untested additions."""
    defined = {
        obj
        for _, obj in inspect.getmembers(exc, inspect.isclass)
        if issubclass(obj, MoiraiError) and obj is not MoiraiError
    }
    assert defined == set(ALL_ERRORS)


@pytest.mark.parametrize("error_cls", [MoiraiError, *ALL_ERRORS])
def test_every_error_has_a_docstring(error_cls):
    assert error_cls.__doc__, f"{error_cls.__name__} needs a docstring"
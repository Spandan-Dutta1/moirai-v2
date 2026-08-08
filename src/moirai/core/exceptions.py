"""
Moirai exception hierarchy.

Every error Moirai raises deliberately inherits from MoiraiError.
Callers can then catch at whatever level of abstraction they need,
while genuine bugs (TypeError, KeyError, ...) stay unhandled and loud.
"""

from __future__ import annotations


class MoiraiError(Exception):
    """Base class for all errors raised intentionally by Moirai."""


# ---- Infrastructure -------------------------------------------------------

class ConfigError(MoiraiError):
    """Invalid, missing, or contradictory configuration."""


class PathError(MoiraiError):
    """Project root could not be resolved, or a path escaped its sandbox."""


# ---- Layer 0: Data Fabric -------------------------------------------------

class DataFabricError(MoiraiError):
    """Base for all Layer 0 failures."""


class IngestionError(DataFabricError):
    """A source could not be fetched or parsed."""


class SchemaValidationError(DataFabricError):
    """Incoming data violated its declared schema."""


class VintageError(DataFabricError):
    """A bitemporal vintage was requested that does not exist."""


# ---- Cross-cutting guarantees ---------------------------------------------

class ProvenanceError(MoiraiError):
    """A result was produced without traceable lineage."""


class ReproducibilityError(MoiraiError):
    """A Class A (bit-reproducible) run failed to reproduce."""


# ---- Engine / Experiments -------------------------------------------------

class EngineError(MoiraiError):
    """Base for computational engine failures."""


class ExperimentError(MoiraiError):
    """Invalid or mutated experiment specification."""
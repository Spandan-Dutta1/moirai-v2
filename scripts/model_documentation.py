"""
Model documentation, generated from the repository itself (Stage 6).

A model validator asks four things: what does it claim, what did you decide
and why, where did every number come from, and what evidence do you have. The
first is a paragraph. The other three are already in the repository, in the
decision records, in the parameter descriptions and in the check scripts, so
they are read out of it rather than written again. A document maintained by
hand drifts from the code; this one cannot.

The parameter inventory works because every field's description declares its
provenance in one word: SOURCED, CALIBRATED, DERIVED, CONTRACTUAL, CONVENTION,
ASSUMED, SWITCH for a field that is not a model value, or METADATA for a label.
A field whose description declares nothing is reported as a gap rather than
quietly passed over, and `test_every_parameter_declares_its_provenance` fails
when one appears.

Prints to standard output, like every other script here. Redirect it to keep a
copy:

    python scripts/model_documentation.py > docs/model_documentation.md
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ROOT / "docs" / "decisions"
TESTS = ROOT / "tests"
SCRIPTS = ROOT / "scripts"

#: Classes whose fields are model choices. Results are excluded, and so is a
#: CsvSpec, whose fields map a publisher's columns rather than set a value.
SUFFIXES = ("Parameters", "Profile")

#: The provenance a description may declare, in the order they are looked for.
PROVENANCE = (
    "SOURCED",
    "CALIBRATED",
    "DERIVED",
    "CONTRACTUAL",
    "CONVENTION",
    "ASSUMED",
    "SWITCH",
    "METADATA",
)

UNLABELLED = "undeclared"

SCOPE = """Moirai traces one shock - the Federal Reserve tightening, or the Reserve
Bank of India responding - from the global policy rate to an individual Indian
borrower's monthly instalment, and from there to the provision its lender must
hold. It is a partial-equilibrium chain of reduced-form links, in the satellite
form used for central-bank stress testing: a macro scenario, a transmission
layer, a sector layer and a borrower microsimulation. It is not a general
equilibrium model and nothing feeds back up the chain.

Intended use is comparative: which borrowers a given shock reaches, through
which contract feature, and what that does to expected credit loss under the
Reserve Bank's directions. It is not calibrated to forecast the level of
defaults for any lender, and no result in it should be read as one."""


@dataclass(frozen=True)
class Decision:
    """One architectural decision record."""

    number: int
    title: str
    status: str


@dataclass(frozen=True)
class Parameter:
    """One declared model choice, and where its value came from."""

    owner: str
    module: str
    name: str
    default: str
    provenance: str
    description: str


def decisions() -> list[Decision]:
    """Every decision record, by number."""
    found: list[Decision] = []
    for path in sorted(DECISIONS.glob("[0-9][0-9][0-9][0-9]-*.md")):
        text = path.read_text(encoding="utf-8").splitlines()
        heading = next((line for line in text if line.startswith("# ")), "")
        status = next(
            (line.split(":", 1)[1].strip() for line in text if line.startswith("Status:")),
            "unrecorded",
        )
        title = re.sub(r"^#\s*\d+\.\s*", "", heading).strip() or path.stem
        found.append(Decision(int(path.name[:4]), title, status))
    return found


def _modules() -> list[Any]:
    """Every importable module of the engine, so nothing is missed by hand."""
    package = importlib.import_module("moirai")
    loaded = [package]
    for info in pkgutil.walk_packages(package.__path__, prefix="moirai."):
        try:
            loaded.append(importlib.import_module(info.name))
        except Exception as err:  # a module that cannot be imported is reported
            print(f"<!-- could not read {info.name}: {type(err).__name__}: {err} -->")
    return loaded


def _provenance(description: str) -> str:
    """The declaration, which must be the description's first word.

    Anywhere else and it is prose: "the common industry convention" explains an
    ASSUMED field rather than declaring a different kind.
    """
    first = re.match(r"\s*([A-Za-z-]+)", description)
    if first and first.group(1).upper() in PROVENANCE:
        return first.group(1).upper().lower()
    return UNLABELLED


def _default(field: FieldInfo) -> str:
    value = field.get_default(call_default_factory=True)
    if value is None or repr(value) == "PydanticUndefined":
        return "required"
    return repr(value)


def parameters() -> list[Parameter]:
    """Every declared field of every parameter class in the engine."""
    seen: set[type] = set()
    found: list[Parameter] = []
    for module in _modules():
        for name in dir(module):
            candidate = getattr(module, name, None)
            if (
                not isinstance(candidate, type)
                or not issubclass(candidate, BaseModel)
                or candidate in seen
                or not name.endswith(SUFFIXES)
                or candidate.__module__ != getattr(module, "__name__", "")
            ):
                continue
            seen.add(candidate)
            for field_name, field in candidate.model_fields.items():
                description = field.description or ""
                found.append(
                    Parameter(
                        owner=name,
                        module=candidate.__module__,
                        name=field_name,
                        default=_default(field),
                        provenance=_provenance(description),
                        description=" ".join(description.split()),
                    )
                )
    return sorted(found, key=lambda p: (p.module, p.owner, p.name))


def test_count() -> int:
    """Test functions in the suite, counted from the files.

    Fewer than pytest collects, because a parametrised function is one
    definition and several cases.
    """
    return sum(
        len(re.findall(r"^def test_", path.read_text(encoding="utf-8"), re.M))
        for path in TESTS.rglob("test_*.py")
    )


def evidence() -> list[tuple[str, str]]:
    """The check scripts, with the first line of each docstring."""
    register: list[tuple[str, str]] = []
    for path in sorted(SCRIPTS.glob("check_*.py")):
        text = path.read_text(encoding="utf-8")
        match = re.search(r'^"""(.*?)(?:\n|""")', text, re.S)
        first = match.group(1).strip() if match else ""
        register.append((path.name, " ".join(first.split())))
    return register


def main() -> None:
    records = decisions()
    fields = parameters()
    counts: dict[str, int] = {}
    for field in fields:
        counts[field.provenance] = counts.get(field.provenance, 0) + 1
    gaps = [f for f in fields if f.provenance == UNLABELLED]

    print("# Moirai: model documentation")
    print()
    print(f"Generated from the repository on {date.today().isoformat()}. Do not edit:")
    print("regenerate with `python scripts/model_documentation.py`.")
    print()
    print(
        f"{len(records)} decision records, {len(fields)} declared parameters, "
        f"{test_count()} test functions, {len(evidence())} evidence scripts."
    )
    print()

    print("## 1. Scope and intended use")
    print()
    print(SCOPE)
    print()

    print("## 2. Provenance of every declared parameter")
    print()
    order = [*(p.lower() for p in PROVENANCE), UNLABELLED]
    summary = ", ".join(f"{counts[k]} {k}" for k in order if counts.get(k))
    print(f"{summary}.")
    print()
    print("| Owner | Parameter | Default | Provenance | Declared as |")
    print("|---|---|---|---|---|")
    for field in fields:
        print(
            f"| `{field.owner}` | `{field.name}` | `{field.default}` "
            f"| {field.provenance} | {field.description} |"
        )
    print()

    print("## 3. Decisions, and why")
    print()
    print("| # | Decision | Status |")
    print("|---|---|---|")
    for record in records:
        print(f"| {record.number:03d} | {record.title} | {record.status} |")
    print()

    print("## 4. Evidence")
    print()
    print("Each script recomputes a result from the data and prints it. Every change")
    print("is compared against the previous version before it merges.")
    print()
    print("| Script | What it checks |")
    print("|---|---|")
    for name, first in evidence():
        print(f"| `{name}` | {first} |")
    print()

    print("## 5. Known gaps")
    print()
    if gaps:
        for field in gaps:
            print(f"- `{field.owner}.{field.name}` declares no provenance")
    else:
        print("Every declared parameter states where its value came from.")
    print()
    print("Gaps recorded in the decision records rather than here: the central banks'")
    print("preference weights are chosen rather than estimated; the India layer inherits")
    print("a response shape estimated on United States data; the household layer is not")
    print("matched to Indian survey data; the credit layer has no arrears channel, so a")
    print("synthetic book has no Stage 3 exposure and the 30-day trigger cannot fire.")


if __name__ == "__main__":
    main()

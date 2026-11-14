"""Model documentation, generated from the repository itself (Stage 6).

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

Writes the file itself when given a path, and prints to standard output when
not:

    python scripts/model_documentation.py docs/model_documentation.md

It writes rather than being redirected because PowerShell's `>` defaults to
UTF-16LE, whose NUL bytes make git treat the result as a binary file. The
document was committed that way from 13 to 14 November 2026: it diffed as
`Bin 46042 -> 46184 bytes`, which is unreviewable, in a document whose whole
purpose is to be read. Writing it here fixes the encoding in one place
instead of in every shell that calls it.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import sys
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


#: Modules that could not be imported, reported in the document rather than
#: passed over. Filled by `_modules`, read by `main`.
UNREADABLE: list[str] = []


def _modules() -> list[Any]:
    """Every importable module of the engine, so nothing is missed by hand."""
    package = importlib.import_module("moirai")
    loaded = [package]
    UNREADABLE.clear()
    for info in pkgutil.walk_packages(package.__path__, prefix="moirai."):
        try:
            loaded.append(importlib.import_module(info.name))
        except Exception as err:  # a module that cannot be imported is reported
            UNREADABLE.append(f"{info.name}: {type(err).__name__}: {err}")
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
    """The check scripts, with the first written line of each docstring.

    The first *non-empty* line, because a docstring may open on the line
    after its quotes. Taking the first line literally printed an empty cell
    for `check_benchmarks.py` - the silent gap this script exists to
    prevent. `test_every_evidence_script_says_what_it_checks` now fails
    rather than letting one through.
    """
    register: list[tuple[str, str]] = []
    for path in sorted(SCRIPTS.glob("check_*.py")):
        text = path.read_text(encoding="utf-8")
        match = re.match(r'\s*"""(.*?)"""', text, re.S)
        body = match.group(1) if match else ""
        first = next((line for line in body.splitlines() if line.strip()), "")
        register.append((path.name, " ".join(first.split())))
    return register


def document() -> str:
    """The whole document as text, with LF endings and a trailing newline."""
    fields = parameters()  # before anything reads UNREADABLE, which it fills
    records = decisions()
    counts: dict[str, int] = {}
    for field in fields:
        counts[field.provenance] = counts.get(field.provenance, 0) + 1
    gaps = [f for f in fields if f.provenance == UNLABELLED]

    lines: list[str] = []
    say = lines.append

    say("# Moirai: model documentation")
    say("")
    say(f"Generated from the repository on {date.today().isoformat()}. Do not edit:")
    say("regenerate with `python scripts/model_documentation.py <path>`.")
    say("")
    say(
        f"{len(records)} decision records, {len(fields)} declared parameters, "
        f"{test_count()} test functions, {len(evidence())} evidence scripts."
    )
    say("")

    for note in UNREADABLE:
        say(f"<!-- could not read {note} -->")
    if UNREADABLE:
        say("")

    say("## 1. Scope and intended use")
    say("")
    say(SCOPE)
    say("")

    say("## 2. Provenance of every declared parameter")
    say("")
    order = [*(p.lower() for p in PROVENANCE), UNLABELLED]
    summary = ", ".join(f"{counts[k]} {k}" for k in order if counts.get(k))
    say(f"{summary}.")
    say("")
    say("| Owner | Parameter | Default | Provenance | Declared as |")
    say("|---|---|---|---|---|")
    for field in fields:
        say(
            f"| `{field.owner}` | `{field.name}` | `{field.default}` "
            f"| {field.provenance} | {field.description} |"
        )
    say("")

    say("## 3. Decisions, and why")
    say("")
    say("| # | Decision | Status |")
    say("|---|---|---|")
    for record in records:
        say(f"| {record.number:03d} | {record.title} | {record.status} |")
    say("")

    say("## 4. Evidence")
    say("")
    say("Each script recomputes a result from the data and prints it. Every change")
    say("is compared against the previous version before it merges.")
    say("")
    say("| Script | What it checks |")
    say("|---|---|")
    for name, first in evidence():
        say(f"| `{name}` | {first} |")
    say("")

    say("## 5. Known gaps")
    say("")
    if gaps:
        for field in gaps:
            say(f"- `{field.owner}.{field.name}` declares no provenance")
    else:
        say("Every declared parameter states where its value came from.")
    say("")
    say("Gaps recorded in the decision records rather than here: the central banks'")
    say("preference weights are chosen rather than estimated; the India layer inherits")
    say("a response shape estimated on United States data; the household layer is not")
    say("matched to Indian survey data; the credit layer has no arrears channel, so a")
    say("synthetic book has no Stage 3 exposure and the 30-day trigger cannot fire.")

    return "\n".join(lines) + "\n"


def write(destination: Path, text: str) -> None:
    """UTF-8, no byte-order mark, LF endings, whichever shell called this."""
    destination.write_bytes(text.encode("utf-8"))


def main(argv: list[str] | None = None) -> None:
    text = document()
    arguments = sys.argv[1:] if argv is None else argv
    if arguments:
        destination = Path(arguments[0])
        write(destination, text)
        print(f"wrote {destination}: {len(text.encode('utf-8'))} bytes, utf-8, lf")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()

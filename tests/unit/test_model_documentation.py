"""Model documentation generated from the repository (Stage 6).

Written before the script existed. The point of generating it is that it
cannot drift from the code, so the tests check the reading rather than the
wording: that every decision record is found, that every parameter of the
credit engine declares where its value came from, and that a declaration is
only counted when it is actually a declaration.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import model_documentation as doc  # noqa: E402

CREDIT = "moirai.engine.credit"


def test_every_decision_record_is_found_and_parsed() -> None:
    records = doc.decisions()
    assert len(records) == len(list(doc.DECISIONS.glob("[0-9][0-9][0-9][0-9]-*.md")))
    assert [r.number for r in records] == sorted(r.number for r in records)
    assert all(r.title and not r.title.startswith("#") for r in records)


def test_the_decision_numbers_have_no_gaps_or_repeats() -> None:
    numbers = [r.number for r in doc.decisions()]
    assert len(numbers) == len(set(numbers))
    assert numbers == list(range(numbers[0], numbers[-1] + 1))


def test_the_parameter_classes_of_the_credit_engine_are_all_found() -> None:
    owners = {p.owner for p in doc.parameters() if p.module.startswith(CREDIT)}
    assert {"RepricingParameters", "DefaultParameters", "EclParameters", "BookProfile"} <= owners


def test_every_parameter_of_the_credit_engine_declares_its_provenance() -> None:
    """The control this script exists to enforce: no unexplained numbers."""
    undeclared = [
        f"{p.owner}.{p.name}"
        for p in doc.parameters()
        if p.module.startswith(CREDIT) and p.provenance == doc.UNLABELLED
    ]
    assert not undeclared, f"undeclared: {undeclared}"


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("SOURCED. The RBI's fallback.", "sourced"),
        ("ASSUMED. Doubling is the common industry convention; vary it.", "assumed"),
        ("Contractual. Repo plus a spread.", "contractual"),
        ("METADATA. A label.", "metadata"),
        ("Off to see the modelled loss on its own.", doc.UNLABELLED),
        ("", doc.UNLABELLED),
    ],
)
def test_a_declaration_counts_only_as_the_first_word(description: str, expected: str) -> None:
    assert doc._provenance(description) == expected


def test_the_evidence_register_lists_the_check_scripts() -> None:
    register = dict(doc.evidence())
    assert "check_loan_book.py" in register
    assert register["check_loan_book.py"]


def test_the_suite_counts_itself() -> None:
    assert doc.test_count() > 100

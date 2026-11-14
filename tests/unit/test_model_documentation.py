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


def test_every_evidence_script_says_what_it_checks() -> None:
    """One blank cell got through because only one script was ever checked.

    `check_benchmarks.py` opened its docstring on the line after the quotes,
    the register took the first line literally, and the generated document
    carried an empty description. A document that reports its gaps cannot
    have one of its own.
    """
    silent = [name for name, says in doc.evidence() if not says]
    assert not silent, f"no description: {silent}"


def test_a_description_is_found_whichever_line_the_docstring_opens_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both house styles read the same, and a bare docstring is still a gap."""
    monkeypatch.setattr(doc, "SCRIPTS", tmp_path)
    (tmp_path / "check_same_line.py").write_text('"""Says it here.\n\nMore.\n"""\n')
    (tmp_path / "check_next_line.py").write_text('"""\nSays it here.\n\nMore.\n"""\n')
    (tmp_path / "check_nothing.py").write_text('"""\n\n"""\n')
    register = dict(doc.evidence())
    assert register["check_same_line.py"] == "Says it here."
    assert register["check_next_line.py"] == "Says it here."
    assert register["check_nothing.py"] == ""


def test_the_document_is_written_as_utf_8_text_not_a_binary_blob(
    tmp_path: Path,
) -> None:
    """The document was committed as UTF-16 for a day and git called it binary.

    `python ... > file.md` in PowerShell writes UTF-16LE, whose NUL bytes make
    git treat the file as binary: it diffed as `Bin 46042 -> 46184 bytes`, so
    the one artefact built to be read could not be reviewed. The script writes
    the file itself now, and these are the properties that must hold.
    """
    destination = tmp_path / "model_documentation.md"
    doc.main([str(destination)])
    raw = destination.read_bytes()

    assert b"\x00" not in raw, "NUL bytes: git will treat this as binary"
    assert not raw.startswith(b"\xef\xbb\xbf"), "byte-order mark"
    assert b"\r\n" not in raw, "CRLF endings"
    assert raw.endswith(b"\n")
    assert raw.decode("utf-8").startswith("# Moirai: model documentation")


def test_printing_and_writing_give_the_same_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "written.md"
    doc.main([str(destination)])
    capsys.readouterr()
    doc.main([])
    printed = capsys.readouterr().out
    assert destination.read_text(encoding="utf-8") == printed


def test_the_document_reports_a_module_it_could_not_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable module is a gap in the document, not a silent omission."""
    monkeypatch.setattr(doc, "UNREADABLE", ["moirai.broken: ImportError: no"])
    monkeypatch.setattr(doc, "parameters", lambda: [])
    assert "could not read moirai.broken" in doc.document()


def test_the_suite_counts_itself() -> None:
    assert doc.test_count() > 100

"""
RAG document loader
====================
Parse a DOCX recovery document into structured records:
    {error, cause, recovery}

Six of the 7 docs in `data/RAG/` follow a similar structure with
"Error Message" / "Root Cause" / "Recovery Steps" headings. The seventh
(`E-Scentz.docx`) is product overview only — skip it.

Dependency: python-docx
    pip install python-docx
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict


class RecoveryRecord(TypedDict):
    error: str
    cause: str
    recovery: str
    source_doc: str        # the DOCX filename, for traceability


def load_docx(docx_path: Path) -> list[RecoveryRecord]:
    """
    Parse one DOCX into a list of RecoveryRecord dicts.

    TODO (teammate):
      1. Open the DOCX with `from docx import Document`.
      2. Walk paragraphs/tables and detect the heading pattern
         (Error Message / Root Cause / Recovery Steps). Headings vary —
         inspect the doc structure first.
      3. Group each (error, cause, recovery) trio into one record.
      4. Return list[RecoveryRecord].

    Heads up:
      - Some docs use tables, some use paragraphs.
      - Pick ONE doc to start with (e.g. Screen Printer.docx — Dion read
        it earlier and confirmed it has 19 structured error blocks).
    """
    raise NotImplementedError("TODO: implement load_docx")


def load_all_recovery_docs(rag_dir: Path) -> list[RecoveryRecord]:
    """
    Convenience wrapper: load every *.docx in `rag_dir` (skipping
    E-Scentz.docx) and concatenate the records.

    TODO (teammate):
      1. Iterate rag_dir.glob("*.docx").
      2. Skip "E-Scentz.docx".
      3. Call load_docx on each, extend the result list.
      4. Return.
    """
    raise NotImplementedError("TODO: implement load_all_recovery_docs")

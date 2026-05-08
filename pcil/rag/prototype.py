"""
RAG prototype CLI
==================
End-to-end demo: take a model-output string, retrieve the best-matching
recovery record from a DOCX, and print the suggestion.

Run from PCIL_dev/:
    python -m pcil.rag.prototype \\
        --doc "../data/RAG/Screen Printer.docx" \\
        --query "OEE dropped, vibration spike and pressure deviation"

Or against all 6 useful docs:
    python -m pcil.rag.prototype \\
        --rag-dir ../data/RAG/ \\
        --query "OEE dropped, vibration spike and pressure deviation"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pcil.rag.loader import load_all_recovery_docs, load_docx
from pcil.rag.lookup import lookup_keywords


def main():
    parser = argparse.ArgumentParser(description="RAG prototype demo.")
    parser.add_argument("--doc", default=None, help="Path to one DOCX file.")
    parser.add_argument("--rag-dir", default=None, help="Directory of DOCX files (skips E-Scentz.docx).")
    parser.add_argument("--query", required=True, help="Model output / error description string.")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if not args.doc and not args.rag_dir:
        raise SystemExit("Pass either --doc <path.docx> or --rag-dir <directory>.")

    if args.doc:
        records = load_docx(Path(args.doc))
        print(f"Loaded {len(records)} records from {args.doc}")
    else:
        records = load_all_recovery_docs(Path(args.rag_dir))
        print(f"Loaded {len(records)} records from {args.rag_dir}")

    matches = lookup_keywords(args.query, records, top_k=args.top_k)

    print(f"\nQuery: {args.query!r}")
    print(f"\nTop {len(matches)} matches:")
    for i, rec in enumerate(matches, 1):
        print(f"\n--- Match {i} (source: {rec.get('source_doc', 'unknown')}) ---")
        print(f"Error:    {rec['error']}")
        print(f"Cause:    {rec['cause']}")
        print(f"Recovery: {rec['recovery']}")


if __name__ == "__main__":
    main()

# pcil/rag/

Pipeline #3 prototype — recovery-document retrieval. Picks up where
the menu's task 4 leaves off.

## Job

The context model (Pipeline #2) outputs something like:

> "OEE dropped by 20%, 0.8 change driven by vibration spike and
> set_pressure deviation."

RAG's job: take that explanation, look up matching recovery procedures
from one of Winardi's reference DOCX files, and return something
operator-actionable like:

> "The pressure valve may be partially blocked. Recommended steps:
> (1) shut off the line, (2) check valve V-12, (3) clean and reseat."

## Source documents

`data/RAG/` (gitignored — sit alongside the repo):

| File | Useful? |
|---|---|
| `Infrared Oven.docx` | yes — Error / Root Cause / Recovery blocks |
| `Injection Molding.docx` | yes |
| `Laser Trimming.docx` | yes |
| `Laser Welding.docx` | yes |
| `Membrane Assembly.docx` | yes |
| `Screen Printer.docx` | yes |
| `E-Scentz.docx` | **skip** — product overview only, no error/recovery content |

For v1, pick **one** machine doc and prototype against it.

## Status

| File | What it does | Status |
|---|---|---|
| `loader.py` | parse a DOCX into structured (error, cause, recovery) records | **TODO** |
| `lookup.py` | keyword search across loaded records | **TODO** |
| `prototype.py` | end-to-end demo: model output string -> recovery suggestion | **TODO** |

## Recommended order

1. `loader.py` — use `python-docx` (`pip install python-docx`) to parse
   one DOCX. Identify the heading patterns ("Error Message", "Root
   Cause", "Recovery Steps") and pull the associated paragraphs into a
   list of dicts:
   ```python
   [
     {"error": "Pressure low", "cause": "Valve blocked", "recovery": "Clean valve V-12..."},
     ...
   ]
   ```
2. `lookup.py` — given a query string, return matching records. Plain
   keyword/string match is fine for v1. Vector embeddings + cosine
   similarity is a stretch goal.
3. `prototype.py` — small CLI that takes a model-output string,
   extracts a keyword (e.g. "pressure" / "vibration"), looks up
   matching records, and prints the recovery steps.

## Stretch goals

- Vector embeddings (`sentence-transformers` + cosine similarity).
- Real LLM call — wrap the retrieved records with a system prompt and
  send to an LLM for the final operator-facing summary.
- Compare results across all 6 DOCX files at once (multi-doc retrieval).

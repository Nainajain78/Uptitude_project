"""Load and parse raw contract documents (PDF/TXT) from the CUAD subset."""

import logging
import re
import sys
from collections import Counter
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_PROCESSED_DIR = BASE_DIR / "data" / "processed"

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

# Marker joined between per-page text when extracting a PDF, so normalize_text
# can reason about repeated headers/footers on a per-page basis.
_PAGE_BREAK = "\x0c"

_PAGE_NUM_RE = re.compile(r"^\s*(page\s+)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.IGNORECASE)
# A line that's just a run of dashes/underscores/asterisks/equals: PDF divider
# rules or redaction bars (e.g. "------------------------"), not real content.
_DASH_LINE_RE = re.compile(r"^[-_*=]{3,}$")
_NON_PRINTABLE_RE = re.compile(r"[^\x09\x0A\x0D\x20-\x7E]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
# Joins a word split across a PDF line-wrap (e.g. "non-\ncompensatory" ->
# "non-compensatory"). Requires a word character on both sides of the break so
# genuine list markers/dashes on their own line aren't affected.
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")

# Typographic characters common in PDF extraction, mapped to ASCII equivalents
# so stripping non-ASCII junk afterwards doesn't fuse adjacent words together.
_TYPOGRAPHIC_MAP = {
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "–": "-", "—": "-",
    "…": "...",
    "\xa0": " ",
}
_TYPOGRAPHIC_RE = re.compile("|".join(re.escape(ch) for ch in _TYPOGRAPHIC_MAP))

# Header/footer lines must be short and repeat on at least this fraction of pages.
_REPEAT_LINE_MAX_LEN = 100
_REPEAT_LINE_MIN_FRACTION = 0.5


def _resolve_dir(dir_path: str) -> Path:
    """Resolve a directory path relative to the project root if not already absolute."""
    path = Path(dir_path)
    return path if path.is_absolute() else BASE_DIR / path


def load_contract(file_path: str) -> str:
    """Load a single contract file and return its raw text content.

    Args:
        file_path: Path to the contract file (PDF or TXT).

    Returns:
        The extracted raw text of the contract. For PDFs, per-page text is
        joined with a form-feed character so downstream normalization can
        detect repeated headers/footers.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1")

    if suffix == ".pdf":
        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        return _PAGE_BREAK.join(pages)

    raise ValueError(f"Unsupported file type '{suffix}' for {file_path}")


def normalize_text(raw_text: str) -> str:
    """Clean raw extracted contract text for downstream LLM processing.

    Strips repeated page headers/footers and page numbers, removes non-ASCII
    junk artifacts left over from PDF extraction, and collapses excessive
    whitespace while preserving paragraph breaks.

    Args:
        raw_text: Raw text as returned by load_contract. May contain
            form-feed (\\x0c) page-break markers if extracted from a PDF.

    Returns:
        Cleaned, normalized contract text.
    """
    pages = raw_text.split(_PAGE_BREAK) if _PAGE_BREAK in raw_text else [raw_text]
    page_lines = [[line.strip() for line in page.splitlines()] for page in pages]

    repeated_lines: set[str] = set()
    if len(pages) > 1:
        line_counts = Counter()
        for lines in page_lines:
            line_counts.update({line for line in lines if line})
        threshold = max(2, int(len(pages) * _REPEAT_LINE_MIN_FRACTION))
        repeated_lines = {
            line
            for line, count in line_counts.items()
            if count >= threshold and len(line) < _REPEAT_LINE_MAX_LEN
        }

    cleaned_pages = []
    for lines in page_lines:
        kept = [
            line
            for line in lines
            if line not in repeated_lines
            and not _PAGE_NUM_RE.match(line)
            and not _DASH_LINE_RE.match(line)
        ]
        cleaned_pages.append("\n".join(kept))

    text = "\n\n".join(cleaned_pages)
    text = _HYPHEN_LINEBREAK_RE.sub(r"\1-\2", text)
    text = _TYPOGRAPHIC_RE.sub(lambda m: _TYPOGRAPHIC_MAP[m.group(0)], text)
    text = _NON_PRINTABLE_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MULTI_BLANK_RE.sub("\n\n", text)

    return text.strip()


def load_contract_batch(input_dir: str, n: int = 50) -> list[dict]:
    """Load, normalize, and cache a batch of contracts from a directory.

    Normalized text is cached as .txt files under data/processed/, keyed by
    the source file's stem, so re-runs skip re-parsing unchanged contracts.
    Files that fail to load or parse are logged as warnings and skipped.

    Args:
        input_dir: Directory containing raw contract files (e.g. data/raw).
        n: Maximum number of contracts to load.

    Returns:
        A list of dicts of the form {"contract_id": str, "text": str}.
    """
    input_path = _resolve_dir(input_dir)
    processed_dir = DEFAULT_PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        p for p in input_path.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )[:n]

    results = []
    for path in candidates:
        contract_id = path.stem
        cached_path = processed_dir / f"{contract_id}.txt"

        try:
            if cached_path.exists():
                text = cached_path.read_text(encoding="utf-8")
            else:
                raw_text = load_contract(str(path))
                text = normalize_text(raw_text)
                cached_path.write_text(text, encoding="utf-8")

            results.append({"contract_id": contract_id, "text": text})
        except Exception as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            continue

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    raw_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    sample_n = 2

    contracts = load_contract_batch(raw_dir, n=sample_n)

    if not contracts:
        print(f"No contracts loaded from '{raw_dir}'. Add a couple of sample "
              f".pdf/.txt files there and re-run: python src/loader.py")
    else:
        for contract in contracts:
            print(f"--- {contract['contract_id']} ({len(contract['text'])} chars) ---")
            print(contract["text"][:500])
            print()

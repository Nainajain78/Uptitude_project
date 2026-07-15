"""Clause extraction logic (termination, confidentiality, liability) using the LLM client."""

import json
import logging
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ValidationError

try:
    from src.llm_client import LLMClient, get_default_llm_client
    from src.utils import split_paragraphs
except ImportError:
    from llm_client import LLMClient, get_default_llm_client
    from utils import split_paragraphs

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_PATH = BASE_DIR / "prompts" / "clause_extraction_prompt.txt"
FEWSHOT_PROMPT_PATH = BASE_DIR / "prompts" / "clause_extraction_fewshot_prompt.txt"

# Above this word count, the full contract is not sent to the LLM. Instead, only
# paragraphs matching clause keyword stems (plus one paragraph of surrounding
# context) are sent. This trades recall for latency/cost/context-window safety:
# clauses phrased without any of these stems (e.g. an unusually worded wind-down
# provision) could be missed, but for CUAD-style commercial contracts these terms
# reliably anchor the relevant sections, and sending 100+ pages of unrelated
# boilerplate to the LLM on every call is not worth the risk of truncation.
_LONG_CONTRACT_WORD_THRESHOLD = 6000
_CLAUSE_STEMS = ("terminat", "confidential", "liab", "indemnif")

_RETRY_INSTRUCTION = (
    "\n\nYour previous response was not valid JSON. "
    "Return ONLY a valid JSON object with the exact keys requested."
)


class ClauseExtractionResult(BaseModel):
    """Structured result of extracting key clauses from a contract."""

    termination_clause: str
    confidentiality_clause: str
    liability_clause: str


@lru_cache(maxsize=2)
def _load_prompt_template(use_fewshot: bool = False) -> str:
    """Read the clause extraction prompt template from prompts/.

    Cached since a batch run calls this on every contract; the template file
    itself doesn't change mid-run.

    Args:
        use_fewshot: If True, load the few-shot variant (worked examples of
            contract excerpt -> correct JSON before the contract text) instead
            of the zero-shot base prompt.
    """
    path = FEWSHOT_PROMPT_PATH if use_fewshot else PROMPT_PATH
    return path.read_text(encoding="utf-8")


def _localize_relevant_paragraphs(contract_text: str) -> str:
    """Narrow a long contract down to paragraphs likely to contain target clauses.

    Args:
        contract_text: Full normalized contract text, paragraphs separated by
            blank lines.

    Returns:
        The matched paragraphs (each with one paragraph of surrounding context),
        in original order, joined back into a single string. Falls back to the
        full text if no paragraph matches any clause keyword stem.
    """
    paragraphs = split_paragraphs(contract_text)

    matched_indices = {
        i for i, p in enumerate(paragraphs)
        if any(stem in p.lower() for stem in _CLAUSE_STEMS)
    }
    if not matched_indices:
        return contract_text

    context_indices = set()
    for i in matched_indices:
        context_indices.update((i - 1, i, i + 1))
    context_indices = sorted(i for i in context_indices if 0 <= i < len(paragraphs))

    return "\n\n".join(paragraphs[i] for i in context_indices)


def _strip_code_fence(text: str) -> str:
    """Remove a surrounding ```/```json markdown code fence, if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_json_response(response_text: str) -> dict | None:
    """Best-effort parse of an LLM response into a JSON dict.

    Handles responses wrapped in markdown code fences or with leading/trailing
    prose around the JSON object.

    Args:
        response_text: Raw text returned by the LLM.

    Returns:
        The parsed dict, or None if no valid JSON object could be extracted.
    """
    candidate = _strip_code_fence(response_text)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(candidate[start:end + 1])
        except json.JSONDecodeError:
            return None

    return None


def extract_clauses(
    contract_text: str, llm_client: LLMClient, use_fewshot: bool = False
) -> ClauseExtractionResult:
    """Extract termination, confidentiality, and liability clauses from a contract.

    Loads prompts/clause_extraction_prompt.txt (or the few-shot variant), fills
    in the {{CONTRACT_TEXT}} placeholder (localizing to relevant paragraphs
    first if the contract is long), calls the LLM, and validates the JSON
    response. Retries once with a stricter instruction if the first response
    isn't valid JSON; falls back to an EXTRACTION_FAILED result if the retry
    also fails.

    Args:
        contract_text: Full normalized contract text.
        llm_client: An LLMClient instance (e.g. OllamaClient) to generate with.
        use_fewshot: If True, use prompts/clause_extraction_fewshot_prompt.txt
            (seeded with worked examples) instead of the zero-shot base prompt.

    Returns:
        A ClauseExtractionResult with the three extracted clauses, or
        "EXTRACTION_FAILED" in each field if extraction could not be completed.
    """
    template = _load_prompt_template(use_fewshot=use_fewshot)

    text_for_prompt = contract_text
    word_count = len(contract_text.split())
    if word_count > _LONG_CONTRACT_WORD_THRESHOLD:
        text_for_prompt = _localize_relevant_paragraphs(contract_text)
        logger.info(
            "Contract has %d words (> %d); localized to %d words of relevant paragraphs.",
            word_count, _LONG_CONTRACT_WORD_THRESHOLD, len(text_for_prompt.split()),
        )

    prompt = template.replace("{{CONTRACT_TEXT}}", text_for_prompt)

    response = llm_client.generate(prompt)
    parsed = _parse_json_response(response)

    if parsed is not None:
        try:
            return ClauseExtractionResult(**parsed)
        except ValidationError as exc:
            logger.warning("Clause extraction response did not match expected schema: %s", exc)

    # Retry once with a stricter instruction telling the model to fix its output.
    retry_response = llm_client.generate(prompt + _RETRY_INSTRUCTION)
    retry_parsed = _parse_json_response(retry_response)

    if retry_parsed is not None:
        try:
            return ClauseExtractionResult(**retry_parsed)
        except ValidationError as exc:
            logger.error("Retry response still did not match expected schema: %s", exc)
    else:
        logger.error("Retry response was still not valid JSON: %r", retry_response[:500])

    return ClauseExtractionResult(
        termination_clause="EXTRACTION_FAILED",
        confidentiality_clause="EXTRACTION_FAILED",
        liability_clause="EXTRACTION_FAILED",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        from src.loader import load_contract_batch
    except ImportError:
        from loader import load_contract_batch

    contracts = load_contract_batch("data/raw", n=1)

    if not contracts:
        print("No contracts found in data/raw. Add a sample .pdf/.txt contract and re-run.")
    else:
        sample = contracts[0]
        print(f"Extracting clauses from '{sample['contract_id']}'...")
        result = extract_clauses(sample["text"], get_default_llm_client())
        print(result.model_dump_json(indent=2))

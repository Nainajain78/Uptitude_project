"""Plain-language summary generation for contracts."""

import logging
from functools import lru_cache
from pathlib import Path

try:
    from src.llm_client import LLMClient, get_default_llm_client
    from src.utils import truncate_to_word_budget
except ImportError:
    from llm_client import LLMClient, get_default_llm_client
    from utils import truncate_to_word_budget

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_PATH = BASE_DIR / "prompts" / "summary_prompt.txt"

# summary_prompt.txt asks for a 100-150 word summary; these bounds add a buffer
# for natural LLM variance before we consider it worth a warning.
_MIN_EXPECTED_WORDS = 90
_MAX_EXPECTED_WORDS = 160

# Contracts longer than this are truncated before summarization. Unlike
# extractor.py's keyword-based localization (which targets specific named
# clauses), a summary needs holistic coverage of purpose/obligations/risk, so
# we keep the head and tail of the document via truncate_to_word_budget
# (shared with extractor.py's long-contract handling) instead of matching
# clause keywords.
_LONG_CONTRACT_WORD_THRESHOLD = 6000


def count_words(text: str) -> int:
    """Count whitespace-separated words in a piece of text."""
    return len(text.split())


@lru_cache(maxsize=1)
def _load_prompt_template() -> str:
    """Read the summary prompt template from prompts/.

    Cached since a batch run calls this on every contract; the template file
    itself doesn't change mid-run.
    """
    return PROMPT_PATH.read_text(encoding="utf-8")


def generate_summary(contract_text: str, llm_client: LLMClient) -> str:
    """Generate a plain-language summary of a contract.

    Fills prompts/summary_prompt.txt's {{CONTRACT_TEXT}} placeholder with the
    (possibly truncated) contract text and calls the LLM. Logs a warning
    without failing if the returned summary falls outside the expected
    90-160 word range.

    Args:
        contract_text: Full normalized contract text.
        llm_client: An LLMClient instance (e.g. OllamaClient) to generate with.

    Returns:
        The generated summary text, stripped of leading/trailing whitespace.
    """
    template = _load_prompt_template()

    text_for_prompt = contract_text
    if count_words(contract_text) > _LONG_CONTRACT_WORD_THRESHOLD:
        text_for_prompt = truncate_to_word_budget(contract_text, _LONG_CONTRACT_WORD_THRESHOLD)
        logger.info(
            "Contract has %d words; truncated to ~%d words before summarization.",
            count_words(contract_text), count_words(text_for_prompt),
        )

    prompt = template.replace("{{CONTRACT_TEXT}}", text_for_prompt)
    summary = llm_client.generate(prompt).strip()

    word_count = count_words(summary)
    if not (_MIN_EXPECTED_WORDS <= word_count <= _MAX_EXPECTED_WORDS):
        logger.warning(
            "Summary word count (%d) is outside the expected %d-%d word range.",
            word_count, _MIN_EXPECTED_WORDS, _MAX_EXPECTED_WORDS,
        )

    return summary


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
        print(f"Summarizing '{sample['contract_id']}'...")
        summary = generate_summary(sample["text"], get_default_llm_client())
        print(f"\nWord count: {count_words(summary)}\n")
        print(summary)

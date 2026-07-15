"""Tests for src/extractor.py's JSON parsing, localization, and retry/fallback logic.

Uses a fake LLMClient throughout so these tests never make real network calls.
"""

from src.extractor import (
    ClauseExtractionResult,
    _load_prompt_template,
    _localize_relevant_paragraphs,
    _parse_json_response,
    extract_clauses,
)
from src.llm_client import LLMClient


class FakeLLMClient(LLMClient):
    """Returns a fixed sequence of canned responses, one per call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts_seen = []

    def generate(self, prompt, system=None):
        self.prompts_seen.append(prompt)
        return self._responses.pop(0)


GOOD_JSON = (
    '{"termination_clause": "t", "confidentiality_clause": "c", "liability_clause": "l"}'
)


def test_parse_json_response_clean_json():
    assert _parse_json_response(GOOD_JSON) == {
        "termination_clause": "t", "confidentiality_clause": "c", "liability_clause": "l",
    }


def test_parse_json_response_fenced_json():
    fenced = f"```json\n{GOOD_JSON}\n```"
    assert _parse_json_response(fenced)["termination_clause"] == "t"


def test_parse_json_response_prose_wrapped_json():
    wrapped = f"Sure, here is the JSON:\n{GOOD_JSON}\nLet me know if you need anything else."
    assert _parse_json_response(wrapped)["liability_clause"] == "l"


def test_parse_json_response_invalid_returns_none():
    assert _parse_json_response("not json at all") is None


def test_localize_relevant_paragraphs_keeps_matched_plus_context():
    paragraphs = [f"Filler paragraph {i} about pricing and delivery." for i in range(10)]
    paragraphs.insert(4, "Termination. Either party may terminate upon notice.")
    text = "\n\n".join(paragraphs)

    localized = _localize_relevant_paragraphs(text)

    assert "Termination. Either party may terminate upon notice." in localized
    assert len(localized.split()) < len(text.split())


def test_localize_relevant_paragraphs_falls_back_to_full_text_when_no_match():
    text = "\n\n".join(f"Filler paragraph {i}." for i in range(5))
    assert _localize_relevant_paragraphs(text) == text


def test_extract_clauses_success_on_first_try():
    client = FakeLLMClient([GOOD_JSON])
    result = extract_clauses("some contract text", client)

    assert isinstance(result, ClauseExtractionResult)
    assert result.termination_clause == "t"
    assert len(client.prompts_seen) == 1


def test_extract_clauses_retries_then_succeeds():
    client = FakeLLMClient(["this is not json", GOOD_JSON])
    result = extract_clauses("some contract text", client)

    assert result.confidentiality_clause == "c"
    assert len(client.prompts_seen) == 2
    assert "previous response was not valid JSON" in client.prompts_seen[1]


def test_extract_clauses_falls_back_after_retry_fails():
    client = FakeLLMClient(["nope", "still nope"])
    result = extract_clauses("some contract text", client)

    assert result.termination_clause == "EXTRACTION_FAILED"
    assert result.confidentiality_clause == "EXTRACTION_FAILED"
    assert result.liability_clause == "EXTRACTION_FAILED"


def test_extract_clauses_use_fewshot_selects_the_fewshot_template():
    client = FakeLLMClient([GOOD_JSON])
    extract_clauses("some contract text", client, use_fewshot=True)

    fewshot_template = _load_prompt_template(use_fewshot=True)
    base_template = _load_prompt_template(use_fewshot=False)
    assert fewshot_template != base_template
    assert "Example 1" in fewshot_template

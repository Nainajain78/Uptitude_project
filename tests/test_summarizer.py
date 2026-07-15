"""Tests for src/summarizer.py's word counting, truncation, and range-warning logic."""

import logging

from src.llm_client import LLMClient
from src.summarizer import count_words, generate_summary


class FakeLLMClient(LLMClient):
    """Returns a fixed response and records the prompt it was called with."""

    def __init__(self, response):
        self._response = response
        self.last_prompt = None

    def generate(self, prompt, system=None):
        self.last_prompt = prompt
        return self._response


def test_count_words():
    assert count_words("one two three") == 3
    assert count_words("") == 0
    assert count_words("   spaced   out   ") == 2


def test_generate_summary_returns_stripped_text():
    client = FakeLLMClient("  a summary with padding  \n")
    result = generate_summary("some contract text", client)
    assert result == "a summary with padding"


def test_generate_summary_warns_when_word_count_out_of_range(caplog):
    short_summary = " ".join(["word"] * 40)
    client = FakeLLMClient(short_summary)

    with caplog.at_level(logging.WARNING):
        result = generate_summary("some contract text", client)

    assert result == short_summary
    assert any("outside the expected" in record.message for record in caplog.records)


def test_generate_summary_no_warning_when_in_range(caplog):
    good_summary = " ".join(["word"] * 120)
    client = FakeLLMClient(good_summary)

    with caplog.at_level(logging.WARNING):
        generate_summary("some contract text", client)

    assert not any("outside the expected" in record.message for record in caplog.records)


def test_generate_summary_truncates_long_contract():
    long_text = " ".join(f"word{i}" for i in range(10000))
    client = FakeLLMClient(" ".join(["word"] * 120))

    generate_summary(long_text, client)

    assert "{{CONTRACT_TEXT}}" not in client.last_prompt
    assert len(client.last_prompt.split()) < len(long_text.split())

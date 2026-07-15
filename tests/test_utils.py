"""Tests for src/utils.py's shared paragraph-splitting and truncation helpers."""

from src.utils import split_paragraphs, truncate_to_word_budget


def test_split_paragraphs_drops_blank_entries():
    text = "First paragraph.\n\n\n\nSecond paragraph.\n\n"
    assert split_paragraphs(text) == ["First paragraph.", "Second paragraph."]


def test_truncate_to_word_budget_noop_when_under_budget():
    text = "one two three"
    assert truncate_to_word_budget(text, max_words=10) == text


def test_truncate_to_word_budget_keeps_head_and_tail_drops_middle():
    text = "HEAD " + " ".join(f"middle{i}" for i in range(1000)) + " TAIL"

    result = truncate_to_word_budget(text, max_words=100)

    assert "HEAD" in result
    assert "TAIL" in result
    assert "middle500" not in result
    assert len(result.split()) < len(text.split())

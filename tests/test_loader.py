"""Tests for src/loader.py's text loading and normalization."""

import pytest

from src.loader import _PAGE_BREAK, load_contract, normalize_text


def test_normalize_text_strips_repeated_headers_and_page_numbers():
    page1 = "CONFIDENTIAL AGREEMENT\n\nThis Agreement is between Acme and Widget.\nPage 1 of 2"
    page2 = "CONFIDENTIAL AGREEMENT\n\n1. Termination. Either party may terminate.\nPage 2 of 2"
    raw = _PAGE_BREAK.join([page1, page2])

    result = normalize_text(raw)

    assert "CONFIDENTIAL AGREEMENT" not in result
    assert "Page 1 of 2" not in result
    assert "Page 2 of 2" not in result
    assert "This Agreement is between Acme and Widget." in result
    assert "1. Termination. Either party may terminate." in result


def test_normalize_text_preserves_paragraph_breaks():
    raw = "First paragraph.\n\n\n\nSecond paragraph."
    result = normalize_text(raw)
    assert result == "First paragraph.\n\nSecond paragraph."


def test_normalize_text_joins_hyphenated_linebreak_but_keeps_hyphen():
    raw = "Neither party shall be liable for special, non-\ncompensatory damages."
    result = normalize_text(raw)
    assert "non-compensatory" in result
    assert "non-\ncompensatory" not in result


def test_normalize_text_strips_redaction_dash_lines():
    raw = "Some clause text.\n------------------------\nMore clause text."
    result = normalize_text(raw)
    assert "------------------------" not in result
    assert "Some clause text." in result
    assert "More clause text." in result


def test_normalize_text_normalizes_typographic_chars_without_fusing_words():
    raw = "Each party shall keep confidential all\xa0‘information’ disclosed."
    result = normalize_text(raw)
    assert "allinformation" not in result
    assert "all 'information'" in result


def test_load_contract_reads_txt_file(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello contract text.", encoding="utf-8")

    result = load_contract(str(file_path))

    assert result == "Hello contract text."


def test_load_contract_unsupported_extension_raises(tmp_path):
    file_path = tmp_path / "sample.docx"
    file_path.write_text("irrelevant", encoding="utf-8")

    with pytest.raises(ValueError):
        load_contract(str(file_path))

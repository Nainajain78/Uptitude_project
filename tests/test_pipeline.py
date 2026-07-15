"""Tests for src/pipeline.py's per-contract error isolation and output writing."""

import csv
import json

from src.extractor import ClauseExtractionResult
from src.pipeline import RESULT_FIELDS, _process_contract, _write_outputs


class FakeLLMClient:
    """Minimal stand-in; not a real LLMClient since these tests monkeypatch
    extract_clauses/generate_summary directly rather than going through a
    real prompt/JSON round trip (already covered in test_extractor.py /
    test_summarizer.py)."""


def test_process_contract_success(monkeypatch):
    monkeypatch.setattr(
        "src.pipeline.extract_clauses",
        lambda text, client: ClauseExtractionResult(
            termination_clause="t", confidentiality_clause="c", liability_clause="l",
        ),
    )
    monkeypatch.setattr("src.pipeline.generate_summary", lambda text, client: "a summary")

    result = _process_contract({"contract_id": "contract_1", "text": "..."}, FakeLLMClient())

    assert result == {
        "contract_id": "contract_1",
        "summary": "a summary",
        "termination_clause": "t",
        "confidentiality_clause": "c",
        "liability_clause": "l",
        "status": "success",
    }


def test_process_contract_isolates_failure(monkeypatch):
    def boom(text, client):
        raise RuntimeError("simulated LLM failure")

    monkeypatch.setattr("src.pipeline.extract_clauses", boom)

    result = _process_contract({"contract_id": "contract_2", "text": "..."}, FakeLLMClient())

    assert result["contract_id"] == "contract_2"
    assert result["status"] == "failed: simulated LLM failure"
    assert result["summary"] == ""
    assert result["termination_clause"] == ""


def test_write_outputs_creates_valid_csv_and_json(tmp_path):
    results = [
        {
            "contract_id": "c1",
            "summary": "s1",
            "termination_clause": "multi\nline clause",
            "confidentiality_clause": "conf1",
            "liability_clause": "liab1",
            "status": "success",
        },
        {
            "contract_id": "c2",
            "summary": "",
            "termination_clause": "",
            "confidentiality_clause": "",
            "liability_clause": "",
            "status": "failed: boom",
        },
    ]
    output_path = tmp_path / "results"

    _write_outputs(results, str(output_path))

    csv_path = tmp_path / "results.csv"
    json_path = tmp_path / "results.json"
    assert csv_path.exists()
    assert json_path.exists()

    with json_path.open(encoding="utf-8") as f:
        assert json.load(f) == results

    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["termination_clause"] == "multi\nline clause"
    assert list(rows[0].keys()) == RESULT_FIELDS

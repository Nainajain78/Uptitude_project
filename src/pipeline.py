"""End-to-end orchestration: load contracts, extract clauses, summarize, and persist outputs."""

import csv
import json
import logging
from pathlib import Path

from tqdm import tqdm

try:
    from src.extractor import extract_clauses
    from src.llm_client import LLMClient, get_default_llm_client
    from src.loader import load_contract_batch
    from src.summarizer import generate_summary
except ImportError:
    from extractor import extract_clauses
    from llm_client import LLMClient, get_default_llm_client
    from loader import load_contract_batch
    from summarizer import generate_summary

logger = logging.getLogger(__name__)

RESULT_FIELDS = [
    "contract_id",
    "summary",
    "termination_clause",
    "confidentiality_clause",
    "liability_clause",
    "status",
]


def _process_contract(contract: dict, llm_client: LLMClient) -> dict:
    """Run clause extraction + summarization for a single contract.

    Args:
        contract: Dict with "contract_id" and "text" (from load_contract_batch).
        llm_client: An LLMClient instance shared across the batch.

    Returns:
        A result dict matching RESULT_FIELDS. On failure, clause/summary
        fields are empty strings and status is "failed: <error message>".
    """
    contract_id = contract["contract_id"]
    try:
        clauses = extract_clauses(contract["text"], llm_client)
        summary = generate_summary(contract["text"], llm_client)
        return {
            "contract_id": contract_id,
            "summary": summary,
            "termination_clause": clauses.termination_clause,
            "confidentiality_clause": clauses.confidentiality_clause,
            "liability_clause": clauses.liability_clause,
            "status": "success",
        }
    except Exception as exc:
        logger.error("Failed to process contract '%s': %s", contract_id, exc)
        return {
            "contract_id": contract_id,
            "summary": "",
            "termination_clause": "",
            "confidentiality_clause": "",
            "liability_clause": "",
            "status": f"failed: {exc}",
        }


def _write_outputs(results: list[dict], output_path: str) -> None:
    """Write results to both "{output_path}.csv" and "{output_path}.json".

    Args:
        results: List of per-contract result dicts (see RESULT_FIELDS).
        output_path: Output path without extension.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out.with_name(out.name + ".csv")
    json_path = out.with_name(out.name + ".json")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Wrote %d result(s) to %s and %s", len(results), csv_path, json_path)


def run_pipeline(input_dir: str, n: int, output_path: str) -> list[dict]:
    """Run the full contract clause extraction and summarization pipeline.

    Loads up to n contracts from input_dir, runs extraction + summarization on
    each (isolating failures per contract so one bad contract doesn't stop the
    batch), and writes the combined results to output_path.csv/.json.

    Args:
        input_dir: Directory containing raw CUAD contract files (e.g. data/raw).
        n: Maximum number of contracts to process.
        output_path: Output path without extension (e.g. outputs/results).

    Returns:
        The list of per-contract result dicts that were written out.
    """
    contracts = load_contract_batch(input_dir, n=n)
    logger.info("Loaded %d contract(s) from %s", len(contracts), input_dir)

    llm_client = get_default_llm_client()
    results = [
        _process_contract(contract, llm_client)
        for contract in tqdm(contracts, desc="Processing contracts")
    ]

    _write_outputs(results, output_path)
    return results

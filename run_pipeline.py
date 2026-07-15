"""CLI entry point for running the contract clause extraction pipeline."""

import argparse
import logging

from src.pipeline import run_pipeline


def main() -> None:
    """Parse CLI args and invoke the pipeline."""
    parser = argparse.ArgumentParser(
        description="Extract clauses and generate summaries for a batch of contracts."
    )
    parser.add_argument(
        "--input_dir", default="data/raw",
        help="Directory containing raw contract files (default: data/raw).",
    )
    parser.add_argument(
        "--n", type=int, default=50,
        help="Number of contracts to process (default: 50).",
    )
    parser.add_argument(
        "--output", default="outputs/results",
        help="Output path without extension; writes .csv and .json (default: outputs/results).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    run_pipeline(args.input_dir, args.n, args.output)


if __name__ == "__main__":
    main()

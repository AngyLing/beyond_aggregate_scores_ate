#!/usr/bin/env python3
"""Rebuild the article's 25 full-test LLM repeat comparisons (A17)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ate_paper.analysis.repeats import reproduce_llm_repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/local/processed")
    parser.add_argument("--prediction-root", type=Path,
                        help="Optional fresh outputs under <root>/<acl|genia|cl>/<group>/<run>/predictions.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "build/llm-repeats")
    args = parser.parse_args()
    try:
        result = reproduce_llm_repeats(ROOT, args.data_dir, args.output, args.prediction_root)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"LLM repeat reproduction failed: {error}\n")
    print(json.dumps({key: result[key] for key in ("status", "mode", "scope", "checks")}, indent=2))


if __name__ == "__main__":
    main()

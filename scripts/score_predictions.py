#!/usr/bin/env python3
"""Score exact offsets and exact offset/class pairs against local gold JSONL."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ate_paper.evaluation import score_predictions
from ate_paper.predictions import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="Prepared local gold mention JSONL, including empty-unit sentinels")
    parser.add_argument("--predictions", type=Path, required=True, help="Released .jsonl.gz or text-free .jsonl")
    parser.add_argument("--dataset")
    parser.add_argument("--output", type=Path, help="Optional new score JSON (no corpus text)")
    args = parser.parse_args()
    try:
        result = score_predictions(read_jsonl(args.gold), read_jsonl(args.predictions), args.dataset)
        serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
        print(serialized, end="")
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Scoring failed: {error}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Join text-free predictions to a user-obtained, prepared local corpus.

The output contains corpus text and should stay in data/local/ or outside Git.
No model inference is performed. The original `norm` is not reconstructed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ate_paper.predictions import CorpusIndex, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True, help="Prepared local units or mention JSONL")
    parser.add_argument("--predictions", type=Path, required=True, help="Released .jsonl.gz")
    parser.add_argument("--output", type=Path, required=True, help="New local .jsonl file (contains corpus text)")
    parser.add_argument("--dataset")
    args = parser.parse_args()
    output_created = False
    try:
        if args.output.suffix != ".jsonl":
            raise ValueError("Output path must end in .jsonl")
        index = CorpusIndex.build(read_jsonl(args.corpus), args.dataset)
        count = 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            output_created = True
            for prediction in read_jsonl(args.predictions):
                if args.dataset is not None and prediction.get("dataset") != args.dataset:
                    continue
                record = index.hydrate(prediction)
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
        print(json.dumps({"rows": count, "units_available": len(index.units),
                          "output": str(args.output), "contains_corpus_text": True}, indent=2))
    except (ValueError, OSError, KeyError) as error:
        if output_created:
            args.output.unlink(missing_ok=True)
        parser.exit(1, f"Hydration failed: {error}\n")


if __name__ == "__main__":
    main()

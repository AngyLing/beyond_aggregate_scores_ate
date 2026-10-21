#!/usr/bin/env python3
"""Rebuild local corpus splits from separately obtained, checksum-verified data."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ate_paper.prepare import prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "data/local/sources",
                        help="Directory containing acl-rd-tec-2.0/, GENIA_term_3.02/, and CL-RuTerm3/")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/local/processed",
                        help="Private local output directory; generated files contain corpus text and gold")
    parser.add_argument("--datasets", nargs="+", choices=("acl", "genia", "cl"),
                        default=["acl", "genia", "cl"])
    args = parser.parse_args()
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Each dataset must be selected once")
    try:
        summary = prepare_data(args.source_root, args.output_root, args.datasets,
                               ROOT / "data/sources.json")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Data preparation failed: {error}\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

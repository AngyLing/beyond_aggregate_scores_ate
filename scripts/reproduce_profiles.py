#!/usr/bin/env python3
"""Rebuild supervised profiles and compare all coordinates and screens to saved results."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ate_paper.analysis.profiles import reproduce_profiles
from ate_paper.analysis.io import comparison_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", "--data-dir", dest="gold_dir", type=Path,
                        default=ROOT / "data/local/processed")
    parser.add_argument("--prediction-root", type=Path, help="New run_experiments.py output root")
    parser.add_argument("--out", type=Path, default=ROOT / "build/profiles")
    args = parser.parse_args()
    try:
        result = reproduce_profiles(ROOT, args.gold_dir.resolve(), args.out, args.prediction_root)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Profile reproduction failed: {error}\n")
    print(json.dumps({"status": result["status"], "scope": result["scope"],
                      "checks": comparison_summary(result["checks"])}, indent=2))


if __name__ == "__main__":
    main()

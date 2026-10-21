#!/usr/bin/env python3
"""Reproduce fixed-output A12/A13/A15/A16 sensitivities in a new build folder."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-output", type=Path, default=ROOT / "build/gold_agreement")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/local/processed")
    parser.add_argument("--prediction-root", type=Path, help="Optional fresh experiment predictions root")
    parser.add_argument("--output", type=Path, default=ROOT / "build/sensitivity",
                        help="New output directory under this repository's build/")
    parser.add_argument("--dataset", action="append", choices=("acl", "genia", "cl"))
    parser.add_argument("--skip-bootstrap", action="store_true", help="A15 point estimates only")
    parser.add_argument("--check-reference", action="store_true", help="Compare saved-release summaries and CL tie table")
    args = parser.parse_args()
    from ate_paper.analysis.sensitivity import run
    from ate_paper.analysis.io import DATASETS, SLUGS
    datasets = [d for d in DATASETS if not args.dataset or SLUGS[d] in args.dataset]
    try:
        result = run(ROOT, args.gold_output.resolve(), args.data_dir.resolve(), args.output.resolve(),
                     datasets, args.prediction_root.resolve() if args.prediction_root else None,
                     args.skip_bootstrap, args.check_reference)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, str(error) + "\n")
    print("Sensitivity computations complete: " + str(args.output.resolve()))
    print("Reference comparison: " + result["checks_against_saved_results"]["status"])


if __name__ == "__main__":
    main()

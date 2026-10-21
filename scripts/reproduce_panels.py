#!/usr/bin/env python3
"""Recompute full-panel scores, profiles, output sets and document intervals."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/local/processed")
    parser.add_argument("--prediction-root", type=Path, help="Fresh build/experiments/<tag> outputs; otherwise use saved release")
    parser.add_argument("--out", type=Path, default=ROOT / "build/panels")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Compute point outputs without document intervals")
    args = parser.parse_args(argv)
    from ate_paper.analysis.panels import reproduce_panels
    try:
        manifest = reproduce_panels(ROOT, args.data_dir, args.out, args.prediction_root, args.skip_bootstrap)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, "Panel reproduction failed: " + str(error) + "\n")
    print(json.dumps({"status": manifest["status"], "scope": manifest["scope"]}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify text-free prediction files, their schema, hashes, and manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ate_paper.predictions import verify_release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps({"status": "PASS", **verify_release(
            args.release_root, args.manifest, require_summary=True)}, indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Verification failed: {error}\n")


if __name__ == "__main__":
    main()

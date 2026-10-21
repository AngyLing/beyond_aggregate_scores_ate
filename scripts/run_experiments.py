#!/usr/bin/env python3
"""List or plan article experiments; use --execute only for an authorized run."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ate_paper.experiments.launcher import main

if __name__ == "__main__":
    main()

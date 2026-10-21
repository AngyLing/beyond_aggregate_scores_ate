#!/usr/bin/env python3
"""Run the no witness analysis."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ate_paper.analysis.no_witness import main


if __name__ == "__main__":
    main()

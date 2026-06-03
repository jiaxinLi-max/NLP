"""Generate all figures from results JSON files.

Usage:
    python scripts/make_figures.py --results-dir results/ --output-dir report/figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.visualize.plots import (
    plot_kv_vs_ratio,
    plot_score_vs_kv,
    plot_score_vs_ratio,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--output-dir", default="report/figures")
    args = ap.parse_args()

    plot_score_vs_ratio(args.results_dir, args.output_dir)
    plot_kv_vs_ratio(args.results_dir, args.output_dir)
    plot_score_vs_kv(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()

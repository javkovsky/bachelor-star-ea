"""Analyse event-activity observables from saved MC or data events.

Usage::

    python scripts/analyze.py --input data/mc_events_det.npz \\
        --out-dir results/analysis

Produces:
* ``results/analysis/summary.png`` – 2×2 summary figure
* ``results/analysis/multiplicity_dist.csv`` – P(N) table
* ``results/analysis/fb_correlation.txt`` – FB correlation summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from star_ea.io import load_events, save_histogram_csv
from star_ea.observables import EventActivityAnalyzer
from star_ea.plotting import plot_analysis_summary, plot_fb_scatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute event-activity observables from saved events."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input .npz event file.",
    )
    parser.add_argument(
        "--out-dir", type=str, default="results/analysis",
        help="Output directory (default: results/analysis).",
    )
    parser.add_argument(
        "--eta-min", type=float, default=-1.0,
        help="Central η minimum (default: -1.0).",
    )
    parser.add_argument(
        "--eta-max", type=float, default=1.0,
        help="Central η maximum (default: 1.0).",
    )
    parser.add_argument(
        "--pt-min", type=float, default=0.15,
        help="Minimum pT cut in GeV/c (default: 0.15).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading events from {args.input} …")
    events = load_events(args.input)
    print(f"Loaded {len(events)} events.")

    analyzer = EventActivityAnalyzer(
        eta_min=args.eta_min,
        eta_max=args.eta_max,
        pt_min=args.pt_min,
    )

    print("Computing observables …")
    results = analyzer.analyse(events)

    # Print summary
    fb = results["fb_correlation"]
    nbd = results["nbd_fit"]
    print(f"\n=== Event-activity summary ===")
    print(f"  ⟨N_ch⟩           = {results['mean_multiplicity']:.3f}")
    print(f"  ⟨⟨pT⟩⟩          = {results['mean_mean_pt']:.4f} GeV/c")
    print(f"  b_corr (FB)      = {fb['b_corr']:.4f}")
    if not np.isnan(nbd.get("mu", float("nan"))):
        print(f"  NBD μ            = {nbd['mu']:.3f} ± {nbd['mu_err']:.3f}")
        print(f"  NBD k            = {nbd['k']:.3f} ± {nbd['k_err']:.3f}")
        print(f"  χ²/ndf           = {nbd['chi2_ndf']:.2f}")

    # Save summary figure
    fig_path = out_dir / "summary.png"
    plot_analysis_summary(results, save_path=str(fig_path))
    print(f"\nSaved summary figure → {fig_path}")

    # Save P(N) histogram
    n_vals, p_n = results["multiplicity_dist"]
    csv_path = out_dir / "multiplicity_dist.csv"
    save_histogram_csv(n_vals, p_n, csv_path)
    print(f"Saved multiplicity distribution → {csv_path}")

    # Save FB correlation text
    fb_path = out_dir / "fb_correlation.txt"
    with fb_path.open("w") as f:
        f.write(f"b_corr   = {fb['b_corr']:.6f}\n")
        f.write(f"mean_fwd = {fb['mean_fwd']:.4f}\n")
        f.write(f"mean_bwd = {fb['mean_bwd']:.4f}\n")
        f.write(f"cov      = {fb['cov']:.6f}\n")
    print(f"Saved FB correlation → {fb_path}")


if __name__ == "__main__":
    main()

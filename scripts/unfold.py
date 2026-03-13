"""Run MultiFold unfolding to correct for detector effects.

The script:
1. Loads paired (particle-level, detector-level) MC events.
2. Loads real data (or uses MC detector events as a stand-in).
3. Runs MultiFold to obtain particle-level unfolding weights.
4. Saves the weights and produces comparison plots.

Usage::

    python scripts/unfold.py \\
        --mc-gen   data/mc_events_gen.npz \\
        --mc-det   data/mc_events_det.npz \\
        --data     data/mc_events_det.npz \\
        --out-dir  results/unfolding \\
        --n-iter   3 --n-epochs 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from star_ea.io import load_events
from star_ea.multifold import MultiFold
from star_ea.observables import EventActivityAnalyzer
from star_ea.plotting import plot_unfolding_comparison, plot_convergence


def events_to_features(events, analyzer: EventActivityAnalyzer) -> np.ndarray:
    """Extract a 1-D feature vector (multiplicity) from each event."""
    mults = np.array([analyzer.event_multiplicity(ev) for ev in events],
                     dtype=np.float32)
    # Column vector for the classifier
    return mults.reshape(-1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MultiFold unfolding on pp collision events."
    )
    parser.add_argument("--mc-gen", required=True,
                        help="MC particle-level events (.npz).")
    parser.add_argument("--mc-det", required=True,
                        help="MC detector-level events (.npz).")
    parser.add_argument("--data", required=True,
                        help="Data / pseudo-data detector-level events (.npz).")
    parser.add_argument("--out-dir", default="results/unfolding",
                        help="Output directory.")
    parser.add_argument("--n-iter", type=int, default=3,
                        help="Number of MultiFold iterations (default: 3).")
    parser.add_argument("--n-epochs", type=int, default=30,
                        help="Training epochs per classifier (default: 30).")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_seed = args.seed
    import torch
    torch.manual_seed(torch_seed)

    print("Loading events …")
    mc_gen = load_events(args.mc_gen)
    mc_det = load_events(args.mc_det)
    data_events = load_events(args.data)
    print(f"  MC (gen):  {len(mc_gen)} events")
    print(f"  MC (det):  {len(mc_det)} events")
    print(f"  Data:      {len(data_events)} events")

    analyzer = EventActivityAnalyzer()

    # Build feature arrays
    x_data = events_to_features(data_events, analyzer)
    x_mc_det = events_to_features(mc_det, analyzer)
    x_mc_gen = events_to_features(mc_gen, analyzer)

    # Run MultiFold
    print(f"\nRunning MultiFold ({args.n_iter} iterations, "
          f"{args.n_epochs} epochs each) …")
    mf = MultiFold(
        n_iter=args.n_iter,
        n_epochs=args.n_epochs,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        lr=args.lr,
        verbose=args.verbose,
    )
    mf.fit(x_data, x_mc_det, x_mc_gen)
    weights = mf.weights
    print(f"Done.  Weight stats: mean={weights.mean():.4f}  "
          f"std={weights.std():.4f}  max={weights.max():.4f}")

    # Save weights
    weights_path = out_dir / "unfolding_weights.npy"
    np.save(str(weights_path), weights)
    print(f"Saved weights → {weights_path}")

    # Convergence plot
    if len(mf.iter_weights()) > 1:
        conv = mf.convergence()
        fig_conv, _ = plot_convergence(conv)
        fig_conv.savefig(str(out_dir / "convergence.png"), dpi=150,
                         bbox_inches="tight")
        print(f"Saved convergence plot → {out_dir}/convergence.png")

    # Comparison plot (multiplicity)
    mults_gen = x_mc_gen[:, 0]
    fig_cmp, _ = plot_unfolding_comparison(
        x_gen=x_data[:, 0],
        x_mc_gen=mults_gen,
        weights_unfolded=weights,
        bins=np.arange(-0.5, 30.5, 1),
        title="Multiplicity unfolding",
        xlabel=r"$N_\mathrm{ch}$",
        ylabel="Normalised counts",
    )
    fig_cmp.savefig(str(out_dir / "unfolding_comparison.png"), dpi=150,
                    bbox_inches="tight")
    print(f"Saved comparison plot → {out_dir}/unfolding_comparison.png")


if __name__ == "__main__":
    main()

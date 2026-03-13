"""Generate toy Monte Carlo pp collision events and save them to disk.

Usage::

    python scripts/generate_mc.py --n-events 50000 --out data/mc_events.npz
    python scripts/generate_mc.py --n-events 50000 --out data/mc_events.npz \\
        --mu 7.0 --k 1.5 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repository root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from star_ea.simulation import ToyMCGenerator, STARDetectorModel, simulate_pp_events
from star_ea.io import save_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate toy pp MC events (particle + detector level)."
    )
    parser.add_argument(
        "--n-events", type=int, default=50_000,
        help="Number of events to generate (default: 50000).",
    )
    parser.add_argument(
        "--out", type=str, default="data/mc_events",
        help="Output file base path (without .npz extension).",
    )
    parser.add_argument(
        "--mu", type=float, default=17.0,
        help="NBD mean multiplicity for |η|<2.5 (default: 17.0).",
    )
    parser.add_argument(
        "--k", type=float, default=1.5,
        help="NBD over-dispersion parameter (default: 1.5).",
    )
    parser.add_argument(
        "--eta-max", type=float, default=2.5,
        help="Generator-level |η| acceptance (default: 2.5).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Generating {args.n_events} pp events (seed={args.seed}) …")
    generator = ToyMCGenerator(
        mu=args.mu, k=args.k, eta_max=args.eta_max, seed=args.seed
    )
    detector = STARDetectorModel(seed=args.seed)
    particle_events, detector_events = simulate_pp_events(
        args.n_events, generator=generator, detector=detector
    )

    out_gen = Path(args.out + "_gen")
    out_det = Path(args.out + "_det")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    save_events(particle_events, out_gen)
    save_events(detector_events, out_det)

    print(f"Saved particle-level events → {out_gen}.npz")
    print(f"Saved detector-level events → {out_det}.npz")

    # Quick summary
    mults_gen = [ev.multiplicity(eta_min=-1.0, eta_max=1.0, pt_min=0.0)
                 for ev in particle_events]
    mults_det = [ev.multiplicity(eta_min=-1.0, eta_max=1.0, pt_min=0.15)
                 for ev in detector_events]
    import numpy as np
    print(
        f"\nGenerator-level  ⟨N_ch⟩ (|η|<1, pT>0)   = {np.mean(mults_gen):.2f}"
    )
    print(
        f"Detector-level   ⟨N_ch⟩ (|η|<1, pT>0.15) = {np.mean(mults_det):.2f}"
    )


if __name__ == "__main__":
    main()

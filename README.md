# bachelor-star-ea

Event activity observables in pp collisions at STAR, with MultiFold unfolding.

## Overview

This repository contains the code for a bachelor's thesis focused on **event
activity observables** in proton-proton (pp) collisions measured by the
[STAR experiment](https://www.star.bnl.gov/) at RHIC.  The project covers three
main areas:

1. **Toy MC simulation** – Minimum-bias pp collision events generated using a
   Negative Binomial Distribution (NBD) for charged-particle multiplicity and a
   Tsallis/modified-Hagedorn parametrisation for the pT spectrum, with a
   simplified STAR TPC detector model (acceptance, efficiency, momentum
   resolution).

2. **Event-activity analysis** – Computation of standard observables:
   * Charged-particle multiplicity distributions P(N)
   * Mean transverse momentum ⟨pT⟩ vs multiplicity
   * Forward–backward (FB) multiplicity correlation coefficient
   * Negative Binomial Distribution parameter extraction (μ, k)

3. **MultiFold unfolding** – Iterative machine-learning-based correction for
   detector effects using the
   [OmniFold/MultiFold algorithm](https://arxiv.org/abs/1911.09107)
   (Andreassen et al., PRL 124, 182001 (2020)).

## Repository structure

```
bachelor-star-ea/
├── src/
│   └── star_ea/
│       ├── __init__.py
│       ├── simulation.py   # Toy MC generator and STAR detector model
│       ├── observables.py  # Event-activity observable computations
│       ├── multifold.py    # MultiFold iterative unfolding
│       ├── plotting.py     # Physics plotting utilities
│       └── io.py           # Data I/O (NumPy .npz, CSV, optional ROOT)
├── tests/
│   ├── test_simulation.py
│   ├── test_observables.py
│   ├── test_multifold.py
│   └── test_io.py
├── scripts/
│   ├── generate_mc.py      # Generate and save MC events
│   ├── analyze.py          # Compute observables and produce plots
│   └── unfold.py           # Run MultiFold unfolding
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

Optional ROOT file support (requires ROOT to be installed separately):
```bash
pip install uproot
```

## Quick start

### 1. Generate Monte Carlo events

```bash
python scripts/generate_mc.py --n-events 50000 --out data/mc_events --seed 42
```

This produces `data/mc_events_gen.npz` (particle level) and
`data/mc_events_det.npz` (detector level).

### 2. Analyse event-activity observables

```bash
python scripts/analyze.py --input data/mc_events_det.npz --out-dir results/analysis
```

Outputs a `summary.png` figure, a `multiplicity_dist.csv` table, and a
`fb_correlation.txt` file in `results/analysis/`.

### 3. Run MultiFold unfolding

```bash
python scripts/unfold.py \
    --mc-gen  data/mc_events_gen.npz \
    --mc-det  data/mc_events_det.npz \
    --data    data/mc_events_det.npz \
    --out-dir results/unfolding \
    --n-iter  3 --n-epochs 30
```

### Using the Python API

```python
from star_ea.simulation import simulate_pp_events
from star_ea.observables import EventActivityAnalyzer
from star_ea.multifold import MultiFold
import numpy as np

# Generate paired (truth, reco) events
particle_events, detector_events = simulate_pp_events(n_events=10_000, seed=42)

# Compute event-activity observables
analyzer = EventActivityAnalyzer()
results = analyzer.analyse(detector_events)

print(f"Mean multiplicity:  {results['mean_multiplicity']:.2f}")
print(f"Mean ⟨pT⟩:          {results['mean_mean_pt']:.3f} GeV/c")
print(f"FB correlation:     {results['fb_correlation']['b_corr']:.4f}")

nbd = results["nbd_fit"]
print(f"NBD fit:  μ = {nbd['mu']:.2f} ± {nbd['mu_err']:.2f},  "
      f"k = {nbd['k']:.2f} ± {nbd['k_err']:.2f}")

# Unfold detector effects with MultiFold
mults_det = np.array([[analyzer.event_multiplicity(ev)] for ev in detector_events],
                     dtype=np.float32)
mults_gen = np.array([[analyzer.event_multiplicity(ev)] for ev in particle_events],
                     dtype=np.float32)

mf = MultiFold(n_iter=3, n_epochs=30)
mf.fit(mults_det, mults_det, mults_gen)   # data=MC det for closure test
print(f"Unfolding weights: mean={mf.weights.mean():.3f}  std={mf.weights.std():.3f}")
```

## Running tests

```bash
pytest tests/
```

## Physics background

### Negative Binomial Distribution

The charged-particle multiplicity distribution in minimum-bias pp collisions is
well described by an NBD:

P(N; μ, k) = Γ(N+k) / (Γ(k) N!) · (μ/k)^N / (1 + μ/k)^(N+k)

where μ is the mean and k controls the width.  The code extracts μ and k by
fitting to measured distributions.

### Forward–backward multiplicity correlation

The FB correlation coefficient measures long-range correlations between particles
produced in separated pseudorapidity intervals:

b_corr = (⟨n_F · n_B⟩ − ⟨n_F⟩⟨n_B⟩) / (σ_F · σ_B)

### MultiFold (OmniFold) unfolding

MultiFold iteratively reweights the MC simulation so that its detector-level
distribution matches the data, then reads off the corresponding particle-level
weights:

1. Train classifier: data vs MC (detector level) → weight w₁
2. Push w₁ to particle level; train classifier: reweighted MC gen vs uniform → weight w₂
3. Repeat using w₂ as new MC weights

After convergence, the particle-level weights provide an unfolded estimate of
the true distribution.

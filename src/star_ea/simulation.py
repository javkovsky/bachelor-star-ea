"""Toy Monte Carlo simulation of minimum-bias pp collisions.

Provides a self-contained generator that reproduces the key features of
proton-proton events seen by the STAR experiment:

* Charged-particle multiplicity drawn from a Negative Binomial Distribution
  (NBD), which describes STAR minimum-bias data well.
* Transverse-momentum (pT) spectrum following the modified Hagedorn (Tsallis)
  parametrisation used in STAR publications.
* Pseudorapidity (η) distribution flat within the generator acceptance.
* Azimuthal angle (φ) uniform in [0, 2π).
* A simple STAR detector model: TPC acceptance cuts, pT-dependent tracking
  efficiency, and Gaussian momentum resolution.
"""

from __future__ import annotations

import dataclasses
from typing import List, Tuple

import numpy as np
from numpy.random import default_rng


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Particle:
    """Four-momentum-like particle container (massless approximation).

    Parameters
    ----------
    pt:
        Transverse momentum in GeV/c.
    eta:
        Pseudorapidity.
    phi:
        Azimuthal angle in radians.
    charge:
        Electric charge (±1 for charged tracks).
    """

    pt: float
    eta: float
    phi: float
    charge: int

    # Derived kinematic helpers --------------------------------------------------

    @property
    def px(self) -> float:
        return self.pt * np.cos(self.phi)

    @property
    def py(self) -> float:
        return self.pt * np.sin(self.phi)

    @property
    def pz(self) -> float:
        return self.pt * np.sinh(self.eta)

    @property
    def p(self) -> float:
        return self.pt * np.cosh(self.eta)


@dataclasses.dataclass
class Event:
    """A single collision event.

    Parameters
    ----------
    particles:
        List of final-state particles.
    event_id:
        Optional integer event identifier.
    """

    particles: List[Particle]
    event_id: int = 0

    def __len__(self) -> int:
        return len(self.particles)

    def multiplicity(self, eta_min: float = -1.0, eta_max: float = 1.0,
                     pt_min: float = 0.15) -> int:
        """Number of charged particles within the given kinematic acceptance."""
        return sum(
            1 for p in self.particles
            if eta_min < p.eta < eta_max and p.pt >= pt_min
        )

    def particles_in(self, eta_min: float = -1.0, eta_max: float = 1.0,
                     pt_min: float = 0.15) -> List[Particle]:
        """Return particles within the given kinematic acceptance."""
        return [
            p for p in self.particles
            if eta_min < p.eta < eta_max and p.pt >= pt_min
        ]


# ---------------------------------------------------------------------------
# pT spectrum
# ---------------------------------------------------------------------------

def _tsallis_pt_sample(n: int, p0: float = 12.6, q: float = 1.111,
                       pt_min: float = 0.0, pt_max: float = 10.0,
                       rng: np.random.Generator = None,
                       seed: int | None = None) -> np.ndarray:
    """Sample pT values from the Tsallis/modified-Hagedorn distribution.

    The differential cross section is proportional to::

        pT * (1 + pT / ((q-1) * p0)) ** (-1/(q-1))

    which is equivalent to ``pT * (1 + pT / C) ** (-n)`` with
    ``C = (q-1)*p0`` and ``n = 1/(q-1)``.  Default parameters correspond
    to ``C = 1.40 GeV/c``, ``n = 9``, matching the published STAR
    charged-hadron pT spectrum in minimum-bias pp collisions at
    √s = 200 GeV.  This gives a mean pT ≈ 0.47 GeV/c (unconstrained from
    below) or ≈ 0.50 GeV/c for pT > 0.15 GeV/c.

    Sampling is done via rejection sampling.
    """
    if rng is None:
        rng = default_rng(seed)

    def pdf(pt: np.ndarray) -> np.ndarray:
        return pt * (1.0 + pt / ((q - 1.0) * p0)) ** (-1.0 / (q - 1.0))

    # Normalise by the maximum of the pdf on [pt_min, pt_max].
    # The Tsallis pdf peaks at pT = p0*(q-1) / (1/(q-1) - 1); probe up to
    # a few times that to ensure we capture the true maximum.
    peak_approx = max((q - 1.0) * p0 / (1.0 / (q - 1.0) - 1.0), pt_min)
    pts_probe = np.linspace(max(pt_min, 0.0), max(peak_approx * 5, 2.0), 500)
    pdf_max = float(np.max(pdf(pts_probe))) * 1.05

    accepted: List[np.ndarray] = []
    total_accepted = 0
    while total_accepted < n:
        batch = max(n * 4, 1000)
        pt_cand = rng.uniform(pt_min, pt_max, batch)
        u = rng.uniform(0.0, pdf_max, batch)
        mask = u < pdf(pt_cand)
        accepted.append(pt_cand[mask])
        total_accepted += int(np.sum(mask))

    return np.concatenate(accepted)[:n]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class ToyMCGenerator:
    """Minimum-bias pp collision generator using an NBD multiplicity model.

    Parameters
    ----------
    mu:
        Mean of the Negative Binomial Distribution for charged-particle
        multiplicity within |η| < 2.5, pT > 0 (generator level).
        The default value of 17 gives roughly 6–7 charged particles per
        event within |η| < 1 after detector acceptance, consistent with
        STAR minimum-bias pp data at √s = 200 GeV.
    k:
        Over-dispersion parameter of the NBD.  k → ∞ gives Poisson;
        small k gives broad distributions.  Typical value ≈ 1.5.
    eta_max:
        Half-width of the pseudorapidity interval for generated particles.
    pt_min_gen:
        Generator-level minimum pT cut (GeV/c).
    pt_max_gen:
        Generator-level maximum pT cut (GeV/c).
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        mu: float = 17.0,
        k: float = 1.5,
        eta_max: float = 2.5,
        pt_min_gen: float = 0.05,
        pt_max_gen: float = 10.0,
        seed: int | None = None,
    ) -> None:
        self.mu = mu
        self.k = k
        self.eta_max = eta_max
        self.pt_min_gen = pt_min_gen
        self.pt_max_gen = pt_max_gen
        self._rng = default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, n_events: int) -> List[Event]:
        """Generate *n_events* particle-level pp collision events.

        Returns
        -------
        events:
            List of :class:`Event` objects.
        """
        events: List[Event] = []
        for i in range(n_events):
            n_ch = self._sample_multiplicity()
            particles = self._generate_particles(n_ch)
            events.append(Event(particles=particles, event_id=i))
        return events

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_multiplicity(self) -> int:
        """Draw multiplicity from an NBD(mu, k) distribution.

        Uses the Gamma-Poisson mixture, which supports continuous k:
        N | λ ~ Poisson(λ),  λ ~ Gamma(k, mu/k).
        """
        lam = self._rng.gamma(shape=self.k, scale=self.mu / self.k)
        return max(int(self._rng.poisson(lam)), 0)

    def _generate_particles(self, n: int) -> List[Particle]:
        if n == 0:
            return []
        pt_vals = _tsallis_pt_sample(
            n, pt_min=self.pt_min_gen, pt_max=self.pt_max_gen, rng=self._rng
        )
        eta_vals = self._rng.uniform(-self.eta_max, self.eta_max, n)
        phi_vals = self._rng.uniform(0.0, 2.0 * np.pi, n)
        charges = self._rng.choice([-1, 1], size=n)
        return [
            Particle(pt=float(pt), eta=float(eta), phi=float(phi),
                     charge=int(ch))
            for pt, eta, phi, ch in zip(pt_vals, eta_vals, phi_vals, charges)
        ]


# ---------------------------------------------------------------------------
# Detector model
# ---------------------------------------------------------------------------

class STARDetectorModel:
    """Simplified STAR TPC detector model.

    Applies the following effects to a particle-level event:

    1. **Acceptance**: only particles with ``pt_min <= pT <= pt_max`` and
       ``|η| <= eta_max`` pass.
    2. **Tracking efficiency**: each accepted particle is kept with probability
       ``efficiency(pT)``, which rises from ~60 % at threshold to ~85 % for
       pT > 0.5 GeV/c.
    3. **Momentum resolution**: the measured pT is smeared by a Gaussian
       with ``σ(pT)/pT = a + b * pT``.

    Parameters
    ----------
    eta_max:
        TPC pseudorapidity acceptance (default 1.0 for STAR TPC).
    pt_min:
        Minimum reconstructed pT (GeV/c).
    pt_max:
        Maximum reconstructed pT (GeV/c).
    eff_plateau:
        Tracking efficiency for high-pT particles.
    eff_threshold:
        pT value (GeV/c) at which efficiency reaches half its plateau.
    momentum_res_a:
        Constant term of σ(pT)/pT.
    momentum_res_b:
        Linear term of σ(pT)/pT.
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        eta_max: float = 1.0,
        pt_min: float = 0.15,
        pt_max: float = 10.0,
        eff_plateau: float = 0.85,
        eff_threshold: float = 0.20,
        momentum_res_a: float = 0.005,
        momentum_res_b: float = 0.005,
        seed: int | None = None,
    ) -> None:
        self.eta_max = eta_max
        self.pt_min = pt_min
        self.pt_max = pt_max
        self.eff_plateau = eff_plateau
        self.eff_threshold = eff_threshold
        self.momentum_res_a = momentum_res_a
        self.momentum_res_b = momentum_res_b
        self._rng = default_rng(seed)

    def apply(self, event: Event) -> Event:
        """Apply detector effects to *event* and return a new :class:`Event`."""
        reco_particles: List[Particle] = []
        for p in event.particles:
            # 1. Acceptance cut
            if abs(p.eta) > self.eta_max:
                continue
            if p.pt < self.pt_min or p.pt > self.pt_max:
                continue
            # 2. Tracking efficiency
            eff = self._efficiency(p.pt)
            if self._rng.random() > eff:
                continue
            # 3. Momentum smearing
            sigma_rel = self.momentum_res_a + self.momentum_res_b * p.pt
            pt_smear = p.pt * (1.0 + self._rng.normal(0.0, sigma_rel))
            pt_smear = max(pt_smear, 1e-6)
            reco_particles.append(
                Particle(pt=pt_smear, eta=p.eta, phi=p.phi, charge=p.charge)
            )
        return Event(particles=reco_particles, event_id=event.event_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _efficiency(self, pt: float) -> float:
        """pT-dependent tracking efficiency (logistic rise)."""
        x = (pt - self.eff_threshold) / 0.05
        return self.eff_plateau / (1.0 + np.exp(-x))

    @property
    def pt_threshold_50(self) -> float:
        """pT at which efficiency is 50 % of plateau."""
        return self.eff_threshold


def simulate_pp_events(
    n_events: int,
    generator: ToyMCGenerator | None = None,
    detector: STARDetectorModel | None = None,
    seed: int | None = None,
) -> Tuple[List[Event], List[Event]]:
    """Generate paired (particle-level, detector-level) pp events.

    Parameters
    ----------
    n_events:
        Number of events to generate.
    generator:
        :class:`ToyMCGenerator` instance.  A default one is created if
        ``None``.
    detector:
        :class:`STARDetectorModel` instance.  A default one is created if
        ``None``.
    seed:
        Random seed (applied when creating default instances).

    Returns
    -------
    particle_events, detector_events:
        Two lists of equal length containing particle-level and
        detector-level :class:`Event` objects.
    """
    if generator is None:
        generator = ToyMCGenerator(seed=seed)
    if detector is None:
        detector = STARDetectorModel(seed=seed)

    particle_events = generator.generate(n_events)
    detector_events = [detector.apply(ev) for ev in particle_events]
    return particle_events, detector_events

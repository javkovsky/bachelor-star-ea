"""Event activity observables for pp collisions.

Provides functions and a high-level analyser class for computing the
standard event-activity observables studied in the STAR pp programme:

* Charged-particle multiplicity distributions P(N).
* Mean transverse momentum ⟨pT⟩ as a function of multiplicity.
* Forward–backward (FB) multiplicity correlations.
* Negative Binomial Distribution (NBD) parameter extraction.
* Two-particle pseudorapidity correlation function.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gammaln

from .simulation import Event


# ---------------------------------------------------------------------------
# Low-level observable functions
# ---------------------------------------------------------------------------

def compute_multiplicity(
    event: Event,
    eta_min: float = -1.0,
    eta_max: float = 1.0,
    pt_min: float = 0.15,
) -> int:
    """Count charged particles within the given kinematic window.

    Parameters
    ----------
    event:
        Input :class:`~star_ea.simulation.Event`.
    eta_min, eta_max:
        Pseudorapidity acceptance window.
    pt_min:
        Minimum transverse momentum cut (GeV/c).

    Returns
    -------
    int
        Charged-particle multiplicity.
    """
    return sum(
        1 for p in event.particles
        if eta_min <= p.eta <= eta_max and p.pt >= pt_min
    )


def compute_mean_pt(
    event: Event,
    eta_min: float = -1.0,
    eta_max: float = 1.0,
    pt_min: float = 0.15,
) -> float:
    """Compute the mean transverse momentum of tracks in a kinematic window.

    Returns ``float('nan')`` if there are no tracks in the window.
    """
    pts = [
        p.pt for p in event.particles
        if eta_min <= p.eta <= eta_max and p.pt >= pt_min
    ]
    return float(np.mean(pts)) if pts else float("nan")


def compute_forward_backward_correlation(
    fwd_multiplicities: np.ndarray,
    bwd_multiplicities: np.ndarray,
) -> Dict[str, float]:
    """Compute the forward–backward (FB) multiplicity correlation.

    The correlation coefficient is defined as::

        b_corr = (⟨n_F · n_B⟩ − ⟨n_F⟩⟨n_B⟩) / (σ_F · σ_B)

    which equals Pearson's r for the two multiplicity series.

    Parameters
    ----------
    fwd_multiplicities:
        Array of forward-hemisphere charged-particle counts per event.
    bwd_multiplicities:
        Array of backward-hemisphere charged-particle counts per event.

    Returns
    -------
    dict with keys:
        * ``"b_corr"`` – FB correlation coefficient.
        * ``"mean_fwd"`` – mean forward multiplicity.
        * ``"mean_bwd"`` – mean backward multiplicity.
        * ``"cov"`` – covariance ⟨n_F · n_B⟩ − ⟨n_F⟩⟨n_B⟩.
    """
    nf = np.asarray(fwd_multiplicities, dtype=float)
    nb = np.asarray(bwd_multiplicities, dtype=float)
    if len(nf) != len(nb):
        raise ValueError("fwd and bwd multiplicity arrays must have the same length.")

    mean_f = float(np.mean(nf))
    mean_b = float(np.mean(nb))
    cov = float(np.mean(nf * nb) - mean_f * mean_b)
    sigma_f = float(np.std(nf))
    sigma_b = float(np.std(nb))

    if sigma_f < 1e-12 or sigma_b < 1e-12:
        b_corr = float("nan")
    else:
        b_corr = cov / (sigma_f * sigma_b)

    return {
        "b_corr": b_corr,
        "mean_fwd": mean_f,
        "mean_bwd": mean_b,
        "cov": cov,
    }


# ---------------------------------------------------------------------------
# Multiplicity distribution & NBD fitting
# ---------------------------------------------------------------------------

def multiplicity_distribution(
    multiplicities: np.ndarray,
    n_max: Optional[int] = None,
    normalise: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the multiplicity distribution P(N).

    Parameters
    ----------
    multiplicities:
        1-D array of per-event charged-particle counts.
    n_max:
        Maximum multiplicity bin (inclusive).  Defaults to ``max(multiplicities)``.
    normalise:
        If ``True`` (default), normalise so that ∑_N P(N) = 1.

    Returns
    -------
    n_values:
        Array of multiplicity values 0, 1, …, n_max.
    probabilities:
        Corresponding probabilities (or raw counts if ``normalise=False``).
    """
    multiplicities = np.asarray(multiplicities, dtype=int)
    if n_max is None:
        n_max = int(multiplicities.max()) if len(multiplicities) > 0 else 0
    n_values = np.arange(0, n_max + 1, dtype=int)
    counts = np.bincount(multiplicities, minlength=n_max + 1)[: n_max + 1]
    if normalise and counts.sum() > 0:
        probs = counts / counts.sum()
    else:
        probs = counts.astype(float)
    return n_values, probs


def _nbd_pmf(n: np.ndarray, mu: float, k: float) -> np.ndarray:
    """Negative Binomial Distribution probability mass function.

    P(N; μ, k) = Γ(N+k) / (Γ(k) N!) · (μ/k)^N / (1 + μ/k)^(N+k)

    Uses log-space computation for numerical stability.
    """
    n = np.asarray(n, dtype=float)
    log_p = (
        gammaln(n + k)
        - gammaln(k)
        - gammaln(n + 1)
        + n * np.log(mu / k)
        - (n + k) * np.log(1.0 + mu / k)
    )
    return np.exp(np.clip(log_p, -700, 0))


def fit_negative_binomial(
    multiplicities: np.ndarray,
    p0: Tuple[float, float] = (5.0, 2.0),
) -> Dict[str, float]:
    """Fit a Negative Binomial Distribution to measured multiplicities.

    Parameters
    ----------
    multiplicities:
        1-D array of per-event multiplicity values.
    p0:
        Initial guess for (μ, k).

    Returns
    -------
    dict with keys ``"mu"``, ``"k"``, ``"mu_err"``, ``"k_err"``, and
    ``"chi2_ndf"`` (χ²/ndf of the fit).
    """
    multiplicities = np.asarray(multiplicities, dtype=int)
    n_values, probs = multiplicity_distribution(multiplicities, normalise=True)
    mask = probs > 0
    n_fit = n_values[mask]
    p_fit = probs[mask]
    sigma = np.sqrt(p_fit / len(multiplicities) + 1e-12)

    def _model(n, mu, k):
        return _nbd_pmf(n, mu, k)

    bounds = ([0.1, 0.1], [100.0, 50.0])
    try:
        popt, pcov = curve_fit(
            _model, n_fit, p_fit, p0=list(p0), sigma=sigma,
            absolute_sigma=True, bounds=bounds, maxfev=5000,
        )
        perr = np.sqrt(np.diag(pcov))
        residuals = (p_fit - _model(n_fit, *popt)) / sigma
        chi2_ndf = float(np.sum(residuals ** 2) / max(len(n_fit) - 2, 1))
        return {
            "mu": float(popt[0]),
            "k": float(popt[1]),
            "mu_err": float(perr[0]),
            "k_err": float(perr[1]),
            "chi2_ndf": chi2_ndf,
        }
    except RuntimeError:
        return {"mu": float("nan"), "k": float("nan"),
                "mu_err": float("nan"), "k_err": float("nan"),
                "chi2_ndf": float("nan")}


# ---------------------------------------------------------------------------
# Mean pT vs multiplicity
# ---------------------------------------------------------------------------

def mean_pt_vs_multiplicity(
    events: List[Event],
    eta_min: float = -1.0,
    eta_max: float = 1.0,
    pt_min: float = 0.15,
    n_max: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ⟨pT⟩ as a function of charged-particle multiplicity N.

    Parameters
    ----------
    events:
        List of :class:`~star_ea.simulation.Event` objects.
    eta_min, eta_max:
        Pseudorapidity window for both N and ⟨pT⟩.
    pt_min:
        Minimum pT cut.
    n_max:
        Maximum multiplicity bin (inclusive).

    Returns
    -------
    n_values:
        Multiplicity values.
    mean_pt:
        Mean pT for each multiplicity bin.  ``nan`` if no events in bin.
    mean_pt_err:
        Statistical uncertainty on ⟨pT⟩ (standard error of the mean).
    """
    multiplicities = np.array(
        [compute_multiplicity(ev, eta_min, eta_max, pt_min) for ev in events]
    )
    if n_max is None:
        n_max = int(multiplicities.max()) if len(multiplicities) > 0 else 0

    n_values = np.arange(0, n_max + 1)
    mean_pt_arr = np.full(n_max + 1, float("nan"))
    mean_pt_err_arr = np.full(n_max + 1, float("nan"))

    for n in n_values:
        mask = multiplicities == n
        if not np.any(mask):
            continue
        ev_n = [events[i] for i in np.where(mask)[0]]
        pts = []
        for ev in ev_n:
            pts.extend(
                p.pt for p in ev.particles
                if eta_min <= p.eta <= eta_max and p.pt >= pt_min
            )
        if pts:
            mean_pt_arr[n] = float(np.mean(pts))
            mean_pt_err_arr[n] = float(np.std(pts) / np.sqrt(len(pts)))

    return n_values, mean_pt_arr, mean_pt_err_arr


# ---------------------------------------------------------------------------
# High-level analyser
# ---------------------------------------------------------------------------

class EventActivityAnalyzer:
    """High-level analyser that computes all event-activity observables.

    Parameters
    ----------
    eta_min, eta_max:
        Pseudorapidity window for the TPC acceptance (default: |η| < 1.0).
    eta_fwd_min, eta_fwd_max:
        Forward pseudorapidity window (used for FB correlations).
    eta_bwd_min, eta_bwd_max:
        Backward pseudorapidity window (used for FB correlations).
    pt_min:
        Minimum pT cut (GeV/c).
    """

    def __init__(
        self,
        eta_min: float = -1.0,
        eta_max: float = 1.0,
        eta_fwd_min: float = 0.5,
        eta_fwd_max: float = 1.0,
        eta_bwd_min: float = -1.0,
        eta_bwd_max: float = -0.5,
        pt_min: float = 0.15,
    ) -> None:
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.eta_fwd_min = eta_fwd_min
        self.eta_fwd_max = eta_fwd_max
        self.eta_bwd_min = eta_bwd_min
        self.eta_bwd_max = eta_bwd_max
        self.pt_min = pt_min

    # ------------------------------------------------------------------
    # Per-event quantities
    # ------------------------------------------------------------------

    def event_multiplicity(self, event: Event) -> int:
        """Charged-particle multiplicity in the central TPC window."""
        return compute_multiplicity(
            event, self.eta_min, self.eta_max, self.pt_min
        )

    def event_mean_pt(self, event: Event) -> float:
        """Mean pT in the central TPC window."""
        return compute_mean_pt(
            event, self.eta_min, self.eta_max, self.pt_min
        )

    # ------------------------------------------------------------------
    # Multi-event quantities
    # ------------------------------------------------------------------

    def multiplicity_distribution(
        self, events: List[Event], normalise: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (N, P(N)) for the given event sample."""
        mults = np.array([self.event_multiplicity(ev) for ev in events])
        return multiplicity_distribution(mults, normalise=normalise)

    def mean_pt_vs_multiplicity(
        self, events: List[Event]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (N, ⟨pT⟩(N), δ⟨pT⟩(N)) for the given event sample."""
        return mean_pt_vs_multiplicity(
            events, self.eta_min, self.eta_max, self.pt_min
        )

    def forward_backward_correlation(
        self, events: List[Event]
    ) -> Dict[str, float]:
        """Compute the FB multiplicity correlation coefficient."""
        fwd = np.array([
            compute_multiplicity(ev, self.eta_fwd_min, self.eta_fwd_max,
                                 self.pt_min)
            for ev in events
        ])
        bwd = np.array([
            compute_multiplicity(ev, self.eta_bwd_min, self.eta_bwd_max,
                                 self.pt_min)
            for ev in events
        ])
        return compute_forward_backward_correlation(fwd, bwd)

    def fit_nbd(
        self, events: List[Event]
    ) -> Dict[str, float]:
        """Fit a Negative Binomial Distribution to the multiplicity distribution."""
        mults = np.array([self.event_multiplicity(ev) for ev in events])
        return fit_negative_binomial(mults)

    def analyse(self, events: List[Event]) -> Dict:
        """Run all observables and return a summary dictionary.

        Returns
        -------
        dict with keys:
            ``"multiplicity_dist"``, ``"mean_pt_vs_mult"``,
            ``"fb_correlation"``, ``"nbd_fit"``,
            ``"mean_multiplicity"``, ``"mean_mean_pt"``.
        """
        n_vals, p_n = self.multiplicity_distribution(events)
        n_vals_pt, mean_pt_arr, mean_pt_err = self.mean_pt_vs_multiplicity(events)
        fb = self.forward_backward_correlation(events)
        nbd = self.fit_nbd(events)
        mults = np.array([self.event_multiplicity(ev) for ev in events])
        mean_pt_vals = [self.event_mean_pt(ev) for ev in events]
        mean_pt_vals_finite = [v for v in mean_pt_vals if not np.isnan(v)]

        return {
            "multiplicity_dist": (n_vals, p_n),
            "mean_pt_vs_mult": (n_vals_pt, mean_pt_arr, mean_pt_err),
            "fb_correlation": fb,
            "nbd_fit": nbd,
            "mean_multiplicity": float(np.mean(mults)),
            "mean_mean_pt": (
                float(np.mean(mean_pt_vals_finite)) if mean_pt_vals_finite
                else float("nan")
            ),
        }

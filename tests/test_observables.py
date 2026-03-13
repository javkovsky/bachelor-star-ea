"""Tests for star_ea.observables module."""

import numpy as np
import pytest

from star_ea.simulation import Event, Particle, ToyMCGenerator
from star_ea.observables import (
    compute_multiplicity,
    compute_mean_pt,
    compute_forward_backward_correlation,
    multiplicity_distribution,
    fit_negative_binomial,
    mean_pt_vs_multiplicity,
    EventActivityAnalyzer,
    _nbd_pmf,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_event_with_pts(pts, etas=None):
    if etas is None:
        etas = [0.0] * len(pts)
    return Event(
        particles=[Particle(pt=pt, eta=eta, phi=0.0, charge=1)
                   for pt, eta in zip(pts, etas)],
        event_id=0,
    )


# ---------------------------------------------------------------------------
# compute_multiplicity
# ---------------------------------------------------------------------------

class TestComputeMultiplicity:
    def test_empty_event(self):
        ev = Event(particles=[], event_id=0)
        assert compute_multiplicity(ev) == 0

    def test_all_pass(self):
        ev = _make_event_with_pts([0.5, 1.0, 2.0])
        assert compute_multiplicity(ev, eta_min=-1.0, eta_max=1.0,
                                    pt_min=0.15) == 3

    def test_pt_cut(self):
        ev = _make_event_with_pts([0.05, 0.10, 0.50])
        assert compute_multiplicity(ev, pt_min=0.15) == 1

    def test_eta_cut(self):
        ev = _make_event_with_pts([1.0, 1.0, 1.0], etas=[0.0, 1.5, -1.5])
        assert compute_multiplicity(ev, eta_min=-1.0, eta_max=1.0) == 1


# ---------------------------------------------------------------------------
# compute_mean_pt
# ---------------------------------------------------------------------------

class TestComputeMeanPt:
    def test_simple_mean(self):
        ev = _make_event_with_pts([1.0, 2.0, 3.0])
        assert pytest.approx(compute_mean_pt(ev, pt_min=0.0), rel=1e-6) == 2.0

    def test_empty_returns_nan(self):
        ev = Event(particles=[], event_id=0)
        assert np.isnan(compute_mean_pt(ev))

    def test_all_below_pt_min(self):
        ev = _make_event_with_pts([0.05, 0.10])
        assert np.isnan(compute_mean_pt(ev, pt_min=0.15))


# ---------------------------------------------------------------------------
# compute_forward_backward_correlation
# ---------------------------------------------------------------------------

class TestFBCorrelation:
    def test_perfect_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_forward_backward_correlation(x, x)
        assert pytest.approx(result["b_corr"], abs=1e-9) == 1.0

    def test_perfect_anticorrelation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_forward_backward_correlation(x, -x)
        assert pytest.approx(result["b_corr"], abs=1e-9) == -1.0

    def test_uncorrelated(self):
        rng = np.random.default_rng(0)
        x = rng.integers(0, 10, 10000).astype(float)
        y = rng.integers(0, 10, 10000).astype(float)
        result = compute_forward_backward_correlation(x, y)
        assert abs(result["b_corr"]) < 0.05

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            compute_forward_backward_correlation(np.array([1, 2]), np.array([1]))

    def test_constant_arrays_nan(self):
        x = np.array([3.0, 3.0, 3.0])
        result = compute_forward_backward_correlation(x, x)
        assert np.isnan(result["b_corr"])

    def test_covariance_sign(self):
        fwd = np.array([1.0, 2.0, 3.0])
        bwd = np.array([3.0, 2.0, 1.0])
        result = compute_forward_backward_correlation(fwd, bwd)
        assert result["cov"] < 0


# ---------------------------------------------------------------------------
# multiplicity_distribution
# ---------------------------------------------------------------------------

class TestMultiplicityDistribution:
    def test_normalised_sums_to_one(self):
        mults = np.array([0, 1, 1, 2, 3, 3, 3])
        _, probs = multiplicity_distribution(mults)
        assert pytest.approx(probs.sum(), abs=1e-10) == 1.0

    def test_unnormalised_is_counts(self):
        mults = np.array([0, 1, 1, 2])
        _, counts = multiplicity_distribution(mults, normalise=False)
        assert counts[0] == 1
        assert counts[1] == 2
        assert counts[2] == 1

    def test_n_max_respected(self):
        mults = np.array([0, 1, 5])
        n_vals, probs = multiplicity_distribution(mults, n_max=3)
        assert n_vals[-1] == 3
        assert len(probs) == 4

    def test_empty_input(self):
        n_vals, probs = multiplicity_distribution(np.array([], dtype=int))
        assert len(n_vals) == 1
        assert probs[0] == 0


# ---------------------------------------------------------------------------
# _nbd_pmf
# ---------------------------------------------------------------------------

class TestNbdPmf:
    def test_normalisation(self):
        n = np.arange(0, 50)
        p = _nbd_pmf(n, mu=5.0, k=2.0)
        assert pytest.approx(p.sum(), rel=1e-3) == 1.0

    def test_nonnegative(self):
        n = np.arange(0, 30)
        p = _nbd_pmf(n, mu=3.0, k=1.0)
        assert np.all(p >= 0)

    def test_mode_location(self):
        # For NBD with mu>0, mode >= 0
        n = np.arange(0, 30)
        p = _nbd_pmf(n, mu=5.0, k=2.0)
        assert n[np.argmax(p)] >= 0


# ---------------------------------------------------------------------------
# fit_negative_binomial
# ---------------------------------------------------------------------------

class TestFitNegativeBinomial:
    def _generate_nbd_sample(self, mu, k, n=10000, seed=0):
        rng = np.random.default_rng(seed)
        lam = rng.gamma(shape=k, scale=mu / k, size=n)
        return rng.poisson(lam).astype(int)

    def test_recovers_parameters(self):
        mults = self._generate_nbd_sample(mu=6.0, k=2.0, n=20000)
        result = fit_negative_binomial(mults)
        assert pytest.approx(result["mu"], rel=0.05) == 6.0
        assert pytest.approx(result["k"], rel=0.15) == 2.0

    def test_returns_expected_keys(self):
        mults = self._generate_nbd_sample(mu=5.0, k=1.5, n=5000)
        result = fit_negative_binomial(mults)
        for key in ("mu", "k", "mu_err", "k_err", "chi2_ndf"):
            assert key in result

    def test_error_estimates_positive(self):
        mults = self._generate_nbd_sample(mu=5.0, k=1.5, n=5000)
        result = fit_negative_binomial(mults)
        assert result["mu_err"] > 0
        assert result["k_err"] > 0


# ---------------------------------------------------------------------------
# mean_pt_vs_multiplicity
# ---------------------------------------------------------------------------

class TestMeanPtVsMultiplicity:
    def test_returns_correct_shapes(self):
        gen = ToyMCGenerator(mu=5.0, k=1.5, seed=7)
        events = gen.generate(500)
        n_vals, mean_pt, mean_pt_err = mean_pt_vs_multiplicity(events)
        assert len(n_vals) == len(mean_pt) == len(mean_pt_err)

    def test_mean_pt_positive_where_defined(self):
        gen = ToyMCGenerator(mu=5.0, k=1.5, seed=8)
        events = gen.generate(500)
        _, mean_pt, _ = mean_pt_vs_multiplicity(events)
        valid = mean_pt[~np.isnan(mean_pt)]
        assert np.all(valid > 0)


# ---------------------------------------------------------------------------
# EventActivityAnalyzer
# ---------------------------------------------------------------------------

class TestEventActivityAnalyzer:
    def _events(self, n=2000, seed=10):
        gen = ToyMCGenerator(mu=6.0, k=1.5, seed=seed)
        return gen.generate(n)

    def test_analyse_returns_expected_keys(self):
        analyzer = EventActivityAnalyzer()
        events = self._events()
        result = analyzer.analyse(events)
        for key in ("multiplicity_dist", "mean_pt_vs_mult", "fb_correlation",
                    "nbd_fit", "mean_multiplicity", "mean_mean_pt"):
            assert key in result

    def test_mean_multiplicity_positive(self):
        analyzer = EventActivityAnalyzer()
        events = self._events()
        result = analyzer.analyse(events)
        assert result["mean_multiplicity"] > 0

    def test_mean_mean_pt_in_range(self):
        analyzer = EventActivityAnalyzer()
        events = self._events()
        result = analyzer.analyse(events)
        # Typical STAR pT for charged hadrons: 0.3 – 1.5 GeV/c
        assert 0.1 < result["mean_mean_pt"] < 3.0

    def test_fb_correlation_in_range(self):
        analyzer = EventActivityAnalyzer()
        events = self._events(n=3000)
        result = analyzer.analyse(events)
        fb = result["fb_correlation"]
        assert -1.0 <= fb["b_corr"] <= 1.0

    def test_multiplicity_distribution_normalised(self):
        analyzer = EventActivityAnalyzer()
        events = self._events()
        _, p_n = analyzer.multiplicity_distribution(events)
        assert pytest.approx(p_n.sum(), abs=1e-9) == 1.0

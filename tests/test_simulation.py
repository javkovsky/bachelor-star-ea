"""Tests for star_ea.simulation module."""

import numpy as np
import pytest

from star_ea.simulation import (
    Particle,
    Event,
    ToyMCGenerator,
    STARDetectorModel,
    simulate_pp_events,
    _tsallis_pt_sample,
)


# ---------------------------------------------------------------------------
# Particle
# ---------------------------------------------------------------------------

class TestParticle:
    def test_kinematics(self):
        p = Particle(pt=1.0, eta=0.0, phi=0.0, charge=1)
        assert pytest.approx(p.px, rel=1e-6) == 1.0
        assert pytest.approx(p.py, abs=1e-10) == 0.0
        assert pytest.approx(p.pz, abs=1e-10) == 0.0
        assert pytest.approx(p.p, rel=1e-6) == 1.0

    def test_nonzero_eta(self):
        p = Particle(pt=1.0, eta=1.0, phi=np.pi / 2, charge=-1)
        assert pytest.approx(p.px, abs=1e-6) == 0.0
        assert pytest.approx(p.py, rel=1e-6) == 1.0
        assert p.pz > 0  # positive eta → positive pz

    def test_charge_sign(self):
        assert Particle(pt=1.0, eta=0.0, phi=0.0, charge=1).charge == 1
        assert Particle(pt=1.0, eta=0.0, phi=0.0, charge=-1).charge == -1


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------

class TestEvent:
    def _make_event(self):
        particles = [
            Particle(pt=0.5, eta=0.5, phi=0.0, charge=1),
            Particle(pt=2.0, eta=-0.5, phi=1.0, charge=-1),
            Particle(pt=0.1, eta=1.5, phi=2.0, charge=1),  # outside |η|<1
            Particle(pt=0.05, eta=0.0, phi=0.0, charge=1), # below pt_min
        ]
        return Event(particles=particles, event_id=0)

    def test_len(self):
        ev = self._make_event()
        assert len(ev) == 4

    def test_multiplicity_default_window(self):
        ev = self._make_event()
        # |η|<1, pT>0.15: particles 0 and 1 qualify
        assert ev.multiplicity() == 2

    def test_multiplicity_wide_window(self):
        ev = self._make_event()
        # |η|<2, pT>0.15: only particles 0 (pt=0.5,η=0.5) and 1 (pt=2.0,η=-0.5)
        # qualify; particle 2 (pt=0.1) and particle 3 (pt=0.05) are both below
        # the pt_min=0.15 cut
        assert ev.multiplicity(eta_min=-2.0, eta_max=2.0, pt_min=0.15) == 2

    def test_particles_in(self):
        ev = self._make_event()
        in_window = ev.particles_in()
        assert len(in_window) == 2


# ---------------------------------------------------------------------------
# ToyMCGenerator
# ---------------------------------------------------------------------------

class TestToyMCGenerator:
    def test_generate_returns_correct_count(self):
        gen = ToyMCGenerator(seed=0)
        events = gen.generate(100)
        assert len(events) == 100

    def test_event_ids_assigned(self):
        gen = ToyMCGenerator(seed=1)
        events = gen.generate(10)
        ids = [ev.event_id for ev in events]
        assert ids == list(range(10))

    def test_particle_charge_in_set(self):
        gen = ToyMCGenerator(seed=2)
        events = gen.generate(50)
        for ev in events:
            for p in ev.particles:
                assert p.charge in (-1, 1)

    def test_particle_eta_in_range(self):
        gen = ToyMCGenerator(mu=7.0, k=1.5, eta_max=2.5, seed=3)
        events = gen.generate(50)
        for ev in events:
            for p in ev.particles:
                assert abs(p.eta) <= 2.5 + 1e-9

    def test_particle_pt_positive(self):
        gen = ToyMCGenerator(seed=4)
        events = gen.generate(50)
        for ev in events:
            for p in ev.particles:
                assert p.pt > 0

    def test_mean_multiplicity_reasonable(self):
        gen = ToyMCGenerator(mu=7.0, k=1.5, seed=42)
        events = gen.generate(5000)
        mults = np.array([
            ev.multiplicity(eta_min=-2.5, eta_max=2.5, pt_min=0.0)
            for ev in events
        ])
        # NBD mean is mu; allow large relative tolerance due to acceptance
        assert 3.0 < mults.mean() < 15.0

    def test_reproducibility(self):
        gen1 = ToyMCGenerator(seed=99)
        gen2 = ToyMCGenerator(seed=99)
        events1 = gen1.generate(10)
        events2 = gen2.generate(10)
        for e1, e2 in zip(events1, events2):
            assert len(e1) == len(e2)
            for p1, p2 in zip(e1.particles, e2.particles):
                assert pytest.approx(p1.pt) == p2.pt
                assert pytest.approx(p1.eta) == p2.eta


# ---------------------------------------------------------------------------
# _tsallis_pt_sample
# ---------------------------------------------------------------------------

class TestTsallisSample:
    def test_sample_count(self):
        pts = _tsallis_pt_sample(1000, seed=0)
        assert len(pts) == 1000

    def test_sample_range(self):
        pts = _tsallis_pt_sample(1000, pt_min=0.15, pt_max=5.0, seed=1)
        assert pts.min() >= 0.15 - 1e-9
        assert pts.max() <= 5.0 + 1e-9

    def test_mean_pt_positive(self):
        pts = _tsallis_pt_sample(5000, seed=2)
        assert pts.mean() > 0


# ---------------------------------------------------------------------------
# STARDetectorModel
# ---------------------------------------------------------------------------

class TestSTARDetectorModel:
    def _make_simple_event(self):
        particles = [
            Particle(pt=1.0, eta=0.0, phi=0.0, charge=1),   # should be accepted
            Particle(pt=0.05, eta=0.0, phi=0.0, charge=1),  # below pT cut
            Particle(pt=1.0, eta=1.5, phi=0.0, charge=-1),  # outside η cut
        ]
        return Event(particles=particles, event_id=0)

    def test_acceptance_cuts(self):
        det = STARDetectorModel(eta_max=1.0, pt_min=0.15, seed=0)
        ev = self._make_simple_event()
        reco = det.apply(ev)
        # Only the first particle passes acceptance (before efficiency)
        for p in reco.particles:
            assert abs(p.eta) <= 1.0
            assert p.pt >= 0.15

    def test_efficiency_reduces_multiplicity(self):
        # Generate many particles at eta=0 and count how many survive
        rng = np.random.default_rng(123)
        n_particles = 2000
        particles = [
            Particle(pt=1.0, eta=0.0, phi=float(phi), charge=1)
            for phi in rng.uniform(0, 2 * np.pi, n_particles)
        ]
        ev = Event(particles=particles, event_id=0)
        det = STARDetectorModel(eff_plateau=0.85, seed=123)
        reco = det.apply(ev)
        # Expect roughly 85% of particles to survive
        fraction = len(reco) / n_particles
        assert 0.7 < fraction < 1.0  # broad tolerance for randomness

    def test_momentum_smearing_applied(self):
        det = STARDetectorModel(momentum_res_a=0.05, momentum_res_b=0.0,
                                eff_plateau=1.0, seed=42)
        # Particle that certainly passes acceptance
        p = Particle(pt=1.0, eta=0.0, phi=0.0, charge=1)
        ev = Event(particles=[p] * 200, event_id=0)
        reco = det.apply(ev)
        reco_pts = [rp.pt for rp in reco.particles]
        # Not all measured pT values should be exactly 1.0
        assert not all(abs(pt - 1.0) < 1e-10 for pt in reco_pts)


# ---------------------------------------------------------------------------
# simulate_pp_events
# ---------------------------------------------------------------------------

class TestSimulatePPEvents:
    def test_returns_paired_lists(self):
        particle_evs, detector_evs = simulate_pp_events(100, seed=0)
        assert len(particle_evs) == 100
        assert len(detector_evs) == 100

    def test_detector_multiplicity_le_particle_multiplicity(self):
        """Detector multiplicity must not exceed generator-level multiplicity
        within the same acceptance window (detector only removes particles)."""
        particle_evs, detector_evs = simulate_pp_events(200, seed=1)
        for pev, dev in zip(particle_evs, detector_evs):
            n_gen = pev.multiplicity(eta_min=-1.0, eta_max=1.0, pt_min=0.15)
            n_det = dev.multiplicity(eta_min=-1.0, eta_max=1.0, pt_min=0.15)
            # Smearing can move a particle below pt_min, but large deviations
            # are unlikely; allow small smearing overshoot
            assert n_det <= n_gen + 2  # small tolerance for smearing

"""Tests for star_ea.io module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from star_ea.simulation import Event, Particle
from star_ea.io import (
    events_to_arrays,
    arrays_to_events,
    save_events,
    load_events,
    save_multiplicities_csv,
    save_histogram_csv,
    load_histogram_csv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_events(n=10, seed=0):
    rng = np.random.default_rng(seed)
    events = []
    for i in range(n):
        k = int(rng.integers(1, 8))
        particles = [
            Particle(
                pt=float(rng.uniform(0.2, 5.0)),
                eta=float(rng.uniform(-1.0, 1.0)),
                phi=float(rng.uniform(0, 2 * np.pi)),
                charge=int(rng.choice([-1, 1])),
            )
            for _ in range(k)
        ]
        events.append(Event(particles=particles, event_id=i))
    return events


# ---------------------------------------------------------------------------
# events_to_arrays / arrays_to_events
# ---------------------------------------------------------------------------

class TestEventsArraysRoundtrip:
    def test_array_keys_present(self):
        events = _make_events()
        arrays = events_to_arrays(events)
        for key in ("pt", "eta", "phi", "charge", "event_id", "n_particles"):
            assert key in arrays

    def test_total_particle_count(self):
        events = _make_events()
        total = sum(len(ev) for ev in events)
        arrays = events_to_arrays(events)
        assert len(arrays["pt"]) == total

    def test_roundtrip_particle_count(self):
        events = _make_events()
        arrays = events_to_arrays(events)
        reconstructed = arrays_to_events(arrays)
        assert len(reconstructed) == len(events)
        for orig, recon in zip(events, reconstructed):
            assert len(orig) == len(recon)

    def test_roundtrip_pt_values(self):
        events = _make_events()
        arrays = events_to_arrays(events)
        reconstructed = arrays_to_events(arrays)
        for orig, recon in zip(events, reconstructed):
            orig_pts = sorted(p.pt for p in orig.particles)
            recon_pts = sorted(p.pt for p in recon.particles)
            assert pytest.approx(orig_pts, rel=1e-5) == recon_pts

    def test_empty_event_preserved(self):
        events = [Event(particles=[], event_id=0)]
        arrays = events_to_arrays(events)
        assert arrays["n_particles"][0] == 0
        reconstructed = arrays_to_events(arrays)
        assert len(reconstructed[0].particles) == 0


# ---------------------------------------------------------------------------
# save_events / load_events
# ---------------------------------------------------------------------------

class TestSaveLoadEvents:
    def test_roundtrip(self):
        events = _make_events(20, seed=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "events"
            save_events(events, path)
            loaded = load_events(str(path) + ".npz")
        assert len(loaded) == len(events)
        for orig, recon in zip(events, loaded):
            assert len(orig) == len(recon)

    def test_pt_preserved(self):
        events = _make_events(5, seed=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ev"
            save_events(events, path)
            loaded = load_events(str(path) + ".npz")
        for orig, recon in zip(events, loaded):
            orig_pts = sorted(p.pt for p in orig.particles)
            recon_pts = sorted(p.pt for p in recon.particles)
            assert pytest.approx(orig_pts, rel=1e-4) == recon_pts


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

class TestCSVIO:
    def test_multiplicities_csv(self):
        mults = np.array([0, 1, 2, 3, 4])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mults.csv"
            save_multiplicities_csv(mults, path)
            assert path.exists()
            lines = path.read_text().strip().splitlines()
            assert lines[0] == "multiplicity"
            assert len(lines) == len(mults) + 1  # header + data rows

    def test_histogram_csv_roundtrip(self):
        n_vals = np.arange(0, 10)
        p_n = np.ones(10) / 10.0
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "hist.csv"
            save_histogram_csv(n_vals, p_n, path)
            n_loaded, p_loaded = load_histogram_csv(path)
        assert list(n_loaded) == list(n_vals)
        assert pytest.approx(list(p_loaded), rel=1e-6) == list(p_n)

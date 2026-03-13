"""Data I/O utilities.

Provides lightweight helpers for:

* Saving and loading event data in NumPy's ``.npz`` format (no ROOT required).
* Exporting multiplicity and pT arrays to CSV for further analysis.
* Optional ROOT/uproot support (enabled automatically when ``uproot`` is
  installed).
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .simulation import Event, Particle


# ---------------------------------------------------------------------------
# NumPy .npz I/O
# ---------------------------------------------------------------------------

def events_to_arrays(events: List[Event]) -> Dict[str, np.ndarray]:
    """Convert a list of events to flat NumPy arrays.

    Returns a dict with keys:

    * ``"pt"``     – pT values of all particles in all events (GeV/c).
    * ``"eta"``    – η values.
    * ``"phi"``    – φ values (radians).
    * ``"charge"`` – electric charge (±1).
    * ``"event_id"`` – event identifier, broadcast to all particles.
    * ``"n_particles"`` – number of particles per event.
    """
    pts, etas, phis, charges, event_ids = [], [], [], [], []
    n_particles = []
    for ev in events:
        n = len(ev.particles)
        n_particles.append(n)
        for p in ev.particles:
            pts.append(p.pt)
            etas.append(p.eta)
            phis.append(p.phi)
            charges.append(p.charge)
            event_ids.append(ev.event_id)
    return {
        "pt": np.array(pts, dtype=np.float32),
        "eta": np.array(etas, dtype=np.float32),
        "phi": np.array(phis, dtype=np.float32),
        "charge": np.array(charges, dtype=np.int8),
        "event_id": np.array(event_ids, dtype=np.int32),
        "n_particles": np.array(n_particles, dtype=np.int32),
    }


def arrays_to_events(arrays: Dict[str, np.ndarray]) -> List[Event]:
    """Reconstruct a list of :class:`~star_ea.simulation.Event` objects from
    flat NumPy arrays (inverse of :func:`events_to_arrays`).
    """
    # n_particles tracks the true event count (including empty events)
    n_particles = arrays.get("n_particles")
    event_ids_per_particle = arrays["event_id"]

    if n_particles is not None and len(n_particles) > 0:
        # Reconstruct using the per-event particle counts so that empty events
        # (with no particles in the flat arrays) are preserved.
        n_events = len(n_particles)
        # Determine event ids: use sequential integers (0..n_events-1) since
        # empty events have no entries in the per-particle arrays.
        events: List[Event] = []
        particle_idx = 0
        pts = arrays["pt"]
        etas = arrays["eta"]
        phis = arrays["phi"]
        charges = arrays["charge"]

        # Build a mapping from event_id to particles using the flat arrays
        from collections import defaultdict
        id_to_particles: dict = defaultdict(list)
        for i in range(len(pts)):
            eid = int(event_ids_per_particle[i])
            id_to_particles[eid].append(
                Particle(pt=float(pts[i]), eta=float(etas[i]),
                         phi=float(phis[i]), charge=int(charges[i]))
            )

        # Reconstruct events in the order given by n_particles.
        # When the per-particle event_id array is empty (all events are
        # empty), the unique IDs in id_to_particles will be absent.
        unique_ids_sorted = sorted(set(int(e) for e in event_ids_per_particle))
        # If all events are empty, n_particles has entries but there are no
        # particle rows, so we fall back to sequential IDs.
        if len(unique_ids_sorted) == 0:
            for ev_idx in range(n_events):
                events.append(Event(particles=[], event_id=ev_idx))
        else:
            # Use unique_ids_sorted; empty events are those missing from
            # id_to_particles.
            for ev_id in unique_ids_sorted:
                events.append(
                    Event(particles=id_to_particles.get(ev_id, []),
                          event_id=ev_id)
                )
            # If there are more events in n_particles than unique ids, the
            # trailing ones are empty events.
            n_seen = len(unique_ids_sorted)
            for ev_idx in range(n_seen, n_events):
                events.append(Event(particles=[], event_id=ev_idx))
        return events

    # Fallback: reconstruct from unique event IDs in the particle arrays
    unique_ids = np.unique(event_ids_per_particle)
    events_out: List[Event] = []
    pts = arrays["pt"]
    etas = arrays["eta"]
    phis = arrays["phi"]
    charges = arrays["charge"]
    for eid in unique_ids:
        mask = event_ids_per_particle == eid
        particles = [
            Particle(pt=float(pt), eta=float(eta), phi=float(phi),
                     charge=int(ch))
            for pt, eta, phi, ch in zip(pts[mask], etas[mask],
                                        phis[mask], charges[mask])
        ]
        events_out.append(Event(particles=particles, event_id=int(eid)))
    return events_out


def save_events(events: List[Event], path: str | Path) -> None:
    """Save events to a ``.npz`` file.

    Parameters
    ----------
    events:
        List of :class:`~star_ea.simulation.Event` objects.
    path:
        Output file path (the ``.npz`` extension is added automatically
        if absent).
    """
    arrays = events_to_arrays(events)
    np.savez_compressed(str(path), **arrays)


def load_events(path: str | Path) -> List[Event]:
    """Load events from a ``.npz`` file created by :func:`save_events`.

    Parameters
    ----------
    path:
        Path to the ``.npz`` file.

    Returns
    -------
    list of :class:`~star_ea.simulation.Event`.
    """
    data = np.load(str(path))
    return arrays_to_events(dict(data))


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_multiplicities_csv(
    multiplicities: np.ndarray,
    path: str | Path,
    header: str = "multiplicity",
) -> None:
    """Write a 1-D multiplicity array to a CSV file (one value per row)."""
    path = Path(path)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([header])
        for n in multiplicities:
            writer.writerow([int(n)])


def save_histogram_csv(
    n_values: np.ndarray,
    probabilities: np.ndarray,
    path: str | Path,
) -> None:
    """Write a multiplicity histogram (N, P(N)) to a CSV file."""
    path = Path(path)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "P(N)"])
        for n, p in zip(n_values, probabilities):
            writer.writerow([int(n), float(p)])


def load_histogram_csv(
    path: str | Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a histogram written by :func:`save_histogram_csv`.

    Returns
    -------
    n_values, probabilities
    """
    n_vals, probs = [], []
    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_vals.append(int(row["N"]))
            probs.append(float(row["P(N)"]))
    return np.array(n_vals, dtype=int), np.array(probs, dtype=float)


# ---------------------------------------------------------------------------
# Optional ROOT / uproot support
# ---------------------------------------------------------------------------

def _uproot_available() -> bool:
    try:
        import uproot  # noqa: F401
        return True
    except ImportError:
        return False


def load_root_histogram(
    path: str | Path,
    hist_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a 1-D TH1 histogram from a ROOT file using *uproot*.

    Parameters
    ----------
    path:
        Path to the ROOT file.
    hist_name:
        Name of the histogram object inside the ROOT file.

    Returns
    -------
    values:
        Bin values (content).
    edges:
        Bin edges (length = len(values) + 1).

    Raises
    ------
    ImportError
        If ``uproot`` is not installed.
    """
    if not _uproot_available():
        raise ImportError(
            "uproot is required to read ROOT files.  "
            "Install it with:  pip install uproot"
        )
    import uproot  # noqa: F811

    with uproot.open(str(path)) as f:
        hist = f[hist_name]
        values, edges = hist.to_numpy()
    return values, edges


def load_root_tree(
    path: str | Path,
    tree_name: str,
    branches: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Load branches from a ROOT TTree as NumPy arrays.

    Parameters
    ----------
    path:
        Path to the ROOT file.
    tree_name:
        Name of the TTree.
    branches:
        List of branch names to load.  ``None`` loads all branches.

    Returns
    -------
    dict mapping branch name → NumPy array.

    Raises
    ------
    ImportError
        If ``uproot`` is not installed.
    """
    if not _uproot_available():
        raise ImportError(
            "uproot is required to read ROOT files.  "
            "Install it with:  pip install uproot"
        )
    import uproot  # noqa: F811

    with uproot.open(str(path)) as f:
        tree = f[tree_name]
        if branches is None:
            arrays = tree.arrays(library="np")
        else:
            arrays = tree.arrays(branches, library="np")
    return dict(arrays)

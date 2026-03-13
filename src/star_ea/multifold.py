"""MultiFold: iterative machine-learning-based unfolding of detector effects.

Implements the MultiFold (OmniFold) algorithm introduced in:

    Andreassen, Komiske, Metodiev, Thaler, Zurek,
    "OmniFold: A Method to Simultaneously Unfold All Observables",
    Phys. Rev. Lett. 124, 182001 (2020), arXiv:1911.09107.

The algorithm iteratively reweights a Monte Carlo simulation so that its
detector-level distribution matches the observed data, then reads off the
corresponding particle-level (truth) weights.

Usage example::

    import numpy as np
    from star_ea.multifold import MultiFold

    # x_data:   (n_data,  n_features) array of detector-level observables
    # x_mc_det: (n_mc,    n_features) array of MC detector-level observables
    # x_mc_gen: (n_mc,    n_features) array of MC particle-level observables
    # (each MC row is a paired detector/particle event)

    mf = MultiFold(n_iter=3, n_epochs=10)
    mf.fit(x_data, x_mc_det, x_mc_gen)
    weights = mf.weights   # shape (n_mc,) – particle-level event weights
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Neural network classifier
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    """Fully connected network for binary classification.

    Architecture: input → [hidden_size → ReLU] * n_layers → 1 → Sigmoid.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_size
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_classifier(
    x_0: np.ndarray,
    x_1: np.ndarray,
    w_0: Optional[np.ndarray],
    w_1: Optional[np.ndarray],
    *,
    hidden_size: int,
    n_layers: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    patience: int,
    dropout: float,
) -> _MLP:
    """Train a binary classifier to distinguish class-0 from class-1 samples.

    Parameters
    ----------
    x_0, x_1:
        Feature arrays for the two classes (shape (n_i, d)).
    w_0, w_1:
        Per-sample weights.  ``None`` means uniform weights.
    **kwargs:
        Training hyper-parameters.

    Returns
    -------
    Trained :class:`_MLP` in eval mode.
    """
    n0, n1 = len(x_0), len(x_1)
    input_dim = x_0.shape[1]

    # Labels: 0 for class 0, 1 for class 1
    x_all = np.vstack([x_0, x_1]).astype(np.float32)
    y_all = np.concatenate([np.zeros(n0), np.ones(n1)]).astype(np.float32)

    if w_0 is None:
        w_0 = np.ones(n0, dtype=np.float32)
    if w_1 is None:
        w_1 = np.ones(n1, dtype=np.float32)

    # Normalise weights so that each class has the same total weight
    w_0_norm = w_0 / (w_0.sum() + 1e-12) * n0
    w_1_norm = w_1 / (w_1.sum() + 1e-12) * n1
    w_all = np.concatenate([w_0_norm, w_1_norm]).astype(np.float32)

    x_t = torch.from_numpy(x_all).to(device)
    y_t = torch.from_numpy(y_all).to(device)
    w_t = torch.from_numpy(w_all).to(device)

    dataset = TensorDataset(x_t, y_t, w_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = _MLP(input_dim, hidden_size=hidden_size, n_layers=n_layers,
                 dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction="none")

    best_loss = float("inf")
    patience_counter = 0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for xb, yb, wb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = (criterion(pred, yb) * wb).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(loader)

        # Early stopping
        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def _predict_weights(
    model: _MLP,
    x: np.ndarray,
    device: torch.device,
    clip: float = 10.0,
) -> np.ndarray:
    """Convert classifier output probabilities to likelihood-ratio weights.

    w(x) = f(x) / (1 - f(x))

    where f(x) is the probability that sample x belongs to class 1.
    Clipped to [1/clip, clip] for numerical stability.
    """
    x_t = torch.from_numpy(x.astype(np.float32)).to(device)
    with torch.no_grad():
        p = model(x_t).cpu().numpy()
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    w = p / (1.0 - p)
    return np.clip(w, 1.0 / clip, clip)


# ---------------------------------------------------------------------------
# Feature normalisation helper
# ---------------------------------------------------------------------------

class _FeatureScaler:
    """Zero-mean, unit-variance normalisation fitted on training data."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "_FeatureScaler":
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None, "Call fit() first."
        return (x - self.mean_) / self.std_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


# ---------------------------------------------------------------------------
# MultiFold
# ---------------------------------------------------------------------------

class MultiFold:
    """MultiFold (OmniFold) iterative unfolding.

    Parameters
    ----------
    n_iter:
        Number of MultiFold iterations.
    n_epochs:
        Maximum number of training epochs per classifier.
    hidden_size:
        Hidden layer width of the neural network.
    n_layers:
        Number of hidden layers.
    batch_size:
        Mini-batch size.
    lr:
        Adam learning rate.
    patience:
        Early-stopping patience (epochs without improvement).
    weight_clip:
        Maximum allowed weight ratio ``f / (1-f)``.
    dropout:
        Dropout probability in the classifier.
    device:
        ``"cpu"`` or ``"cuda"``.  Auto-detected if ``None``.
    verbose:
        If ``True``, print per-iteration loss summaries.
    """

    def __init__(
        self,
        n_iter: int = 3,
        n_epochs: int = 50,
        hidden_size: int = 64,
        n_layers: int = 2,
        batch_size: int = 512,
        lr: float = 1e-3,
        patience: int = 10,
        weight_clip: float = 10.0,
        dropout: float = 0.0,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.n_iter = n_iter
        self.n_epochs = n_epochs
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.weight_clip = weight_clip
        self.dropout = dropout
        self.verbose = verbose

        if device is None:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self._device = torch.device(device)

        # Results filled by fit()
        self.weights_: Optional[np.ndarray] = None
        self._iter_weights_: List[np.ndarray] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        x_data: np.ndarray,
        x_mc_det: np.ndarray,
        x_mc_gen: np.ndarray,
    ) -> "MultiFold":
        """Run MultiFold unfolding.

        Parameters
        ----------
        x_data:
            Detector-level observables from real data.
            Shape ``(n_data, n_features)``.
        x_mc_det:
            Detector-level observables from MC simulation.
            Shape ``(n_mc, n_features)``.
        x_mc_gen:
            Particle-level observables from MC simulation (paired with
            ``x_mc_det`` – row *i* corresponds to the same event).
            Shape ``(n_mc, n_features_gen)``.

        Returns
        -------
        self
        """
        x_data = np.asarray(x_data, dtype=np.float32)
        x_mc_det = np.asarray(x_mc_det, dtype=np.float32)
        x_mc_gen = np.asarray(x_mc_gen, dtype=np.float32)
        # Ensure 2-D column layout: (n,) → (n, 1)
        if x_data.ndim == 1:
            x_data = x_data.reshape(-1, 1)
        if x_mc_det.ndim == 1:
            x_mc_det = x_mc_det.reshape(-1, 1)
        if x_mc_gen.ndim == 1:
            x_mc_gen = x_mc_gen.reshape(-1, 1)

        if x_mc_det.shape[0] != x_mc_gen.shape[0]:
            raise ValueError(
                "x_mc_det and x_mc_gen must have the same number of rows "
                f"(got {x_mc_det.shape[0]} and {x_mc_gen.shape[0]})."
            )

        # Normalise features independently for detector and generator spaces
        scaler_det = _FeatureScaler().fit(
            np.vstack([x_data, x_mc_det])
        )
        scaler_gen = _FeatureScaler().fit(x_mc_gen)

        xd_norm = scaler_det.transform(x_data)
        xm_det_norm = scaler_det.transform(x_mc_det)
        xm_gen_norm = scaler_gen.transform(x_mc_gen)

        n_mc = x_mc_det.shape[0]
        omega = np.ones(n_mc, dtype=np.float32)  # particle-level weights

        self._iter_weights_ = []

        for it in range(self.n_iter):
            if self.verbose:
                print(f"[MultiFold] Iteration {it + 1}/{self.n_iter}")

            # Step 1: Reweight MC detector to match data
            clf1 = _train_classifier(
                x_0=xd_norm,           # class 0: data
                x_1=xm_det_norm,       # class 1: MC detector (current weights)
                w_0=None,
                w_1=omega,
                hidden_size=self.hidden_size,
                n_layers=self.n_layers,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                device=self._device,
                patience=self.patience,
                dropout=self.dropout,
            )
            # Detector-level weight: how much more data-like than MC-like
            w_det = _predict_weights(clf1, xm_det_norm, self._device,
                                     clip=self.weight_clip)
            # Push weights to particle level
            omega_pushed = omega * w_det

            # Step 2: Reweight unweighted MC gen to match pushed weights
            clf2 = _train_classifier(
                x_0=xm_gen_norm,       # class 0: MC gen with pushed weights
                x_1=xm_gen_norm,       # class 1: MC gen uniform
                w_0=omega_pushed,
                w_1=None,
                hidden_size=self.hidden_size,
                n_layers=self.n_layers,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                device=self._device,
                patience=self.patience,
                dropout=self.dropout,
            )
            # Particle-level weight from the step-2 classifier
            omega = _predict_weights(clf2, xm_gen_norm, self._device,
                                     clip=self.weight_clip)
            # Normalise weights to unit mean
            omega = omega / (omega.mean() + 1e-12)

            self._iter_weights_.append(omega.copy())
            if self.verbose:
                print(
                    f"  weight mean={omega.mean():.4f}  "
                    f"std={omega.std():.4f}  "
                    f"max={omega.max():.4f}"
                )

        self.weights_ = omega
        return self

    @property
    def weights(self) -> np.ndarray:
        """Particle-level unfolding weights after the last iteration."""
        if self.weights_ is None:
            raise RuntimeError("Call fit() before accessing weights.")
        return self.weights_

    def iter_weights(self) -> List[np.ndarray]:
        """List of particle-level weights after each iteration (for convergence checks)."""
        if not self._iter_weights_:
            raise RuntimeError("Call fit() before accessing iter_weights().")
        return self._iter_weights_

    def convergence(self) -> List[float]:
        """Mean absolute change in weights between successive iterations.

        Returns
        -------
        List of length ``n_iter - 1``.  Converged when values are small.
        """
        iw = self.iter_weights()
        return [
            float(np.mean(np.abs(iw[i + 1] - iw[i])))
            for i in range(len(iw) - 1)
        ]

    # ------------------------------------------------------------------
    # Convenience: reweight a histogram
    # ------------------------------------------------------------------

    def reweight_histogram(
        self,
        x_mc_gen: np.ndarray,
        bins: int | np.ndarray = 30,
        range_: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return weighted and unweighted histograms of a 1-D observable.

        Parameters
        ----------
        x_mc_gen:
            1-D observable values at particle level (length must match the
            ``x_mc_gen`` used in :meth:`fit`).
        bins:
            Number of bins or bin edges.
        range_:
            ``(lo, hi)`` range for the histogram.

        Returns
        -------
        hist_unfolded:
            Weighted histogram (unfolded).
        bin_edges:
            Bin edges.
        """
        if self.weights_ is None:
            raise RuntimeError("Call fit() before reweight_histogram().")
        x_mc_gen = np.asarray(x_mc_gen, dtype=float)
        hist_unfolded, bin_edges = np.histogram(
            x_mc_gen, bins=bins, range=range_, weights=self.weights_
        )
        return hist_unfolded.astype(float), bin_edges

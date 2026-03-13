"""Tests for star_ea.multifold module."""

import numpy as np
import pytest
import torch

from star_ea.multifold import MultiFold, _MLP, _FeatureScaler, _predict_weights


# ---------------------------------------------------------------------------
# _FeatureScaler
# ---------------------------------------------------------------------------

class TestFeatureScaler:
    def test_fit_transform_zero_mean(self):
        from star_ea.multifold import _FeatureScaler
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler = _FeatureScaler()
        x_scaled = scaler.fit_transform(x)
        assert pytest.approx(x_scaled.mean(axis=0), abs=1e-6) == [0.0, 0.0]

    def test_fit_transform_unit_std(self):
        from star_ea.multifold import _FeatureScaler
        x = np.random.default_rng(0).normal(5, 2, (100, 3))
        scaler = _FeatureScaler()
        x_scaled = scaler.fit_transform(x)
        assert pytest.approx(x_scaled.std(axis=0), abs=0.1) == [1.0, 1.0, 1.0]

    def test_transform_uses_fit_stats(self):
        from star_ea.multifold import _FeatureScaler
        x_train = np.array([[0.0], [1.0], [2.0]])
        x_test = np.array([[3.0]])
        scaler = _FeatureScaler()
        scaler.fit(x_train)
        x_scaled = scaler.transform(x_test)
        assert pytest.approx(x_scaled[0, 0], rel=0.1) == (3.0 - 1.0) / (
            x_train.std() + 1e-8
        )

    def test_transform_before_fit_raises(self):
        from star_ea.multifold import _FeatureScaler
        scaler = _FeatureScaler()
        with pytest.raises(AssertionError):
            scaler.transform(np.array([[1.0]]))


# ---------------------------------------------------------------------------
# _MLP
# ---------------------------------------------------------------------------

class TestMLP:
    def test_output_shape(self):
        model = _MLP(input_dim=4, hidden_size=16, n_layers=2)
        x = torch.randn(32, 4)
        out = model(x)
        assert out.shape == (32,)

    def test_output_in_zero_one(self):
        model = _MLP(input_dim=4, hidden_size=16, n_layers=2)
        x = torch.randn(100, 4)
        out = model(x)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_different_input_sizes(self):
        for d in [1, 3, 10]:
            model = _MLP(input_dim=d, hidden_size=32)
            x = torch.randn(8, d)
            out = model(x)
            assert out.shape == (8,)


# ---------------------------------------------------------------------------
# MultiFold
# ---------------------------------------------------------------------------

def _make_toy_data(n_data=300, n_mc=500, seed=0):
    """Create toy 1-D multiplicity data for testing."""
    rng = np.random.default_rng(seed)
    # "True" distribution: Poisson(5)
    x_data = rng.poisson(5, size=(n_data, 1)).astype(np.float32)
    # MC generator: Poisson(4) — slightly different from data
    x_mc_gen = rng.poisson(4, size=(n_mc, 1)).astype(np.float32)
    # MC detector: add small Gaussian smearing
    x_mc_det = np.clip(x_mc_gen + rng.normal(0, 0.5, x_mc_gen.shape),
                       0, None).astype(np.float32)
    return x_data, x_mc_det, x_mc_gen


class TestMultiFold:
    def test_fit_returns_self(self):
        x_data, x_mc_det, x_mc_gen = _make_toy_data()
        mf = MultiFold(n_iter=1, n_epochs=2, hidden_size=16, batch_size=64,
                       patience=2)
        result = mf.fit(x_data, x_mc_det, x_mc_gen)
        assert result is mf

    def test_weights_shape(self):
        x_data, x_mc_det, x_mc_gen = _make_toy_data(n_mc=200)
        mf = MultiFold(n_iter=1, n_epochs=2, hidden_size=16, batch_size=64,
                       patience=2)
        mf.fit(x_data, x_mc_det, x_mc_gen)
        assert mf.weights.shape == (200,)

    def test_weights_positive(self):
        x_data, x_mc_det, x_mc_gen = _make_toy_data()
        mf = MultiFold(n_iter=1, n_epochs=2, hidden_size=16, batch_size=64,
                       patience=2)
        mf.fit(x_data, x_mc_det, x_mc_gen)
        assert np.all(mf.weights > 0)

    def test_iter_weights_length(self):
        x_data, x_mc_det, x_mc_gen = _make_toy_data()
        mf = MultiFold(n_iter=2, n_epochs=2, hidden_size=16, batch_size=64,
                       patience=2)
        mf.fit(x_data, x_mc_det, x_mc_gen)
        assert len(mf.iter_weights()) == 2

    def test_convergence_length(self):
        x_data, x_mc_det, x_mc_gen = _make_toy_data()
        mf = MultiFold(n_iter=3, n_epochs=2, hidden_size=16, batch_size=64,
                       patience=2)
        mf.fit(x_data, x_mc_det, x_mc_gen)
        assert len(mf.convergence()) == 2  # n_iter - 1

    def test_weights_before_fit_raises(self):
        mf = MultiFold(n_iter=1, n_epochs=2)
        with pytest.raises(RuntimeError):
            _ = mf.weights

    def test_iter_weights_before_fit_raises(self):
        mf = MultiFold(n_iter=1, n_epochs=2)
        with pytest.raises(RuntimeError):
            _ = mf.iter_weights()

    def test_mismatched_mc_raises(self):
        x_data, x_mc_det, x_mc_gen = _make_toy_data()
        mf = MultiFold(n_iter=1, n_epochs=2)
        with pytest.raises(ValueError, match="same number of rows"):
            mf.fit(x_data, x_mc_det, x_mc_gen[:10])

    def test_reweight_histogram(self):
        x_data, x_mc_det, x_mc_gen = _make_toy_data()
        mf = MultiFold(n_iter=1, n_epochs=2, hidden_size=16, batch_size=64,
                       patience=2)
        mf.fit(x_data, x_mc_det, x_mc_gen)
        hist, edges = mf.reweight_histogram(x_mc_gen[:, 0], bins=10,
                                            range_=(0.0, 15.0))
        assert len(hist) == 10
        assert len(edges) == 11
        assert np.all(hist >= 0)

    def test_1d_input_promoted_to_2d(self):
        """fit() should accept 1-D arrays and promote them to (n, 1)."""
        rng = np.random.default_rng(0)
        x_data = rng.poisson(5, 100).astype(np.float32)
        x_mc_det = rng.poisson(5, 200).astype(np.float32)
        x_mc_gen = rng.poisson(5, 200).astype(np.float32)
        mf = MultiFold(n_iter=1, n_epochs=2, hidden_size=16, batch_size=64,
                       patience=2)
        mf.fit(x_data, x_mc_det, x_mc_gen)
        assert mf.weights.shape == (200,)

    def test_unfolding_shifts_distribution(self):
        """After unfolding MC(mu=4) towards data(mu=6), the weighted mean of
        x_mc_gen should move closer to the data mean."""
        rng = np.random.default_rng(5)
        n = 2000
        x_data = rng.poisson(6, (n, 1)).astype(np.float32)
        x_mc_gen = rng.poisson(4, (n, 1)).astype(np.float32)
        x_mc_det = x_mc_gen.copy()  # no detector smearing for simplicity

        mf = MultiFold(n_iter=3, n_epochs=20, hidden_size=32, batch_size=128,
                       patience=5, verbose=False)
        mf.fit(x_data, x_mc_det, x_mc_gen)

        unweighted_mean = float(x_mc_gen[:, 0].mean())
        weighted_mean = float(
            np.average(x_mc_gen[:, 0], weights=mf.weights)
        )
        data_mean = float(x_data[:, 0].mean())

        # Weighted mean should be closer to data mean than unweighted
        assert abs(weighted_mean - data_mean) < abs(unweighted_mean - data_mean)

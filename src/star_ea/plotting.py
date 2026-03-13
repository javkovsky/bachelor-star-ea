"""Physics plotting utilities for STAR event-activity analysis.

All plot functions accept optional ``ax`` keyword arguments so they can
be embedded in larger figure layouts.  When ``ax`` is ``None`` a new
``(fig, ax)`` pair is created.

The module uses ``matplotlib`` exclusively and does not require ROOT or
any other non-standard plotting library.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np

from .observables import _nbd_pmf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ax(ax: Optional[plt.Axes], figsize: tuple) -> Tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


# ---------------------------------------------------------------------------
# Multiplicity distribution P(N)
# ---------------------------------------------------------------------------

def plot_multiplicity_distribution(
    n_values: np.ndarray,
    probabilities: np.ndarray,
    *,
    nbd_params: Optional[dict] = None,
    label: str = "Data",
    ax: Optional[plt.Axes] = None,
    color: str = "steelblue",
    title: str = "Charged-particle multiplicity distribution",
    xlabel: str = r"$N_\mathrm{ch}$",
    ylabel: str = r"$P(N_\mathrm{ch})$",
    figsize: tuple = (7, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot P(N) and optionally an NBD fit.

    Parameters
    ----------
    n_values:
        Multiplicity values (x axis).
    probabilities:
        Corresponding P(N) values.
    nbd_params:
        If given, must be a dict with ``"mu"`` and ``"k"`` keys.  The
        corresponding NBD curve is drawn on top of the data.
    label:
        Legend label for the data points.
    """
    fig, ax = _make_ax(ax, figsize)
    ax.semilogy(n_values, np.clip(probabilities, 1e-10, None),
                "o", ms=4, color=color, label=label, zorder=3)

    if nbd_params is not None:
        mu = nbd_params["mu"]
        k = nbd_params["k"]
        n_fine = np.arange(0, n_values[-1] + 1)
        p_nbd = _nbd_pmf(n_fine, mu, k)
        mu_err = nbd_params.get("mu_err", 0.0)
        k_err = nbd_params.get("k_err", 0.0)
        nbd_label = (
            rf"NBD $\mu={mu:.2f}\pm{mu_err:.2f}$, "
            rf"$k={k:.2f}\pm{k_err:.2f}$"
        )
        ax.semilogy(n_fine, p_nbd, "-", lw=2, color="crimson",
                    label=nbd_label, zorder=2)

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Mean pT vs multiplicity
# ---------------------------------------------------------------------------

def plot_mean_pt_vs_multiplicity(
    n_values: np.ndarray,
    mean_pt: np.ndarray,
    mean_pt_err: Optional[np.ndarray] = None,
    *,
    label: str = "Simulation",
    ax: Optional[plt.Axes] = None,
    color: str = "steelblue",
    title: str = r"$\langle p_T \rangle$ vs. $N_\mathrm{ch}$",
    xlabel: str = r"$N_\mathrm{ch}$",
    ylabel: str = r"$\langle p_T \rangle$ [GeV/$c$]",
    figsize: tuple = (7, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot ⟨pT⟩ as a function of charged-particle multiplicity."""
    fig, ax = _make_ax(ax, figsize)
    valid = ~np.isnan(mean_pt)
    x = n_values[valid]
    y = mean_pt[valid]
    if mean_pt_err is not None:
        ye = mean_pt_err[valid]
        ax.errorbar(x, y, yerr=ye, fmt="o-", ms=4, color=color,
                    capsize=3, label=label)
    else:
        ax.plot(x, y, "o-", ms=4, color=color, label=label)

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Forward–backward correlation scatter
# ---------------------------------------------------------------------------

def plot_fb_scatter(
    fwd_mult: np.ndarray,
    bwd_mult: np.ndarray,
    *,
    b_corr: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    color: str = "steelblue",
    alpha: float = 0.3,
    title: str = "Forward–backward multiplicity correlation",
    xlabel: str = r"$N_\mathrm{fwd}$",
    ylabel: str = r"$N_\mathrm{bwd}$",
    figsize: tuple = (6, 6),
) -> Tuple[plt.Figure, plt.Axes]:
    """Scatter plot of forward vs backward multiplicity per event."""
    fig, ax = _make_ax(ax, figsize)
    ax.scatter(fwd_mult, bwd_mult, s=5, alpha=alpha, color=color)
    if b_corr is not None:
        ax.set_title(
            f"{title}\n$b_{{\\rm corr}} = {b_corr:.3f}$", fontsize=13
        )
    else:
        ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Unfolding comparison
# ---------------------------------------------------------------------------

def plot_unfolding_comparison(
    x_gen: np.ndarray,
    x_mc_gen: np.ndarray,
    weights_unfolded: np.ndarray,
    *,
    bins: int = 30,
    range_: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Unfolding comparison",
    xlabel: str = "Observable",
    ylabel: str = "Normalised counts",
    figsize: tuple = (7, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """Compare the unfolded MC distribution with the true particle-level.

    Parameters
    ----------
    x_gen:
        True particle-level observable (1-D array).
    x_mc_gen:
        MC particle-level observable (1-D array, same variable).
    weights_unfolded:
        Unfolding weights for each MC event.
    """
    fig, ax = _make_ax(ax, figsize)
    kw = dict(bins=bins, range=range_, density=True, histtype="step", lw=2)
    ax.hist(x_gen, **kw, color="black", label="Truth")
    ax.hist(x_mc_gen, **kw, color="steelblue", ls="--", label="MC (unweighted)")
    ax.hist(x_mc_gen, **kw, weights=weights_unfolded,
            color="crimson", label="MC (unfolded)")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------

def plot_convergence(
    convergence_values: List[float],
    *,
    ax: Optional[plt.Axes] = None,
    color: str = "steelblue",
    title: str = "MultiFold convergence",
    xlabel: str = "Iteration",
    ylabel: str = r"$\langle | \Delta w | \rangle$",
    figsize: tuple = (6, 4),
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot mean weight change between successive MultiFold iterations."""
    fig, ax = _make_ax(ax, figsize)
    iters = np.arange(1, len(convergence_values) + 1)
    ax.plot(iters, convergence_values, "o-", ms=6, color=color)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(iters)
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Combined summary figure
# ---------------------------------------------------------------------------

def plot_analysis_summary(
    analysis_results: dict,
    *,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """Produce a 2×2 summary figure from the output of
    :meth:`~star_ea.observables.EventActivityAnalyzer.analyse`.

    Parameters
    ----------
    analysis_results:
        Dictionary returned by
        :meth:`~star_ea.observables.EventActivityAnalyzer.analyse`.
    save_path:
        If given, the figure is saved to this path.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Multiplicity distribution
    n_vals, p_n = analysis_results["multiplicity_dist"]
    nbd = analysis_results.get("nbd_fit")
    plot_multiplicity_distribution(n_vals, p_n, nbd_params=nbd, ax=axes[0, 0])

    # 2. Mean pT vs multiplicity
    n_vals_pt, mean_pt, mean_pt_err = analysis_results["mean_pt_vs_mult"]
    plot_mean_pt_vs_multiplicity(n_vals_pt, mean_pt, mean_pt_err,
                                 ax=axes[0, 1])

    # 3. FB correlation as a text summary (no scatter data here)
    fb = analysis_results["fb_correlation"]
    axes[1, 0].axis("off")
    summary_text = (
        "Forward–Backward Correlation Summary\n\n"
        f"$b_{{\\rm corr}}$ = {fb['b_corr']:.4f}\n"
        f"$\\langle N_{{\\rm fwd}} \\rangle$ = {fb['mean_fwd']:.2f}\n"
        f"$\\langle N_{{\\rm bwd}} \\rangle$ = {fb['mean_bwd']:.2f}\n"
        f"Covariance = {fb['cov']:.4f}"
    )
    axes[1, 0].text(0.5, 0.5, summary_text, ha="center", va="center",
                    transform=axes[1, 0].transAxes, fontsize=13,
                    bbox=dict(boxstyle="round", facecolor="lightyellow",
                              edgecolor="gray"))

    # 4. NBD fit summary
    axes[1, 1].axis("off")
    if nbd is not None and not np.isnan(nbd.get("mu", float("nan"))):
        nbd_text = (
            "NBD Fit Parameters\n\n"
            f"$\\mu$ = {nbd['mu']:.3f} ± {nbd['mu_err']:.3f}\n"
            f"$k$ = {nbd['k']:.3f} ± {nbd['k_err']:.3f}\n"
            f"$\\chi^2/\\mathrm{{ndf}}$ = {nbd['chi2_ndf']:.2f}\n\n"
            f"$\\langle N_{{\\rm ch}} \\rangle$ = "
            f"{analysis_results['mean_multiplicity']:.2f}\n"
            f"$\\langle\\langle p_T\\rangle\\rangle$ = "
            f"{analysis_results['mean_mean_pt']:.3f} GeV/$c$"
        )
    else:
        nbd_text = (
            f"$\\langle N_{{\\rm ch}} \\rangle$ = "
            f"{analysis_results['mean_multiplicity']:.2f}\n"
            f"$\\langle\\langle p_T\\rangle\\rangle$ = "
            f"{analysis_results['mean_mean_pt']:.3f} GeV/$c$"
        )
    axes[1, 1].text(0.5, 0.5, nbd_text, ha="center", va="center",
                    transform=axes[1, 1].transAxes, fontsize=13,
                    bbox=dict(boxstyle="round", facecolor="lightblue",
                              edgecolor="gray"))

    fig.suptitle("STAR pp Event-Activity Analysis Summary", fontsize=15,
                 fontweight="bold")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig

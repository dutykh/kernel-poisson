# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Figure of the Newton systems: conditioning against the size of the counts, cost against the
number of observations, and convergence histories. Reads results/newton/{spectra,cost,trace}.csv."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figure_style as fs  # noqa: E402

RESULTS = HERE.parent / "results" / "newton"

LABELS = {
    "fisher": "Fisher landmarks",
    "uniform": "uniform landmarks",
    "rkhs": r"conjugate gradients in $\mathcal{H}$",
    "truncated_newton": "truncated Newton (Maalouf et al.)",
    "lbfgs": "quasi-Newton",
    "exact": "Newton, direct solves",
}
STYLE = {"fisher": 0, "rkhs": 1, "uniform": 2, "truncated_newton": 3, "lbfgs": 4, "exact": 5}


def style(method, **extra):
    st = fs.series(STYLE[method])
    st.update(markersize=4.0, linewidth=1.1)
    st.update(extra)
    return st


def median(frame, by, column):
    """Median over the samples of each count scale (robust to a sample whose tuned penalty is
    far from the others)."""
    return frame.groupby(by)[column].median().reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "figures" / "newton.pdf"))
    ap.add_argument("--scale", type=float, default=30.0, help="count scale of the cost panel")
    args = ap.parse_args()
    fs.configure()
    spectra = pd.read_csv(RESULTS / "spectra.csv")
    cost = pd.read_csv(RESULTS / "cost.csv")
    trace = pd.read_csv(RESULTS / "trace.csv")

    fig, axes = plt.subplots(1, 3, figsize=(fs.WIDTH, 2.55), layout="constrained")

    # (a) condition numbers against the largest fitted count
    ax = axes[0]
    mu = median(spectra, "scale", "mu_max")
    series_a = [("truncated_newton", "kappa_truncated_newton"), ("rkhs", "kappa_rkhs"),
                ("uniform", "kappa_uniform_4"), ("fisher", "kappa_fisher_4")]
    for method, column in series_a:
        g = median(spectra, "scale", column)
        ax.plot(mu["mu_max"], g[column], **style(method))
    bound = median(spectra, "scale", "bound_rkhs")
    ax.plot(mu["mu_max"], bound["bound_rkhs"], color=fs.SLATE, linewidth=0.8, linestyle="-",
            label=r"bound $1+w_{\max}\sigma_1/\lambda$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"largest fitted count $w_{\max}$")
    ax.set_ylabel("condition number")
    ax.set_ylim(0.5, 3e15)
    fs.grid(ax)
    fs.panel(ax, "(a)")

    # (b) time to the solution against N, at one count scale
    ax = axes[1]
    sub = cost[cost["scale"] == args.scale]
    for method in ("truncated_newton", "lbfgs", "exact", "rkhs", "uniform", "fisher"):
        g = sub[sub["method"] == method].sort_values("N")
        if len(g):
            ax.plot(g["N"], g["seconds"], **style(method))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"number of observations $N$")
    ax.set_ylabel("time (s)")
    fs.grid(ax)
    fs.panel(ax, "(b)")

    # (c) convergence histories at N = 4000
    ax = axes[2]
    for method in ("truncated_newton", "lbfgs", "rkhs", "uniform", "fisher"):
        g = trace[trace["method"] == method].sort_values("seconds")
        gap = np.maximum(g["gap"].to_numpy(), 1e-16)
        st = style(method)
        if method == "lbfgs":
            st.update(marker=None)
        elif method == "truncated_newton":
            st.update(markevery=max(1, len(g) // 12))
        ax.plot(g["seconds"], gap, **st)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("relative objective gap")
    ax.set_ylim(5e-17, 1.0)
    fs.grid(ax)
    fs.panel(ax, "(c)")

    handles, labels = [], []
    for method in ("fisher", "uniform", "rkhs", "truncated_newton", "lbfgs", "exact"):
        (h,) = axes[1].plot([], [], **style(method))
        handles.append(h)
        labels.append(LABELS[method])
    hb = axes[0].get_lines()[-1]
    handles.append(hb)
    labels.append(hb.get_label())
    fs.legend_outside(fig, where="upper", ncol=3, handles=handles, labels=labels, columnspacing=1.2, handlelength=2.6)
    fs.save(fig, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

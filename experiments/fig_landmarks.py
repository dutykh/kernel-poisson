# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Supplementary figure on the landmark preconditioner: condition number of the preconditioned
Newton system and Nystrom error of the Fisher-weighted Gram matrix, for landmarks drawn from Fisher
leverage scores or uniformly, with m = d, 2d and 4d landmarks. Reads results/newton/spectra.csv."""
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


def median_over_samples(v):
    """Median over the samples of a count scale (robust to a sample whose tuned penalty is far
    from the others)."""
    return float(np.median(np.asarray(v, dtype=float)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "figures" / "landmarks.pdf"))
    a = ap.parse_args()
    fs.configure()
    sp = pd.read_csv(RESULTS / "spectra.csv")
    g = sp.groupby("scale")
    mu = g["mu_max"].apply(median_over_samples)
    fig, axes = plt.subplots(1, 2, figsize=(fs.WIDTH, 2.6), layout="constrained")
    for c in (1, 2, 4):
        for scheme, base in (("fisher", 0), ("uniform", 2)):
            st = fs.series(base if c == 4 else (1 if scheme == "fisher" else 3) if c == 2 else (4 if scheme == "fisher" else 5))
            lab = f"{'Fisher' if scheme == 'fisher' else 'uniform'}, $m={'' if c == 1 else c}\\hat d$"
            for ax, col in zip(axes, (f"kappa_{scheme}_{c}", f"eps_{scheme}_{c}")):
                ax.plot(mu.values, g[col].apply(median_over_samples).values, color=st["color"], linestyle=st["linestyle"],
                        marker=st["marker"], markersize=3.5, markerfacecolor="white",
                        markeredgecolor=st["color"], linewidth=1.0, label=lab)
    axes[0].set_ylabel("condition number, preconditioned")
    nystrom = r'Nystr\"om' if plt.rcParams["text.usetex"] else "Nyström"
    axes[1].set_ylabel(nystrom + r" error $\|\mathbf{E}_L\|/\lambda$")
    for ax, t in zip(axes, ("(a)", "(b)")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"largest fitted count $w_{\max}$")
        fs.grid(ax)
        fs.panel(ax, t)
    axes[1].axhline(1.0, color=fs.SLATE, linewidth=0.7)
    fs.legend_outside(fig, where="upper", ncol=3, handles=axes[0].get_lines(),
                      labels=[l.get_label() for l in axes[0].get_lines()])
    fs.save(fig, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

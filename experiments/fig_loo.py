# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Figure of the leave-one-out theorem and of its extension to blocks: one-step error against its
bound, and the exact and one-step deviances along the penalty path, first for the deletion of one
observation (top row), then for the deletion of a block of neighbouring observations (bottom row).
Reads results/loo/{points,path}.csv and results/loo/{blocks_points,blocks_path}.csv."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figure_style as fs  # noqa: E402

RESULTS = HERE.parent / "results" / "loo"


def error_panel(ax, pts, error, xlabel, ylabel, ok_label, bad_label, title):
    ok = pts["condition"] <= 0.25
    lo = min(pts["bound"].min(), pts[error][pts[error] > 0].min()) / 2
    hi = max(pts["bound"].max(), pts[error].max()) * 2
    ax.plot([lo, hi], [lo, hi], color=fs.SLATE, linewidth=0.8, label="error $=$ bound")
    s0 = fs.series(0)
    ax.plot(pts["bound"][ok], pts[error][ok], linestyle="none", marker="o", markersize=3.0,
            markerfacecolor="white", markeredgecolor=s0["color"], markeredgewidth=0.7,
            label=ok_label)
    s1 = fs.series(1)
    ax.plot(pts["bound"][~ok], pts[error][~ok], linestyle="none", marker="s", markersize=3.2,
            markerfacecolor="white", markeredgecolor=s1["color"], markeredgewidth=0.8,
            label=bad_label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fs.grid(ax)
    fs.panel(ax, title)
    ax.legend(loc="upper left", frameon=False, handletextpad=0.4, borderaxespad=0.3)


def path_panel(ax, path, exact, onestep, labels, title, loc):
    ax.plot(path["t"], path[exact], **{**fs.series(0), "markersize": 3.5, "linewidth": 1.1},
            label=labels[0])
    ax.plot(path["t"], path[onestep], **{**fs.series(1), "markersize": 3.5, "linewidth": 1.1},
            label=labels[1])
    ax.plot(path["t"], path["train_deviance"], **{**fs.series(2), "markersize": 3.0, "linewidth": 1.0},
            label="training")
    ax.set_xscale("log")
    ax.set_xlabel(r"penalty per event $t=\lambda/\sum_iy_i$")
    ax.set_ylabel("mean deviance")
    fs.panel(ax, title)
    ax.legend(loc=loc, frameon=False, handletextpad=0.4, borderaxespad=0.3)


def quoted_numbers(pts, path, bpts, bpath):
    """Print the numbers that the supplementary material quotes about the figure."""
    ok = pts["condition"] <= 0.25
    ratio = (pts["bound"] / pts["error"])[ok]
    print(f"single observations: the hypothesis holds for {100 * ok.mean():.0f} percent; the bound is "
          f"{ratio.min():.1f} to {ratio.max():.0f} times the error (median {ratio.median():.1f})")
    gap = (path["alo_deviance"] - path["loo_deviance"]).abs() / path["loo_deviance"]
    print(f"  over the penalties of panel (b) the two deviances differ by at most {100 * gap.max():.2f} "
          f"percent and select t = {path.loc[path['alo_deviance'].idxmin(), 't']:.1e} (one-step) and "
          f"{path.loc[path['loo_deviance'].idxmin(), 't']:.1e} (exact)")
    okb = bpts["condition"] <= 0.25
    ratio_b = (bpts["bound"] / bpts["error"])[okb]
    print(f"blocks: the hypothesis holds for {bpts.loc[okb, 'block'].nunique()} of {bpts['block'].nunique()} "
          f"blocks, which hold {100 * okb.mean():.0f} percent of the observations; the bound is "
          f"{ratio_b.min():.1f} to {ratio_b.max():.0f} times the error (median {ratio_b.median():.0f})")
    half = bpath[bpath["share_condition"] >= 0.5]
    gap = (half["onestep_block_deviance"] - half["exact_block_deviance"]).abs() / half["exact_block_deviance"]
    print(f"  for t from {half['t'].max():.1e} down to {half['t'].min():.1e}, where the hypothesis holds for "
          f"at least half of the observations, the two block deviances differ by at most {100 * gap.max():.2f} "
          f"percent and select t = {bpath.loc[bpath['onestep_block_deviance'].idxmin(), 't']:.1e} (one-step) "
          f"and {bpath.loc[bpath['exact_block_deviance'].idxmin(), 't']:.1e} (exact)")
    over = 100 * (bpath["onestep_block_deviance"] - bpath["exact_block_deviance"]) / bpath["exact_block_deviance"]
    print(f"  over the penalties of panel (d) the one-step deviance exceeds the exact one by at most "
          f"{over.max():.2f} percent")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "figures" / "loo.pdf"))
    ap.add_argument("--tmin", type=float, default=1e-5)
    args = ap.parse_args()
    fs.configure()
    pts = pd.read_csv(RESULTS / "points.csv")
    path = pd.read_csv(RESULTS / "path.csv")
    path = path[path["t"] >= args.tmin * 0.999]
    bpts = pd.read_csv(RESULTS / "blocks_points.csv")
    bpath = pd.read_csv(RESULTS / "blocks_path.csv")
    bpath = bpath[bpath["t"] >= args.tmin * 0.999]
    quoted_numbers(pts, path, bpts, bpath)

    fig, axes = plt.subplots(2, 2, figsize=(fs.WIDTH, 5.2), layout="constrained")
    error_panel(axes[0, 0], pts, "error", r"bound $1.04\,A_i\sqrt{q_i/(1-h_i)}\;c_i$",
                r"error $|\tilde\eta_i-\hat\eta_i^{(-i)}|$",
                r"$A_i\sqrt{c_i}\leqslant\frac{1}{4}$", r"$A_i\sqrt{c_i}>\frac{1}{4}$", "(a)")
    path_panel(axes[0, 1], path, "loo_deviance", "alo_deviance",
               ("exact leave-one-out", "one-step leave-one-out"), "(b)", "upper left")
    error_panel(axes[1, 0], bpts, "error", r"bound $1.04\,A_B\,d_i\,c_B$",
                r"error $|\tilde\eta_i-\hat\eta_i^{(-B)}|$",
                r"$A_B\sqrt{c_B}\leqslant\frac{1}{4}$", r"$A_B\sqrt{c_B}>\frac{1}{4}$", "(c)")
    path_panel(axes[1, 1], bpath, "exact_block_deviance", "onestep_block_deviance",
               ("exact leave-one-block-out", "one-step leave-one-block-out"), "(d)", "upper center")
    fs.save(fig, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

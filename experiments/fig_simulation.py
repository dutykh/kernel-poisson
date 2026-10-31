# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Figure and table of the simulation study: excess risk per event of each method against the
truth, for four log-rates and two mean counts. Reads results/simulation/*.csv; writes
figures/simulation.pdf and results/simulation/summary.csv."""
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

RESULTS = HERE.parent / "results" / "simulation"
TRUTHS = ["interaction", "additive", "threshold", "linear"]
TLABEL = {"interaction": "smooth\ninteraction", "additive": "additive", "threshold": "thresholds",
          "linear": "linear"}
METHODS = ["glm_poisson", "gam_poisson", "boosting", "forest", "ours_poisson"]
MLABEL = {"glm_poisson": "GLM", "gam_poisson": "GAM", "boosting": "boosting", "forest": "forest",
          "ours_poisson": "kernel Poisson"}


def geometric_mean(ratios):
    return float(np.exp(np.mean(np.log(ratios))))


def quoted_numbers(piv):
    """Print the numbers that the article quotes about the simulation: geometric means over the
    replicates of ratios of excess risks per event."""
    relative = piv.div(piv.min(axis=1), axis=0)
    print(f"risk divided by the smallest risk of the methods on the same replicate, geometric mean over "
          f"{len(piv)} replicates:")
    print("  " + ", ".join(f"{MLABEL[m]} {geometric_mean(relative[m]):.2f}"
                           for m in relative.mean().sort_values().index))
    for mc in sorted(piv.index.get_level_values("mean_count").unique()):
        s = piv.loc[("interaction", mc)]
        trees = min(geometric_mean(s[m] / s["ours_poisson"]) for m in ("forest", "boosting"))
        print(f"smooth interaction, mean count {mc:g}: best tree ensemble over kernel Poisson {trees:.2f}, "
              f"additive model over kernel Poisson {geometric_mean(s['gam_poisson'] / s['ours_poisson']):.2f}")
        s = piv.loc[("linear", mc)]
        print(f"linear, mean count {mc:g}: kernel Poisson exceeds the linear model by "
              f"{100 * (geometric_mean(s['ours_poisson'] / s['glm_poisson']) - 1):.0f} percent")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "figures" / "simulation.pdf"))
    a = ap.parse_args()
    files = sorted(RESULTS.glob("*_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files])
    df = df[df["status"] == "ok"]
    summary = (df.groupby(["truth", "mean_count", "method"])["risk"]
               .agg(reps="size", mean="mean", median="median",
                    se=lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v))))
               .reset_index())
    summary.to_csv(RESULTS / "summary.csv", index=False)
    print(summary.pivot_table(index=["truth", "mean_count"], columns="method", values="mean").round(4).to_string())

    # ratio of each method's risk to the kernel estimator's, replicate by replicate
    piv = df.pivot_table(index=["truth", "mean_count", "rep"], columns="method", values="risk")
    quoted_numbers(piv)
    fs.configure()
    means = sorted(df["mean_count"].unique())
    fig, axes = plt.subplots(1, len(means), figsize=(fs.WIDTH, 2.6), layout="constrained", sharey=True)
    axes = np.atleast_1d(axes)
    shown = [m for m in METHODS if m != "ours_poisson" and m in piv.columns]
    off = np.linspace(-0.27, 0.27, len(shown))
    for ax, mc in zip(axes, means):
        for k, m in enumerate(shown):
            st = fs.series(k + 1)
            xs, ys, lo, hi = [], [], [], []
            for i, t in enumerate(TRUTHS):
                key = (t, mc)
                if key not in piv.index.droplevel("rep").unique():
                    continue
                sub = piv.loc[(t, mc)]
                r = (sub[m] / sub["ours_poisson"]).dropna().to_numpy()
                if len(r) == 0:
                    continue
                g = np.exp(np.mean(np.log(r)))
                xs.append(i + off[k]); ys.append(g)
                lo.append(np.quantile(r, 0.25)); hi.append(np.quantile(r, 0.75))
            if xs:
                ys, lo, hi = map(np.asarray, (ys, lo, hi))
                ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], linestyle="none", marker=st["marker"],
                            markersize=4, markerfacecolor="white", markeredgecolor=st["color"],
                            ecolor=st["color"], elinewidth=0.7, label=MLABEL[m])
        ax.axhline(1.0, color=fs.SLATE, linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xticks(range(len(TRUTHS)))
        ax.set_xticklabels([TLABEL[t] for t in TRUTHS])
        fs.grid(ax)
        fs.panel(ax, f"({chr(97 + list(means).index(mc))}) mean count {mc:g}")
    axes[0].set_ylabel("risk relative to kernel Poisson")
    fs.legend_outside(fig, where="upper", ncol=4)
    fs.save(fig, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Figure of the rate theorem: excess risk per event against the expected number of events, for
three regimes of the mean count per observation (smooth and rough truths), for overdispersed
counts, and at a fixed number of observations as the exposures grow. Reads results/rates/rates.csv
and results/rates/plateau.csv."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figure_style as fs  # noqa: E402

RESULTS = HERE.parent / "results" / "rates"
A = 2.0
SLOPES = {"smooth": 2 * A / (1 + 2 * A), "rough": A / (1 + A)}


def summarise(df, by):
    g = df.groupby(by)
    spec = dict(Lambda=("Lambda", "mean"), n=("risk_oracle", "size"),
                gm=("risk_oracle", lambda v: float(np.exp(np.mean(np.log(v))))),
                q1=("risk_oracle", lambda v: float(np.quantile(v, 0.25))),
                q3=("risk_oracle", lambda v: float(np.quantile(v, 0.75))),
                med=("risk_oracle", "median"))
    if "risk_alo" in df.columns:
        spec["gm_alo"] = ("risk_alo", lambda v: float(np.exp(np.mean(np.log(v)))))
    return g.agg(**spec).reset_index()


def fitted_slope(s):
    x, y = np.log(s["Lambda"].to_numpy()), np.log(s["gm"].to_numpy())
    return float(np.polyfit(x, y, 1)[0])


def reference(ax, s, slope, label):
    x = np.array([s["Lambda"].min(), s["Lambda"].max()])
    x0, y0 = np.exp(np.mean(np.log(s["Lambda"]))), np.exp(np.mean(np.log(s["gm"])))
    ax.plot(x, 2.2 * y0 * (x / x0) ** (-slope), color=fs.SLATE, linewidth=0.8, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "figures" / "rates.pdf"))
    ap.add_argument("--reps", type=int, default=15)
    args = ap.parse_args()
    fs.configure()
    df = pd.read_csv(RESULTS / "rates.csv")
    df = df[df["rep"] < args.reps]          # the same replicates in every cell of the design
    pois = df[df["kappa_d"] == 0.0]
    fig, grid = plt.subplots(2, 2, figsize=(fs.WIDTH, 4.7), layout="constrained")
    axes = grid.ravel()
    report = []
    for ax, kind, title in zip(axes[:2], ("smooth", "rough"), ("(a)", "(b)")):
        sub = pois[pois["kind"] == kind]
        if sub.empty:
            continue
        for k, mc in enumerate(sorted(sub["mean_count"].unique())):
            s = summarise(sub[sub["mean_count"] == mc], "Lambda_target")
            st = fs.series(k)
            ax.errorbar(s["Lambda"], s["med"], yerr=[s["med"] - s["q1"], s["q3"] - s["med"]],
                        color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                        markersize=3.5, markerfacecolor="white", markeredgecolor=st["color"],
                        linewidth=1.0, elinewidth=0.6, capsize=1.5,
                        label=f"mean count {mc:g}")
            report.append((kind, mc, len(s), fitted_slope(s) if len(s) > 1 else float("nan")))
        s_all = summarise(sub, "Lambda_target")
        reference(ax, s_all, SLOPES[kind], rf"slope $-{SLOPES[kind]:.2f}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"expected events $\Lambda$")
        fs.grid(ax)
        fs.panel(ax, title)
    axes[0].set_ylabel("excess risk per event")
    axes[2].set_ylabel("excess risk per event")

    ax = axes[2]
    od = df[(df["kind"] == "smooth") & (df["mean_count"] == 1.0)]
    for k, kd in enumerate(sorted(od["kappa_d"].unique())):
        s = summarise(od[od["kappa_d"] == kd], "Lambda_target")
        st = fs.series(k)
        ax.plot(s["Lambda"], s["gm"], color=st["color"], linestyle=st["linestyle"],
                marker=st["marker"], markersize=3.5, markerfacecolor="white",
                markeredgecolor=st["color"], linewidth=1.0,
                label=("Poisson" if kd == 0 else rf"$\kappa_{{\mathrm{{d}}}}={kd:g}$"))
        report.append(("overdispersed", kd, len(s), fitted_slope(s) if len(s) > 1 else float("nan")))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"expected events $\Lambda$")
    fs.grid(ax)
    fs.panel(ax, "(c)")

    ax = axes[3]
    pl_path = RESULTS / "plateau.csv"
    if pl_path.exists():
        pl = pd.read_csv(pl_path)
        pl = pl[pl["rep"] < args.reps]
        rough1 = pois[(pois["kind"] == "rough") & (pois["mean_count"] == 1.0)]
        if not rough1.empty:
            s = summarise(rough1, "Lambda_target")
            ax.plot(s["Lambda"], s["med"], color=fs.SLATE, linestyle=(0, (5, 2)), linewidth=0.9,
                    label="mean count 1, as in (b)")
        for k, n in enumerate(sorted(pl["N"].unique())):
            s = summarise(pl[pl["N"] == n], "Lambda_target")
            st = fs.series(k + 3)
            ax.errorbar(s["Lambda"], s["med"], yerr=[s["med"] - s["q1"], s["q3"] - s["med"]],
                        color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                        markersize=3.5, markerfacecolor="white", markeredgecolor=st["color"],
                        linewidth=1.0, elinewidth=0.6, capsize=1.5, label=f"$N={n}$")
            tail = s[s["Lambda"] >= s["Lambda"].max() / 30]
            report.append(("plateau", n, len(s), fitted_slope(tail) if len(tail) > 1 else float("nan")))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"expected events $\Lambda$")
    fs.grid(ax)
    fs.panel(ax, "(d)")

    def collect(axs):
        handles, labels = [], []
        for a in axs:
            for h, l in zip(*a.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        return handles, labels

    h, l = collect(axes[:2])
    fs.legend_outside(fig, where="upper", ncol=5, handles=h, labels=l)
    h, l = collect(axes[2:])
    fs.legend_outside(fig, where="lower", ncol=4, handles=h, labels=l)
    for ax in axes:
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    fs.save(fig, args.out)
    for r in report:
        print("%-14s %-6s points=%d fitted slope=%.3f" % (r[0], r[1], r[2], r[3]))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

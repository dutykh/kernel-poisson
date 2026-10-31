# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Case study: deep-sea demersal fish counts against depth and period (Bailey et al. 2009).

Fits the kernel negative-binomial and kernel Poisson estimators with the swept area as exposure,
tuned by approximate leave-one-out, and records the fitted rate curves with their pointwise 90 per
cent bands (Gaussian reading of the penalized fit), the design law and the event law of depth, and
the identities of the exact-structure proposition. Writes results/fish/*.csv and the figure
figures/fish.pdf.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import figure_style as fs  # noqa: E402
from data import datasets  # noqa: E402
from kernelpoisson.estimator import KernelPoissonRegressor  # noqa: E402
from kernelpoisson.kernels import gram  # noqa: E402
from kernelpoisson.uncertainty import log_rate_variance  # noqa: E402

RESULTS = HERE.parent / "results" / "fish"
PERIODS = ["1977-1989", "2000-2002"]
HECTARE = 1.0e4          # square metres; rates are reported per hectare swept


def fit(family, X, y, e):
    return KernelPoissonRegressor(family=family, categorical=(1,), seed=0).fit(X, y, e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE.parent / "figures" / "fish.pdf"))
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    d = datasets.load("fishing")
    X = d["X"][["meandepth", "period"]].to_numpy(dtype=object)
    y, e = d["y"].astype(float), d["exposure"].astype(float)
    depth = X[:, 0].astype(float)

    summary = {}
    curves = []
    grid = np.linspace(depth.min(), depth.max(), 200)
    for family in ("negbin", "poisson"):
        est = fit(family, X, y, e)
        eta = est.predict_eta(X, e)
        mu = np.exp(eta)
        # exact structure: balance and the observed-minus-expected identity at the data
        Z = est.encoder_.transform(X)
        K = gram(Z, lengthscale=est.lengthscale_, family=est.kernel)
        K[np.diag_indices_from(K)] += est.nugget
        fhat = K @ est.best_["alpha"]
        loss = est._loss(est.kappa_)
        _, g, w = loss.derivatives(y, eta)
        oe = K @ (-g)                          # sum_i k(x, x_i) (-l'(y_i, eta_i)) at the data
        summary[family] = dict(kappa=est.kappa_, lengthscale=est.lengthscale_, lam=est.lam_,
                               t=est.lam_ / y.sum(), alo=est.best_["alo"],
                               balance=float(np.sum(mu) / np.sum(y)),
                               weighted_balance=float(np.sum(-g)),
                               oe_identity=float(np.max(np.abs(oe - est.lam_ * fhat))
                                                 / np.max(np.abs(est.lam_ * fhat))))
        for p in PERIODS:
            Xg = np.empty((len(grid), 2), dtype=object)
            Xg[:, 0] = grid
            Xg[:, 1] = p
            eg = est.predict_eta(Xg)
            sd = np.sqrt(log_rate_variance(est, X, y, e, Xg))
            curves.append(pd.DataFrame(dict(family=family, period=p, depth=grid,
                                            rate=np.exp(eg) * HECTARE,
                                            lower=np.exp(eg - 1.645 * sd) * HECTARE,
                                            upper=np.exp(eg + 1.645 * sd) * HECTARE)))
        if family == "negbin":
            mu_nb = mu
    curves = pd.concat(curves)
    curves.to_csv(RESULTS / "curves.csv", index=False)
    points = pd.DataFrame(dict(depth=depth, period=X[:, 1], count=y, swept=e,
                               rate=y / e * HECTARE, fitted=mu_nb))
    points.to_csv(RESULTS / "points.csv", index=False)
    edges = np.arange(750, 5250, 250.0)
    hauls, _ = np.histogram(depth, bins=edges)
    events, _ = np.histogram(depth, bins=edges, weights=mu_nb)
    laws = pd.DataFrame(dict(left=edges[:-1], right=edges[1:], design=hauls / hauls.sum(),
                             event=events / events.sum()))
    laws.to_csv(RESULTS / "laws.csv", index=False)
    half = np.median(depth)
    summary["laws"] = dict(median_depth=float(half),
                           event_share_below_median=float(mu_nb[depth <= half].sum() / mu_nb.sum()),
                           hauls=int(len(y)), events=float(y.sum()))
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    # figure
    fs.configure()
    fig, axes = plt.subplots(1, 2, figsize=(fs.WIDTH, 2.7), layout="constrained",
                             gridspec_kw=dict(width_ratios=[1.35, 1.0]))
    ax = axes[0]
    for k, p in enumerate(PERIODS):
        st = fs.series(k)
        c = curves[(curves["family"] == "negbin") & (curves["period"] == p)]
        ax.fill_between(c["depth"] / 1000, c["lower"], c["upper"], color=st["color"], alpha=0.12,
                        linewidth=0)
        ax.plot(c["depth"] / 1000, c["rate"], color=st["color"], linestyle=st["linestyle"],
                linewidth=1.2, label=f"fitted, {p.replace('-', '--')}")
        sel = points["period"] == p
        ax.plot(points["depth"][sel] / 1000, points["rate"][sel], linestyle="none",
                marker=st["marker"], markersize=3.0, markerfacecolor="white",
                markeredgecolor=st["color"], markeredgewidth=0.7, label=f"observed, {p.replace('-', '--')}")
    ax.set_yscale("log")
    ax.set_xlabel("mean depth of the haul (km)")
    ax.set_ylabel("fish per hectare swept")
    fs.grid(ax)
    fs.panel(ax, "(a)")

    ax = axes[1]
    mids = 0.5 * (laws["left"] + laws["right"]) / 1000
    w = 0.25
    ax.step(np.r_[laws["left"] / 1000, laws["right"].iloc[-1] / 1000],
            np.r_[laws["design"], laws["design"].iloc[-1]], where="post", color=fs.SLATE,
            linewidth=1.0, label="hauls (design law)")
    ax.bar(mids, laws["event"], width=w * 0.92, color=fs.NAVY, alpha=0.18, edgecolor=fs.NAVY,
           linewidth=0.6, label="fitted fish (event law)")
    ax.set_xlabel("mean depth (km)")
    ax.set_ylabel("share per 250 m")
    fs.panel(ax, "(b)")
    fs.legend_outside(fig, where="upper", ncol=3)
    fs.save(fig, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

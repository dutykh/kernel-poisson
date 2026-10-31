# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Summary of the benchmark: mean scores per data set and method, relative deviances against the
Poisson generalized linear model, corrected paired tests, ranks, and the figure and tables of the
article. Reads results/benchmark/NAME.csv and writes results/benchmark/summary*.csv,
figures/benchmark.pdf and the tables tables/benchmark*.tex.

The paired comparison of two methods on the J folds of a repeated K-fold design uses the corrected
resampled t statistic of Nadeau and Bengio (2003): with d_j the difference of mean test deviances on
fold j, t = mean(d) / sqrt((1/J + n_test/n_train) var(d)), referred to Student's law with J - 1
degrees of freedom. The single split of the French claims gives no such comparison.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figure_style as fs  # noqa: E402

RESULTS = HERE.parent / "results" / "benchmark"
ROOT = HERE.parent
ORDER = ["fishing", "solder", "fatalities", "biochemists", "abalone", "chicago", "eclipse", "bike",
         "datacar", "fremtpl2"]
LABEL = {"fishing": "deep-sea fish", "solder": "wave soldering", "fatalities": "traffic fatalities",
         "biochemists": "biochemists", "abalone": "abalone rings", "chicago": "Chicago mortality",
         "eclipse": "Eclipse defects", "bike": "bike rentals", "datacar": "Australian claims",
         "fremtpl2": "French claims"}
SHORT = {"fishing": "fish", "solder": "solder", "fatalities": "fatal.", "biochemists": "bioch.",
         "abalone": "abalone", "chicago": "Chicago", "eclipse": "Eclipse", "bike": "bike",
         "datacar": "AU", "fremtpl2": "FR"}
METHODS = ["glm_poisson", "glm_negbin", "gam_poisson", "gam_negbin", "gss", "ktweedie", "krr_log",
           "gp_laplace", "forest", "boosting", "network", "ours_poisson", "ours_negbin"]
MLABEL = {"glm_poisson": "GLM", "glm_negbin": "NB GLM", "gam_poisson": "GAM", "gam_negbin": "NB GAM",
          "gss": "SS-ANOVA", "ktweedie": "kernel Tweedie", "krr_log": "KRR log", "gp_laplace": "GP Laplace",
          "forest": "forest", "boosting": "boosting", "network": "network",
          "ours_poisson": "kernel Poisson", "ours_negbin": "kernel NB"}
SCORES = ["deviance", "log_score", "rps", "dss", "d2", "balance", "coverage90", "interval90", "gini"]


def load(name):
    path = RESULTS / f"{name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"]
    # A task run twice (two runners on one data set) keeps its first record.
    return df.drop_duplicates(subset=["method", "repeat", "fold", "law"], keep="first")


def corrected_t(d, n_train, n_test):
    d = np.asarray(d, dtype=float)
    J = len(d)
    if J < 2:
        return np.nan, np.nan
    var = np.var(d, ddof=1)
    if var == 0:
        return np.nan, np.nan
    t = d.mean() / np.sqrt((1.0 / J + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def summarise(df):
    """Mean scores per method and predictive law, and fold-level deviances (law-free)."""
    mean = df.groupby(["method", "law"])[SCORES + ["fit_seconds", "peak_rss_mb"]].mean().reset_index()
    count = df.groupby(["method", "law"]).size().rename("folds").reset_index()
    return mean.merge(count, on=["method", "law"])


def relative_deviance(df, ref="glm_poisson"):
    """Per fold: 100 (dev_ref - dev_method) / dev_ref, using the Poisson law rows (the deviance
    depends on the mean only)."""
    pois = df[df["law"] == "poisson"]
    key = ["repeat", "fold"]
    base = pois[pois["method"] == ref].set_index(key)["deviance"]
    out = []
    for m, g in pois.groupby("method"):
        g = g.set_index(key)
        common = g.index.intersection(base.index)
        rel = 100.0 * (base.loc[common] - g.loc[common, "deviance"]) / base.loc[common]
        out.append(pd.DataFrame(dict(method=m, rel=rel.values)))
    return pd.concat(out) if out else pd.DataFrame(columns=["method", "rel"])


MISSING = r"$\cdot$"


def fmt(x, digits=3):
    """Significant digits with trailing zeros kept; MISSING for a missing value."""
    if x is None or not np.isfinite(x):
        return MISSING
    if abs(x) >= 1e4:
        return f"{x:.2e}".replace("e+0", "e").replace("e+", "e")
    out = f"{x:#.{digits + 1}g}"
    return out.rstrip(".")


def fmt_seconds(x):
    """A time in seconds: three significant digits below a hundred, whole seconds above."""
    if x is None or not np.isfinite(x):
        return MISSING
    return f"{x:,.0f}".replace(",", "{,}") if x >= 100 else fmt(x)


def fmt_p(p):
    """A p-value with two significant digits, or an upper bound below one in a thousand."""
    if p is None or not np.isfinite(p):
        return MISSING
    return r"$<0.001$" if p < 1e-3 else f"{p:#.2g}"


def write_tables(summary, tests, path):
    """Main-text table: per data set, the deviance of the Poisson GLM, of the best competitor and of
    the two kernel estimators, the best negative-binomial log score among competitors and ours, and
    the corrected p-value of kernel NB against the best competitor. Tables of the supplementary material: every method."""
    pois = summary[summary["law"] == "poisson"].set_index(["dataset", "method"])
    nb = summary[summary["law"] == "negbin"].set_index(["dataset", "method"])
    ours = ("ours_poisson", "ours_negbin")
    lines = []
    for d in ORDER:
        if d not in pois.index.get_level_values(0):
            continue
        dev = pois.loc[d]["deviance"]
        comp = dev.drop([m for m in ours if m in dev.index])
        best = comp.idxmin()
        row = [LABEL[d], fmt(dev.get("glm_poisson", np.nan)), f"{MLABEL[best]} {fmt(comp[best])}",
               fmt(dev.get("ours_poisson", np.nan)), fmt(dev.get("ours_negbin", np.nan))]
        if d in nb.index.get_level_values(0):
            ls = nb.loc[d]["log_score"]
            lc = ls.drop([m for m in ours if m in ls.index])
            row += [fmt(lc.min()), fmt(ls.get("ours_negbin", np.nan))]
        else:
            row += [MISSING, MISSING]
        t = tests[(tests["dataset"] == d) & (tests["ours"] == "ours_negbin") & (tests["other"] == best)]
        row.append(fmt_p(float(t["p"].iloc[0])) if len(t) else MISSING)
        lines.append(" & ".join(row) + r" \\")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    head = ("\\begin{tabular}{@{}lrlrrrrr@{}}\n\\toprule\n"
            "data set & GLM & best competitor & KP & KNB & log score & KNB & $p$ \\\\\n\\midrule\n")
    Path(path).write_text(head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    # supplementary material: mean deviance, mean negative-binomial log score and mean fitting time of every
    # method on every data set
    for law, col, name in (("poisson", "deviance", "deviance"), ("negbin", "log_score", "logscore"),
                           ("poisson", "fit_seconds", "seconds")):
        piv = summary[summary["law"] == law].pivot(index="method", columns="dataset", values=col)
        piv = piv.reindex(columns=[c for c in ORDER if c in piv.columns])
        piv = piv.reindex([m for m in METHODS if m in piv.index])
        rows = []
        for m, r in piv.iterrows():
            cell = fmt_seconds if name == "seconds" else fmt
            rows.append(" & ".join([MLABEL[m]] + [cell(v) for v in r.values]) + r" \\")
        cols = list(piv.columns)
        head = ("\\begin{tabular}{@{}l" + "r" * len(cols) + "@{}}\n\\toprule\n"
                + " & ".join([""] + [SHORT[c] for c in cols]) + " \\\\\n\\midrule\n")
        Path(path).with_name(f"benchmark_{name}_all.tex").write_text(
            head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", default=str(ROOT / "figures" / "benchmark.pdf"))
    ap.add_argument("--table", default=str(ROOT / "tables" / "benchmark.tex"))
    a = ap.parse_args()
    rows, rel_all, tests = [], [], []
    for name in ORDER:
        df = load(name)
        if df is None or df.empty:
            continue
        s = summarise(df)
        s.insert(0, "dataset", name)
        rows.append(s)
        r = relative_deviance(df)
        r.insert(0, "dataset", name)
        rel_all.append(r)
        # corrected paired tests of each kernel estimator against every other method (deviance)
        pois = df[df["law"] == "poisson"]
        key = ["repeat", "fold"]
        n_train, n_test = float(pois["n_train"].mean()), float(pois["n_test"].mean())
        for ours in ("ours_poisson", "ours_negbin"):
            go = pois[pois["method"] == ours].set_index(key)["deviance"]
            for m in METHODS:
                if m == ours:
                    continue
                gm = pois[pois["method"] == m].set_index(key)["deviance"]
                common = go.index.intersection(gm.index)
                if len(common) == 0:
                    continue
                d = (gm.loc[common] - go.loc[common]).values       # positive when ours is better
                t, pval = corrected_t(d, n_train, n_test) if len(common) > 1 else (np.nan, np.nan)
                tests.append(dict(dataset=name, ours=ours, other=m, folds=len(common),
                                  mean_difference=float(d.mean()), t=t, p=pval))
    if not rows:
        print("no results yet")
        return
    summary = pd.concat(rows)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    rel = pd.concat(rel_all)
    rel.to_csv(RESULTS / "summary_relative_deviance.csv", index=False)
    pd.DataFrame(tests).to_csv(RESULTS / "summary_tests.csv", index=False)

    # ranks of the mean deviance (Poisson rows) and of the negative-binomial log score per data set
    piv_dev = summary[summary["law"] == "poisson"].pivot(index="dataset", columns="method", values="deviance")
    piv_log = summary[summary["law"] == "negbin"].pivot(index="dataset", columns="method", values="log_score")
    piv_dev = piv_dev.reindex([d for d in ORDER if d in piv_dev.index])
    piv_log = piv_log.reindex([d for d in ORDER if d in piv_log.index])
    piv_dev.to_csv(RESULTS / "summary_deviance.csv")
    piv_log.to_csv(RESULTS / "summary_logscore_nb.csv")
    print(piv_dev.round(4).to_string())
    print(piv_log.round(4).to_string())
    ranks = piv_dev.rank(axis=1)
    print("mean rank (deviance):")
    print(ranks.mean().sort_values().round(2).to_string())
    # ranks among the methods run on every data set, by deviance and by negative-binomial log score
    dev_all = [m for m in METHODS if m in piv_dev.columns and piv_dev[m].notna().all()]
    log_all = [m for m in METHODS if m in piv_log.columns and piv_log[m].notna().all()]
    rdev, rlog = piv_dev[dev_all].rank(axis=1), piv_log[log_all].rank(axis=1)
    rank_table = pd.DataFrame({
        "deviance": rdev.mean(), "first_by_deviance": (rdev == 1).sum(),
        "log_score_nb": rlog.mean(), "first_by_log_score_nb": (rlog == 1).sum(),
    }).sort_values("deviance")
    rank_table.index.name = "method"
    rank_table.to_csv(RESULTS / "summary_ranks.csv")
    print(f"mean ranks of the deviance among the {len(dev_all)} methods run on every data set, and of the")
    print(f"negative-binomial log score among the {len(log_all)} of them with a negative-binomial law:")
    print(rank_table.round(2).to_string())

    write_tables(summary, pd.DataFrame(tests), a.table)

    # figure: relative deviance against the Poisson GLM, per data set and method (medians and
    # quartiles over folds; medians beyond the axis range are drawn at its edge as arrows)
    fs.configure()
    styles = {
        "ours_negbin": dict(color=fs.NAVY, marker="s", fill=fs.NAVY),
        "ours_poisson": dict(color=fs.NAVY, marker="o", fill="white"),
        "gam_negbin": dict(color=fs.WINE, marker="D", fill="white"),
        "ktweedie": dict(color=fs.TEAL, marker="^", fill="white"),
        "forest": dict(color=fs.OCHRE, marker="v", fill="white"),
        "boosting": dict(color=fs.PLUM, marker="p", fill="white"),
        "network": dict(color=fs.LILAC, marker="h", fill="white"),
        "gp_laplace": dict(color=fs.SLATE, marker="X", fill="white"),
        "gss": dict(color=fs.TEAL, marker="d", fill=fs.TEAL),
        "krr_log": dict(color=fs.WINE, marker="*", fill="white"),
        "glm_poisson": dict(color=fs.INK, marker="P", fill="white"),
    }
    shown = list(styles)
    devs = summary[summary["law"] == "poisson"].pivot(index="dataset", columns="method", values="deviance")
    present = [d for d in ORDER if d in devs.index]
    xhi = 1.6
    xbeyond = xhi * 1.1
    fig, ax = plt.subplots(figsize=(fs.WIDTH, 0.42 * len(present) + 1.2), layout="constrained")
    off = np.linspace(-0.3, 0.3, len(shown))
    for k, m in enumerate(shown):
        st = styles[m]
        for i, dname in enumerate(present):
            if m not in devs.columns or not np.isfinite(devs.loc[dname, m]):
                continue
            ratio = devs.loc[dname, m] / np.nanmin(devs.loc[dname].to_numpy())
            y = i + off[k]
            # ratios beyond the axis range keep their marker, in a strip right of a dashed line
            x = xbeyond * np.exp(0.01 * (k - (len(shown) - 1) / 2)) if ratio > xhi else ratio
            ax.plot([x], [y], linestyle="none", marker=st["marker"], markersize=4.4,
                    markerfacecolor=st["fill"], markeredgecolor=st["color"])
    ax.axvline(1.0, color=fs.SLATE, linewidth=0.7)
    ax.axvline(xhi * 1.02, color=fs.SLATE, linewidth=0.6, linestyle=(0, (3, 2)))
    ax.set_xscale("log")
    ax.set_xlim(0.99, xbeyond * 1.075)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ticks = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, xbeyond]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.1f}" for t in ticks[:-1]] + [f"$>{xhi:.1f}$"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels([LABEL[d] for d in present])
    ax.invert_yaxis()
    ax.set_xlabel("mean test deviance relative to the best method on the data set")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], linestyle="none", marker=styles[m]["marker"], markersize=4.4,
                      markerfacecolor=styles[m]["fill"], markeredgecolor=styles[m]["color"],
                      label=MLABEL[m]) for m in shown]
    fs.legend_outside(fig, where="upper", ncol=6, handles=handles, labels=[h.get_label() for h in handles])
    Path(a.figure).parent.mkdir(parents=True, exist_ok=True)
    fs.save(fig, a.figure)
    print(f"wrote {a.figure}")


if __name__ == "__main__":
    main()

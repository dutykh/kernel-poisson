#!/usr/bin/env python3
# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Number of landmarks against out-of-sample accuracy of the kernel Poisson estimator.

In the benchmark the kernel Poisson estimator is restricted, above 6000 training rows, to the span
of 1500 landmarks. On the hourly bike rentals it selected the shortest length scale of its grid on 24 of
its 25 folds, with a median of 1365 effective parameters, close to the 1500 that the landmarks
allow. This experiment refits it on the folds of the first repetition of the benchmark design with
m = 1500, 3000 and 6000 landmarks, the length-scale factors 0.25 and 0.5 of the median heuristic
and the penalties per event 10^-1 down to 10^-9 (three per decade), everything else as in the
benchmark (Fisher-weighted leverage landmarks from a pilot fit on 500 uniform landmarks, path
stopped three penalties after the minimum of the leave-one-out deviance), and records the test
deviance with the selected tuning and the effective number of parameters.

Usage, from the root folder of the repository:
    python -I experiments/landmark_budget.py [--data bike] [--landmarks 1500 3000 6000]
                                             [--folds 0 1 2 3 4] [--jobs J --threads T]
    python -I experiments/landmark_budget.py --table tables/landmark_budget.tex
Results: results/landmark_budget/NAME.csv, one row per (landmarks, fold); finished rows are kept
and not run again. With --table, nothing is fitted: the table of the supplementary material is
written from the results, with the deviances of the forest, the boosted trees and the network of
the benchmark on the same folds.
"""
from __future__ import annotations

import os
import sys


def _threads_from_argv(argv, default=2):
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            return int(argv[i + 1])
    return default


_T = str(max(1, _threads_from_argv(sys.argv)))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T

import argparse  # noqa: E402
import csv  # noqa: E402
import fcntl  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.common import as_frame, exposure, factor_names  # noqa: E402
from data import datasets  # noqa: E402
from experiments.benchmark import BASE_SEED, default_design, folds  # noqa: E402

RESULTS = ROOT / "results" / "landmark_budget"
COLUMNS = ["dataset", "landmarks", "repeat", "fold", "n_train", "n_test", "deviance", "d2",
           "lengthscale_factor", "t", "df", "alo", "fit_seconds"]
FACTORS = (0.25, 0.5)
T_GRID = np.logspace(-1, -9, 25)


def run_one(name, m, fold):
    from kernelpoisson import scores
    from kernelpoisson.estimator import KernelPoissonRegressor, median_lengthscale

    d = datasets.load(name)
    K, R = default_design(name, len(d["y"]))
    r, f, tr, te = [s for s in folds(d, K, 1) if s[1] == fold][0]
    Xtr = as_frame(d["X"].iloc[tr].reset_index(drop=True))
    Xte = as_frame(d["X"].iloc[te].reset_index(drop=True))
    idx = tuple(Xtr.columns.get_loc(c) for c in factor_names(Xtr, list(d["categorical"])))
    dtype = object if idx else float
    ytr, yte = d["y"][tr].astype(float), d["y"][te].astype(float)
    etr = exposure(None if d["exposure"] is None else d["exposure"][tr], len(tr))
    ete = exposure(None if d["exposure"] is None else d["exposure"][te], len(te))
    est = KernelPoissonRegressor(family="poisson", categorical=idx, seed=BASE_SEED + r,
                                 landmarks=m, lengthscale_factors=FACTORS, lam_rel=T_GRID)
    start = time.perf_counter()
    est.fit(Xtr.to_numpy(dtype=dtype), ytr, etr)
    mu = est.predict(Xte.to_numpy(dtype=dtype), ete)
    seconds = time.perf_counter() - start
    null_mu = ete * float(np.sum(ytr) / np.sum(etr))
    s = scores.summary(yte, mu, ete, None, null_mu)
    ell0 = median_lengthscale(est.Z_, seed=BASE_SEED + r)
    df = [q["df"] for q in est.path_ if q["lam"] == est.lam_ and q["lengthscale"] == est.lengthscale_]
    return dict(dataset=name, landmarks=m, repeat=r, fold=f, n_train=len(tr), n_test=len(te),
                deviance=s["deviance"], d2=s["d2"], lengthscale_factor=est.lengthscale_ / ell0,
                t=est.lam_ / float(np.sum(ytr)), df=df[0] if df else "", alo=est.best_["alo"],
                fit_seconds=round(seconds, 1))


def sci(x):
    """A positive number as $a\\cdot10^{b}$ with one significant digit, or $10^{b}$ when a is one."""
    b = int(np.floor(np.log10(x)))
    a = int(round(x / 10.0 ** b))
    if a == 10:
        a, b = 1, b + 1
    return f"$10^{{{b}}}$" if a == 1 else f"${a}\\cdot10^{{{b}}}$"


def thousands(n):
    """An integer with thin thousands separators, as in the text of the article."""
    return f"{int(n):,}".replace(",", "{,}")


def benchmark_lengthscale0(name, fold):
    """The median-heuristic length scale of the training part of a fold of the first repetition."""
    from kernelpoisson.estimator import Encoder, median_lengthscale

    d = datasets.load(name)
    K, R = default_design(name, len(d["y"]))
    r, f, tr, te = [s for s in folds(d, K, 1) if s[1] == fold][0]
    Xtr = as_frame(d["X"].iloc[tr].reset_index(drop=True))
    idx = tuple(Xtr.columns.get_loc(c) for c in factor_names(Xtr, list(d["categorical"])))
    A = Xtr.to_numpy(dtype=object if idx else float)
    return median_lengthscale(Encoder(idx).fit(A).transform(A), seed=BASE_SEED + r)


def write_table(name, out, table):
    """Mean test deviance, leave-one-out deviance at the selected point, effective number of
    parameters and selected tuning per number of landmarks, over the folds present for every number,
    and the competitors on the same folds."""
    import pandas as pd
    res = pd.read_csv(out)
    folds_all = sorted(set.intersection(*[set(g["fold"]) for _, g in res.groupby("landmarks")]))
    res = res[res["fold"].isin(folds_all)]
    bench = pd.read_csv(ROOT / "results" / "benchmark" / f"{name}.csv")
    bench = bench[(bench["status"] == "ok") & (bench["law"] == "poisson") & (bench["repeat"] == 0)
                  & (bench["fold"].isin(folds_all))]
    lines = []
    for m, g in res.groupby("landmarks"):
        lines.append(f"$m={thousands(m)}$ & {g['deviance'].mean():.2f} & {g['alo'].mean():.2f} & "
                     + f"{thousands(round(g['df'].mean()))} & {g['lengthscale_factor'].median():g} & "
                     + f"{sci(g['t'].median())} \\\\")
    labels = {"ours_poisson": "benchmark fit, $m=1{,}500$", "forest": "random forest",
              "boosting": "gradient-boosted trees", "network": "neural network"}
    for method, label in labels.items():
        g = bench[bench["method"] == method]
        if not len(g):
            continue
        if method == "ours_poisson":
            info = [json.loads(v) for v in g["info"]]
            factors = [q["lengthscale"] / benchmark_lengthscale0(name, int(f))
                       for q, f in zip(info, g["fold"])]
            lines.append(f"{label} & {g['deviance'].mean():.2f} & {np.mean([q['alo'] for q in info]):.2f} & "
                         + f"{thousands(round(np.mean([q['df'] for q in info])))} & "
                         + f"{np.median(factors):.2g} & "
                         + f"{sci(np.median([q['lam'] / q['events'] for q in info]))} \\\\")
        else:
            lines.append(f"{label} & {g['deviance'].mean():.2f} & & & & \\\\")
    head = ("\\begin{tabular}{@{}lrrrrr@{}}\n\\toprule\n"
            "& test & leave-one-out & parameters & length scale & $t$ \\\\\n\\midrule\n")
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    print(f"wrote {table} ({len(folds_all)} folds)")


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0, os.SEEK_END)
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            if fh.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
            fh.flush()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", default="bike")
    p.add_argument("--landmarks", type=int, nargs="+", default=[1500, 3000, 6000])
    p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--table", default=None)
    a = p.parse_args(argv)
    out = RESULTS / f"{a.data}.csv"
    if a.table:
        write_table(a.data, out, Path(a.table))
        return
    done = set()
    if out.exists():
        with open(out, newline="") as fh:
            done = {(int(r["landmarks"]), int(r["fold"])) for r in csv.DictReader(fh)}
    # the largest landmark sets first, so that the longest fits start early
    tasks = [(m, f) for m in sorted(a.landmarks, reverse=True) for f in a.folds if (m, f) not in done]
    print(f"{a.data}: {len(tasks)} fits to run ({len(done)} done) -> {out}", flush=True)
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        futures = {pool.submit(run_one, a.data, m, f): (m, f) for m, f in tasks}
        for fut in as_completed(futures):
            m, f = futures[fut]
            row = fut.result()
            append_row(out, row)
            print(f"  m={m} fold {f}: deviance {row['deviance']:.4f}, factor "
                  f"{row['lengthscale_factor']:.2f}, t {row['t']:.1e}, df {row['df']:.1f}, "
                  f"{row['fit_seconds']} s", flush=True)


if __name__ == "__main__":
    main()

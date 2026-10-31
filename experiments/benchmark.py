#!/usr/bin/env python3
# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Out-of-sample comparison of count-regression methods on the open data sets.

Every method of baselines.METHODS is fitted on the training part of each fold and scored on its
test part with kernelpoisson.scores.summary: once with the Poisson predictive law of the predicted
means, and once with a negative-binomial law when a dispersion is available (the model's own
kappa for the negative-binomial models, the moment estimate kappa_nb for the mean-only methods).
The null predictions of the deviance ratio d2 are those of the intercept-plus-offset model of the
training fold, mu0 = e sum(y_train) / sum(e_train).

Folds. Repeated K-fold cross-validation, the r-th repetition shuffled with the seed
20261018 + r and identical for all methods; grouped K-fold (GroupKFold with a shuffled
assignment of the groups) when the data set defines groups (states, years, days); the fixed
split of the scikit-learn example for freMTPL2. Defaults: 10 x 5 below 2000 rows, 5 x 5 up to
20000 rows, 5 x 2 above. --smoke runs the first fold of a 5-fold split of a 500-row subsample
(--rows to change it) of every data set, for every method, and writes to
results/benchmark/smoke/ (results/benchmark/smokeN/ for N rows).

The ranked probability score is the sum over k of (F(k) - 1{y ⩽ k})^2. The reference
implementation sums it over 0 ⩽ k ⩽ the largest 1 - 1e-12 quantile of all test rows at once,
which never ends when a method predicts an absurdly large mean for one row (a smooth
extrapolated far outside the training range). The scores are therefore computed with the
reference implementation whenever that sum is affordable, and otherwise row by row: exactly
inside the window between the 1e-12 and 1 - 1e-12 quantiles, adding one for each k between y
and the lower end of the window (where F(k) < 1e-12) and for each k between the upper end and y;
when even the window is too wide, by the identity RPS = E|Y - y| - (1/2) E|Y - Y'|, with
E|Y - y| = mu - y + 2 sum_{k < y} (y - k) P(Y = k) summed exactly, and E|Y - Y'| equal to
2 mu e^{-2 mu} (I0(2 mu) + I1(2 mu)) for the Poisson law and approximated by the gamma limit
2 kappa mu Gamma(1/kappa + 1/2) / (sqrt(pi) Gamma(1/kappa)) for the negative binomial. The
number of rows scored by the identity is recorded in the message column. The logarithmic score
of the negative-binomial law is computed from the closed form
log P(Y = y) = lgamma(y + r) - lgamma(r) - lgamma(y + 1) - r log(1 + mu/r) + y log(mu/(r + mu)),
r = 1/kappa, which stays finite for the tiny means of nearly separated fits, where the
parametrisation p = r/(r + mu) rounds to one.

Each (data set, method, repetition, fold) is one task, run in a worker process of its own (this
script started again with --worker; at most --jobs at a time, each with --threads BLAS/OpenMP
threads, jobs x threads at most 16, all at nice 15), timed by the wall clock and measured by its
peak resident memory (the larger of the worker and of its R or PyTorch subprocesses). A task
that exceeds --timeout seconds is stopped and recorded as failed. Rows are appended to
results/benchmark/NAME.csv under a file lock as tasks finish; tasks already present with status
ok or skipped are not run again, so an interrupted run resumes where it stopped.

Usage, from the root folder of the repository:
    python -I experiments/benchmark.py --data NAME [--methods M ...] [--folds K --repeats R]
                                       [--jobs J --threads T] [--timeout S] [--smoke]
                                       [--only R:F ...]
NAME is one data set of data.datasets.NAMES, a comma-separated list, or all.
"""
from __future__ import annotations

import os
import sys


def _threads_from_argv(argv, default=4):
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith("--threads="):
            return int(a.split("=", 1)[1])
    return default


# the BLAS and OpenMP pools read these when numpy is first imported, here and in every worker
_T = str(max(1, _threads_from_argv(sys.argv)))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _T

import argparse  # noqa: E402
import csv  # noqa: E402
import fcntl  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import baselines  # noqa: E402
from data import datasets  # noqa: E402

BASE_SEED = 20261018
SMOKE_ROWS = 500
MAX_THREADS = 16
SMOKE_TIMEOUT = 900.0
RESULTS = ROOT / "results" / "benchmark"
SCORE_KEYS = ["deviance", "log_score", "rps", "dss", "rmse", "mae", "balance", "coverage90",
              "interval90", "d2", "gini"]
COLUMNS = (["dataset", "method", "law", "repeat", "fold", "n_train", "n_test", "status",
            "fit_seconds", "peak_rss_mb", "kappa", "kappa_source"] + SCORE_KEYS
           + ["finished", "message", "info"])
# rough cost order: the slowest methods are started first
COST_ORDER = ["forest", "ktweedie", "gss", "gp_laplace", "boosting", "network", "krr_log",
              "ours_negbin", "gam_negbin", "ours_poisson", "gam_poisson", "glm_negbin",
              "glm_poisson"]


# ================================================================================================
def default_design(name, n):
    """(K, R) of the repeated K-fold design, or None for the fixed split."""
    if name == "fremtpl2":
        return None
    if n < 2000:
        return 10, 5
    if n <= 20000:
        return 5, 5
    return 5, 2


def folds(d, K, R):
    """(repeat, fold, train indices, test indices) of the design."""
    from sklearn.model_selection import GroupKFold, KFold
    n = len(d["y"])
    for r in range(R):
        seed = BASE_SEED + r
        if d["groups"] is None:
            split = KFold(n_splits=K, shuffle=True, random_state=seed).split(np.arange(n))
        else:
            split = GroupKFold(n_splits=K, shuffle=True, random_state=seed).split(
                np.arange(n), groups=d["groups"])
        for f, (tr, te) in enumerate(split):
            yield r, f, tr, te


def subsample(d, m, seed=BASE_SEED):
    """The data set restricted to m rows drawn without replacement (all rows if fewer)."""
    n = len(d["y"])
    if n <= m:
        return d
    rows = np.sort(np.random.default_rng(seed).choice(n, size=m, replace=False))
    out = dict(d)
    out["X"] = d["X"].iloc[rows].reset_index(drop=True)
    out["y"] = d["y"][rows]
    for k in ("exposure", "groups"):
        out[k] = None if d[k] is None else d[k][rows]
    return out


# ================================================================================================
RPS_REFERENCE_LIMIT = 2.0e7      # rows x quantile range summed by the reference implementation
RPS_WINDOW_LIMIT = 2.0e5         # widest quantile window summed exactly for one row
RPS_TAIL = 1e-12


def _law(mu, kappa):
    from scipy import stats
    mu = max(float(mu), 1e-300)
    if not kappa:
        return stats.poisson(mu)
    r = 1.0 / kappa
    return stats.nbinom(r, r / (r + mu))


def _rps_one(y, mu, kappa):
    """Ranked probability score of one row; returns (value, True if by the identity)."""
    from scipy.special import gammaln, i0e, i1e
    d = _law(mu, kappa)
    y = int(round(y))
    lo, hi = int(d.ppf(RPS_TAIL)), int(d.ppf(1.0 - RPS_TAIL))
    if hi - lo <= RPS_WINDOW_LIMIT:
        total = 0.0
        if y > 0:                                   # k < y: F(k)^2, negligible below lo
            k = np.arange(max(0, lo), min(y - 1, hi) + 1)
            total += float(np.sum(d.cdf(k) ** 2))
            total += max(0, y - 1 - hi)             # F(k) = 1 above hi
        k = np.arange(max(y, lo), hi + 1)           # k ⩾ y: (1 - F(k))^2, negligible above hi
        total += float(np.sum(d.sf(k) ** 2))
        total += max(0, lo - y)                     # 1 - F(k) = 1 below lo
        return total, False
    k = np.arange(0, y)
    mad_y = mu - y + 2.0 * float(np.sum((y - k) * d.pmf(k))) if y > 0 else mu
    if not kappa:
        mad = 2.0 * mu * (i0e(2.0 * mu) + i1e(2.0 * mu))
    else:
        a = 1.0 / kappa
        mad = 2.0 * kappa * mu * np.exp(gammaln(a + 0.5) - gammaln(a)) / np.sqrt(np.pi)
    return mad_y - 0.5 * mad, True


class GuardedRPS:
    """Drop-in for kernelpoisson.scores.ranked_probability that falls back on row-by-row
    evaluation when the reference sum is unaffordable; counts the rows scored by the identity."""

    def __init__(self, reference):
        self.reference = reference
        self.by_identity = 0

    def __call__(self, y, mu, kappa=None, tail=RPS_TAIL):
        from scipy import stats
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        m = np.maximum(mu, 1e-300)
        if kappa:
            r = 1.0 / kappa
            upper = stats.nbinom(r, r / (r + m)).ppf(1.0 - tail)
        else:
            upper = stats.poisson(m).ppf(1.0 - tail)
        reach = np.maximum(upper, y) + 1.0
        if len(y) * float(np.max(reach)) <= RPS_REFERENCE_LIMIT:
            return self.reference(y, mu, kappa, tail)
        # the rows of smallest reach go to the reference sum as long as it stays affordable
        order = np.argsort(reach, kind="mergesort")
        cost = np.arange(1, len(y) + 1) * reach[order]
        n_ref = int(np.searchsorted(cost, RPS_REFERENCE_LIMIT, side="right"))
        out = np.empty(len(y))
        if n_ref:
            ref = order[:n_ref]
            out[ref] = self.reference(y[ref], mu[ref], kappa, tail)
        for i in order[n_ref:]:
            out[i], approx = _rps_one(y[i], mu[i], kappa)
            self.by_identity += int(approx)
        return out


def stable_log_score(reference):
    """Logarithmic score -log P(Y = y): the reference for the Poisson law, the closed form above
    for the negative binomial."""
    from scipy.special import gammaln

    def log_score(y, mu, kappa=None):
        if kappa is None or kappa == 0:
            return reference(y, mu, kappa)
        y = np.asarray(y, dtype=float)
        mu = np.maximum(np.asarray(mu, dtype=float), 1e-300)
        r = 1.0 / kappa
        ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0) - r * np.log1p(mu / r)
              + y * (np.log(mu) - np.log(r + mu)))
        return -ll

    log_score.stable = True
    return log_score


def _peak_rss_mb():
    """Peak resident memory of the task: the high-water mark of this process's own address space
    (VmHWM; getrusage would also count the parent's memory, which survives exec in ru_maxrss)
    and the largest peak among its finished R or PyTorch subprocesses."""
    own = None
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    own = float(line.split()[1])
                    break
    except OSError:
        pass
    if own is None:
        own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return max(own, children) / 1024.0


def _row(task, **fields):
    row = {c: "" for c in COLUMNS}
    row.update(dataset=task["dataset"], method=task["method"], repeat=task["repeat"],
               fold=task["fold"], n_train=len(task["ytr"]), n_test=len(task["yte"]),
               finished=datetime.now().isoformat(timespec="seconds"))
    row.update(fields)
    return row


def run_task(task):
    """Fit, predict and score one (data set, method, repetition, fold); returns CSV rows."""
    from kernelpoisson import scores
    if not isinstance(scores.ranked_probability, GuardedRPS):
        scores.ranked_probability = GuardedRPS(scores.ranked_probability)
    if not getattr(scores.log_score, "stable", False):
        scores.log_score = stable_log_score(scores.log_score)
    fn, kwargs = baselines.get(task["method"])
    t0 = time.perf_counter()
    try:
        out = fn(task["Xtr"], task["ytr"], task["etr"], task["Xte"], task["ete"],
                 task["categorical"], task["seed"], task["threads"], **kwargs)
        status, message = ("ok", "") if out is not None else ("skipped", "not run at this size")
    except Exception as exc:
        out, status = None, "failed"
        message = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"[-4000:]
    seconds = time.perf_counter() - t0
    rss = _peak_rss_mb()
    if status != "ok":
        return [_row(task, law="poisson", status=status, fit_seconds=round(seconds, 3),
                     peak_rss_mb=round(rss, 1), message=message)]
    yte = task["yte"]
    mu = np.asarray(out["mu"], dtype=float)
    if mu.shape != yte.shape or not np.all(np.isfinite(mu)) or np.any(mu <= 0.0):
        return [_row(task, law="poisson", status="failed", fit_seconds=round(seconds, 3),
                     peak_rss_mb=round(rss, 1),
                     message="predicted means are not positive and finite for every test row")]
    etr = np.ones(len(task["ytr"])) if task["etr"] is None else task["etr"]
    ete = np.ones(len(yte)) if task["ete"] is None else task["ete"]
    null_mu = ete * float(np.sum(task["ytr"]) / np.sum(etr))
    laws = [("poisson", None, "")]
    if out.get("kappa") is not None:
        laws.append(("negbin", float(out["kappa"]), "model"))
    elif out.get("kappa_nb") is not None:
        laws.append(("negbin", float(out["kappa_nb"]), "moment"))
    info = json.dumps(out.get("info", {}), default=str)
    rows = []
    for law, kappa, source in laws:
        before = scores.ranked_probability.by_identity
        s = scores.summary(yte, mu, ete, kappa, null_mu)
        approx = scores.ranked_probability.by_identity - before
        note = f"rps of {approx} of {len(yte)} rows by the identity" if approx else ""
        rows.append(_row(task, law=law, status="ok", fit_seconds=round(seconds, 3),
                         peak_rss_mb=round(rss, 1), kappa="" if kappa is None else kappa,
                         kappa_source=source, info=info, message=note,
                         **{k: s.get(k, "") for k in SCORE_KEYS}))
    return rows


# ================================================================================================
def append_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0, os.SEEK_END)
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            if fh.tell() == 0:
                writer.writeheader()
            writer.writerows(rows)
            fh.flush()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def done_tasks(path):
    if not path.exists():
        return set()
    with open(path, newline="") as fh:
        return {(r["method"], int(r["repeat"]), int(r["fold"])) for r in csv.DictReader(fh)
                if r["status"] in ("ok", "skipped")}


def _renice(target=15):
    try:
        current = os.nice(0)
        if current < target:
            os.nice(target - current)
    except OSError:
        pass


def worker_main(task_path, rows_path):
    """Run one pickled task and pickle its rows (the --worker mode of this script)."""
    _renice()
    with open(task_path, "rb") as fh:
        task = pickle.load(fh)
    rows = run_task(task)
    tmp = f"{rows_path}.tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(rows, fh)
    os.replace(tmp, rows_path)


def _tail(path, n=2000):
    try:
        return Path(path).read_text(errors="replace")[-n:]
    except OSError:
        return ""


def run_tasks(tasks, out_csv, jobs, threads, timeout, log=print):
    """Run the tasks in at most `jobs` worker processes; append their rows as they finish."""
    with tempfile.TemporaryDirectory(prefix="benchmark-") as tmp:
        folder = Path(tmp)
        pending, running, counter = list(tasks), [], 0
        while pending or running:
            while pending and len(running) < jobs:
                t = pending.pop(0)
                counter += 1
                tp, rp, lp = (folder / f"{kind}{counter}" for kind in ("task", "rows", "log"))
                with open(tp, "wb") as fh:
                    pickle.dump(t, fh)
                with open(lp, "w") as logfh:
                    proc = subprocess.Popen(
                        [sys.executable, "-I", str(Path(__file__).resolve()), "--worker", str(tp),
                         str(rp), "--threads", str(threads)],
                        stdout=logfh, stderr=subprocess.STDOUT, cwd=str(ROOT))
                running.append(dict(proc=proc, task=t, paths=(tp, rp, lp), start=time.time()))
            time.sleep(0.2)
            for entry in list(running):
                proc, t, (tp, rp, lp) = entry["proc"], entry["task"], entry["paths"]
                elapsed = time.time() - entry["start"]
                status = proc.poll()
                if status is None and not (timeout and elapsed > timeout):
                    continue
                if status is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    rows = [_row(t, law="poisson", status="failed", fit_seconds=round(elapsed, 1),
                                 message=f"stopped after the time limit of {timeout:g} s")]
                elif rp.exists():
                    with open(rp, "rb") as fh:
                        rows = pickle.load(fh)
                else:
                    rows = [_row(t, law="poisson", status="failed", fit_seconds=round(elapsed, 1),
                                 message=f"worker exited with status {status}: {_tail(lp)}")]
                append_rows(out_csv, rows)
                running.remove(entry)
                for path in (tp, rp, lp):
                    Path(path).unlink(missing_ok=True)
                first = rows[0]
                dev = first.get("deviance", "")
                dev = f"{dev:.4f}" if isinstance(dev, float) else "-"
                msg = first["message"].splitlines()[0][:160] if first["message"] else ""
                log(f"  {t['dataset']:11s} {t['method']:13s} r{t['repeat']} f{t['fold']} "
                    f"{first['status']:7s} {first['fit_seconds']}s {first['peak_rss_mb']}MB "
                    f"deviance {dev}" + (f"  [{msg}]" if msg else ""))


def run_dataset(name, methods, K, R, jobs, threads, smoke, timeout=None, log=print,
                rows=SMOKE_ROWS, only=None):
    d = datasets.load(name)
    if smoke:
        d = subsample(d, rows)
        design = [next(folds(d, 5, 1))]
        tag = "smoke" if rows == SMOKE_ROWS else f"smoke{rows}"
        out_csv = RESULTS / tag / f"{name}.csv"
        if out_csv.exists():
            out_csv.unlink()
        timeout = timeout or SMOKE_TIMEOUT
    else:
        out_csv = RESULTS / f"{name}.csv"
        plan = default_design(name, len(d["y"]))
        if plan is None and K is None and R is None:
            tr, te = datasets.fremtpl2_split(d)
            design = [(0, 0, tr, te)]
        else:
            k0, r0 = plan if plan is not None else (5, 1)
            design = list(folds(d, K or k0, R or r0))
    done = done_tasks(out_csv)
    ordered = sorted(methods, key=lambda m: COST_ORDER.index(m) if m in COST_ORDER else -1)
    tasks = []
    for method in ordered:
        for r, f, tr, te in design:
            if (method, r, f) in done or (only is not None and (r, f) not in only):
                continue
            tasks.append(dict(dataset=name, method=method, repeat=r, fold=f,
                              Xtr=d["X"].iloc[tr].reset_index(drop=True), ytr=d["y"][tr],
                              etr=None if d["exposure"] is None else d["exposure"][tr],
                              Xte=d["X"].iloc[te].reset_index(drop=True), yte=d["y"][te],
                              ete=None if d["exposure"] is None else d["exposure"][te],
                              categorical=list(d["categorical"]), seed=BASE_SEED + r,
                              threads=threads))
    log(f"{name}: {len(d['y'])} rows, {len(design)} folds, {len(tasks)} tasks to run "
        f"({len(done)} done) -> {out_csv}")
    if tasks:
        run_tasks(tasks, out_csv, jobs, threads, timeout, log=log)
    return out_csv


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data")
    p.add_argument("--methods", nargs="+", default=None, choices=list(baselines.METHODS))
    p.add_argument("--folds", type=int, default=None)
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--timeout", type=float, default=None,
                   help="seconds after which a task is stopped (default: none; 900 with --smoke)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--rows", type=int, default=SMOKE_ROWS,
                   help="size of the subsample of a smoke run (default 500)")
    p.add_argument("--only", nargs="+", default=None, metavar="R:F",
                   help="run only these (repetition, fold) pairs, written as R:F")
    p.add_argument("--worker", nargs=2, metavar=("TASK", "ROWS"), help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.worker:
        worker_main(*a.worker)
        return
    if a.jobs < 1 or a.threads < 1 or a.jobs * a.threads > MAX_THREADS:
        p.error(f"jobs x threads must lie between 1 and {MAX_THREADS}")
    _renice()
    if not a.data:
        p.error("--data is required")
    names = list(datasets.NAMES) if a.data == "all" else a.data.split(",")
    for nm in names:
        if nm not in datasets.NAMES:
            p.error(f"unknown data set {nm}")
    methods = a.methods or list(baselines.METHODS)
    only = None if a.only is None else {tuple(int(v) for v in x.split(":")) for x in a.only}
    for nm in names:
        run_dataset(nm, methods, a.folds, a.repeats, a.jobs, a.threads, a.smoke, a.timeout,
                    log=lambda s: print(s, flush=True), rows=a.rows, only=only)


if __name__ == "__main__":
    main()

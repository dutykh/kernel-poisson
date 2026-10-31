# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Preprocessing shared by the competitor methods, fitted on the training fold only.

Continuous covariates are standardised by the training mean and standard deviation. The levels of
a factor are those seen in the training fold; a level of new data that the training fold does not
contain is replaced by the most frequent training level, which is also the reference level of
the treatment coding (so that it receives the all-zero indicator row). Columns that are constant
on the training fold carry no information there and are dropped.

The dispersion of a negative-binomial law with given means mu_i, Var Y_i = mu_i + kappa mu_i^2,
is estimated by the moment equation sum_i (y_i - mu_i)^2 = sum_i (mu_i + kappa mu_i^2), that is
kappa = sum_i ((y_i - mu_i)^2 - mu_i) / sum_i mu_i^2, floored at 1e-6.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R_DIR = Path(__file__).resolve().parent / "r"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernelpoisson import scores  # noqa: E402

KAPPA_FLOOR = 1e-6


def exposure(e, n):
    """Exposures as a float array (ones when e is None)."""
    return np.ones(n) if e is None else np.asarray(e, dtype=float)


def as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True)
    X = np.asarray(X)
    return pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])])


def factor_names(X: pd.DataFrame, categorical) -> list:
    """Names of the factor columns, given as names or as positions."""
    out = []
    for c in categorical or ():
        out.append(X.columns[c] if isinstance(c, (int, np.integer)) else c)
    return [c for c in X.columns if c in set(out)]


class Preprocessor:
    """Training-fold standardisation of continuous columns and coding of factors."""

    def fit(self, X, categorical=()):
        X = as_frame(X)
        factors = factor_names(X, categorical)
        self.continuous_, self.mean_, self.scale_ = [], {}, {}
        for c in X.columns:
            if c in factors:
                continue
            v = X[c].to_numpy(dtype=float)
            sd = float(np.std(v))
            if sd > 0.0:
                self.continuous_.append(c)
                self.mean_[c], self.scale_[c] = float(np.mean(v)), sd
        self.factors_, self.levels_, self.mode_ = [], {}, {}
        for c in factors:
            counts = X[c].astype(str).value_counts()
            if len(counts) < 2:
                continue
            self.factors_.append(c)
            self.mode_[c] = str(counts.index[0])
            # reference (most frequent) level first, then the others in sorted order
            self.levels_[c] = [self.mode_[c]] + sorted(lv for lv in counts.index if lv != self.mode_[c])
        self.min_ = {c: float(X[c].astype(float).min()) for c in self.continuous_}
        self.max_ = {c: float(X[c].astype(float).max()) for c in self.continuous_}
        return self

    @property
    def columns(self):
        return self.continuous_ + self.factors_

    def clean(self, X, clip=False) -> pd.DataFrame:
        """Kept columns: continuous as floats (clipped to the training range if clip), factors
        as strings with unseen levels replaced by the training mode."""
        X = as_frame(X)
        out = pd.DataFrame(index=range(len(X)))
        for c in self.continuous_:
            v = X[c].to_numpy(dtype=float)
            out[c] = np.clip(v, self.min_[c], self.max_[c]) if clip else v
        for c in self.factors_:
            v = X[c].astype(str).to_numpy(dtype=object)
            known = np.isin(v, self.levels_[c])
            v[~known] = self.mode_[c]
            out[c] = v
        return out

    def matrix(self, X, coding="treatment", factor_scale=1.0):
        """Numeric design: standardised continuous columns, then indicator columns of the factor
        levels ("treatment": all levels but the reference; "full": every level), multiplied by
        factor_scale. Returns (matrix, column names)."""
        Xc = self.clean(X)
        blocks, names = [], []
        if self.continuous_:
            A = np.column_stack([(Xc[c].to_numpy(float) - self.mean_[c]) / self.scale_[c]
                                 for c in self.continuous_])
            blocks.append(A)
            names += list(self.continuous_)
        for c in self.factors_:
            levels = self.levels_[c][1:] if coding == "treatment" else self.levels_[c]
            v = Xc[c].to_numpy(dtype=object)
            B = np.column_stack([(v == lv).astype(float) for lv in levels]) * factor_scale
            blocks.append(B)
            names += [f"{c}={lv}" for lv in levels]
        A = np.hstack(blocks) if blocks else np.zeros((len(Xc), 0))
        return A, names


def independent_columns(A, tol=1e-9):
    """Indices (in increasing order) of a maximal set of linearly independent columns of A,
    found by a QR factorisation with column pivoting; the first column (the intercept) is
    always kept."""
    import scipy.linalg as sla
    if A.shape[1] == 0:
        return np.arange(0)
    norms = np.linalg.norm(A, axis=0)
    norms[norms == 0] = 1.0
    _, R, piv = sla.qr(A / norms, mode="economic", pivoting=True)
    d = np.abs(np.diag(R))
    rank = int(np.sum(d > tol * d[0])) if len(d) else 0
    keep = set(piv[:rank].tolist()) | {0}
    return np.array(sorted(keep))


def kappa_moment(y, mu):
    """Moment estimate of the negative-binomial dispersion given the means mu."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    k = float(np.sum((y - mu) ** 2 - mu) / max(float(np.sum(mu ** 2)), 1e-300))
    return max(k, KAPPA_FLOOR)


def mean_deviance(y, mu):
    return float(np.mean(scores.deviance(y, mu)))


def kfold(n, k, seed):
    """Shuffled k-fold split of range(n) as a list of (train, validation) index arrays."""
    from sklearn.model_selection import KFold
    return list(KFold(n_splits=k, shuffle=True, random_state=seed).split(np.arange(n)))


def thread_env(n_threads):
    n = str(max(1, int(n_threads)))
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[v] = n
    return env


def run_rscript(script, args, n_threads, cwd, timeout=None):
    """Run an R script with --vanilla in folder cwd; RuntimeError with R's messages on failure."""
    cmd = ["Rscript", "--vanilla", str(R_DIR / script), *map(str, args)]
    out = subprocess.run(cmd, cwd=str(cwd), env=thread_env(n_threads), capture_output=True,
                         text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"{script} failed:\n{out.stderr.strip()[-3000:]}")
    return out.stdout


def read_r_info(path):
    """Key-value pairs written by an R script as a two-column CSV file."""
    t = pd.read_csv(path)
    out = {}
    for k, v in zip(t["key"], t["value"]):
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            out[str(k)] = str(v)
    return out


def threadpool(n_threads):
    """Context limiting the BLAS and OpenMP pools of this process to n_threads."""
    from threadpoolctl import threadpool_limits
    return threadpool_limits(limits=max(1, int(n_threads)))

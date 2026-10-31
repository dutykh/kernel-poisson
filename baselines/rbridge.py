# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Exchange of a training and a test fold with the R scripts of r/ through plain CSV files.

The covariates are renamed v1, v2, ... (R-safe names) and described in spec.csv by their kind:
"factor" (strings), "smooth" (a continuous covariate with at least ten distinct training
values) or "linear" (a continuous covariate with fewer distinct values). The response y and
the offset log e are the first two columns of train.csv and test.csv.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .common import Preprocessor, exposure, read_r_info, run_rscript

SMOOTH_MIN_DISTINCT = 10


def prepare(Xtr, ytr, etr, Xte, ete, categorical, clip=False):
    """Cleaned training and test frames with R-safe names, and the specification table."""
    pre = Preprocessor().fit(Xtr, categorical)
    tr = pre.clean(Xtr)
    te = pre.clean(Xte, clip=clip)
    names = {c: f"v{j + 1}" for j, c in enumerate(pre.columns)}
    kinds = []
    for c in pre.columns:
        if c in pre.factors_:
            kinds.append("factor")
        elif tr[c].nunique() >= SMOOTH_MIN_DISTINCT:
            kinds.append("smooth")
        else:
            kinds.append("linear")
    spec = pd.DataFrame({"name": [names[c] for c in pre.columns], "kind": kinds,
                         "column": list(pre.columns)})
    tr = tr.rename(columns=names)
    te = te.rename(columns=names)
    tr.insert(0, "offset", np.log(exposure(etr, len(tr))))
    tr.insert(0, "y", np.asarray(ytr, dtype=float))
    te.insert(0, "offset", np.log(exposure(ete, len(te))))
    te.insert(0, "y", 0.0)
    return tr, te, spec


def run(script, tr, te, spec, extra_args, n_threads, timeout=None):
    """Write the folds, run the script, return (linear predictor, info dict)."""
    with tempfile.TemporaryDirectory(prefix="rbridge-") as tmp:
        folder = Path(tmp)
        spec[["name", "kind"]].to_csv(folder / "spec.csv", index=False)
        tr.to_csv(folder / "train.csv", index=False)
        te.to_csv(folder / "test.csv", index=False)
        run_rscript(script, [folder, *extra_args], n_threads, cwd=folder, timeout=timeout)
        eta = pd.read_csv(folder / "pred.csv")["eta"].to_numpy(dtype=float)
        info = read_r_info(folder / "info.csv")
    return eta, info

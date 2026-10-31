# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Neural-network Poisson regression: a two-hidden-layer tanh perceptron for the log-rate, the
offset log e added to its output, trained with the Poisson likelihood by Adam with weight decay
1e-4 and early stopping on a 15 per cent validation split of the training fold; the width (32
or 64 units per layer) is chosen by the validation loss. Inputs are the standardised continuous
covariates and the factor indicators of the kernel estimator's encoding.

The network is trained by network_worker.py in the interpreter PYTHON, which must have PyTorch
and exchanges .npz files with this module. PYTHON is the interpreter named by the environment
variable NETWORK_PYTHON, by default the interpreter that runs this module, so that PyTorch can
live in an environment of its own.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .common import as_frame, exposure, factor_names, kappa_moment, thread_env

PYTHON = os.environ.get("NETWORK_PYTHON", sys.executable)
WORKER = Path(__file__).resolve().parent / "network_worker.py"
SETTINGS = dict(val_fraction=0.15, weight_decay=1e-4, lr=1e-3, max_epochs=2000, patience=30)
WIDTHS = (32, 64)


def available() -> bool:
    """True when the interpreter PYTHON can import PyTorch and NumPy, which the worker needs."""
    try:
        done = subprocess.run([PYTHON, "-I", "-c", "import numpy, torch"], capture_output=True,
                              timeout=300)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def _batch(n):
    return 64 if n <= 2000 else (256 if n <= 20000 else 1024)


def fit_predict(Xtr, ytr, etr, Xte, ete, categorical=(), seed=0, n_threads=1):
    from kernelpoisson.kernels import Encoder

    ytr = np.asarray(ytr, dtype=float)
    Xtr, Xte = as_frame(Xtr), as_frame(Xte)
    idx = [Xtr.columns.get_loc(c) for c in factor_names(Xtr, categorical)]
    enc = Encoder(idx).fit(Xtr.to_numpy(dtype=object))
    Ztr = enc.transform(Xtr.to_numpy(dtype=object))
    Zte = enc.transform(Xte.to_numpy(dtype=object))
    otr = np.log(exposure(etr, len(ytr)))
    ote = np.log(exposure(ete, len(Zte)))
    with tempfile.TemporaryDirectory(prefix="network-") as tmp:
        src, dst = Path(tmp) / "input.npz", Path(tmp) / "output.npz"
        np.savez(src, Ztr=Ztr, ytr=ytr, otr=otr, Zte=Zte, ote=ote, widths=np.array(WIDTHS),
                 seed=np.array(int(seed)), threads=np.array(int(n_threads)),
                 batch=np.array(_batch(len(ytr))), **{k: np.array(v) for k, v in SETTINGS.items()})
        out = subprocess.run([PYTHON, "-I", str(WORKER), str(src), str(dst)], cwd=tmp,
                             env=thread_env(n_threads), capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"network worker failed:\n{out.stderr.strip()[-3000:]}")
        res = np.load(dst, allow_pickle=False)
        mu, mu_in = res["mu_test"], res["mu_train"]
        info = json.loads(str(res["info"]))
    info.update(batch=_batch(len(ytr)), **SETTINGS)
    return dict(mu=mu, kappa=None, kappa_nb=kappa_moment(ytr, mu_in), info=info)

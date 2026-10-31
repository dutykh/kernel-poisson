# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""The truncated-Newton method of Maalouf, Trafalis and Adrianto (2011) carried to counts.

Newton (IRLS) outer iteration on theta = (b, alpha) with Poisson weights W = diag(mu); each
Newton system

    [ 1'W1     1'W K        ] [db    ]     [ 1'(mu - y)               ]
    [ K W 1    K W K + lam K] [dalpha] = - [ K (mu - y) + lam K alpha  ]

is solved by Euclidean conjugate gradients on the coefficients, stopped at a relative residual
or after max_cg iterations, and the step is damped by Armijo backtracking. This is the reference
against which the preconditioned method of `newton.py` is compared.
"""
from __future__ import annotations

import time

import numpy as np


def _hvp(K, mu, lam, pb, pa):
    Kpa = K @ pa
    v = pb + Kpa
    Wv = mu * v
    return float(np.sum(Wv)), K @ Wv + lam * Kpa


def fit_truncated_newton(K, y, offset=None, lam=1.0, max_outer=100, max_cg=200, cg_tol=None,
                         tol=1e-10, eta_clip=60.0):
    """Return a dict with the fit and per-iteration records (objective, CG steps, time)."""
    y = np.asarray(y, dtype=float)
    N = len(y)
    o = np.zeros(N) if offset is None else np.asarray(offset, dtype=float)
    alpha = np.zeros(N)
    Ka = np.zeros(N)
    b = float(np.log(np.sum(y) / np.sum(np.exp(o))))

    def phi(b, a, Ka):
        eta = np.minimum(o + b + Ka, eta_clip)
        return float(np.sum(np.exp(eta) - y * eta) + 0.5 * lam * a @ Ka)

    F = phi(b, alpha, Ka)
    records = []
    scale = max(1.0, float(np.sum(y)))
    for it in range(max_outer):
        t0 = time.perf_counter()
        mu = np.exp(np.minimum(o + b + Ka, eta_clip))
        gb = float(np.sum(mu - y))
        ga = K @ (mu - y) + lam * Ka
        gnorm = np.sqrt(gb * gb + ga @ ga)
        forcing = cg_tol if cg_tol is not None else min(0.5, np.sqrt(gnorm))
        # conjugate gradients on the augmented system, Euclidean inner product
        xb, xa = 0.0, np.zeros(N)
        rb, ra = -gb, -ga.copy()
        pb, pa = rb, ra.copy()
        rs = rb * rb + ra @ ra
        rs0 = rs
        ncg = 0
        for k in range(max_cg):
            ncg += 1
            hb, ha = _hvp(K, mu, lam, pb, pa)
            pHp = pb * hb + pa @ ha
            if pHp <= 0:
                break
            a = rs / pHp
            xb += a * pb
            xa += a * pa
            rb -= a * hb
            ra -= a * ha
            rs_new = rb * rb + ra @ ra
            if np.sqrt(rs_new) <= forcing * np.sqrt(rs0):
                break
            beta = rs_new / rs
            pb = rb + beta * pb
            pa = ra + beta * pa
            rs = rs_new
        slope = gb * xb + ga @ xa
        Kx = K @ xa
        step = 1.0
        while True:
            b_new = b + step * xb
            a_new = alpha + step * xa
            Ka_new = Ka + step * Kx
            F_new = phi(b_new, a_new, Ka_new)
            if F_new <= F + 1e-4 * step * slope or step < 1e-10:
                break
            step *= 0.5
        b, alpha, Ka = b_new, a_new, Ka_new
        records.append(dict(objective=F_new, decrement=-slope, step=step, cg=ncg,
                            seconds=time.perf_counter() - t0))
        done = (-slope) <= tol * scale and step == 1.0
        F = F_new
        if done:
            break
    eta = o + b + Ka
    return dict(alpha=alpha, b=b, f=Ka, eta=eta, mu=np.exp(np.minimum(eta, eta_clip)),
                objective=F, records=records)

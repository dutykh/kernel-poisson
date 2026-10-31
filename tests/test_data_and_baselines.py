# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Checks of the data loaders and of the competitor methods of the benchmark study.

The loaders must return consistent shapes, non-negative integer counts, positive finite
exposures, factor columns as strings, no missing covariate and no column derived from the
response. Every competitor method must run on a small simulated Poisson problem with exposures
and return positive finite means; the Poisson GLM must recover the coefficients of the
simulation. Run from the root folder of the repository:  python -I -m pytest -q tests
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import baselines  # noqa: E402
from data import datasets  # noqa: E402

KEYS = {"X", "y", "exposure", "categorical", "groups", "name", "description", "source", "licence",
        "citation"}
NEEDS_R = {"gam_poisson", "gam_negbin", "gss", "ktweedie"}
HAS_R = shutil.which("Rscript") is not None


def r_has(packages):
    """True when Rscript is installed and so is every R package of the list."""
    if not HAS_R:
        return False
    names = ", ".join(f"'{p}'" for p in packages)
    code = f"quit(status = !all(vapply(c({names}), requireNamespace, logical(1), quietly = TRUE)))"
    return subprocess.run(["Rscript", "--vanilla", "-e", code], capture_output=True).returncode == 0


HAS_R_PACKAGES = r_has(("mgcv", "gss", "ktweedie"))


def load_or_skip(name):
    """The data set, or a skipped test when its export needs R or an R package that is missing."""
    try:
        return datasets.load(name)
    except RuntimeError as exc:
        if "Rscript is not installed" in str(exc) or "no package called" in str(exc):
            pytest.skip(str(exc).splitlines()[0][:200])
        raise


@pytest.fixture(scope="module", params=datasets.NAMES)
def loaded(request):
    return load_or_skip(request.param)


def test_loader_shapes_and_values(loaded):
    d = loaded
    assert KEYS <= set(d)
    n = len(d["y"])
    X, y = d["X"], d["y"]
    assert isinstance(X, pd.DataFrame) and len(X) == n and X.shape[1] >= 1
    assert y.dtype == float and np.all(np.isfinite(y))
    assert np.all(y >= 0) and np.all(y == np.round(y)) and y.sum() > 0
    if d["exposure"] is not None:
        e = d["exposure"]
        assert e.shape == (n,) and np.all(np.isfinite(e)) and np.all(e > 0)
    if d["groups"] is not None:
        assert len(d["groups"]) == n and len(np.unique(d["groups"])) >= 5
    assert not X.isna().any().any()
    assert set(d["categorical"]) <= set(X.columns)
    for c in X.columns:
        if c in d["categorical"]:
            assert all(isinstance(v, str) for v in X[c].head(50))
            assert X[c].nunique() >= 2
        else:
            assert np.issubdtype(X[c].dtype, np.floating)
    for text in ("description", "source", "licence", "citation"):
        assert isinstance(d[text], str) and len(d[text]) > 5


def test_loader_has_no_response_derived_columns(loaded):
    forbidden = set(datasets.FORBIDDEN[loaded["name"]])
    assert not forbidden & set(loaded["X"].columns)
    # no covariate reproduces the response
    y = loaded["y"]
    for c in loaded["X"].columns:
        if c not in loaded["categorical"]:
            v = loaded["X"][c].to_numpy(float)
            if np.std(v) > 0:
                assert abs(np.corrcoef(v, y)[0, 1]) < 0.999


@pytest.mark.parametrize("name,n", [("fishing", 147), ("solder", 900), ("fatalities", 335),
                                    ("abalone", 4177), ("chicago", 4863), ("eclipse", 10593),
                                    ("bike", 17379), ("datacar", 67856), ("fremtpl2", 678013),
                                    ("biochemists", 915)])
def test_row_counts_of_the_sources(name, n):
    assert len(load_or_skip(name)["y"]) == n


def test_fremtpl2_caps_and_split():
    d = datasets.load("fremtpl2")
    assert d["y"].max() <= 4 and d["exposure"].max() <= 1.0
    assert d["X"]["VehAge"].max() <= 20 and d["X"]["DrivAge"].max() <= 90
    assert d["X"]["BonusMalus"].max() <= 150
    tr, te = datasets.fremtpl2_split(d)
    n = len(d["y"])
    assert len(np.intersect1d(tr, te)) == 0 and len(tr) + len(te) == n
    assert abs(len(te) / n - 0.33) < 1e-4
    # the test rows are those of the scikit-learn example in the original row order
    from sklearn.model_selection import train_test_split
    _, te0 = train_test_split(np.arange(n), test_size=0.33, random_state=0)
    assert np.array_equal(np.sort(d["row_id"][te]), np.sort(te0))


def test_a_download_that_differs_from_the_record_is_refused(tmp_path, monkeypatch):
    good, bad = b"a,b\n1,2\n3,4\n", b"a,b\n1,2\n3,5\n"
    record = tmp_path / "checksums.json"
    record.write_text(json.dumps({"fake/fake.csv": dict(
        sha256=hashlib.sha256(good).hexdigest(), bytes=len(good), origin="test",
        recorded="2026-10-31")}))
    first, second = "https://first.invalid/fake.csv", "https://second.invalid/fake.csv"
    served = {first: bad, second: good}
    monkeypatch.setattr(datasets, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(datasets, "CHECKSUMS", record)
    monkeypatch.setitem(datasets.RAW, "fake", ("fake.csv", [first, second], None, "csv"))
    monkeypatch.setattr(datasets, "_download",
                        lambda url, dest, timeout=60.0: dest.write_bytes(served[url]))
    # the first source serves a file that differs from the record: it is refused, the next
    # source is tried, and only the verified file is kept
    path = datasets._raw("fake")
    assert path.read_bytes() == good
    assert [p.name for p in (tmp_path / "cache").iterdir()] == ["fake"]
    # when no source serves the recorded file, the load fails and nothing is kept
    path.unlink()
    served[second] = bad
    with pytest.raises(RuntimeError, match="SHA256"):
        datasets._raw("fake")
    assert not path.exists()


# ================================================================================================
def synthetic(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"x1": rng.uniform(-1, 1, n), "x2": rng.normal(0, 1, n),
                      "f": rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2])})
    e = np.exp(rng.normal(0.0, 0.5, n))
    eta = 0.3 + 0.6 * X["x1"] - 0.4 * X["x2"] + np.where(X["f"] == "b", 0.5, 0.0) \
        + np.where(X["f"] == "c", -0.3, 0.0)
    y = rng.poisson(e * np.exp(eta)).astype(float)
    return X, y, e, np.asarray(eta)


@pytest.mark.parametrize("method", list(baselines.METHODS))
def test_baseline_runs_and_returns_positive_means(method):
    if method in NEEDS_R and not HAS_R_PACKAGES:
        pytest.skip("Rscript or one of the R packages mgcv, gss and ktweedie is not available")
    if method == "network":
        from baselines.network import available
        if not available():
            pytest.skip("PyTorch is not available to the interpreter of the network")
    X, y, e, _ = synthetic(n=260, seed=1)
    tr, te = np.arange(200), np.arange(200, 260)
    fn, kwargs = baselines.get(method)
    out = fn(X.iloc[tr], y[tr], e[tr], X.iloc[te], e[te], ["f"], 0, 2, **kwargs)
    assert out is not None
    mu = np.asarray(out["mu"])
    assert mu.shape == (60,) and np.all(np.isfinite(mu)) and np.all(mu > 0)
    assert isinstance(out["info"], dict)
    if out.get("kappa") is not None:
        assert np.isfinite(out["kappa"]) and out["kappa"] > 0
    if method in ("boosting", "forest", "network", "krr_log"):
        assert out["kappa_nb"] >= 1e-6
    # the predictions carry the exposure: doubling it doubles the predicted means
    if method in ("glm_poisson", "ours_poisson", "gp_laplace"):
        out2 = fn(X.iloc[tr], y[tr], e[tr], X.iloc[te], 2.0 * e[te], ["f"], 0, 2, **kwargs)
        assert np.allclose(out2["mu"], 2.0 * mu, rtol=1e-8)


def test_baselines_without_exposure_or_factors():
    X, y, e, _ = synthetic(n=200, seed=2)
    X = X[["x1", "x2"]]
    for method in ("glm_negbin", "boosting", "ours_poisson"):
        fn, kwargs = baselines.get(method)
        out = fn(X.iloc[:150], y[:150], None, X.iloc[150:], None, [], 0, 2, **kwargs)
        assert np.all(np.isfinite(out["mu"])) and np.all(out["mu"] > 0)


def test_poisson_glm_recovers_the_coefficients():
    X, y, e, _ = synthetic(n=6000, seed=3)
    fn, kwargs = baselines.get("glm_poisson")
    out = fn(X, y, e, X.iloc[:10], e[:10], ["f"], 0, 2, **kwargs)
    coef, b0 = out["info"]["coef"], out["info"]["intercept"]
    # the most frequent level a is the reference: intercept 0.3, b: +0.5, c: -0.3
    assert abs(b0 - 0.3) < 0.05
    assert abs(coef["x1"] - 0.6) < 0.05 and abs(coef["x2"] + 0.4) < 0.03
    assert abs(coef["f=b"] - 0.5) < 0.05 and abs(coef["f=c"] + 0.3) < 0.06


def test_negative_binomial_glm_recovers_the_dispersion():
    rng = np.random.default_rng(4)
    n = 6000
    X = pd.DataFrame({"x1": rng.uniform(-1, 1, n)})
    mu = 2.0 * np.exp(0.5 * X["x1"].to_numpy())
    kappa = 0.4
    y = rng.negative_binomial(1.0 / kappa, 1.0 / (1.0 + kappa * mu)).astype(float)
    fn, kwargs = baselines.get("glm_negbin")
    out = fn(X, y, None, X.iloc[:5], None, [], 0, 2, **kwargs)
    assert abs(out["kappa"] - kappa) < 0.06
    assert abs(out["info"]["coef"]["x1"] - 0.5) < 0.06


def test_moment_dispersion():
    from baselines.common import kappa_moment
    rng = np.random.default_rng(5)
    mu = rng.uniform(1, 5, 200000)
    kappa = 0.3
    y = rng.negative_binomial(1.0 / kappa, 1.0 / (1.0 + kappa * mu))
    assert abs(kappa_moment(y, mu) - kappa) < 0.01
    assert kappa_moment(rng.poisson(mu), mu) < 0.01


def test_guarded_ranked_probability_score_matches_the_reference():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import benchmark
    from kernelpoisson import scores
    rng = np.random.default_rng(6)
    mu = rng.gamma(2.0, 20.0, 40)
    y = rng.poisson(mu).astype(float)
    y[:3] = [0.0, 400.0, 3.0]                     # counts far in both tails
    for kappa in (None, 0.3):
        ref = scores.ranked_probability(y, mu, kappa)
        rowwise = np.array([benchmark._rps_one(a, b, kappa)[0] for a, b in zip(y, mu)])
        assert np.allclose(rowwise, ref, rtol=1e-9, atol=1e-9)
    # the identity RPS = E|Y - y| - E|Y - Y'|/2 for wide laws, against the exact window sum
    for kappa, m, yy in ((None, 3.0e5, 2.9e5), (None, 3.0e5, 10.0), (0.05, 2.0e4, 1.5e4),
                         (0.05, 2.0e4, 3.0)):
        exact, flag = benchmark._rps_one(yy, m, kappa)
        assert not flag
        old = benchmark.RPS_WINDOW_LIMIT
        try:
            benchmark.RPS_WINDOW_LIMIT = 10.0
            identity, flag = benchmark._rps_one(yy, m, kappa)
        finally:
            benchmark.RPS_WINDOW_LIMIT = old
        assert flag and abs(identity - exact) <= 1e-3 * exact, (kappa, m, yy, identity, exact)


def test_guarded_score_splits_ordinary_and_wide_rows():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import benchmark
    from kernelpoisson import scores
    rng = np.random.default_rng(7)
    mu = rng.gamma(2.0, 2.0, 300)
    y = rng.poisson(mu).astype(float)
    wild = mu.copy()
    wild[5] = 8.8e10                               # one absurd prediction
    guard = benchmark.GuardedRPS(scores.ranked_probability)
    out = guard(y, wild)
    keep = np.arange(300) != 5
    assert np.allclose(out[keep], scores.ranked_probability(y[keep], wild[keep]), rtol=1e-12)
    assert guard.by_identity == 1 and abs(out[5] - (8.8e10 - y[5])) <= 1e-3 * 8.8e10


def test_stable_negative_binomial_log_score():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import benchmark
    from kernelpoisson import scores
    stable = benchmark.stable_log_score(scores.log_score)
    rng = np.random.default_rng(8)
    mu = rng.gamma(2.0, 2.0, 200)
    y = rng.poisson(mu).astype(float)
    for kappa in (0.1, 2.0):
        assert np.allclose(stable(y, mu, kappa), scores.log_score(y, mu, kappa), rtol=1e-10)
    assert np.allclose(stable(y, mu, None), scores.log_score(y, mu, None))
    # a tiny mean: the scipy parametrisation gives an infinite score, the closed form ~ -log mu
    tiny = np.array([1e-20])
    val = stable(np.array([1.0]), tiny, 0.5)[0]
    assert np.isfinite(val) and abs(val - (-np.log(1e-20))) < 1e-6

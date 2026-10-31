<!-- Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE) -->

# Kernel Poisson regression with exposures

This repository accompanies the article *Kernel Poisson Regression: Events Set the Accuracy and the
Cost* by Denys Dutykh, Dirar Homouz and Maher Maalouf (Khalifa University of Science and Technology,
Abu Dhabi). The article shows that, once the observations are numerous enough, the expected number
of events, not the number of observations, governs the accuracy and the cost of Poisson regression
in a reproducing kernel Hilbert space. The repository holds the estimator of the article, the code
of every experiment, the saved results behind every figure and table, and the means to rebuild
them. A reader who wants to use the estimator needs only the folder `kernelpoisson/`. A reader who
wants to check a number of the article can redraw every figure and table from the saved results in
under a minute, and can re-run any experiment.

## The model and the estimator

The data are counts `y_i` observed over exposures `e_i > 0` (a duration, an area, a length of code)
at covariates `x_i`, for `i = 1, …, N`. The model is

```text
Y_i ~ Poisson(mu_i),     log(mu_i) = log(e_i) + b + f(x_i),
```

where `b` is an intercept and `f` lies in the reproducing kernel Hilbert space of a Gaussian kernel
with length scale `ell`. The estimator minimizes

```text
J(b, f) = Σ_i [ exp(log(e_i) + b + f(x_i)) − y_i (log(e_i) + b + f(x_i)) ] + (lam/2) ‖f‖²
```

over the intercept `b`, which is not penalized, and over `f`. The negative binomial variant replaces
the Poisson loss by the negative binomial one, whose variance is `mu + kappa mu²` at a fixed
dispersion `kappa`. The symbols are named as in the code: `lam` is the penalty λ of the article,
`ell` the length scale, and `kappa` the dispersion of the negative binomial law.

Three features distinguish the code. First, for the Poisson loss the intercept is eliminated in
closed form,

```text
b(f) = log( Σ_i y_i / Σ_i e_i exp(f(x_i)) ),
```

so that the fitted counts add up to the observed total at every iteration; for the negative binomial
loss the same elimination is a scalar Newton iteration, and a weighted balance replaces the equality
of totals. Second, each Newton system is solved by conjugate gradients in the inner product of the
kernel space, preconditioned by the projection of the data operator on a few landmark observations
drawn with probabilities proportional to Fisher-weighted ridge leverage scores; above 6000
observations the estimator itself is restricted to the span of 1500 landmarks. Third, the penalty
`lam = t Σ_i y_i` and the length scale are chosen by the one-step leave-one-out deviance (the
logarithmic score for the negative binomial law), which is computed in closed form from the leverages
of the fit.

## Installation

The code was run on Linux, with Python 3.13 and 3.14. The experiments lock their result files with
POSIX file locks and therefore need a Unix system; the estimator itself is plain NumPy and SciPy.

```bash
git clone https://github.com/dutykh/kernel-poisson
cd kernel-poisson
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The file `requirements.txt` pins the versions used for the article (numpy, scipy, pandas,
scikit-learn, statsmodels, matplotlib). The estimator alone needs only numpy and scipy, and
`pip install .` installs it as the package `kernelpoisson`.

The competitors of the benchmark add two requirements. The generalized additive models, the
smoothing-spline ANOVA and kernel Tweedie regression call R (4.5.2 for the article) with the packages
mgcv 1.9.4, gss 2.2.10 and ktweedie 1.0.3, and the data sets of the biochemists (package pscl
1.5.9) and of Chicago mortality are exported by R as well. All four packages come from CRAN, whose
current version of gss (3.0.0 at the time of writing) differs from the version of the article, which
is installed from the CRAN archive:

```r
install.packages(c("mgcv", "pscl", "ktweedie"))
install.packages("https://cran.r-project.org/src/contrib/Archive/gss/gss_2.2-10.tar.gz",
                 repos = NULL, type = "source")
```

The smoothing-spline ANOVA is the competitor whose results depend most on the build: another version
or build of the package changes its predictions in the fourth digit. Where R or one of these
packages is missing, the tests of the R competitors and of the two data sets that R exports are
skipped, and `make data` stops at those two data sets.

The neural network needs PyTorch (2.14.1, CPU build, for the article) and numpy in the Python
interpreter named by the environment variable `NETWORK_PYTHON`, by default the interpreter that
runs the code; its test is skipped where either is missing.

The figures are set in Latin Modern: with LaTeX installed (`latex`, `dvipng` and the package
`lmodern`) their text goes through LaTeX, as in the article, and otherwise through the Computer
Modern fonts of matplotlib, with slightly different typography; `FIGURE_USETEX=0` switches LaTeX off.

## Using the estimator

The deep-sea fish counts of Bailey et al. (2009) are fitted in about a second. The example runs from
the root folder of the repository:

```python
from data import datasets
from kernelpoisson import KernelPoissonRegressor

d = datasets.load("fishing")                      # 147 trawl hauls, swept area as exposure
X = d["X"][["meandepth", "period"]].to_numpy(dtype=object)
est = KernelPoissonRegressor(family="negbin", categorical=(1,)).fit(X, d["y"], d["exposure"])
counts = est.predict(X, d["exposure"])            # fitted counts at the observed exposures
rate = est.predict(X)                             # fitted rate per unit of exposure
```

The argument `categorical` lists the columns of `X` that are factors, `family` is `"poisson"` or
`"negbin"`, and after the fit the attributes `lam_`, `lengthscale_` and `kappa_` hold the selected
tuning. The solver is also available on its own, for a Gram matrix `K` of observations, counts `y`
and exposures `e`:

```python
import numpy as np
from kernelpoisson import KernelNewton

fit = KernelNewton(K, y, offset=np.log(e)).solve(lam)
# fit.mu: fitted counts, fit.eta: log-means, fit.b: intercept, fit.alpha: kernel coefficients
```

## Rebuilding the article

The folder `figures/` holds the figures of the article and `tables/` its tables. They are drawn from
the numbers saved in `results/`, which are described in `results/README.md`. The command
`make figures` redraws all of them in under a minute, after fetching the two small data sets that it
needs (the fish counts and the bike rentals, 0.3 MB in all), and with LaTeX installed the figures
come out identical to those of the article and the tables identical character by character. As it
draws, it also prints the numbers that the article quotes from these results: the fitted slopes of
the rates, the mean ranks of the benchmark, the ratios of the simulation and the bounds of the
leave-one-out experiments. Every other command recomputes results and then redraws the figures that
depend on them. All commands run from the root folder of the repository, and the heavy ones run at
nice 15 with at most 16 threads.

| command | what it does | time |
|---|---|---|
| `make test` | the tests described below | about 20 seconds once the data are on disk |
| `make data` | downloads and checks the ten data sets | a few minutes, 48 MB |
| `make figures` | redraws every figure and table from `results/` | under a minute |
| `make fish` | the deep-sea fish case study and its figure | seconds |
| `make newton` | condition numbers, cost and convergence of the solvers, with the landmark figure | about 30 minutes |
| `make loo` | one-step against exact leave-one-out, for single observations and for blocks of neighbors | about four minutes |
| `make rates` | excess risk per event against the number of events, and at a fixed number of observations | several hours |
| `make simulation` | the simulation study with a known truth | about an hour |
| `make benchmark` | the comparison on the ten open data sets | about a day |
| `make summary` | the tables and the figure of the benchmark, from its results | seconds |
| `make landmark-budget` | test deviance on the bike rentals with 1500, 3000 and 6000 landmarks | several hours |
| `make all` | every step above in turn, the tests first | days |

The times are those of a machine with 16 cores; the leave-one-out and simulation experiments use a
single thread.

The figures and tables of the article correspond to the commands as follows.

| in the article | file | redraw from the saved results | recompute the results |
|---|---|---|---|
| main text, deep-sea fish counts | `figures/fish.pdf` | `make fish` | `make fish` |
| main text, the Newton systems | `figures/newton.pdf` | `make fig-newton` | `make newton` |
| main text, rates in the number of events | `figures/rates.pdf` | `make fig-rates` | `make rates` |
| main text, benchmark on ten data sets | `figures/benchmark.pdf` | `make summary` | `make benchmark` |
| supplement, leave-one-out | `figures/loo.pdf` | `make fig-loo` | `make loo` |
| supplement, simulation with a known truth | `figures/simulation.pdf` | `make fig-simulation` | `make simulation` |
| supplement, landmark preconditioners | `figures/landmarks.pdf` | `make fig-landmarks` | `make newton` |
| supplement, tables of the benchmark | `tables/benchmark.tex` and `tables/benchmark_*_all.tex` (three files) | `make summary` | `make benchmark` |
| supplement, landmarks on the bike rentals | `tables/landmark_budget.tex` | `make table-landmark-budget` | `make landmark-budget` |

Every random choice has a fixed seed: the folds of the benchmark (the seed 20261018 plus the number
of the repetition), the simulated designs and the landmarks. The long experiments (the benchmark,
the simulation, the rates and the landmarks on the bike rentals) resume where they stopped when they
are interrupted, since the finished rows stay in their result files.

Re-running an experiment reproduces the saved numbers up to the rounding errors of the linear
algebra libraries; the fitting times, the memory and the dates recorded in the results are those of
the machine that ran them, and the timings of the article are likewise those of its own machine. A
complete re-run of the benchmark on two data sets, the fish counts and the traffic fatalities (13
methods on 50 folds each), reproduces 1997 of the 2000 saved rows to within 1e-9, and 1800 of them
bit for bit; the other three belong to the smoothing-spline ANOVA of the R package gss, whose fit is
fragile on strongly overdispersed counts and is then retried with another random basis.

Two experiments differ from a re-run by more than that. The first is the cost experiment of the
Newton systems, whose counts of products and of iterations change as well as its times
(`results/README.md` explains). The second is the pair of leave-one-out experiments: the exact
refits at their smallest penalties (outside the range that their figure shows) are so ill
conditioned that the exact deviances there, dominated by a few observations, change with the number
of linear algebra threads and from one run to another. The command `make loo` uses one thread, so
that they do not depend on the number of cores; every other number of these experiments agrees with
the saved files to at least six digits.

The largest single fit, on the French claims, peaks near 7 GB of memory.

## Data

No data are redistributed. The module `data/datasets.py` downloads each of the ten data sets from its
public source on first use, checks it against the SHA256 recorded in `data/checksums.json`, and keeps
it in a cache that is not part of the repository. The sources, the licenses and the choices made for
each data set (the covariates kept, the rows dropped, the caps applied) are described in
`data/README.md`. The case study of the fish also writes the observed values of its 147 hauls to
`results/fish/points.csv`, which git ignores for the same reason.

## Tests

The command `make test` checks, among others, that the derivatives of the losses match finite
differences and satisfy the self-concordance inequality `|ℓ'''| ⩽ ℓ''`; that conjugate gradients in
the kernel geometry reproduce the exact Newton direction; that every iterate reproduces the observed
total; that the kernel-weighted residuals equal `lam f(x)` at the fit; that the fitted counts solve
the entropy problem dual to the estimator; that the linear kernel reproduces the ridge Poisson
regression of scikit-learn; that truncated conjugate gradient directions decrease the objective; that
the one-step leave-one-out predictor agrees with exact refits to second order; that the projected
model with all observations as landmarks is the full estimator; that the negative binomial loss tends
to the Poisson loss as `kappa → 0`; and that the scores agree with direct formulas. Further tests
check that every data loader returns the documented data, without any covariate derived from the
response, and that a downloaded file that differs from its record is refused; that every competitor
runs on a simulated problem and returns positive finite means; that the Poisson and negative binomial
models recover the coefficients and the dispersion of the simulation; and that the ranked probability
score and the logarithmic score stay finite and exact for extreme predictions.

## Layout

| folder | content |
|---|---|
| `kernelpoisson/` | the estimator: losses, kernels, Newton solver and preconditioner, landmarks, leave-one-out, projected model, scores, uncertainty bands, and the truncated Newton method of Maalouf, Trafalis and Adrianto (2011) carried to counts |
| `baselines/` | the competitors (generalized linear and additive models, smoothing-spline ANOVA, kernel Tweedie, kernel ridge on the log scale, Gaussian-process Laplace, random forest, gradient boosting, neural network) |
| `data/` | download, checks and loading of the ten data sets; see `data/README.md` |
| `experiments/` | one Python file per experiment and per figure, and the style of the figures |
| `results/` | the numbers behind every figure and table; see `results/README.md` |
| `figures/`, `tables/` | the figures and tables of the article |
| `tests/` | the tests of the estimator, the data loaders and the competitors |

## Citation

If you use the code, please cite the article. The file `CITATION.cff` gives the reference that GitHub
shows under "Cite this repository".

## License

The code is distributed under the GNU Lesser General Public License, version 2.1 (the file
`LICENSE`). The data sets are not part of the repository and keep their own licenses, which are listed
in `data/README.md`.

## Authors

Denys Dutykh (Mathematics Department), Dirar Homouz (Physics Department) and Maher Maalouf
(Department of Management Science and Engineering), Khalifa University of Science and Technology,
Abu Dhabi, United Arab Emirates.

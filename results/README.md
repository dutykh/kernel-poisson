<!-- Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE) -->

# Saved results

Every figure and every table of the article is drawn from the files of this folder: the command
`make figures` redraws them, and the command that recomputes a folder is named below. Each
experiment writes one sub-folder, and the long ones (the benchmark, the simulation, the rates and
the landmarks on the bike rentals) resume from the rows already written when they are interrupted.
The columns that record times (`seconds`, `fit_seconds`), memory (`peak_rss_mb`) and dates
(`finished`) are those of the machine that ran the experiment.

## benchmark/

The command `make benchmark` writes one file `NAME.csv` for each of the ten data sets (`fishing`,
`solder`, `fatalities`, `biochemists`, `abalone`, `chicago`, `eclipse`, `bike`, `datacar`,
`fremtpl2`), and `make summary` condenses them. A row of `NAME.csv` is one method on one fold of one
repetition of the cross-validation, scored under one predictive law. The columns are:

| column | meaning |
|---|---|
| `dataset`, `method` | the data set and the method (`glm_poisson`, `glm_negbin`, `gam_poisson`, `gam_negbin`, `gss`, `ktweedie`, `krr_log`, `gp_laplace`, `forest`, `boosting`, `network`, `ours_poisson`, `ours_negbin`) |
| `law` | the predictive law behind the scores that need a distribution: `poisson` with the predicted mean, or `negbin` with the dispersion `kappa` |
| `repeat`, `fold` | the repetition and the fold of the cross-validation (the French claims have one split, `0` and `0`) |
| `n_train`, `n_test` | the sizes of the training and of the test part |
| `status` | `ok`, or `skipped` where a method is not run at the size of the data set |
| `fit_seconds`, `peak_rss_mb` | the time to fit and tune on the training part, and the peak memory |
| `kappa`, `kappa_source` | the dispersion of the `negbin` law, from the model itself (`model`) or by the moment estimate on the training part (`moment`) |
| `deviance` | the mean Poisson deviance of the test counts |
| `log_score` | the mean of `−log P(Y = y)` under the predictive law |
| `rps`, `dss` | the mean ranked probability score and the mean Dawid–Sebastiani score |
| `rmse`, `mae` | the root mean squared error and the mean absolute error of the predicted means |
| `balance` | the sum of the predicted means over the sum of the observed test counts |
| `coverage90`, `interval90` | the share of test counts inside the central 90 percent predictive interval, and the mean interval score of that interval |
| `d2` | one minus the ratio of the mean deviance to that of the model with an intercept and the exposure only, fitted on the training part |
| `gini` | the Gini index of the concentration curve of the observed counts, ordered by predicted rate |
| `finished`, `message`, `info` | the date of the row, notes of the run, and the tuned settings of the method as JSON |

The file `summary.csv` holds the mean of each score per data set, method and law, with the number of
folds; `summary_deviance.csv` and `summary_logscore_nb.csv` hold the mean deviance and the mean
negative binomial logarithmic score as tables with the data sets in rows and the methods in columns;
`summary_ranks.csv` gives the mean rank of each method among those run on every data set, and the
number of data sets on which it is first; `summary_relative_deviance.csv` gives, for each fold in turn, the
percentage `100 (dev_GLM − dev_method) / dev_GLM` by which each method lowers the deviance of the
Poisson generalized linear model; and `summary_tests.csv` gives the corrected resampled t tests of
Nadeau and Bengio (2003) of each kernel estimator (`ours`) against each other method (`other`), with
the mean difference of the deviances (positive where the kernel estimator is better), the statistic
`t` and the two-sided `p`.

A complete re-run of the benchmark on the fish counts and on the traffic fatalities (13 methods on
50 folds each, 1000 rows per data set) agrees with the saved files in 1997 of the 2000 rows to
within 1e-9, and 1800 rows agree bit for bit in every score. The three other rows are all of `gss`,
the smoothing-spline ANOVA of the R package of that name, on the fish counts: its fit breaks down
on strongly overdispersed counts and is then retried with another random basis (see the column
`info`). On one fold the re-run needed a second attempt where the saved run succeeded at the
first, and on two folds the predictions differ in the fourth digit at the same smoothing
parameter; the package changes its predictions in the fourth digit from one version or build to
the next.

## fish/

The command `make fish` writes the case study of the deep-sea fish. The file `curves.csv` holds the
fitted rate per hectare swept, with its pointwise 90 percent band (`lower`, `upper`), on a grid of
200 depths in meters, for each family (`negbin`, `poisson`) and each period; `laws.csv` holds the
share of the hauls (`design`) and of the fitted counts (`event`) in bins of 250 meters of depth;
and `summary.json` holds the tuning selected by each fit, its balance, the two identities of
the exact structure of the estimator, and the share of the fitted counts below the median depth.
The command also writes the observed values of the 147 hauls to `points.csv`, which git ignores
because it holds data.

## loo/

The command `make loo` writes the leave-one-out experiment, on simulated data with 400 observations.
The file `path.csv` has one row for each of 16 penalties per event `t = lam / Σ_i y_i`: the one-step
(`alo_deviance`) and the exact (`loo_deviance`) leave-one-out mean deviances, the training deviance,
the effective dimension `dof`, the largest error `max_error` of the one-step predictor of the
log-mean, and `share_condition`, the share of the observations that satisfy the hypothesis of the
theorem. The file `points.csv` has one row for each observation, at the penalty selected by the
one-step criterion: the count `y`, the fitted log-mean `eta`, the one-step log-mean `eta_alo` and the
exact log-mean `eta_loo` without the observation, the leverage, the Cook-type distance `cook`, the
`bound` of the theorem, the quantity `condition` that the hypothesis compares with one quarter, the
`error` of the one-step predictor and the `shift` that deleting the observation causes.

The same data and the same penalties serve for the deletion of a block of neighboring observations,
the cells of a grid with 5 × 5 cells on the square of the covariates. The file `blocks_path.csv` has
one row for each penalty: the one-step (`onestep_block_deviance`) and the exact
(`exact_block_deviance`) leave-one-block-out mean deviances, the exact leave-one-out deviance, the
training deviance, the share of the observations whose block satisfies the hypothesis of the block
theorem (`share_condition`), the largest error `max_error`, and the largest ratio of the error to the
bound over the observations that satisfy the hypothesis. The file `blocks_points.csv` has one row
for each observation, at the penalty selected by the one-step block criterion: its `block`, the
count, the fitted log-mean `eta`, the one-step log-mean `eta_onestep` and the exact log-mean
`eta_exact` without the block, the `bound`, the `condition` and the `error`.

The exact refits at the smallest penalties of the paths (`t` of 4.6e-6 and below, outside the
penalties of the figure) are so ill conditioned that the exact deviances there, which a few
observations dominate, change with the number of linear algebra threads and from one run to another.
At `t = 1e-6` the exact leave-one-out deviance is 5.97 in the saved `path.csv`, 5.11 in the saved
`blocks_path.csv`, where it is computed again for the same data, 5.61 when the experiment is re-run
with one thread and 4.02 with four threads. The one-step deviances and `share_condition` do not
change, and every number at the penalties of the figure agrees with the saved files to at least six
digits, relative to the largest value of its column.

## newton/

The command `make newton` writes the experiment on the Newton systems. The file `spectra.csv` has one
row for each count scale `scale` and each of three simulated samples (`rep`): the largest fitted
count `mu_max`, the exact effective dimension `d`, the condition numbers of the system of the
truncated Newton method (`kappa_truncated_newton`) and of the same operator in the geometry of the
kernel space (`kappa_rkhs`) with its bound `bound_rkhs`, and, for uniform and for Fisher landmarks,
the condition number `kappa_*` of the preconditioned system and the Nyström error `eps_*` in units of
`lam`, with `m` landmarks equal to one, two and four times the effective dimension (the last number
of each column name). The file `cost.csv` gives, for each method, number of observations `N` and
count scale, the number of products with the kernel matrix (`matvecs`, in units of one product of
the full matrix), the time `seconds` and the number of inner iterations `inner`, with the relative
gap `rel_error` of the final objective to a reference solution; each method runs until its own
stopping rule. The file `trace.csv` gives the same gap against the elapsed time along each run, for
one problem with 4000 observations.

The two files `cost.csv` and `trace.csv` record one run on one machine, and their times depend on
the machine and on the number of threads. A re-run gives the same accuracies and the same picture.
The times of the solvers of the article agree to within a factor of about 1.6 (2.9 seconds against
the saved 2.5 seconds for the Fisher landmarks with 16000 observations and count scale 30), whereas
those of the truncated Newton method of Maalouf, Trafalis and Adrianto vary by larger factors from
run to run (24 seconds against 35 seconds with 8000 observations and count scale 30), because its
number of iterations depends on rounding errors in a system whose condition number is near 1e13. The
saved counts `matvecs` and `inner` of the two landmark preconditioners were recorded with a
formulation of their matrix by explicit products; the code of this repository forms the same matrix
as a sum of Gram matrices, which stays positive semidefinite in floating point, so that a re-run
counts fewer products and more inner iterations (27 against 6 in total for the Fisher landmarks with
16000 observations and count scale 30).

## rates/

The command `make rates` writes the experiment on the rates. The file `rates.csv` has one row for
each cell of the design and each replicate: the kind of the truth (`smooth` or `rough`), the
capacity exponent `a`, the target and the realized expected number of events (`Lambda_target`,
`Lambda`), the mean count per observation, the dispersion `kappa_d` of the counts (zero for the
Poisson law), the number of observations `N` and of events, the penalty per event of the oracle
(`lam_oracle`) with its excess risk per event `risk_oracle`, and the penalty per event chosen by
one-step leave-one-out (`lam_alo`) with its excess risk `risk_alo`; each cell has the 15 replicates
`rep` 0 to 14. The file `plateau.csv` has the same quantities for the experiment at a fixed number
of observations `N` with growing exposures.

## simulation/

The command `make simulation` writes one file `TRUTH_MEAN.csv` for each of the four log-rates
(`interaction`, `additive`, `threshold`, `linear`) and each of the two mean counts (`0.3`, `3`). A
row is one method on one replicate (`rep`): the excess risk per event `risk`, estimated on 20000
fresh covariate values, with the time `seconds`, the number of events and the `status` of the run.
The file `summary.csv` gives the mean, the median and the standard error of the risk for each
design and method.

## landmark_budget/

The command `make landmark-budget` writes `bike.csv`, with one row for each number of landmarks and
each of the five folds of the first repetition on the bike rentals: the test `deviance` and `d2`, the
selected length scale as a multiple of the median distance (`lengthscale_factor`), the selected
penalty per event `t`, the effective number of parameters `df`, the one-step leave-one-out deviance
`alo` at the selected point, and the fitting time `fit_seconds`.

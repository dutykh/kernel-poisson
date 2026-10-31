# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
# Rebuilds the results, figures and tables of the article. Every target is run from this folder.
# The targets that recompute results run at nice 15 with at most 16 threads (jobs times threads).
PY ?= python3 -I
HEAVY_THREADS ?= OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
FIG = figures
TAB = tables
# The exact leave-one-out refits at the smallest penalties of the path are so ill conditioned that
# their results change with the number of linear algebra threads: the experiment uses one, so that
# they do not depend on the number of cores of the machine.
LOO_THREADS = OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

.PHONY: test data figures fish fig-newton fig-landmarks fig-loo fig-rates fig-simulation summary \
        table-landmark-budget newton loo rates simulation benchmark landmark-budget all

# Tests of the estimator, the data loaders and the competitors.
test:
	$(PY) -m pytest -q tests

# Downloads and checks the ten data sets, and prints their statistics.
data:
	$(PY) data/datasets.py

# Every figure and every table of the article, drawn from the saved results (about a minute).
figures: fish fig-newton fig-landmarks fig-loo fig-rates fig-simulation summary table-landmark-budget

fish:
	$(PY) experiments/case_fish.py --out $(FIG)/fish.pdf

fig-newton:
	$(PY) experiments/fig_newton.py --out $(FIG)/newton.pdf

fig-landmarks:
	$(PY) experiments/fig_landmarks.py --out $(FIG)/landmarks.pdf

fig-loo:
	$(PY) experiments/fig_loo.py --out $(FIG)/loo.pdf

fig-rates:
	$(PY) experiments/fig_rates.py --out $(FIG)/rates.pdf

fig-simulation:
	$(PY) experiments/fig_simulation.py --out $(FIG)/simulation.pdf

summary:
	$(PY) experiments/summarise_benchmark.py --figure $(FIG)/benchmark.pdf --table $(TAB)/benchmark.tex

table-landmark-budget:
	$(PY) experiments/landmark_budget.py --data bike --table $(TAB)/landmark_budget.tex

# The experiments themselves: each recomputes its results and then redraws its figures.
newton:
	$(HEAVY_THREADS) nice -n 15 $(PY) experiments/newton_systems.py --part all
	$(MAKE) fig-newton fig-landmarks

loo:
	$(LOO_THREADS) nice -n 15 $(PY) experiments/loo_accuracy.py
	$(LOO_THREADS) nice -n 15 $(PY) experiments/block_loo.py
	$(MAKE) fig-loo

rates:
	$(HEAVY_THREADS) nice -n 15 $(PY) experiments/rates.py --reps 15
	$(HEAVY_THREADS) nice -n 15 $(PY) experiments/plateau.py --reps 15
	$(MAKE) fig-rates

simulation:
	for c in "interaction 0.3" "interaction 3" "additive 0.3" "additive 3" \
	         "threshold 0.3" "threshold 3" "linear 0.3" "linear 3"; do \
	  set -- $$c; nice -n 15 $(PY) experiments/simulation.py --truth $$1 --mean $$2 --reps 20 || exit 1; done
	$(MAKE) fig-simulation

benchmark:
	nice -n 15 $(PY) experiments/benchmark.py --data all --jobs 4 --threads 4
	$(MAKE) summary

landmark-budget:
	nice -n 15 $(PY) experiments/landmark_budget.py --data bike --jobs 3 --threads 2
	$(MAKE) table-landmark-budget

all: test newton loo rates simulation benchmark landmark-budget figures

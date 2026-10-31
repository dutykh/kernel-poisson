# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Kernel Poisson regression with exposures: profiled Newton iteration, conjugate gradients in
the reproducing-kernel geometry with a Fisher-leverage preconditioner, approximate
leave-one-out tuning, and the negative-binomial variant."""
from .kernels import Encoder, gram, median_lengthscale, squared_distances
from .losses import NegativeBinomial, Poisson
from .newton import Fit, KernelNewton, rkhs_pcg, exact_newton_direction
from .alo import leverages, alo_eta, cook_distances
from .truncated_newton import fit_truncated_newton
from .estimator import KernelPoissonRegressor

__all__ = ["Encoder", "gram", "median_lengthscale", "squared_distances", "NegativeBinomial",
           "Poisson", "Fit", "KernelNewton", "rkhs_pcg", "exact_newton_direction", "leverages",
           "alo_eta", "cook_distances", "fit_truncated_newton", "KernelPoissonRegressor"]

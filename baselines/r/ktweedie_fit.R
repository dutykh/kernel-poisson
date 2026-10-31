# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
# Kernel Tweedie regression (ktweedie): the log-mean of a Tweedie law with power rho lies in the
# reproducing-kernel space of the Gaussian kernel exp(-sigma |x - x'|^2), without intercept,
# penalty lambda alpha' K alpha; lambda is chosen by ktd_cv (Tweedie log-likelihood over nfolds
# folds). The fit is redone at the next best lambda when the optimiser reports failure.
# Usage: Rscript --vanilla ktweedie_fit.R FOLDER SEED
# FOLDER holds xtr.csv, xte.csv (numeric designs), ytr.csv (response), lambda.csv and par.csv
# (sigma, rho, nfolds); the script writes pred.csv (log-mean at the test points) and info.csv.
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
seed <- as.integer(args[2])
suppressMessages(library(ktweedie))
xtr <- as.matrix(read.csv(file.path(dir, "xtr.csv")))
xte <- as.matrix(read.csv(file.path(dir, "xte.csv")))
ytr <- read.csv(file.path(dir, "ytr.csv"))$y
lams <- read.csv(file.path(dir, "lambda.csv"))$lambda
par <- read.csv(file.path(dir, "par.csv"))
kern <- rbfdot(sigma = par$sigma)
set.seed(seed)

tweedie_ll <- function(y, eta, rho) {
  -(y * exp(-(rho - 1) * eta) / (rho - 1) + exp((2 - rho) * eta) / (2 - rho))
}

# ktd_cv, and the same criterion fold by fold when one of its fits fails
cv <- tryCatch(ktd_cv(x = xtr, y = ytr, kern = kern, lambda = lams, nfolds = par$nfolds,
                      rho = par$rho, loss = "LL"), error = function(e) NULL)
if (!is.null(cv)) {
  score <- cv$LL
  route <- "ktd_cv"
} else {
  route <- "folds"
  folds <- sample(rep(seq_len(par$nfolds), length.out = nrow(xtr)))
  score <- setNames(rep(0, length(lams)), lams)
  for (k in seq_len(par$nfolds)) {
    w <- folds == k
    m <- ktd_estimate(x = xtr[!w, , drop = FALSE], y = ytr[!w], kern = kern, lam1 = lams,
                      rho = par$rho)
    K <- as.matrix(kernelMatrix(kern, xtr[!w, , drop = FALSE], xtr[w, , drop = FALSE]))
    for (l in lams) {
      fit <- m$estimates[[paste("lambda", l)]]
      if (is.null(fit$coefficient)) {
        score[paste(l)] <- -Inf
      } else {
        score[paste(l)] <- score[paste(l)] +
          sum(tweedie_ll(ytr[w], drop(crossprod(K, fit$coefficient)), par$rho)) / nrow(xtr)
      }
    }
  }
}
order_l <- as.numeric(names(score))[order(-score)]
eta <- NULL
for (l in order_l) {
  m <- ktd_estimate(x = xtr, y = ytr, kern = kern, lam1 = l, rho = par$rho)
  if (!is.null(m$estimates[[1]]$coefficient) && m$estimates[[1]]$convergence <= 3) {
    eta <- as.numeric(ktd_predict(m, newdata = xte, type = "link")$prediction)
    chosen <- l
    conv <- m$estimates[[1]]$convergence
    break
  }
}
if (is.null(eta)) stop("ktd_estimate failed at every lambda")
write.csv(data.frame(eta = eta), file.path(dir, "pred.csv"), row.names = FALSE)
info <- data.frame(key = c("lambda", "cv_loglik", "convergence", "route"),
                   value = c(chosen, score[paste(chosen)], conv, route))
write.csv(info, file.path(dir, "info.csv"), row.names = FALSE)

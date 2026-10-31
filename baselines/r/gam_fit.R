# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
# Generalised additive model for counts (mgcv) with the offset log e:
#   log mu = offset + b + sum_j s_j(x_j) + linear terms + factor effects,
# with thin-plate regression splines s(x, k = 10) and smoothing parameters chosen by REML
# (gam), or by fast REML on discretised covariates (bam) for large training sets.
# Usage: Rscript --vanilla gam_fit.R FOLDER poisson|negbin NTHREADS LARGE(0|1)
# FOLDER holds spec.csv (name, kind = smooth|linear|factor), train.csv and test.csv
# (columns y, offset and the covariates); the script writes pred.csv (linear predictor,
# offset included) and info.csv.
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
family <- args[2]
nthreads <- as.integer(args[3])
large <- as.integer(args[4]) == 1L
suppressMessages(library(mgcv))
spec <- read.csv(file.path(dir, "spec.csv"), stringsAsFactors = FALSE)
cc <- c(y = "numeric", offset = "numeric")
for (i in seq_len(nrow(spec))) {
  cc[spec$name[i]] <- if (spec$kind[i] == "factor") "character" else "numeric"
}
tr <- read.csv(file.path(dir, "train.csv"), colClasses = cc, na.strings = "")
te <- read.csv(file.path(dir, "test.csv"), colClasses = cc, na.strings = "")
for (v in spec$name[spec$kind == "factor"]) {
  lv <- sort(unique(tr[[v]]))
  tr[[v]] <- factor(tr[[v]], levels = lv)
  te[[v]] <- factor(te[[v]], levels = lv)
}
terms <- ifelse(spec$kind == "smooth", sprintf("s(%s, k = 10)", spec$name), spec$name)
form <- as.formula(paste("y ~", paste(c(terms, "offset(offset)"), collapse = " + ")))
fam <- if (family == "poisson") poisson() else nb()
if (large) {
  fit <- bam(form, family = fam, data = tr, method = "fREML", discrete = TRUE, nthreads = nthreads)
} else {
  fit <- gam(form, family = fam, data = tr, method = "REML",
             control = gam.control(nthreads = nthreads))
}
eta <- as.numeric(predict(fit, newdata = te, type = "link"))
theta <- if (family == "poisson") NA else fit$family$getTheta(TRUE)
write.csv(data.frame(eta = eta), file.path(dir, "pred.csv"), row.names = FALSE)
info <- data.frame(
  key = c("theta", "edf", "converged", "smooths", "linear", "factors", "fitter"),
  value = c(theta, sum(fit$edf), as.integer(isTRUE(fit$converged)), sum(spec$kind == "smooth"),
            sum(spec$kind == "linear"), sum(spec$kind == "factor"),
            if (large) "bam-fREML" else "gam-REML"))
write.csv(info, file.path(dir, "info.csv"), row.names = FALSE)

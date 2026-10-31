# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
# Smoothing-spline ANOVA model for counts (gss): Poisson likelihood with the offset log e and
# main effects only, log mu = offset + b + sum_j f_j(x_j), each f_j a cubic smoothing spline
# (continuous covariate) or a nominal effect (factor), with the smoothing parameters chosen by
# the cross-validation score of gssanova on a random subset of basis functions (drawn with
# SEED), or kept at their initial values when SKIP is 1 (skip.iter = TRUE).
# Usage: Rscript --vanilla gss_fit.R FOLDER SEED SKIP
# FOLDER holds spec.csv, train.csv and test.csv (columns y, offset, covariates); the script
# writes pred.csv (linear predictor without the offset) and info.csv.
args <- commandArgs(trailingOnly = TRUE)
dir <- args[1]
seed <- as.integer(args[2])
skip <- as.integer(args[3]) == 1L
suppressMessages(library(gss))
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
form <- as.formula(paste("y ~", paste(spec$name, collapse = " + ")))
fit <- suppressWarnings(gssanova(form, family = "poisson", data = tr, offset = offset,
                                 seed = seed, skip.iter = skip))
if (!all(is.finite(fit$c)) || !all(is.finite(fit$d))) {
  stop("gssanova returned non-finite coefficients")
}
labels <- setdiff(c(fit$terms$labels, fit$lab.p), "offset")
eta <- as.numeric(predict(fit, newdata = te, include = labels))
write.csv(data.frame(eta = eta), file.path(dir, "pred.csv"), row.names = FALSE)
info <- data.frame(key = c("nbasis", "nlambda", "terms"),
                   value = c(length(fit$id.basis), fit$nlambda, length(labels) - 1))
write.csv(info, file.path(dir, "info.csv"), row.names = FALSE)

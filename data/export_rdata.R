# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
# Writes one R data object as a plain CSV file (no row names).
#   Rscript --vanilla export_rdata.R rda FILE.rda OBJECT OUT.csv      object stored in an .rda file
#   Rscript --vanilla export_rdata.R package PACKAGE OBJECT OUT.csv   data set of an installed package
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) stop("usage: export_rdata.R rda|package SOURCE OBJECT OUT.csv")
mode <- args[1]; src <- args[2]; obj <- args[3]; out <- args[4]
env <- new.env()
if (mode == "rda") {
  load(src, envir = env)
} else if (mode == "package") {
  utils::data(list = obj, package = src, envir = env)
} else {
  stop("unknown mode ", mode)
}
if (!exists(obj, envir = env, inherits = FALSE)) stop("object ", obj, " not found")
d <- get(obj, envir = env)
d <- as.data.frame(d)
for (v in names(d)) if (inherits(d[[v]], "Date")) d[[v]] <- format(d[[v]], "%Y-%m-%d")
utils::write.csv(d, out, row.names = FALSE)
cat(nrow(d), ncol(d), "\n")

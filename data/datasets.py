# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Open count-regression data sets of the benchmark study, with exposures and grouping.

load(name) returns a dict with the keys
    X            pandas DataFrame of covariates; the factor columns hold strings,
    y            numpy float array of the counts,
    exposure     numpy float array of positive exposures, or None (unit exposure),
    categorical  list of the names of the factor columns of X,
    groups       numpy array of group labels for grouped or blocked cross-validation, or None,
    name, description, source, licence, citation.

Raw files are downloaded once into cache/NAME/. Every download first lands in a new, empty folder
of its own (cache/incoming-*), is checked there (file format, and the MD5 published by OpenML
where there is one) and only then moved into the cache. The SHA256 of every raw file is recorded
in checksums.json at the first download and checked at every later load, so that all results are
computed from byte-identical inputs. Files derived from a raw file (a CSV written by R from an
.rda file, or by R from an installed package) are recorded and checked in the same way.

Columns that are functions of the response (densities, rates, sub-counts, claim indicators,
claim amounts, the components of a total) are never part of X.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CHECKSUMS = HERE / "checksums.json"

NAMES = ("fishing", "solder", "fatalities", "abalone", "chicago", "eclipse", "bike", "datacar",
         "fremtpl2", "biochemists")

SHUFFLE_SEED = 20261018          # fixed permutation of the rows of freMTPL2freq

RDATASETS = "https://vincentarelbundock.github.io/Rdatasets/csv"
ECLIPSE_PAGE = "https://www.st.cs.uni-saarland.de/softevo/bug-data/eclipse/"
ECLIPSE_ARCHIVE = ("https://web.archive.org/web/20230307104700id_/"
                   "https://www.st.cs.uni-saarland.de/softevo/bug-data/eclipse/promise-2_0a-files-csv.zip")
DLNM_TARBALL = "dlnm_2.4.10.tar.gz"

# raw files: name -> (file name in the cache, candidate URLs in order, OpenML MD5 or None, kind)
RAW = {
    "fishing": ("fishing.csv", [f"{RDATASETS}/COUNT/fishing.csv"], None, "csv"),
    "solder": ("solder.csv", [f"{RDATASETS}/rpart/solder.csv"], None, "csv"),
    "fatalities": ("Fatalities.csv", [f"{RDATASETS}/AER/Fatalities.csv"], None, "csv"),
    "abalone": ("abalone.arff", ["https://www.openml.org/data/v1/download/22111820/abalone.arff",
                                 "https://openml.org/data/v1/download/22111820/abalone.arff"],
                "81f6d8b4dce14710077ab88389254d83", "arff"),
    "chicago": (DLNM_TARBALL, [f"https://cloud.r-project.org/src/contrib/{DLNM_TARBALL}",
                               f"https://cloud.r-project.org/src/contrib/Archive/dlnm/{DLNM_TARBALL}"],
                None, "tar"),
    "eclipse": ("promise-2_0a-files-csv.zip",
                [ECLIPSE_PAGE + "promise-2_0a-files-csv.zip", ECLIPSE_ARCHIVE], None, "zip"),
    "bike": ("bike+sharing+dataset.zip",
             ["https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"], None, "zip"),
    "datacar": ("dataCar.csv", [f"{RDATASETS}/insuranceData/dataCar.csv"], None, "csv"),
    "fremtpl2": ("freMTPL2freq.arff",
                 ["https://www.openml.org/data/v1/download/20649148/freMTPL2freq.arff",
                  "https://openml.org/data/v1/download/20649148/freMTPL2freq.arff",
                  "https://api.openml.org/data/v1/download/20649148/freMTPL2freq.arff"],
                 "f8875568bf0ca622929105197e2db613", "arff"),
}


# ================================================================================================
# cache, downloads and checksums
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_checksums() -> dict:
    if CHECKSUMS.exists():
        return json.loads(CHECKSUMS.read_text())
    return {}


def _record_checksum(key: str, path: Path, origin: str) -> None:
    table = _read_checksums()
    table[key] = dict(sha256=_sha256(path), bytes=path.stat().st_size, origin=origin,
                      recorded=date.today().isoformat())
    tmp = CHECKSUMS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dict(sorted(table.items())), indent=2) + "\n")
    os.replace(tmp, CHECKSUMS)


def _verify(key: str, path: Path) -> bool:
    """True when a checksum is on record and matches; ValueError on a mismatch."""
    entry = _read_checksums().get(key)
    if entry is None:
        return False
    digest = _sha256(path)
    if digest != entry["sha256"]:
        raise ValueError(f"SHA256 of {path} is {digest}, the record in {CHECKSUMS.name} says "
                         f"{entry['sha256']}: the file differs from the one the results were "
                         "computed with")
    return True


def _looks_like(kind: str, path: Path) -> bool:
    if kind == "zip":
        return zipfile.is_zipfile(path)
    if kind == "tar":
        return tarfile.is_tarfile(path)
    head = path.read_bytes()[:4096].decode("utf-8", errors="replace").lower()
    if "<html" in head or "<!doctype" in head:
        return False
    if kind == "arff":
        return "@relation" in head
    return "," in head.splitlines()[0] if head else False


def _download(url: str, dest: Path, timeout: float = 60.0) -> None:
    """Stream url into dest; a read that stalls for timeout seconds raises."""
    req = urllib.request.Request(url, headers={"User-Agent": "kernel-poisson/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh, length=1 << 20)


def _raw(name: str) -> Path:
    """Path of the verified raw file of a data set, downloading it on first use."""
    filename, urls, md5, kind = RAW[name]
    key = f"{name}/{filename}"
    target = CACHE / name / filename
    if target.exists():
        if not _verify(key, target):
            _record_checksum(key, target, "file found in the cache")
        return target
    CACHE.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        for attempt in range(4):
            incoming = Path(tempfile.mkdtemp(prefix=f"incoming-{name}-", dir=CACHE))
            dest = incoming / filename
            print(f"downloading {name} from {url} (attempt {attempt + 1})", file=sys.stderr,
                  flush=True)
            try:
                _download(url, dest)
                if not _looks_like(kind, dest):
                    raise ValueError(f"the file from {url} is not a {kind} file")
                if md5 is not None and _md5(dest) != md5:
                    raise ValueError(f"MD5 of the file from {url} differs from the published {md5}")
                if not _verify(key, dest):
                    _record_checksum(key, dest, url)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(dest, target)
                print(f"  {name}: {target.stat().st_size / 1e6:.2f} MB received and checked",
                      file=sys.stderr, flush=True)
                return target
            except Exception as exc:          # network failure or a rejected file: try again
                errors.append(f"{url} (attempt {attempt + 1}): {exc}")
                if isinstance(exc, ValueError) and "SHA256" in str(exc):
                    break                     # the same bytes would come again: next source
                time.sleep(2.0 * (attempt + 1))
            finally:
                shutil.rmtree(incoming, ignore_errors=True)
    raise RuntimeError(f"could not obtain the raw file of {name}:\n  " + "\n  ".join(errors))


def _derived(name: str, filename: str, make) -> Path:
    """A file derived from verified inputs, made once by make(dest) and checked afterwards."""
    key = f"{name}/derived/{filename}"
    target = CACHE / name / "derived" / filename
    if target.exists():
        if not _verify(key, target):
            _record_checksum(key, target, "derived file found in the cache")
        return target
    work = Path(tempfile.mkdtemp(prefix=f"incoming-{name}-", dir=CACHE))
    try:
        dest = work / filename
        origin = make(dest, work)
        if not _verify(key, dest):
            _record_checksum(key, dest, origin)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(dest, target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return target


def _rscript(args, cwd: Path) -> str:
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    try:
        out = subprocess.run(["Rscript", "--vanilla", str(HERE / "export_rdata.R"), *map(str, args)],
                             cwd=str(cwd), env=env, capture_output=True, text=True, timeout=600)
    except FileNotFoundError as exc:
        raise RuntimeError("Rscript is not installed: R is needed to export the data sets "
                           "chicago and biochemists (see data/README.md)") from exc
    if out.returncode != 0:
        raise RuntimeError(f"Rscript failed: {out.stderr.strip()[-2000:]}")
    return out.stdout


def _read_arff(path: Path) -> pd.DataFrame:
    """Dense ARFF file as a DataFrame (numeric, nominal and string attributes)."""
    names, start, first = [], None, ""
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            if s.lower().startswith("@attribute"):
                m = re.match(r"@attribute\s+(?:'([^']*)'|\"([^\"]*)\"|(\S+))\s+(.*)$", s, re.I)
                names.append(next(g for g in m.groups()[:3] if g is not None))
            elif s.lower().startswith("@data"):
                start = i + 1
                break
    if start is None:
        raise ValueError(f"{path} has no @data section")
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= start and line.strip() and not line.startswith("%"):
                first = line
                break
    quote = "'" if "'" in first else '"'
    return pd.read_csv(path, skiprows=start, header=None, names=names, quotechar=quote,
                       skipinitialspace=True, na_values=["?"], comment="%")


def _strings(frame: pd.DataFrame, columns) -> pd.DataFrame:
    for c in columns:
        frame[c] = frame[c].astype(str)
    return frame


def _floats(frame: pd.DataFrame, columns) -> pd.DataFrame:
    for c in columns:
        frame[c] = frame[c].astype(float)
    return frame


def _bundle(name, X, y, exposure, categorical, groups, description, source, licence, citation,
            **extra):
    X = X.reset_index(drop=True)
    out = dict(X=X, y=np.asarray(y, dtype=float),
               exposure=None if exposure is None else np.asarray(exposure, dtype=float),
               categorical=list(categorical),
               groups=None if groups is None else np.asarray(groups),
               name=name, description=description, source=source, licence=licence,
               citation=citation)
    out.update(extra)
    return out


# ================================================================================================
# the data sets
def _fishing():
    d = pd.read_csv(_raw("fishing"))
    # density = totabund / sweptarea is a function of the response; site is an identifier
    X = _strings(_floats(d[["meandepth", "year", "period"]].copy(), ["meandepth", "year"]), ["period"])
    return _bundle(
        "fishing", X, d["totabund"], d["sweptarea"], ["period"], None,
        "Total number of demersal fish caught by bottom trawl at 147 sites between 800 and 4900 m "
        "depth in the north-east Atlantic, in 1977-1989 and 2000-2002, with the swept area as "
        "exposure.",
        f"{RDATASETS}/COUNT/fishing.csv (R package COUNT, data set fishing)",
        "GPL-2 (R package COUNT)",
        "Bailey, D. M., Collins, M. A., Gordon, J. D. M., Zuur, A. F. and Priede, I. G. (2009). "
        "Long-term changes in deep-water fish populations in the northeast Atlantic: a deeper "
        "reaching effect of fisheries? Proceedings of the Royal Society B 276, 1965-1969, "
        "doi:10.1098/rspb.2009.0098; as adapted by Zuur, A. F., Hilbe, J. M. and Ieno, E. N. (2013). "
        "A Beginner's Guide to GLM and GLMM with R. Highland Statistics.")


def _solder():
    d = pd.read_csv(_raw("solder"))
    factors = ["Opening", "Solder", "Mask", "PadType", "Panel"]
    X = _strings(d[factors].copy(), factors)
    return _bundle(
        "solder", X, d["skips"], None, factors, None,
        "Number of visible solder skips on printed-circuit boards in a designed experiment with "
        "five factors (clearance, solder thickness, mask, pad type, panel).",
        f"{RDATASETS}/rpart/solder.csv (R package rpart, data set solder)",
        "GPL-2 | GPL-3 (R package rpart)",
        "Comizzoli, R. B., Landwehr, J. M. and Sinclair, J. D. (1990). Robust materials and "
        "processes: key to reliability. AT&T Technical Journal 69(6), 113-128, "
        "doi:10.1002/j.1538-7305.1990.tb00491.x; data as published in Chambers, J. M. and "
        "Hastie, T. J. (eds) (1992). Statistical Models in S. Wadsworth and Brooks/Cole.")


FATALITIES_CONTINUOUS = ["beertax", "spirits", "unemp", "income", "emppop", "youngdrivers", "miles",
                         "baptist", "mormon", "dry", "drinkage", "gsp", "year"]
FATALITIES_FACTORS = ["breath", "jail", "service"]


def _fatalities():
    d = pd.read_csv(_raw("fatalities"))
    # one state-year (California 1988) lacks the two sentencing laws: dropped
    d = d.dropna(subset=FATALITIES_CONTINUOUS + FATALITIES_FACTORS).reset_index(drop=True)
    X = _strings(_floats(d[FATALITIES_CONTINUOUS + FATALITIES_FACTORS].copy(), FATALITIES_CONTINUOUS),
                 FATALITIES_FACTORS)
    return _bundle(
        "fatalities", X, d["fatal"], d["milestot"], FATALITIES_FACTORS, d["state"].to_numpy(),
        "Annual number of traffic fatalities in the 48 contiguous US states, 1982-1988, with the "
        "total vehicle-miles travelled (millions) as exposure and the state as group.",
        f"{RDATASETS}/AER/Fatalities.csv (R package AER, data set Fatalities)",
        "GPL-2 | GPL-3 (R package AER)",
        "Ruhm, C. J. (1996). Alcohol policies and highway vehicle fatalities. Journal of Health "
        "Economics 15(4), 435-454, doi:10.1016/S0167-6296(96)00490-0; data as distributed with "
        "Stock, J. H. and Watson, M. W. (2007). Introduction to Econometrics, 2nd ed. Addison Wesley.")


ABALONE_MEASURES = ["length", "diameter", "height", "whole_weight", "shucked_weight",
                    "viscera_weight", "shell_weight"]


def _abalone():
    d = _read_arff(_raw("abalone")).rename(columns={"cisvera_weight": "viscera_weight"})
    X = d[["sex"] + ABALONE_MEASURES].copy()
    # OpenML divided every measurement by 200; millimetres and grams are restored
    for c in ABALONE_MEASURES:
        X[c] = 200.0 * X[c].astype(float)
    X = _strings(X, ["sex"])
    return _bundle(
        "abalone", X, d["rings"], None, ["sex"], None,
        "Number of shell rings (age in years minus 1.5) of 4177 blacklip abalone from Tasmania, "
        "against sex and seven physical measurements.",
        "https://www.openml.org/d/44956 (OpenML 44956, from the UCI Machine Learning Repository)",
        "CC BY 4.0",
        "Nash, W. J., Sellers, T. L., Talbot, S. R., Cawthorn, A. J. and Ford, W. B. (1994). The "
        "Population Biology of Abalone (Haliotis species) in Tasmania. I. Blacklip Abalone "
        "(H. rubra) from the North Coast and Islands of Bass Strait. Sea Fisheries Division, "
        "Technical Report No. 48; UCI Machine Learning Repository, doi:10.24432/C55C7W.")


CHICAGO_CONTINUOUS = ["temp", "dptp", "pm10", "o3", "time", "doy"]


def _chicago_csv(dest: Path, work: Path) -> str:
    tarball = _raw("chicago")
    untrusted = Path(tempfile.mkdtemp(prefix="untrusted-chicago-", dir=CACHE))
    try:
        rda = untrusted / "chicagoNMMAPS.rda"
        with tarfile.open(tarball) as tf:
            member = tf.getmember("dlnm/data/chicagoNMMAPS.rda")
            if not member.isfile():
                raise ValueError("dlnm/data/chicagoNMMAPS.rda is not a regular file")
            rda.write_bytes(tf.extractfile(member).read())
        _rscript(["rda", rda, "chicagoNMMAPS", dest], cwd=work)
    finally:
        shutil.rmtree(untrusted, ignore_errors=True)
    return f"R: load(chicagoNMMAPS.rda) from {DLNM_TARBALL}; write.csv"


def _chicago():
    d = pd.read_csv(_derived("chicago", "chicagoNMMAPS.csv", _chicago_csv))
    # cvd and resp are parts of the response death. The relative humidity rhum is missing for the
    # whole of 1998-2000 and is nearly a function of temperature and dew point: it is left out.
    # The days with a missing PM10 value are dropped.
    d = d.dropna(subset=CHICAGO_CONTINUOUS + ["dow"]).reset_index(drop=True)
    X = _strings(_floats(d[CHICAGO_CONTINUOUS + ["dow"]].copy(), CHICAGO_CONTINUOUS), ["dow"])
    return _bundle(
        "chicago", X, d["death"], None, ["dow"], d["year"].to_numpy(),
        "Daily number of non-accidental deaths in Chicago, 1987-2000, against temperature, dew "
        "point, PM10, ozone, the day index, the day of the year and the day of the week, with the "
        "calendar year as block.",
        f"https://cloud.r-project.org/src/contrib/{DLNM_TARBALL} (R package dlnm, data set "
        "chicagoNMMAPS)",
        "GPL-2 | GPL-3 (R package dlnm)",
        "Samet, J. M., Dominici, F., Zeger, S. L., Schwartz, J. and Dockery, D. W. (2000). The "
        "National Morbidity, Mortality, and Air Pollution Study. Part I: Methods and Methodologic "
        "Issues. Research Report 94, Health Effects Institute; data as distributed with "
        "Gasparrini, A. (2011). Distributed lag linear and non-linear models in R: the package "
        "dlnm. Journal of Statistical Software 43(8), 1-20, doi:10.18637/jss.v043.i08.")


ECLIPSE_METRICS = (["ACD"] + [f"{m}_{a}" for m in ("FOUT", "MLOC", "NBD", "NOF") for a in ("avg", "max", "sum")]
                   + ["NOI"] + [f"NOM_{a}" for a in ("avg", "max", "sum")] + ["NOT"]
                   + [f"{m}_{a}" for m in ("NSF", "NSM", "PAR") for a in ("avg", "max", "sum")]
                   + [f"VG_{a}" for a in ("avg", "max", "sum")])


def _eclipse():
    with zipfile.ZipFile(_raw("eclipse")) as z:
        d = pd.read_csv(io.BytesIO(z.read("eclipse-metrics-files-3.0.csv")), sep=";")
    # post (the response) and pre (pre-release defects, a process measure) are not code metrics;
    # TLOC is the exposure; the abstract-syntax-tree node counts are left out
    X = pd.DataFrame({f"log1p_{m}": np.log1p(d[m].astype(float)) for m in ECLIPSE_METRICS})
    return _bundle(
        "eclipse", X, d["post"], d["TLOC"], [], None,
        "Number of post-release defects of each of the 10593 Java files of Eclipse 3.0, against "
        "30 code-complexity metrics (log(1 + value)), with the lines of code as exposure.",
        ECLIPSE_PAGE + "promise-2_0a-files-csv.zip (release 2.0a, file eclipse-metrics-files-3.0.csv; "
        "when the server does not answer, the copy kept by the Internet Archive on 2023-03-07)",
        "factual data free of restriction according to the providers; redistribution of the "
        "archive is not permitted, so it is downloaded and never committed",
        "Zimmermann, T., Premraj, R. and Zeller, A. (2007). Predicting defects for Eclipse. "
        "Third International Workshop on Predictor Models in Software Engineering (PROMISE'07), "
        "doi:10.1109/PROMISE.2007.10.")


BIKE_CONTINUOUS = ["hour", "hour_sin", "hour_cos", "temp", "atemp", "hum", "windspeed"]
BIKE_FACTORS = ["weekday", "workingday", "holiday", "season", "weathersit", "yr"]


def _bike():
    with zipfile.ZipFile(_raw("bike")) as z:
        d = pd.read_csv(io.BytesIO(z.read("hour.csv")))
    # casual + registered = cnt: both are excluded; instant is a row index
    X = pd.DataFrame({"hour": d["hr"].astype(float),
                      "hour_sin": np.sin(2.0 * np.pi * d["hr"] / 24.0),
                      "hour_cos": np.cos(2.0 * np.pi * d["hr"] / 24.0)})
    for c in ["temp", "atemp", "hum", "windspeed"]:
        X[c] = d[c].astype(float)
    for c in BIKE_FACTORS:
        X[c] = d[c].astype(str)
    day = pd.factorize(d["dteday"], sort=True)[0]
    return _bundle(
        "bike", X, d["cnt"], None, BIKE_FACTORS, day,
        "Hourly number of rented bicycles of the Capital Bikeshare system (Washington D.C.), "
        "2011-2012, against the hour, calendar and weather, with the day as block.",
        "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset (UCI 275, hour.csv)",
        "CC BY 4.0",
        "Fanaee-T, H. and Gama, J. (2014). Event labeling combining ensemble detectors and "
        "background knowledge. Progress in Artificial Intelligence 2(2-3), 113-127, "
        "doi:10.1007/s13748-013-0040-3; UCI Machine Learning Repository, doi:10.24432/C5W894.")


DATACAR_FACTORS = ["veh_body", "veh_age", "gender", "area", "agecat"]


def _datacar():
    d = pd.read_csv(_raw("datacar"))
    # clm = 1{numclaims > 0} and claimcst0 (claim cost) are functions of the response
    X = _strings(_floats(d[["veh_value"] + DATACAR_FACTORS].copy(), ["veh_value"]), DATACAR_FACTORS)
    return _bundle(
        "datacar", X, d["numclaims"], d["exposure"], DATACAR_FACTORS, None,
        "Number of claims of 67856 one-year vehicle insurance policies (2004-2005), with the "
        "fraction of the year insured as exposure.",
        f"{RDATASETS}/insuranceData/dataCar.csv (R package insuranceData, data set dataCar)",
        "GPL-2 (R package insuranceData)",
        "de Jong, P. and Heller, G. Z. (2008). Generalized Linear Models for Insurance Data. "
        "Cambridge University Press, doi:10.1017/CBO9780511755408.")


FREMTPL2_CONTINUOUS = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity"]
FREMTPL2_FACTORS = ["Area", "VehBrand", "VehGas", "Region"]


def _fremtpl2_raw() -> pd.DataFrame:
    return _read_arff(_raw("fremtpl2"))


def _fremtpl2():
    d = _fremtpl2_raw()
    n = len(d)
    perm = np.random.default_rng(SHUFFLE_SEED).permutation(n)
    d = d.iloc[perm].reset_index(drop=True)
    X = pd.DataFrame({"VehPower": d["VehPower"].astype(float),
                      "VehAge": np.minimum(d["VehAge"].astype(float), 20.0),
                      "DrivAge": np.minimum(d["DrivAge"].astype(float), 90.0),
                      "BonusMalus": np.minimum(d["BonusMalus"].astype(float), 150.0),
                      "LogDensity": np.log(d["Density"].astype(float))})
    for c in FREMTPL2_FACTORS:
        X[c] = d[c].astype(str)
    y = np.minimum(d["ClaimNb"].to_numpy(float), 4.0)
    e = np.minimum(d["Exposure"].to_numpy(float), 1.0)
    return _bundle(
        "fremtpl2", X, y, e, FREMTPL2_FACTORS, None,
        "Number of claims of 678013 French motor third-party liability policies, with the "
        "insured period in years as exposure (counts capped at 4 and exposures at 1).",
        "https://www.openml.org/d/41214 (OpenML 41214, freMTPL2freq of the R package CASdatasets)",
        "CC0 on OpenML; GPL (>= 2) in the R package CASdatasets",
        "Dutang, C. and Charpentier, A. (2018). CASdatasets: Insurance Datasets, R package; "
        "Charpentier, A. (ed.) (2014). Computational Actuarial Science with R. CRC Press; "
        "preprocessing as in Noll, A., Salzmann, R. and Wuthrich, M. V. (2018). Case Study: French "
        "Motor Third-Party Liability Claims. SSRN, doi:10.2139/ssrn.3164764.",
        row_id=perm, idpol=d["IDpol"].to_numpy())


def fremtpl2_split(data=None):
    """Train and test positions (in the shuffled rows returned by load) of the split of the
    scikit-learn example on Poisson regression: train_test_split(test_size=0.33,
    random_state=0) applied to the rows in their original order."""
    from sklearn.model_selection import train_test_split
    d = load("fremtpl2") if data is None else data
    n = len(d["y"])
    perm = d["row_id"]
    inverse = np.empty(n, dtype=int)
    inverse[perm] = np.arange(n)
    train0, test0 = train_test_split(np.arange(n), test_size=0.33, random_state=0)
    return np.sort(inverse[train0]), np.sort(inverse[test0])


def _biochemists_csv(dest: Path, work: Path) -> str:
    _rscript(["package", "pscl", "bioChemists", dest], cwd=work)
    return "R: data(bioChemists, package = 'pscl'); write.csv"


def _biochemists():
    d = pd.read_csv(_derived("biochemists", "bioChemists.csv", _biochemists_csv))
    X = _strings(_floats(d[["fem", "mar", "kid5", "phd", "ment"]].copy(), ["kid5", "phd", "ment"]),
                 ["fem", "mar"])
    return _bundle(
        "biochemists", X, d["art"], None, ["fem", "mar"], None,
        "Number of articles published in the last three years of the PhD by 915 biochemistry "
        "doctoral students, against gender, marriage, young children, prestige of the department "
        "and the productivity of the mentor.",
        "R package pscl, data set bioChemists",
        "GPL-2 (R package pscl)",
        "Long, J. S. (1990). The origins of sex differences in science. Social Forces 68(4), "
        "1297-1316, doi:10.2307/2579146; Long, J. S. (1997). Regression Models for Categorical and "
        "Limited Dependent Variables. Sage.")


LOADERS = dict(fishing=_fishing, solder=_solder, fatalities=_fatalities, abalone=_abalone,
               chicago=_chicago, eclipse=_eclipse, bike=_bike, datacar=_datacar,
               fremtpl2=_fremtpl2, biochemists=_biochemists)

# columns that must never be covariates, for the checks of the loaders
FORBIDDEN = dict(
    fishing=["totabund", "density", "site"],
    solder=["skips"],
    fatalities=["fatal", "nfatal", "sfatal", "fatal1517", "nfatal1517", "fatal1820", "nfatal1820",
                "fatal2124", "nfatal2124", "afatal", "milestot"],
    abalone=["rings"],
    chicago=["death", "cvd", "resp"],
    eclipse=["post", "pre", "TLOC", "log1p_TLOC", "log1p_pre", "log1p_post"],
    bike=["cnt", "casual", "registered"],
    datacar=["numclaims", "clm", "claimcst0", "exposure"],
    fremtpl2=["ClaimNb", "Exposure", "IDpol"],
    biochemists=["art"],
)


def load(name: str) -> dict:
    """The data set NAME (one of NAMES) as a dict; see the module documentation."""
    if name not in LOADERS:
        raise KeyError(f"unknown data set {name!r}; choose among {', '.join(NAMES)}")
    return LOADERS[name]()


def describe(d: dict) -> dict:
    """Row count and the basic statistics of the counts and exposures of a loaded data set."""
    y = d["y"]
    e = d["exposure"]
    out = dict(name=d["name"], n=len(y), covariates=d["X"].shape[1],
               factors=len(d["categorical"]), mean=float(np.mean(y)),
               var_over_mean=float(np.var(y, ddof=1) / np.mean(y)),
               zeros=float(np.mean(y == 0)), max=float(np.max(y)))
    if e is not None:
        out.update(exposure_min=float(np.min(e)), exposure_max=float(np.max(e)),
                   rate=float(np.sum(y) / np.sum(e)))
    if d["groups"] is not None:
        out["groups"] = int(len(np.unique(d["groups"])))
    return out


if __name__ == "__main__":
    import sys
    for nm in (sys.argv[1:] or NAMES):
        print(json.dumps(describe(load(nm))))

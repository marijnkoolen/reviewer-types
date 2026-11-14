"""P2: genre and fame effects, within reviewer vs between reviewer (silver).

Frequentist high-dimensional FE regressions (pyfixest) on the full eligible
corpus (dated >=1999, platform_1 excluded: 445,531 reviews). Not the Bayesian
ADVI models of section 14 (unstable across seeds, see _pre_p0_reviewer_key).

Part 1  genre: pooled (genre + year FE + platform FE) vs within-reviewer
        (genre | reviewer + year); pooled also run on the identifying
        subsample (reviewers with reviews in >=2 genres).
Part 2  fame: book_fame + author_fame under (i) platform FE + genre dummies,
        (ii) reviewer FE, (iii) reviewer FE + work FE (all with year FE).
Part 3  book_fame x genre and author_fame x genre slopes under (i) and (ii),
        BH q-values across the interaction family.

Specs: A = year FE; B = year FE + log(num_sentences) (not for the log-length outcome).
SEs: clustered two-way by reviewer and work (CRV1 "rev+work"); genre is a
work-level attribute so work clustering matters for genre coefficients.
Fame construction copied verbatim from 16c/17c/18c (lagged log1p prior review
count per work_id / author_id / user_id, sorted by id then review_date) and
verified against the 16c sample parquet.

Usage: venv/bin/python src/29c_within_reviewer_effects.py [--quick]
Writes results/29_within_reviewer_effects/*.csv
"""
import os
import sys
import time
import warnings
import zlib
import multiprocessing as mp
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "29_within_reviewer_effects"
OUT.mkdir(exist_ok=True)
QUICK = "--quick" in sys.argv
NPROC = 8
B_BOOT = 20 if QUICK else 100
SEED = 29
REF = "Literary_fiction"
MAIN7 = ["Author", "Classification", "Content", "Other_works", "Reader_response", "Recommendations", "Style"]
CLUSTER = {"CRV1": "rev+work"}

# outcome name -> column
P12_OUTCOMES = ([f"y_{c}" for c in MAIN7] + ["y_loglen", "y_Content_Narrative", "y_contig",
                "y_onset_Content", "y_onset_Reader_response"])
P3_OUTCOMES = [f"y_{c}" for c in MAIN7] + ["y_Content_Narrative", "y_Content_Other", "y_Content_Quote"]
if QUICK:
    P12_OUTCOMES = ["y_Content", "y_Author", "y_loglen"]
    P3_OUTCOMES = ["y_Content_Narrative", "y_Author"]


# ----------------------------------------------------------------------------- data
def build_lagged_covariates():
    """Verbatim from 16c/17c/18c."""
    review = pd.read_parquet(RES / "00_data_prep/silver_review_features.parquet")
    review = review[review["source_id"] != "platform_1"].copy()
    review["review_date"] = pd.to_datetime(review["review_date"], errors="coerce")
    review = review[review["review_date"].notna()]
    review = review[review["review_date"].dt.year >= 1999]
    for id_col, out_col in [("work_id", "book_fame"), ("author_id", "author_fame"), ("user_id", "reviewer_experience")]:
        review = review.sort_values([id_col, "review_date"])
        prior_count = review.groupby(id_col).cumcount()
        review[out_col] = np.log1p(prior_count)
    review["year"] = review["review_date"].dt.year
    return review


def prep():
    f = build_lagged_covariates()
    # verification against 16c sample (same construction -> identical values)
    s16 = pd.read_parquet(RES / "16_bayesian_coverage_model/silver_bayesian_coverage_sample.parquet",
                          columns=["review_id", "book_fame", "author_fame", "reviewer_experience"])
    m = s16.merge(f[["review_id", "book_fame", "author_fame", "reviewer_experience"]], on="review_id", suffixes=("_16", ""))
    ver = {c: float(np.abs(m[c] - m[c + "_16"]).max()) for c in ["book_fame", "author_fame", "reviewer_experience"]}
    ver["n_checked"] = len(m)
    print("fame construction check vs 16c sample (max abs diff):", ver, flush=True)
    pd.Series(ver).to_csv(OUT / "fame_construction_check.csv", header=["value"])

    f["loglen"] = np.log(f["num_sentences"])
    f["y_loglen"] = f["loglen"]
    for c in MAIN7 + ["Content--Narrative", "Content--Other", "Content--Quote"]:
        f["y_" + c.replace("--", "_")] = f["prop__" + c]
    c = pd.read_csv(RES / "22_review_contiguity_scaled/silver_review_contiguity_scaled.csv",
                    usecols=["review_id", "excess_over_chance", "eligible_a1"])
    c.loc[~c.eligible_a1, "excess_over_chance"] = np.nan
    f = f.merge(c[["review_id", "excess_over_chance"]].rename(columns={"excess_over_chance": "y_contig"}),
                on="review_id", how="left")
    s = pd.read_parquet(RES / "00_data_prep/silver_sentence_sequences.parquet",
                        columns=["review_id", "num_sentences", "rel_position", "Content", "Reader_response"])
    s = s[s.num_sentences > 1]
    on = pd.DataFrame({f"y_onset_{cat}": s.loc[s[cat] == 1].groupby("review_id").rel_position.min()
                       for cat in ["Content", "Reader_response"]})
    f = f.merge(on, left_on="review_id", right_index=True, how="left")
    f["g"] = f["nur_genre"].str.replace(" ", "_")
    f["rev"] = pd.factorize(f["user_id"])[0]
    f["work"] = pd.factorize(f["work_id"])[0]
    f["plat"] = pd.factorize(f["source_id"])[0]
    f = f.reset_index(drop=True)
    genres = sorted(f["g"].unique())
    assert REF in genres
    f["same_day"] = f.groupby(["work_id", "review_date"]).review_id.transform("size") > 1
    for g in genres:
        if g != REF:
            f["G_" + g] = (f["g"] == g).astype(float)
        f["bf_" + g] = f["book_fame"] * (f["g"] == g)
        f["af_" + g] = f["author_fame"] * (f["g"] == g)
    keep = (["review_id", "rev", "work", "plat", "year", "g", "book_fame", "author_fame", "reviewer_experience",
             "loglen", "same_day", "source_id", "author_id"]
            + [c for c in f.columns if c.startswith(("y_", "G_", "bf_", "af_"))])
    return f[keep], genres


DF, GENRES = None, None
NONREF = None


CACHE = Path(os.environ.get("P2_CACHE", "/tmp/p2_cache.parquet"))


def init(load=False):
    global DF, GENRES, NONREF
    if load:
        DF = pd.read_parquet(CACHE)
        GENRES = sorted(DF["g"].unique())
    else:
        DF, GENRES = prep()
        DF.to_parquet(CACHE)
    NONREF = [g for g in GENRES if g != REF]


# ----------------------------------------------------------------------------- helpers
def fit(data, y, xs, fe, vcov=CLUSTER):
    fml = f"{y} ~ {' + '.join(xs)} | {fe}" if xs else f"{y} ~ 1 | {fe}"
    m = pf.feols(fml, data=data, vcov=vcov)
    names = list(m._coefnames)
    coef = pd.Series(np.asarray(m.coef()), index=names)
    V = pd.DataFrame(np.asarray(m._vcov), index=names, columns=names)
    return m, coef, V


def iter_filter(d, cols=("rev", "work"), minn=2):
    while True:
        n0 = len(d)
        for c in cols:
            d = d[d.groupby(c)[c].transform("size") >= minn]
        if len(d) == n0:
            return d


def genre_stats(b9, V9, counts):
    """b9: coefficient vector for NONREF genres (ref=0). counts: dict genre->n.
    Returns review-weighted SD of genre effects (raw and noise-corrected)."""
    order = [REF] + NONREF
    b = np.r_[0.0, b9]
    K = len(b)
    V = np.zeros((K, K))
    V[1:, 1:] = V9
    w = np.array([counts.get(g, 0) for g in order], float)
    w = w / w.sum()
    A = np.eye(K) - np.outer(np.ones(K), w)
    dev = A @ b
    W = np.diag(w)
    var_raw = dev @ W @ dev
    noise = np.trace(A.T @ W @ A @ V)
    return np.sqrt(var_raw), np.sqrt(max(var_raw - noise, 0.0)), w


def wcorr(a, b, w):
    ma, mb = (w * a).sum() / w.sum(), (w * b).sum() / w.sum()
    cov = (w * (a - ma) * (b - mb)).sum()
    return cov / np.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum())


def sd_only(b9, w10):
    b = np.r_[0.0, b9]
    m = (w10 * b).sum()
    return np.sqrt((w10 * (b - m) ** 2).sum())


# ----------------------------------------------------------------------------- Part 1
def part1(y):
    rng = np.random.default_rng(SEED + zlib.crc32(y.encode()) % 1000)
    gd = ["G_" + g for g in NONREF]
    base = DF.dropna(subset=[y])
    rows, coefrows = [], []
    for spec in ["A", "B"]:
        if spec == "B" and y == "y_loglen":
            continue
        ctrl = ["loglen"] if spec == "B" else []
        xs = gd + ctrl
        t0 = time.time()
        multi = base.groupby("rev")["g"].transform("nunique") >= 2
        sub = base[multi]
        _, cf, Vf = fit(base, y, xs, "year + plat")
        _, cs, Vs = fit(sub, y, xs, "year + plat")
        _, cw, Vw = fit(sub, y, xs, "year + rev")
        cnt_full = base["g"].value_counts().to_dict()
        cnt_sub = sub["g"].value_counts().to_dict()
        sd_f = genre_stats(cf[gd].values, Vf.loc[gd, gd].values, cnt_full)
        sd_s = genre_stats(cs[gd].values, Vs.loc[gd, gd].values, cnt_sub)
        sd_w = genre_stats(cw[gd].values, Vw.loc[gd, gd].values, cnt_sub)
        w10 = sd_s[2]
        r = np.corrcoef(cs[gd].values, cw[gd].values)[0, 1]
        rw = wcorr(cs[gd].values, cw[gd].values, w10[1:])
        # cluster bootstrap over reviewers (one-way, iid SEs inside; coefficients only)
        sub = sub.sort_values("rev")
        revs, starts, lens = np.unique(sub["rev"].values, return_index=True, return_counts=True)
        arr = sub[[y, "year", "plat"] + xs].reset_index(drop=True)
        ratios, corrs, sdps, sdws = [], [], [], []
        for b in range(B_BOOT):
            rr = rng.integers(0, len(revs), len(revs))
            ln = lens[rr]
            offs = np.cumsum(ln) - ln
            idx = np.repeat(starts[rr] - offs, ln) + np.arange(ln.sum())
            bd = arr.iloc[idx].copy()
            bd["brev"] = np.repeat(np.arange(len(rr)), ln)
            try:
                mp = pf.feols(f"{y} ~ {' + '.join(xs)} | year + plat", data=bd, vcov="iid")
                mw = pf.feols(f"{y} ~ {' + '.join(xs)} | year + brev", data=bd, vcov="iid")
            except Exception:
                continue
            bp = pd.Series(np.asarray(mp.coef()), index=mp._coefnames)[gd].values
            bw = pd.Series(np.asarray(mw.coef()), index=mw._coefnames)[gd].values
            sp, sw = sd_only(bp, w10), sd_only(bw, w10)
            sdps.append(sp); sdws.append(sw)
            ratios.append(sw / sp)
            corrs.append(np.corrcoef(bp, bw)[0, 1])
        q = lambda a, p: float(np.percentile(a, p)) if len(a) else np.nan
        rows.append(dict(
            outcome=y, spec=spec, n_full=len(base), n_sub=len(sub), reviewers_sub=len(revs),
            reviewers_full=base["rev"].nunique(), sd_y_sub=float(sub[y].std()),
            sd_pooled_full=sd_f[0], sd_pooled_sub=sd_s[0], sd_within=sd_w[0],
            sd_pooled_sub_corr=sd_s[1], sd_within_corr=sd_w[1],
            ratio_raw=sd_w[0] / sd_s[0], ratio_corr=sd_w[1] / sd_s[1] if sd_s[1] > 0 else np.nan,
            ratio_boot_lo=q(ratios, 2.5), ratio_boot_hi=q(ratios, 97.5),
            corr=r, corr_weighted=rw, corr_boot_lo=q(corrs, 2.5), corr_boot_hi=q(corrs, 97.5),
            n_boot=len(ratios), secs=time.time() - t0))
        for g in NONREF:
            k = "G_" + g
            coefrows.append(dict(outcome=y, spec=spec, genre=g, n_sub_genre=cnt_sub.get(g, 0),
                                 pooled_full=cf[k], se_pooled_full=np.sqrt(Vf.loc[k, k]),
                                 pooled_sub=cs[k], se_pooled_sub=np.sqrt(Vs.loc[k, k]),
                                 within=cw[k], se_within=np.sqrt(Vw.loc[k, k])))
    print(f"part1 done {y}", flush=True)
    return dict(summary=pd.DataFrame(rows), coefs=pd.DataFrame(coefrows))


# ----------------------------------------------------------------------------- Part 2
def part2(y):
    gd = ["G_" + g for g in NONREF]
    base = DF.dropna(subset=[y])
    rows = []
    for spec in ["A", "B"]:
        if spec == "B" and y == "y_loglen":
            continue
        ctrl = ["loglen"] if spec == "B" else []
        for exp in [False, True]:
            ex = ["reviewer_experience"] if exp else []
            designs = {
                "i_platform+genre": (base, gd + ["book_fame", "author_fame"] + ex + ctrl, "year + plat"),
                "ii_reviewer": (base[base.groupby("rev")["rev"].transform("size") >= 2],
                                gd + ["book_fame", "author_fame"] + ex + ctrl, "year + rev"),
                "iii_reviewer+work": (iter_filter(base), ["book_fame", "author_fame"] + ex + ctrl, "year + rev + work"),
            }
            for dn, (d, xs, fe) in designs.items():
                try:
                    _, cf, V = fit(d, y, xs, fe)
                except Exception as e:
                    print("FAIL", y, spec, dn, e, flush=True)
                    continue
                sdy = d[y].std()
                for v in ["book_fame", "author_fame"] + ex:
                    if v not in cf.index:
                        rows.append(dict(outcome=y, spec=spec, design=dn, with_exp=exp, var=v, coef=np.nan))
                        continue
                    se = np.sqrt(V.loc[v, v])
                    rows.append(dict(outcome=y, spec=spec, design=dn, with_exp=exp, var=v, coef=cf[v], se=se,
                                     z=cf[v] / se, n=len(d), reviewers=d["rev"].nunique(), works=d["work"].nunique(),
                                     sd_x=float(d[v].std()), sd_y=float(sdy),
                                     std_effect=cf[v] * d[v].std() / sdy))
    print(f"part2 done {y}", flush=True)
    return dict(p2=pd.DataFrame(rows))


# ----------------------------------------------------------------------------- Part 3
def part3(y):
    gd = ["G_" + g for g in NONREF]
    bf = ["bf_" + g for g in GENRES]
    af = ["af_" + g for g in GENRES]
    base = DF.dropna(subset=[y])
    slopes, contr, joint = [], [], []
    for spec in ["A", "B"]:
        ctrl = ["loglen"] if spec == "B" else []
        for dn, (d, fe) in {"i_platform+genre": (base, "year + plat"),
                            "ii_reviewer": (base[base.groupby("rev")["rev"].transform("size") >= 2], "year + rev")}.items():
            _, cf, V = fit(d, y, bf + af + gd + ctrl, fe)
            for fam, cols in [("book_fame", bf), ("author_fame", af)]:
                pre = cols[0][:3]
                for g in GENRES:
                    k = pre + g
                    slopes.append(dict(outcome=y, spec=spec, design=dn, fame=fam, genre=g, slope=cf[k],
                                       se=np.sqrt(V.loc[k, k]), n_genre=int((d["g"] == g).sum()), n=len(d)))
                kref = pre + REF
                for g in NONREF:
                    k = pre + g
                    c = cf[k] - cf[kref]
                    se = np.sqrt(V.loc[k, k] + V.loc[kref, kref] - 2 * V.loc[k, kref])
                    contr.append(dict(outcome=y, spec=spec, design=dn, fame=fam, contrast=f"{g} - {REF}",
                                      est=c, se=se, kind="vs_ref"))
                if fam == "book_fame":
                    k1, k2 = pre + "Historical_fiction", pre + "Suspense"
                    c = cf[k1] - cf[k2]
                    se = np.sqrt(V.loc[k1, k1] + V.loc[k2, k2] - 2 * V.loc[k1, k2])
                    contr.append(dict(outcome=y, spec=spec, design=dn, fame=fam,
                                      contrast="Historical_fiction - Suspense", est=c, se=se, kind="hist_vs_susp"))
                # joint Wald: all 10 slopes equal
                kk = [pre + g for g in GENRES]
                R = np.zeros((len(kk) - 1, len(kk)))
                for i in range(len(kk) - 1):
                    R[i, i], R[i, i + 1] = 1, -1
                bb = cf[kk].values
                Vk = V.loc[kk, kk].values
                diff = R @ bb
                W = diff @ np.linalg.solve(R @ Vk @ R.T, diff)
                joint.append(dict(outcome=y, spec=spec, design=dn, fame=fam, wald_chi2=W, df=len(kk) - 1,
                                  p=1 - sps.chi2.cdf(W, len(kk) - 1)))
    print(f"part3 done {y}", flush=True)
    return dict(slopes=pd.DataFrame(slopes), contr=pd.DataFrame(contr), joint=pd.DataFrame(joint))


# ----------------------------------------------------------------------------- diagnostics
def identification_diagnostics():
    """Residual variation of book_fame / author_fame under each design (base sample, no outcome NaNs)."""
    gd = ["G_" + g for g in NONREF]
    base = DF
    rows = []
    d3 = iter_filter(base)
    d2 = base[base.groupby("rev")["rev"].transform("size") >= 2]
    for dn, d, fe, xs in [("raw", base, None, None), ("i_year+platform+genre", base, "year + plat", gd),
                          ("ii_year+reviewer", d2, "year + rev", gd),
                          ("iii_year+reviewer+work", d3, "year + rev + work", [])]:
        for v in ["book_fame", "author_fame"]:
            if fe is None:
                sd, n = d[v].std(), len(d)
            else:
                m = pf.feols(f"{v} ~ {' + '.join(xs) if xs else '1'} | {fe}", data=d, vcov="iid")
                sd, n = float(np.std(m.resid(), ddof=1)), len(d)
            rows.append(dict(design=dn, var=v, n=n, reviewers=d["rev"].nunique(), works=d["work"].nunique(),
                             authors=d["author_id"].nunique(), sd=sd))
    out = pd.DataFrame(rows)
    out["sd_share_of_raw"] = out["sd"] / out.groupby("var")["sd"].transform("first")
    extra = dict(
        corr_book_author_fame_raw=float(base[["book_fame", "author_fame"]].corr().iloc[0, 1]),
        share_reviews_same_work_same_day=float(base["same_day"].mean()),
        share_works_book_fame_varies_after_filter=float((d3.groupby("work")["book_fame"].nunique() > 1).mean()),
        share_works_author_fame_varies_within_work=float((d3.groupby("work")["author_fame"].nunique() > 1).mean()),
        n_iii=len(d3), reviewers_iii=int(d3["rev"].nunique()), works_iii=int(d3["work"].nunique()),
        n_ii=len(d2), reviewers_ii=int(d2["rev"].nunique()), n_base=len(base),
        reviewers_base=int(base["rev"].nunique()), works_base=int(base["work"].nunique()))
    return out, pd.Series(extra)


def task(args):
    kind, y = args
    if kind == "diag":
        return kind, y, dict(res=identification_diagnostics())
    return kind, y, {"p1": part1, "p2": part2, "p3": part3}[kind](y)


def main():
    t0 = time.time()
    init()
    print(f"data: {len(DF)} reviews, {DF.rev.nunique()} reviewers, {DF.work.nunique()} works, genres {GENRES}", flush=True)
    tasks = ([("diag", "diag")] + [("p1", y) for y in P12_OUTCOMES] + [("p2", y) for y in P12_OUTCOMES] + [("p3", y) for y in P3_OUTCOMES])
    results = {}
    with mp.get_context('spawn').Pool(NPROC, initializer=init, initargs=(True,)) as pool:
        for kind, y, r in pool.imap_unordered(task, tasks):
            results[(kind, y)] = r
            print(f"[{time.time() - t0:.0f}s] finished {kind} {y}", flush=True)

    ident, extra = results[("diag", "diag")]["res"]
    ident.to_csv(OUT / "identification_fame.csv", index=False)
    extra.to_csv(OUT / "identification_extra.csv", header=["value"])
    print(ident.to_string(), "\n", extra.to_string(), flush=True)

    def cat(kind, key, ys):
        return pd.concat([results[(kind, y)][key] for y in ys], ignore_index=True)

    cat("p1", "summary", P12_OUTCOMES).to_csv(OUT / "part1_summary.csv", index=False)
    cat("p1", "coefs", P12_OUTCOMES).to_csv(OUT / "part1_genre_coefs.csv", index=False)
    cat("p2", "p2", P12_OUTCOMES).to_csv(OUT / "part2_fame.csv", index=False)
    slopes = cat("p3", "slopes", P3_OUTCOMES)
    contr = cat("p3", "contr", P3_OUTCOMES)
    joint = cat("p3", "joint", P3_OUTCOMES)
    contr["z"] = contr["est"] / contr["se"]
    contr["p"] = 2 * sps.norm.sf(np.abs(contr["z"]))
    contr["q_bh"] = np.nan
    for (spec, dn), idx in contr.groupby(["spec", "design"]).groups.items():
        contr.loc[idx, "q_bh"] = multipletests(contr.loc[idx, "p"], method="fdr_bh")[1]
    slopes["z"] = slopes["slope"] / slopes["se"]
    slopes["p"] = 2 * sps.norm.sf(np.abs(slopes["z"]))
    joint["q_bh"] = np.nan
    for (spec, dn), idx in joint.groupby(["spec", "design"]).groups.items():
        joint.loc[idx, "q_bh"] = multipletests(joint.loc[idx, "p"], method="fdr_bh")[1]
    slopes.to_csv(OUT / "part3_slopes.csv", index=False)
    contr.to_csv(OUT / "part3_contrasts.csv", index=False)
    joint.to_csv(OUT / "part3_heterogeneity_wald.csv", index=False)
    print(f"ALL DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

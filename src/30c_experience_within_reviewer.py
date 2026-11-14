"""P4: reviewer experience, within vs between reviewers (silver).

Pooled / within-reviewer (reviewer FE) / Mundlak models of outcome ~ experience,
experience = log1p(# prior dated reviews of the reviewer), lagged as in 16c.
Identification checks (APC problem): within-reviewer corr(experience, time),
tenure control, pace-varying subsample, platform x year FE. Plus the adjusted
early-3 vs last-3 paired comparison of 15c.
Usage: venv/bin/python src/30c_experience_within_reviewer.py
Writes results/30_experience_within_reviewer/*.csv
"""
import sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "30_experience_within_reviewer"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS  # noqa: E402

LABELS = [m for m in MAIN_LABELS if m != "None"]
OUTCOMES = [f"prop__{l}" for l in LABELS] + ["loglen", "rating", "contig_excess", "specificity"]
STEP50 = np.log1p(50)  # experience contrast 0 -> 50 prior reviews


def build():
    r = pd.read_parquet(RES / "00_data_prep/silver_review_features.parquet")
    r = r[r["source_id"] != "platform_1"].copy()
    r["n_total"] = r.groupby("user_id")["review_id"].transform("size")  # incl. undated (15c 'prolific' = >=10)
    r["review_date"] = pd.to_datetime(r["review_date"], errors="coerce")
    r = r[r["review_date"].notna() & (r["review_date"].dt.year >= 1999)]
    r = r.sort_values(["user_id", "review_date", "review_id"]).reset_index(drop=True)
    g = r.groupby("user_id")
    r["n_dated"] = g["review_id"].transform("size")
    r["experience"] = np.log1p(g.cumcount())
    r["year"] = r["review_date"].dt.year
    r["t"] = r["year"] + (r["review_date"].dt.dayofyear - 1) / 365.25  # fractional calendar year
    r["tenure"] = r["t"] - g["t"].transform("min")
    r["loglen"] = np.log(r["num_sentences"])
    r["genre"] = r["nur_genre"].astype(str)
    r["platform"] = r["source_id"].astype(str)
    r["pxy"] = r["platform"] + "_" + r["year"].astype(str)
    ds = pd.read_csv(RES / "09_length_dimension/silver_density_specificity_by_review.csv")[["review_id", "specificity"]]
    c = pd.read_csv(RES / "22_review_contiguity_scaled/silver_review_contiguity_scaled.csv",
                    usecols=["review_id", "excess_over_chance", "eligible_a1"])
    c.loc[~c["eligible_a1"], "excess_over_chance"] = np.nan
    c = c.rename(columns={"excess_over_chance": "contig_excess"})[["review_id", "contig_excess"]]
    r = r.merge(ds, on="review_id", how="left").merge(c, on="review_id", how="left")
    r["rating"] = pd.to_numeric(r["rating"], errors="coerce")
    return r


def prep_samples(r):
    s = r[r["n_dated"] >= 2].copy()
    g = s.groupby("user_id")
    s["exp_bar"] = g["experience"].transform("mean")
    s["exp_dev"] = s["experience"] - s["exp_bar"]
    s["t_bar"] = g["t"].transform("mean")
    # within-reviewer R2 of experience on calendar time (reviewers with >=4 reviews)
    def r2(d):
        if len(d) < 4 or d["t"].std() == 0 or d["experience"].std() == 0:
            return np.nan
        return np.corrcoef(d["t"], d["experience"])[0, 1] ** 2
    R2 = s.groupby("user_id")[["t", "experience"]].apply(r2).rename("r2_time")
    s = s.merge(R2, left_on="user_id", right_index=True, how="left")
    return s


def tidy(fit, term):
    t = fit.tidy()
    row = t.loc[term]
    return float(row["Estimate"]), float(row["Std. Error"])


def run_outcome(args):
    y, spec = args
    S = SAMPLE
    d = S.dropna(subset=[y])
    if spec == "B":
        if y == "loglen":
            return []
        ctrl = " + loglen"
    else:
        ctrl = ""
    rows = []
    cl = {"CRV1": "user_id"}
    def add(model, term, fit, n_rev, extra=""):
        b, se = tidy(fit, term)
        rows.append(dict(outcome=y, spec=spec, model=model, term=term, coef=b, se=se,
                         effect_0_to_50=b * STEP50, n_obs=fit._N, n_reviewers=n_rev, note=extra))
    nr = d["user_id"].nunique()
    # 1 pooled (>=2 sample) and pooled on full sample
    f = pf.feols(f"{y} ~ experience{ctrl} | year + platform + genre", d, vcov=cl); add("1_pooled", "experience", f, nr)
    dfull = FULL.dropna(subset=[y])
    f = pf.feols(f"{y} ~ experience{ctrl} | year + platform + genre", dfull, vcov=cl)
    add("1_pooled_full_sample", "experience", f, dfull["user_id"].nunique())
    # 2 within reviewer
    f = pf.feols(f"{y} ~ experience{ctrl} | user_id + year + genre", d, vcov=cl); add("2_within", "experience", f, nr)
    # 3 Mundlak
    f = pf.feols(f"{y} ~ exp_bar + exp_dev{ctrl} | year + platform + genre", d, vcov=cl)
    add("3_mundlak", "exp_bar", f, nr, "between"); add("3_mundlak", "exp_dev", f, nr, "within")
    f = pf.feols(f"{y} ~ exp_bar + exp_dev + t_bar{ctrl} | year + platform + genre", d, vcov=cl)
    add("3b_mundlak_plus_mean_time", "exp_bar", f, nr, "between"); add("3b_mundlak_plus_mean_time", "exp_dev", f, nr, "within")
    # tenure checks
    f = pf.feols(f"{y} ~ experience + tenure{ctrl} | user_id + year + genre", d, vcov=cl)
    add("4_within_tenure", "experience", f, nr, "fractional tenure (identified only by within-year timing)")
    add("4_within_tenure", "tenure", f, nr, "per year")
    # platform x year FE
    f = pf.feols(f"{y} ~ experience{ctrl} | user_id + pxy + genre", d, vcov=cl)
    add("5_within_platformXyear", "experience", f, nr)
    # pace-varying subsample: >=4 reviews, within-reviewer R2(exp~time) < 0.8
    dp = d[(d["r2_time"] < 0.8)]
    nrp = dp["user_id"].nunique()
    f = pf.feols(f"{y} ~ experience{ctrl} | user_id + year + genre", dp, vcov=cl); add("6_within_pace_varying", "experience", f, nrp)
    f = pf.feols(f"{y} ~ experience + tenure{ctrl} | user_id + year + genre", dp, vcov=cl)
    add("6_within_pace_varying+tenure", "experience", f, nrp)
    return rows


def init_globals():
    global SAMPLE, FULL
    SAMPLE, FULL = pd.read_pickle(OUT / "_sample.pkl"), pd.read_pickle(OUT / "_full.pkl")


def timecorr(S):
    rows = []
    for lab, d in [("all >=2", S), ("pace: >=4 reviews", S[S["r2_time"].notna()])]:
        w = lambda c: d[c] - d.groupby("user_id")[c].transform("mean")
        e, t = w("experience"), w("t")
        # partial out year FE from the within-demeaned variables
        ey = e - e.groupby(d["year"]).transform("mean"); ty = t - t.groupby(d["year"]).transform("mean")
        rows.append(dict(sample=lab, n_obs=len(d), n_reviewers=d["user_id"].nunique(),
                         within_corr_exp_time=np.corrcoef(e, t)[0, 1],
                         within_corr_exp_time_after_year=np.corrcoef(ey, ty)[0, 1],
                         within_sd_exp=e.std(), within_sd_time_years=t.std(),
                         share_reviewers_r2_ge_0_8=float((d.drop_duplicates("user_id")["r2_time"] >= 0.8).mean())
                         if lab != "all >=2" else np.nan,
                         median_r2=float(d.drop_duplicates("user_id")["r2_time"].median())))
    pd.DataFrame(rows).to_csv(OUT / "time_experience_correlation.csv", index=False)
    print(pd.DataFrame(rows).to_string())


def paired(r, N=3):
    outs = []
    for lab, dd0 in [("all >=6 dated", r[r["n_dated"] >= 2 * N]),
                     ("prolific (>=10 total), >=6 dated [15c sample]", r[(r["n_dated"] >= 2 * N) & (r["n_total"] >= 10)])]:
        o = paired_one(dd0, N); o.insert(0, "sample", lab); outs.append(o)
    out = pd.concat(outs); out.to_csv(OUT / "early_late_paired_adjusted.csv", index=False)
    return out


def paired_one(d, N=3):
    d = d.copy()
    g = d.groupby("user_id")
    d["rk"] = g.cumcount() + 1
    d["rke"] = g.cumcount(ascending=False) + 1
    ys = [f"prop__{l}" for l in LABELS] + ["num_sentences", "loglen", "rating", "specificity", "contig_excess"]
    rows = []
    for y in ys:
        for spec in ["unadjusted", "A", "B", "B_within_len", "A_FE", "B_FE"]:
            if spec in ("B", "B_within_len", "B_FE") and y in ("num_sentences", "loglen"):
                continue
            dd = d.dropna(subset=[y]).copy()
            dd = dd[dd.groupby("user_id")["rk"].transform("size") >= 2].copy()
            if spec == "unadjusted":
                dd["e"] = dd[y]
            elif spec == "A":
                dd["e"] = dd[y] - dd.groupby("year")[y].transform("mean")
            elif spec == "B":
                yy = dd[y] - dd.groupby("year")[y].transform("mean")
                xx = dd["loglen"] - dd.groupby("year")["loglen"].transform("mean")
                dd["e"] = yy - (xx * yy).sum() / (xx * xx).sum() * xx
            elif spec in ("A_FE", "B_FE"):
                # year effects (and length slope in B) estimated WITHIN reviewers jointly with experience;
                # e = y - reviewer FE - year FE (- length term): keeps only the experience component
                fm = f"{y} ~ experience" + (" + loglen" if spec == "B_FE" else "") + " | user_id + year"
                fit = pf.feols(fm, dd, fixef_rm="none")
                dd["e"] = fit.resid() + fit.coef()["experience"] * dd["experience"]
            else:  # length slope estimated within reviewer
                dd["e"] = pf.feols(f"{y} ~ loglen | year + user_id", dd, fixef_rm="none").resid()
            # reviewer FE residual drops reviewer level; differences between windows are what we use
            e = dd[dd["rk"] <= N].groupby("user_id")["e"].mean()
            l = dd[dd["rke"] <= N].groupby("user_id")["e"].mean()
            e, l = e.align(l, join="inner")
            diff = (l - e).dropna()
            if len(diff) < 10:
                continue
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            try:
                p = stats.wilcoxon(diff).pvalue
            except Exception:
                p = np.nan
            rows.append(dict(outcome=y, adjustment=spec, n_reviewers=len(diff), mean_early=e.loc[diff.index].mean(),
                             mean_late=l.loc[diff.index].mean(), mean_paired_diff=diff.mean(), se=se,
                             median_paired_diff=diff.median(), wilcoxon_p=p))
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    r = build()
    print("dated non-platform_1 reviews:", len(r), "reviewers:", r["user_id"].nunique(), flush=True)
    S = prep_samples(r)
    print("n_dated>=2 sample:", len(S), S["user_id"].nunique(), flush=True)
    print("n_dated>=6 reviewers:", r[r["n_dated"] >= 6]["user_id"].nunique(), flush=True)
    pd.DataFrame([dict(sample="all dated, no platform_1", n_obs=len(r), n_reviewers=r.user_id.nunique()),
                  dict(sample=">=2 dated reviews", n_obs=len(S), n_reviewers=S.user_id.nunique()),
                  dict(sample=">=6 dated reviews", n_obs=int((r.n_dated >= 6).sum()), n_reviewers=r[r.n_dated >= 6].user_id.nunique()),
                  dict(sample=">=4 dated, within-R2<0.8 (pace-varying)", n_obs=int((S.r2_time < 0.8).sum()),
                       n_reviewers=S[S.r2_time < 0.8].user_id.nunique())]).to_csv(OUT / "sample_sizes.csv", index=False)
    timecorr(S)
    if "--skip-paired" not in sys.argv:
        pr = paired(r)
        print(pr.to_string(), flush=True)
    if "--only-paired" in sys.argv:
        return
    S.to_pickle(OUT / "_sample.pkl"); r.to_pickle(OUT / "_full.pkl")
    jobs = [(y, s) for y in OUTCOMES for s in ["A", "B"]]
    import multiprocessing as mp
    with ProcessPoolExecutor(4, mp_context=mp.get_context('spawn'), initializer=init_globals) as ex:  # fork: globals inherited
        res = []
        for jb, out in zip(jobs, ex.map(run_outcome, jobs)):
            res.append(out); print("finished", jb, f"{(time.time()-t0)/60:.1f} min", flush=True)
    df = pd.DataFrame([x for rr in res for x in rr])
    df.to_csv(OUT / "regressions_long.csv", index=False)
    print(df.to_string(), flush=True)
    (OUT / "_sample.pkl").unlink(); (OUT / "_full.pkl").unlink()
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()

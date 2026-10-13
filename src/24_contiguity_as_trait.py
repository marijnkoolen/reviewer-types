"""Is contiguity a reviewer trait? (ANALYSIS_PLAN.md item 14, Q1.)

Items 3/6 already established that *composition* and *order* are personal
signatures (modal cluster share; per-reviewer similarity on compositional and
sequence-position vectors). Contiguity -- whether a reviewer keeps each
category in one block -- has never been tested as a trait.

Primary statistic: the intraclass correlation, the share of variance in a
per-review measure that sits BETWEEN reviewers rather than within them. This
is a different instrument from items 3/6 (which used modal share and pairwise
similarity), and it answers "is this a trait" directly.

Two things make a bare ICC uninterpretable here, and both are handled:

1. BENCHMARKS. An ICC of 0.12 means nothing in isolation. It is reported
   alongside measures already known to be traits (`Content` /
   `Reader_response` share, items 3/6) and alongside review length, which is
   the strongest personal signature in the data. The comparison is the
   finding, not the raw number.

2. LENGTH CONTROL, which is not optional. Contiguity excess correlates with
   review length (rho=+0.19, item 13 A1b) and length has ICC ~0.61, so part
   of contiguity's between-reviewer variance is really "this reviewer writes
   long reviews". Every measure is therefore reported three ways: raw, and
   residualized on length, and residualized on length + genre + platform
   (platform matters because item 8 found Cramer's V=0.70 between
   prolificness and platform). The residualized ICC is the actual answer.

ICC(1) is computed from one-way ANOVA components with the unequal-group-size
correction k0 = (N - sum(k_i^2)/N) / (a-1); a bootstrap over REVIEWERS (the
independent unit) gives the CI. Also reported: whether consistency itself is
a trait -- do some reviewers vary more review-to-review than others -- tested
against a null that shuffles reviews across reviewers, preserving each
reviewer's review count.

Usage: python src/24_contiguity_as_trait.py [gold|silver]  (default: silver)
Writes results/{source}_contiguity_trait_icc.csv,
  results/{source}_contiguity_trait_consistency.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "silver"
assert SOURCE in ("gold", "silver")

MIN_REVIEWS = 10          # eligible reviews per reviewer, not total reviews
N_BOOT = 500
N_SHUFFLE = 500
RNG_SEED = 0

MEASURES = [
    ("excess_over_chance", "contiguity excess"),
    ("boundary_excess", "boundary excess"),
    ("review_clumpiness", "raw clumpiness"),
    ("n_sentences", "review length [confound]"),
    ("prop__Content", "Content share [benchmark]"),
    ("prop__Reader_response", "Reader_response share [benchmark]"),
]


def load():
    contig = pd.read_csv(RESULTS / "22_review_contiguity_scaled" / f"{SOURCE}_review_contiguity_scaled.csv")
    if SOURCE == "gold":
        rf = pd.read_csv(RESULTS / "00_data_prep/gold_review_features.csv")
    else:
        rf = pd.read_parquet(RESULTS / "00_data_prep/silver_review_features.parquet")
    cols = ["review_id", "user_id", "source_id", "nur_genre",
            "prop__Content", "prop__Reader_response"]
    d = contig[contig["eligible_a1"]].merge(rf[cols], on="review_id")
    d = d[d["source_id"] != "platform_1"]          # placeholder user_id, ANALYSIS_PLAN §0
    counts = d["user_id"].value_counts()
    keep = counts[counts >= MIN_REVIEWS].index
    return d[d["user_id"].isin(keep)].reset_index(drop=True)


def icc1(values, groups):
    """ICC(1) from one-way ANOVA components, with the unequal-group-size
    correction. Returns 0 when between-group variance does not exceed
    within-group (a negative ICC estimate means 'no trait')."""
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    if df["g"].nunique() < 2:
        return np.nan
    grp = df.groupby("g")["v"]
    k, means = grp.size(), grp.mean()
    n, a = len(df), len(k)
    grand = df["v"].mean()
    msb = (k * (means - grand) ** 2).sum() / (a - 1)
    msw = ((df["v"] - df["g"].map(means)) ** 2).sum() / (n - a)
    k0 = (n - (k ** 2).sum() / n) / (a - 1)
    if msb + (k0 - 1) * msw <= 0:
        return 0.0
    return max(0.0, (msb - msw) / (msb + (k0 - 1) * msw))


def residualize(d, col, formula_rhs):
    """Residuals of `col` on the given covariates -- the ICC of these answers
    'is there a personal style beyond what the covariates already explain'."""
    sub = d[["user_id", col, "n_sentences", "nur_genre", "source_id"]].dropna()
    sub = sub.rename(columns={col: "_y"})
    model = smf.ols(f"_y ~ {formula_rhs}", data=sub).fit()
    return pd.Series(model.resid.to_numpy(), index=sub.index), sub["user_id"]


def bootstrap_icc(values, groups, rng, n_boot=N_BOOT):
    """CI by resampling REVIEWERS (the independent unit), not reviews."""
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    idx = {g: sub.to_numpy() for g, sub in df.groupby("g").groups.items()}
    users = np.array(list(idx.keys()))
    out = []
    for _ in range(n_boot):
        pick = rng.choice(len(users), size=len(users), replace=True)
        rows, labels = [], []
        for j, u in enumerate(pick):
            r = idx[users[u]]
            rows.append(r)
            labels.append(np.full(len(r), j))       # relabel so repeats stay distinct
        rows = np.concatenate(rows)
        out.append(icc1(df["v"].loc[rows].to_numpy(), np.concatenate(labels)))
    return float(np.nanpercentile(out, 2.5)), float(np.nanpercentile(out, 97.5))


def consistency_trait(d, col, rng):
    """Is CONSISTENCY itself a trait -- do some reviewers vary less
    review-to-review than others? Compared against a null that shuffles
    reviews across reviewers while preserving each reviewer's review count,
    which holds the marginal distribution fixed and destroys only the
    reviewer-review pairing."""
    sub = d[["user_id", col]].dropna()
    obs = sub.groupby("user_id")[col].std().dropna()
    vals = sub[col].to_numpy()
    codes = pd.factorize(sub["user_id"])[0]
    n_groups = codes.max() + 1
    null_spread = []
    for _ in range(N_SHUFFLE):
        perm = rng.permutation(vals)
        s = pd.Series(perm).groupby(codes).std().dropna()
        null_spread.append(s.std())
    obs_spread = obs.std()
    null_spread = np.array(null_spread)
    p = (1 + np.sum(null_spread >= obs_spread)) / (N_SHUFFLE + 1)
    return {"measure": col, "n_reviewers": int(n_groups),
            "observed_sd_of_per_reviewer_sd": round(float(obs_spread), 4),
            "null_sd_of_per_reviewer_sd": round(float(null_spread.mean()), 4),
            "ratio": round(float(obs_spread / null_spread.mean()), 3),
            "p_value": p}


def main():
    rng = np.random.default_rng(RNG_SEED)
    d = load()
    print(f"{SOURCE}: {d['user_id'].nunique():,} reviewers with >={MIN_REVIEWS} eligible reviews, "
          f"{len(d):,} reviews")

    rows = []
    for col, label in MEASURES:
        raw = icc1(d[col].to_numpy(), d["user_id"].to_numpy())
        lo, hi = bootstrap_icc(d[col].to_numpy(), d["user_id"].to_numpy(), rng)

        if col == "n_sentences":
            res_len = res_full = np.nan          # residualizing length on length is vacuous
        else:
            r1, g1 = residualize(d, col, "n_sentences")
            res_len = icc1(r1.to_numpy(), g1.to_numpy())
            r2, g2 = residualize(d, col, "n_sentences + C(nur_genre) + C(source_id)")
            res_full = icc1(r2.to_numpy(), g2.to_numpy())

        rows.append({"measure": col, "label": label,
                     "icc_raw": round(raw, 4), "icc_ci_lo": round(lo, 4), "icc_ci_hi": round(hi, 4),
                     "icc_resid_length": round(res_len, 4) if np.isfinite(res_len) else np.nan,
                     "icc_resid_length_genre_platform": round(res_full, 4) if np.isfinite(res_full) else np.nan})

    icc = pd.DataFrame(rows)
    icc["pct_of_raw_surviving_length"] = (icc["icc_resid_length"] / icc["icc_raw"] * 100).round(1)
    icc.to_csv(RESULTS / "24_contiguity_as_trait" / f"{SOURCE}_contiguity_trait_icc.csv", index=False)

    print("\n=== ICC: share of variance BETWEEN reviewers ===")
    print(icc[["label", "icc_raw", "icc_ci_lo", "icc_ci_hi", "icc_resid_length",
                "icc_resid_length_genre_platform", "pct_of_raw_surviving_length"]].to_string(index=False))

    print("\n=== is CONSISTENCY itself a trait? ===")
    cons = pd.DataFrame([consistency_trait(d, c, rng)
                          for c in ["excess_over_chance", "boundary_excess", "n_sentences"]])
    cons.to_csv(RESULTS / "24_contiguity_as_trait" / f"{SOURCE}_contiguity_trait_consistency.csv", index=False)
    print(cons.to_string(index=False))
    print("(ratio > 1 = reviewers differ in how variable they are, beyond chance)")

    print(f"\nwrote results to {RESULTS} ({SOURCE}_contiguity_trait_*.csv)")


if __name__ == "__main__":
    main()

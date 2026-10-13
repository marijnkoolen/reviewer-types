"""Why do Historical_fiction and Suspense diverge so sharply in book fame's
effect on Content share? (ANALYSIS_PLAN.md item 11, path 1 of 3 -- cheapest,
tried first.)

Candidate mechanism: spoiler sensitivity. Suspense readers are famously
plot-secretive; a well-known thriller's later reviews might avoid
re-summarizing the twist, while a well-known historical novel's setting and
events aren't "spoilable" the same way. `Content` (item 10) is a pooled main
label -- this reruns the identical book_fame x genre interaction from item
10.2 on Content's three available sub-categories separately
(`Content--Narrative` = plot/synopsis, `Content--Other`, `Content--Quote` --
`Content--Theme` is a WEAK_MODEL_LABEL, excluded from silver). If spoiler
avoidance is the mechanism, the Historical_fiction/Suspense divergence
should concentrate in `Content--Narrative` specifically and be flat (or at
least much smaller) in the other two subs.

Usage: python src/18c_content_spoiler_check.py
Writes results/18_content_spoiler_check/silver_content_subs_by_fame_genre.csv
"""
import sys
import warnings
from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

CONTENT_SUBS = ["Content--Narrative", "Content--Other", "Content--Quote"]
N_SAMPLE = 10_000
RNG_SEED = 0
ADVI_ITERS = 30_000
POSTERIOR_DRAWS = 1000


def build_lagged_covariates():
    review = pd.read_parquet(RESULTS / "00_data_prep/silver_review_features.parquet")
    review = review[review["source_id"] != "platform_1"].copy()
    review["review_date"] = pd.to_datetime(review["review_date"], errors="coerce")
    review = review[review["review_date"].notna()]
    review = review[review["review_date"].dt.year >= 1999]

    for id_col, out_col in [("work_id", "book_fame"), ("author_id", "author_fame"), ("user_id", "reviewer_experience")]:
        review = review.sort_values([id_col, "review_date"])
        prior_count = review.groupby(id_col).cumcount()
        review[out_col] = np.log1p(prior_count)

    review["year"] = review["review_date"].dt.year
    review["year_c"] = review["year"] - review["year"].mean()
    review["length_c"] = (review["num_sentences"] - review["num_sentences"].mean()) / review["num_sentences"].std()
    return review


def subsample(review):
    rng = np.random.default_rng(RNG_SEED)
    ids = rng.choice(review["review_id"].to_numpy(), size=N_SAMPLE, replace=False)
    return review[review["review_id"].isin(ids)].reset_index(drop=True)


def beta_boundary_adjust(y):
    n = len(y)
    return (y * (n - 1) + 0.5) / n


def fit_beta(df, label, formula):
    d = df.copy()
    d["y"] = beta_boundary_adjust(df[f"prop__{label}"].to_numpy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bmb.Model(formula, d, family="beta")
        approx = model.fit(inference_method="vi", random_seed=RNG_SEED, n=ADVI_ITERS)
        idata = approx.sample(POSTERIOR_DRAWS)
    return idata


def summarize(idata, label, var_names):
    summary = az.summary(idata, var_names=var_names, hdi_prob=0.94)
    summary = summary.reset_index().rename(columns={"index": "covariate"})
    summary.insert(0, "label", label)
    return summary


def main():
    review = build_lagged_covariates()
    df = subsample(review)
    print(f"{len(review)} eligible reviews, {len(df)} in subsample")
    for sub in CONTENT_SUBS:
        print(f"  {sub}: prevalence {(df[f'prop__{sub}'] > 0).mean():.3f}, mean share {df[f'prop__{sub}'].mean():.4f}")

    formula = ("y ~ year_c + length_c + book_fame + author_fame + reviewer_experience "
               "+ book_fame:nur_genre + author_fame:nur_genre "
               "+ (year_c|source_id) + (1|nur_genre)")
    var_names = ["year_c", "length_c", "book_fame", "author_fame", "reviewer_experience",
                 "book_fame:nur_genre", "author_fame:nur_genre"]

    all_fe = []
    for label in CONTENT_SUBS:
        print(f"\nfitting {label}...")
        idata = fit_beta(df, label, formula)
        fe = summarize(idata, label, var_names)
        all_fe.append(fe)

    out = pd.concat(all_fe, ignore_index=True)
    out.to_csv(RESULTS / "18_content_spoiler_check/silver_content_subs_by_fame_genre.csv", index=False)

    print("\n=== book_fame x genre on each Content sub-category ===")
    book_fame_genre = out[out["covariate"].str.startswith("book_fame:nur_genre", na=False)].copy()
    book_fame_genre["genre"] = book_fame_genre["covariate"].str.extract(r"\[(.*)\]")
    pivot = book_fame_genre.pivot(index="genre", columns="label", values="mean")
    print(pivot.round(4).to_string())

    print("\n=== Historical_fiction vs. Suspense specifically ===")
    focus = book_fame_genre[book_fame_genre["genre"].isin(["Historical_fiction", "Suspense"])]
    print(focus[["label", "genre", "mean", "hdi_3%", "hdi_97%"]].sort_values(["label", "genre"]).to_string(index=False))

    print(f"\nwrote results to {RESULTS / 'silver_content_subs_by_fame_genre.csv'}")


if __name__ == "__main__":
    main()

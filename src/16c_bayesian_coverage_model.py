"""Bayesian hierarchical model of category coverage: platform, genre, time,
and reputation, with accumulation (ANALYSIS_PLAN.md item 9). Design settled
over several rounds of chat discussion (2026-08-20) before any code was
written -- see the plan entry for the full reasoning; this docstring covers
only the implementation choices.

Outcome, decomposed rather than modeled as one joint object: one hierarchical
Beta regression per main category's `prop__label` (7 parallel models), not a
single Dirichlet-multinomial over the whole compositional vector -- cheaper
to fit, and keeps interpretation in the same "share of Content" language the
rest of this project already uses. The optional 4-category joint model
(Reader_response/Content/Style/Author) is a separate, later step, not built
here.

Two lagged accumulation covariates, at BOTH book and author level: log1p(count
of that work's / that author's reviews strictly before this one), sorted
chronologically per work_id and separately per author_id. Rating-based fame
is deliberately deferred (needs a genre-level cold-start prior that count-
based fame doesn't) -- count-only for this pass. Reviewer experience uses the
identical construction on user_id, reusing item 8's logic.

Random-effect structure kept deliberately small: (year_c|source_id) -- random
intercept AND slope on year by platform -- plus (1|nur_genre), not work/
author/reviewer identity (fame/experience are the mechanism, not an
additional identity effect on top; work/author/reviewer as crossed random
effects at this cardinality would be a much harder sampling problem for no
real gain here).

Collinearity between book_fame and author_fame was flagged at design time as
expected, not a bug -- author fame is partly built from book fame, and the
two are identical at an author's debut book. Reported explicitly here, not
buried: raw covariate correlation, VIF for the full fixed-effects design
matrix, and the posterior correlation between the two coefficients for every
label, saved to its own CSV precisely so it doesn't get lost when the
results come in.

Fit via bambi (PyMC backend), ADVI (`inference_method="vi"`) for speed on a
10,000-review subsample -- same de-risking pattern as items 5/7: prototype
first, sanity-check against the already-published frequentist year-trend
coefficients (silver_length_proportion_year_trends.csv, item 2/§11) before
deciding whether full-corpus / full-NUTS is worth the added cost.

Usage: python src/16c_bayesian_coverage_model.py
Writes results/16_bayesian_coverage_model/silver_bayesian_coverage_fixed_effects.csv,
  results/16_bayesian_coverage_model/silver_bayesian_coverage_collinearity.csv,
  results/16_bayesian_coverage_model/silver_bayesian_coverage_sanity_check.csv,
  results/16_bayesian_coverage_model/silver_bayesian_coverage_sample.parquet (the covariate table used)
"""
import sys
import warnings
from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
N_SAMPLE = 10_000
RNG_SEED = 0
ADVI_ITERS = 30_000
POSTERIOR_DRAWS = 1000
FIXED_EFFECTS = ["year_c", "length_c", "book_fame", "author_fame", "reviewer_experience"]


def build_lagged_covariates():
    """Cumulative prior review count per work_id / author_id / user_id,
    computed only from reviews strictly before the current one -- requires
    the full corpus (not just the eventual subsample) so a review sampled
    into the subsample still gets its correct prior count from ALL earlier
    reviews of that work/author/reviewer, not just the ones that happened
    to also land in the subsample."""
    review = pd.read_parquet(RESULTS / "00_data_prep/silver_review_features.parquet")
    review = review[review["source_id"] != "platform_1"].copy()
    review["review_date"] = pd.to_datetime(review["review_date"], errors="coerce")
    review = review[review["review_date"].notna()]
    review = review[review["review_date"].dt.year >= 1999]  # 4 pre-1999 placeholder dates, see item 1

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
    """Smithson & Verkuilen (2006): standard rescale so exact 0/1 proportions
    (common here -- many reviews have prop__label exactly 0 or 1) fall
    strictly inside (0,1), which a Beta likelihood requires."""
    n = len(y)
    return (y * (n - 1) + 0.5) / n


def collinearity_report(df):
    X = df[FIXED_EFFECTS].to_numpy()
    X = np.column_stack([np.ones(len(X)), X])  # VIF needs an intercept column present
    vif = [variance_inflation_factor(X, i) for i in range(1, X.shape[1])]
    vif_df = pd.DataFrame({"covariate": FIXED_EFFECTS, "vif": vif})

    corr = df[FIXED_EFFECTS].corr()
    book_author_corr = corr.loc["book_fame", "author_fame"]
    print("=== collinearity check: book_fame vs. author_fame (flagged at design time, expected) ===")
    print(f"raw covariate correlation: {book_author_corr:.3f}")
    print(vif_df.to_string(index=False))
    return vif_df, book_author_corr


def fit_one_label(df, label):
    d = df.copy()
    d["y"] = beta_boundary_adjust(df[f"prop__{label}"].to_numpy())
    formula = "y ~ year_c + length_c + book_fame + author_fame + reviewer_experience + (year_c|source_id) + (1|nur_genre)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = bmb.Model(formula, d, family="beta")
        # inference_method="vi" returns the raw PyMC MeanField approximation, not an
        # InferenceData -- draws= is ignored by bambi in this path, so posterior samples
        # have to be drawn from the fitted approximation explicitly.
        approx = model.fit(inference_method="vi", random_seed=RNG_SEED, n=ADVI_ITERS)
        idata = approx.sample(POSTERIOR_DRAWS)
    return model, idata


def summarize_fixed_effects(idata, label):
    summary = az.summary(idata, var_names=FIXED_EFFECTS, hdi_prob=0.94)
    summary = summary.reset_index().rename(columns={"index": "covariate"})
    summary.insert(0, "label", label)
    return summary


def posterior_fame_correlation(idata, label):
    post = idata.posterior
    book = post["book_fame"].to_numpy().flatten()
    author = post["author_fame"].to_numpy().flatten()
    r = np.corrcoef(book, author)[0, 1]
    return {"label": label, "posterior_corr_book_author_fame": r}


def sanity_check(fixed_effects):
    freq = pd.read_csv(RESULTS / "09_length_dimension/silver_length_proportion_year_trends.csv")
    bayes_year = fixed_effects[fixed_effects["covariate"] == "year_c"][["label", "mean", "sd"]]
    bayes_year = bayes_year.rename(columns={"mean": "bayes_year_coef", "sd": "bayes_year_sd"})
    merged = bayes_year.merge(
        freq[["label", "proportion_adjusted_year_coef"]].rename(
            columns={"proportion_adjusted_year_coef": "freq_year_coef_full_corpus"}),
        on="label", how="left")
    merged["same_sign"] = np.sign(merged["bayes_year_coef"]) == np.sign(merged["freq_year_coef_full_corpus"])
    return merged


def main():
    print("building lagged covariates on the full dated, non-platform_1 corpus...")
    review = build_lagged_covariates()
    print(f"{len(review)} eligible reviews")

    df = subsample(review)
    print(f"subsample: {len(df)} reviews")
    df.to_parquet(RESULTS / "16_bayesian_coverage_model/silver_bayesian_coverage_sample.parquet", index=False)

    vif_df, book_author_corr = collinearity_report(df)

    all_fixed_effects = []
    all_fame_corr = []
    for label in MAIN_LABELS_USED:
        print(f"\n=== fitting {label} ===")
        model, idata = fit_one_label(df, label)
        fe = summarize_fixed_effects(idata, label)
        all_fixed_effects.append(fe)
        all_fame_corr.append(posterior_fame_correlation(idata, label))
        print(fe[["covariate", "mean", "sd", "hdi_3%", "hdi_97%"]].to_string(index=False))

    fixed_effects = pd.concat(all_fixed_effects, ignore_index=True)
    fixed_effects.to_csv(RESULTS / "16_bayesian_coverage_model/silver_bayesian_coverage_fixed_effects.csv", index=False)

    fame_corr_df = pd.DataFrame(all_fame_corr)
    vif_df["raw_covariate_corr_book_author_fame"] = book_author_corr
    fame_corr_df.to_csv(RESULTS / "16_bayesian_coverage_model/silver_bayesian_coverage_collinearity.csv", index=False)
    vif_df.to_csv(RESULTS / "16_bayesian_coverage_model/silver_bayesian_coverage_vif.csv", index=False)

    print("\n=== posterior correlation between book_fame and author_fame coefficients, per label ===")
    print(fame_corr_df.to_string(index=False))
    print(f"\n(raw covariate correlation across the sample: {book_author_corr:.3f}; "
          f"VIF: {vif_df.set_index('covariate').loc['book_fame', 'vif']:.2f} / "
          f"{vif_df.set_index('covariate').loc['author_fame', 'vif']:.2f})")

    print("\n=== sanity check: Bayesian year_c coefficient vs. existing frequentist year-trend (item 2/§11) ===")
    check = sanity_check(fixed_effects)
    check.to_csv(RESULTS / "16_bayesian_coverage_model/silver_bayesian_coverage_sanity_check.csv", index=False)
    print(check.to_string(index=False))
    print(f"\nsame sign as frequentist full-corpus trend: {check['same_sign'].sum()}/{len(check)}")

    print(f"\nwrote results to {RESULTS} (silver_bayesian_coverage_*.csv)")


if __name__ == "__main__":
    main()

"""Digging into book fame vs. author fame: divergence, genre, and reviewer
mix (ANALYSIS_PLAN.md item 10) -- direct follow-on from item 9's headline
finding that book fame and author fame are cleanly separable (VIF~1.2,
posterior coefficient correlation <0.05) and often opposite-signed. Design
settled in chat (2026-08-20) before any code was written.

Three sub-parts:
1. When do book fame and author fame diverge? Two new lagged, time-varying
   covariates (recomputed as of each review, not a static per-author
   summary, consistent with item 9's whole design): author_fame_concentration
   (what share of the author's prior reviews belong to their single
   most-reviewed prior book) and is_current_breakout (is THIS book currently
   the author's most-reviewed one, or is a different book of theirs more
   famous). Added as covariates + interactions to item 9's 7-model setup.
2. Fame x genre: does the fame effect differ between Literary_fiction (where
   a famous author's similar new book plausibly reads as creative staleness)
   and "genre fiction" (where delivering more of a known, trusted experience
   is the point)? Fixed-effect book_fame:genre / author_fame:genre
   interactions -- simpler than genre-specific random slopes (the plan's
   original phrasing), a deliberate first-pass simplification, easier to fit
   and to read off directly as per-genre slope estimates.
3. Reviewer prolificness x fame:
   3a. Do prolific reviewers repeat authors? Observed repeat-author rate
       (does this review's author already appear earlier in this reviewer's
       own history) against a permutation null: for each reviewer's own n
       and dominant genre, simulate n random author draws from that genre's
       empirical author pool (weighted by real popularity, not uniform),
       repeated many times, cached per (genre, n) -- same caching pattern as
       item 3's modal-share null.
   3b. Is fame's effect on composition actually about *who* reviews, not
       *how* any given person writes? item 9's model already answers this
       partially without noticing: reviewer_experience sits in the SAME
       regression as book_fame/author_fame, so item 9's fame coefficients
       are already the effect net of reviewer composition (the "direct"
       effect). Two checks here: (i) does fame predict reviewer_experience
       directly (quick OLS)? (ii) refit the 7 models WITHOUT
       reviewer_experience (the "total" effect) and compare against item 9's
       existing coefficients (the "direct" effect) -- the difference is the
       share of fame's effect that runs through reviewer selection.

Usage: python src/17c_fame_extensions.py
Writes results/17_fame_extensions/silver_fame_divergence_fixed_effects.csv,
  results/17_fame_extensions/silver_fame_genre_fixed_effects.csv,
  results/17_fame_extensions/silver_repeat_author_rate.csv,
  results/17_fame_extensions/silver_fame_predicts_experience.csv,
  results/17_fame_extensions/silver_fame_total_vs_direct_effect.csv
"""
import sys
import warnings
from pathlib import Path

import arviz as az
import bambi as bmb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS  # noqa: E402
from stats_utils import benjamini_hochberg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
N_SAMPLE = 10_000
RNG_SEED = 0
ADVI_ITERS = 30_000
POSTERIOR_DRAWS = 1000
N_SIM_REPEAT_AUTHOR = 2000
MIN_PROLIFIC = 10

_repeat_author_null_cache = {}


# --- shared setup, duplicated from 16c_bayesian_coverage_model.py rather than
# importing a numeric-prefixed module (project convention) -------------------

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


# --- sub-part 1: fame divergence --------------------------------------------

def add_fame_divergence(review, sample_df):
    """author_fame_concentration and is_current_breakout, lagged as of each
    review -- needs each subsample author's FULL prior history (not just
    their reviews that happen to also be in the subsample), so this walks
    the full corpus restricted to authors appearing in the subsample."""
    sub_authors = set(sample_df["author_id"])
    hist = review[review["author_id"].isin(sub_authors)].sort_values(["author_id", "review_date"]).reset_index(drop=True)

    author_ids = hist["author_id"].to_numpy()
    work_ids = hist["work_id"].to_numpy()
    review_ids = hist["review_id"].to_numpy()
    n = len(hist)
    concentration = np.zeros(n)
    is_breakout = np.zeros(n, dtype=np.int8)

    counts = {}
    cur_author = None
    for i in range(n):
        a = author_ids[i]
        if a != cur_author:
            counts = {}
            cur_author = a
        w = work_ids[i]
        if counts:
            prior_total = sum(counts.values())
            prior_max = max(counts.values())
        else:
            prior_total = 0
            prior_max = 0
        this_work_prior = counts.get(w, 0)
        concentration[i] = prior_max / prior_total if prior_total > 0 else 0.0
        is_breakout[i] = 1 if (prior_total == 0 or this_work_prior >= prior_max) else 0
        counts[w] = this_work_prior + 1

    fame_df = pd.DataFrame({"review_id": review_ids, "author_fame_concentration": concentration,
                             "is_current_breakout": is_breakout})
    return sample_df.merge(fame_df, on="review_id", how="left")


def run_fame_divergence(review, sample_df):
    print("\n" + "=" * 70)
    print("SUB-PART 1: fame divergence (concentration, breakout)")
    print("=" * 70)
    df = add_fame_divergence(review, sample_df)
    print(f"mean author_fame_concentration: {df['author_fame_concentration'].mean():.3f}, "
          f"mean is_current_breakout: {df['is_current_breakout'].mean():.3f}")

    formula = ("y ~ year_c + length_c + book_fame + author_fame + reviewer_experience "
               "+ author_fame_concentration + is_current_breakout "
               "+ book_fame:author_fame_concentration + author_fame:author_fame_concentration "
               "+ (year_c|source_id) + (1|nur_genre)")
    var_names = ["year_c", "length_c", "book_fame", "author_fame", "reviewer_experience",
                 "author_fame_concentration", "is_current_breakout",
                 "book_fame:author_fame_concentration", "author_fame:author_fame_concentration"]

    all_fe = []
    for label in MAIN_LABELS_USED:
        print(f"fitting {label}...")
        idata = fit_beta(df, label, formula)
        fe = summarize(idata, label, var_names)
        all_fe.append(fe)

    out = pd.concat(all_fe, ignore_index=True)
    out.to_csv(RESULTS / "17_fame_extensions/silver_fame_divergence_fixed_effects.csv", index=False)
    print(out[out["covariate"].isin(["author_fame_concentration", "is_current_breakout",
                                      "book_fame:author_fame_concentration",
                                      "author_fame:author_fame_concentration"])]
          [["label", "covariate", "mean", "sd", "hdi_3%", "hdi_97%"]].to_string(index=False))
    return out


# --- sub-part 2: fame x genre ------------------------------------------------

def run_fame_genre(sample_df):
    print("\n" + "=" * 70)
    print("SUB-PART 2: fame x genre interaction")
    print("=" * 70)
    formula = ("y ~ year_c + length_c + book_fame + author_fame + reviewer_experience "
               "+ book_fame:nur_genre + author_fame:nur_genre "
               "+ (year_c|source_id) + (1|nur_genre)")

    var_names = ["year_c", "length_c", "book_fame", "author_fame", "reviewer_experience",
                 "book_fame:nur_genre", "author_fame:nur_genre"]
    all_fe = []
    for label in MAIN_LABELS_USED:
        print(f"fitting {label}...")
        idata = fit_beta(sample_df, label, formula)
        fe = summarize(idata, label, var_names)
        all_fe.append(fe)

    out = pd.concat(all_fe, ignore_index=True)
    out.to_csv(RESULTS / "17_fame_extensions/silver_fame_genre_fixed_effects.csv", index=False)
    genre_rows = out[out["covariate"].str.contains("nur_genre", na=False)]
    print(genre_rows[["label", "covariate", "mean", "hdi_3%", "hdi_97%"]].to_string(index=False))
    return out


# --- sub-part 3a: repeat-author rate ----------------------------------------

def repeat_author_null(n, genre_author_pool, rng):
    """Vectorized: draws x n array of simulated author picks, repeat rate per
    draw = 1 - (distinct authors / n), computed via sort-and-count-adjacent-
    equal rather than a per-draw Python set loop (repeated across ~thousands
    of unique (n, genre) pairs, so the vectorized version matters here)."""
    key = (n, id(genre_author_pool))
    if key not in _repeat_author_null_cache:
        draws = rng.choice(genre_author_pool, size=(N_SIM_REPEAT_AUTHOR, n), replace=True)
        sorted_draws = np.sort(draws, axis=1)
        n_distinct = 1 + (sorted_draws[:, 1:] != sorted_draws[:, :-1]).sum(axis=1)
        _repeat_author_null_cache[key] = 1 - n_distinct / n
    return _repeat_author_null_cache[key]


def run_repeat_author_rate(review):
    print("\n" + "=" * 70)
    print("SUB-PART 3a: do prolific reviewers repeat authors?")
    print("=" * 70)
    tendency = pd.read_csv(RESULTS / "12_prolific_reviewer_tendency/silver_prolific_reviewer_tendency.csv")
    prolific_ids = set(tendency["user_id"])

    df = review[review["user_id"].isin(prolific_ids)].sort_values(["user_id", "review_date"]).copy()
    df["is_repeat_author"] = df.groupby("user_id")["author_id"].transform(
        lambda s: s.duplicated(keep="first"))

    genre_pools = {g: sub["author_id"].to_numpy() for g, sub in review.groupby("nur_genre")}

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for user_id, sub in df.groupby("user_id"):
        n = len(sub)
        observed_rate = sub["is_repeat_author"].mean()
        dominant_genre = sub["nur_genre"].value_counts().idxmax()
        null_rates = repeat_author_null(n, genre_pools[dominant_genre], rng)
        p_value = (1 + np.sum(null_rates >= observed_rate)) / (N_SIM_REPEAT_AUTHOR + 1)
        rows.append({"user_id": user_id, "n_reviews": n, "dominant_genre": dominant_genre,
                      "observed_repeat_author_rate": round(observed_rate, 4),
                      "null_mean_repeat_author_rate": round(null_rates.mean(), 4),
                      "p_value": p_value})

    out = pd.DataFrame(rows)
    out["q_bh"] = benjamini_hochberg(out["p_value"].to_numpy())
    out.to_csv(RESULTS / "17_fame_extensions/silver_repeat_author_rate.csv", index=False)

    print(f"{len(out)} prolific reviewers")
    print(f"mean observed repeat-author rate: {out['observed_repeat_author_rate'].mean():.4f}")
    print(f"mean null (chance) repeat-author rate: {out['null_mean_repeat_author_rate'].mean():.4f}")
    print(f"reviewers with significantly (q<0.05) high repeat-author rate: "
          f"{(out['q_bh'] < 0.05).sum()}/{len(out)}")
    return out


# --- sub-part 3b: mediation --------------------------------------------------

def run_fame_predicts_experience(sample_df):
    print("\n" + "=" * 70)
    print("SUB-PART 3b(i): does fame predict reviewer experience?")
    print("=" * 70)
    model = smf.ols("reviewer_experience ~ book_fame + author_fame", data=sample_df).fit()
    out = pd.DataFrame({
        "covariate": model.params.index, "coef": model.params.values,
        "p_value": model.pvalues.values,
    })
    out.to_csv(RESULTS / "17_fame_extensions/silver_fame_predicts_experience.csv", index=False)
    print(model.summary().tables[1])
    return out


def run_total_vs_direct_effect(sample_df):
    print("\n" + "=" * 70)
    print("SUB-PART 3b(ii): total effect (no reviewer_experience) vs. item 9's direct effect")
    print("=" * 70)
    formula = "y ~ year_c + length_c + book_fame + author_fame + (year_c|source_id) + (1|nur_genre)"
    var_names = ["year_c", "length_c", "book_fame", "author_fame"]

    all_fe = []
    for label in MAIN_LABELS_USED:
        print(f"fitting {label} (total effect, no reviewer_experience)...")
        idata = fit_beta(sample_df, label, formula)
        fe = summarize(idata, label, var_names)
        all_fe.append(fe)
    total = pd.concat(all_fe, ignore_index=True)
    total = total.rename(columns={"mean": "total_effect_mean", "sd": "total_effect_sd"})

    direct = pd.read_csv(RESULTS / "16_bayesian_coverage_model/silver_bayesian_coverage_fixed_effects.csv")
    direct = direct[direct["covariate"].isin(["book_fame", "author_fame"])]
    direct = direct.rename(columns={"mean": "direct_effect_mean", "sd": "direct_effect_sd"})

    merged = total[total["covariate"].isin(["book_fame", "author_fame"])][
        ["label", "covariate", "total_effect_mean", "total_effect_sd"]].merge(
        direct[["label", "covariate", "direct_effect_mean", "direct_effect_sd"]], on=["label", "covariate"])
    merged["mediated_share"] = (merged["total_effect_mean"] - merged["direct_effect_mean"]) / merged["total_effect_mean"]
    merged.to_csv(RESULTS / "17_fame_extensions/silver_fame_total_vs_direct_effect.csv", index=False)
    print(merged.to_string(index=False))
    return total, merged


def main():
    review = build_lagged_covariates()
    sample_df = subsample(review)
    print(f"{len(review)} eligible reviews, {len(sample_df)} in subsample")

    run_fame_divergence(review, sample_df)
    run_fame_genre(sample_df)
    run_repeat_author_rate(review)
    run_fame_predicts_experience(sample_df)
    run_total_vs_direct_effect(sample_df)

    print(f"\nwrote results to {RESULTS} (silver_fame_*.csv, silver_repeat_author_rate.csv)")


if __name__ == "__main__":
    main()

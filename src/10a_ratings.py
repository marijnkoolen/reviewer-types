"""Ratings (ANALYSIS_PLAN.md §11), gold set only.

`0.0`/blank ratings are confirmed placeholders for "no rating" (see
ANALYSIS_PLAN.md §0) and are excluded throughout, not treated as a real low
score. 1,060/1,152 reviews (92%) have a real 1-5 rating.

Three questions: (1) does what a review discusses correlate with its rating,
watching the length confound explicitly since longer reviews could
mechanically touch more categories; (2) do the §5 typology clusters differ in
average rating; (3) an ordinal regression of rating on category composition,
length, and `Literary_fiction` genre together, to check whether §6's finding
(Literary_fiction rates ~1 point lower on every platform) is actually
explained by what those reviews discuss, or is an independent genre effect
that composition doesn't account for.

Usage: python src/10a_ratings.py
Writes results/10_ratings/gold_rating_correlations.csv, results/10_ratings/gold_rating_by_cluster.csv,
results/10_ratings/gold_rating_ordinal_model.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS, SPARSE_GOLD_LABELS  # noqa: E402
from stats_utils import benjamini_hochberg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
TESTED_LABELS = [l for l in MAIN_LABELS_USED + SUB_LABELS if l not in SPARSE_GOLD_LABELS]


def load_rated():
    df = pd.read_csv(RESULTS / "00_data_prep/gold_review_features.csv")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    rated = df.dropna(subset=["rating"]).copy()
    rated["rating_int"] = rated["rating"].astype(int)
    print(f"{len(rated)}/{len(df)} reviews have a rating")
    return rated


def length_confound(df):
    rho_words, p_words = stats.spearmanr(df["review_num_words"], df["rating"])
    rho_sent, p_sent = stats.spearmanr(df["num_sentences"], df["rating"])
    print(f"length vs. rating: words rho={rho_words:.4f} (p={p_words:.4g}), "
          f"sentences rho={rho_sent:.4f} (p={p_sent:.4g})")
    return rho_words, p_words


def category_correlations(df):
    rows = []
    for label in TESTED_LABELS:
        rho, p = stats.spearmanr(df[f"prop__{label}"], df["rating"])
        rows.append({"label": label, "spearman_rho": round(rho, 4), "p_value": p})
    out = pd.DataFrame(rows)
    out["q_value_bh"] = benjamini_hochberg(out["p_value"].to_numpy())
    return out.sort_values("spearman_rho")


def rating_by_cluster(df):
    clusters = pd.read_csv(RESULTS / "05_typology/gold_typology_clusters.csv")
    d = df.merge(clusters[["review_id", "cluster_composition"]], on="review_id", how="inner")
    groups = [g["rating"].to_numpy() for _, g in d.groupby("cluster_composition")]
    h, p = stats.kruskal(*groups)
    summary = d.groupby("cluster_composition")["rating"].agg(["mean", "median", "std", "count"]).reset_index()
    summary.columns = ["cluster", "mean_rating", "median_rating", "std_rating", "n"]
    print(f"rating ~ cluster: Kruskal-Wallis H={h:.2f}, p={p:.4g}")
    return summary.sort_values("mean_rating", ascending=False), {"H": h, "p_value": p}


def ordinal_model(df):
    d = df.copy()
    d["is_literary_fiction"] = (d["nur_genre"] == "Literary_fiction").astype(int)
    d["length_std"] = (d["num_sentences"] - d["num_sentences"].mean()) / d["num_sentences"].std()
    predictors = [f"prop__{l}" for l in MAIN_LABELS_USED] + ["length_std", "is_literary_fiction"]
    X = d[predictors]
    y = d["rating_int"]

    model = OrderedModel(y, X, distr="logit")
    res = model.fit(method="bfgs", disp=False, maxiter=200)

    out = pd.DataFrame({
        "predictor": res.params.index,
        "coef": res.params.to_numpy(),
        "p_value": res.pvalues.to_numpy(),
    })
    out = out[~out["predictor"].str.contains(r"/")]  # drop the threshold ("cut point") params
    out["q_value_bh"] = benjamini_hochberg(out["p_value"].to_numpy())
    print(f"\nordinal logistic regression (rating 1-5 ~ main-category proportions + length + Literary_fiction)")
    print(f"n={model.endog.shape[0]}, pseudo-R2={res.prsquared:.4f}")
    print(out.sort_values("p_value").to_string(index=False))
    return out.sort_values("p_value"), res.prsquared


def main():
    df = load_rated()
    length_confound(df)

    corr = category_correlations(df)
    corr.to_csv(RESULTS / "10_ratings/gold_rating_correlations.csv", index=False)
    print("\n=== category proportion vs. rating (Spearman) ===")
    print(corr.to_string(index=False))

    cluster_summary, cluster_stats = rating_by_cluster(df)
    cluster_summary.to_csv(RESULTS / "10_ratings/gold_rating_by_cluster.csv", index=False)
    print("\n=== rating by typology cluster ===")
    print(cluster_summary.to_string(index=False))

    model_out, pseudo_r2 = ordinal_model(df)
    model_out.to_csv(RESULTS / "10_ratings/gold_rating_ordinal_model.csv", index=False)

    litfic_row = model_out[model_out["predictor"] == "is_literary_fiction"]
    if len(litfic_row) and litfic_row["p_value"].iloc[0] < 0.05:
        print(f"\nLiterary_fiction remains significant (p={litfic_row['p_value'].iloc[0]:.4g}) "
              f"after controlling for composition and length -- the rating gap is NOT explained by what "
              f"these reviews discuss.")
    else:
        print(f"\nLiterary_fiction is no longer significant once composition and length are controlled "
              f"-- the rating gap may be substantially explained by what these reviews discuss.")

    print(f"\nwrote results to {RESULTS} (gold_rating_correlations.csv, gold_rating_by_cluster.csv, "
          f"gold_rating_ordinal_model.csv)")


if __name__ == "__main__":
    main()

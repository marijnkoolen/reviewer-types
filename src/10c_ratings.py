"""Ratings on the silver set (ANALYSIS_PLAN.md §11, Phase C), raw
(uncorrected) -- 466,973 reviews. Same three questions and method as gold's
10a_ratings.py (category correlations, rating by typology cluster, ordinal
regression checking whether the Literary_fiction rating penalty survives
controlling for composition), run at 437x the rated-review count.

Usage: python src/10c_ratings.py
Writes results/10_ratings/silver_rating_correlations.csv, silver_rating_by_cluster.csv,
results/10_ratings/silver_rating_ordinal_model.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS, SPARSE_GOLD_LABELS, WEAK_MODEL_LABELS  # noqa: E402
from stats_utils import benjamini_hochberg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
TESTED_LABELS = [l for l in MAIN_LABELS_USED + SUB_LABELS if l not in SPARSE_GOLD_LABELS and l not in WEAK_MODEL_LABELS]


def load_rated():
    df = pd.read_parquet(RESULTS / "00_data_prep/silver_review_features.parquet")
    rated = df.dropna(subset=["rating"]).copy()
    rated["rating_int"] = rated["rating"].astype(int)
    print(f"{len(rated)}/{len(df)} reviews have a rating")
    return rated


def length_confound(df):
    rho_words, p_words = stats.spearmanr(df["review_num_words"], df["rating"])
    rho_sent, p_sent = stats.spearmanr(df["num_sentences"], df["rating"])
    print(f"length vs. rating: words rho={rho_words:.4f} (p={p_words:.4g}), "
          f"sentences rho={rho_sent:.4f} (p={p_sent:.4g})")


def category_correlations(df):
    rows = []
    for label in TESTED_LABELS:
        rho, p = stats.spearmanr(df[f"prop__{label}"], df["rating"])
        rows.append({"label": label, "spearman_rho": round(rho, 4), "p_value": p})
    out = pd.DataFrame(rows)
    out["q_value_bh"] = benjamini_hochberg(out["p_value"].to_numpy())
    return out.sort_values("spearman_rho")


def rating_by_cluster(df):
    clusters = pd.read_csv(RESULTS / "05_typology/silver_typology_clusters.csv")
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
    out = out[~out["predictor"].str.contains(r"/")]
    out["q_value_bh"] = benjamini_hochberg(out["p_value"].to_numpy())
    print(f"\nordinal logistic regression: n={model.endog.shape[0]}, pseudo-R2={res.prsquared:.4f}")
    print(out.sort_values("p_value").to_string(index=False))
    return out.sort_values("p_value"), res.prsquared


def main():
    df = load_rated()
    length_confound(df)

    corr = category_correlations(df)
    corr.to_csv(RESULTS / "10_ratings/silver_rating_correlations.csv", index=False)
    print("\n=== category proportion vs. rating (Spearman) ===")
    print(corr.to_string(index=False))

    cluster_summary, cluster_stats = rating_by_cluster(df)
    cluster_summary.to_csv(RESULTS / "10_ratings/silver_rating_by_cluster.csv", index=False)
    print("\n=== rating by typology cluster ===")
    print(cluster_summary.to_string(index=False))

    print("\nfitting ordinal model (this can take a few minutes at this scale)...")
    model_out, pseudo_r2 = ordinal_model(df)
    model_out.to_csv(RESULTS / "10_ratings/silver_rating_ordinal_model.csv", index=False)

    litfic_row = model_out[model_out["predictor"] == "is_literary_fiction"]
    if len(litfic_row) and litfic_row["p_value"].iloc[0] < 0.05:
        print(f"\nLiterary_fiction remains significant (p={litfic_row['p_value'].iloc[0]:.4g}) "
              f"after controlling for composition and length.")
    else:
        print(f"\nLiterary_fiction is no longer significant once composition and length are controlled.")

    print(f"\nwrote results to {RESULTS} (silver_rating_correlations.csv, silver_rating_by_cluster.csv, "
          f"silver_rating_ordinal_model.csv)")


if __name__ == "__main__":
    main()

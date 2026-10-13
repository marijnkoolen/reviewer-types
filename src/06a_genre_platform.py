"""Genre and platform effects (ANALYSIS_PLAN.md §7), gold set only.

Scope decision: a full re-run of co-occurrence/contiguity/order per genre x
platform stratum (10 genres x 4 platforms = 40 cells, several genres under
100 reviews, `Romance` x `platform_3` has zero reviews) would mostly measure
noise at gold-set scale -- those finer analyses are deferred to Phase C
(670K-review silver set), where the cells are actually big enough. This
script covers what gold *can* support at a stratified level:

1. Descriptives per genre/platform (length, rating) -- also the confound
   check the plan calls for: if genre/platform correlate strongly with
   review length, downstream category differences could just be a length
   artifact even though every category feature elsewhere in this project is
   already length-normalized.
2. Category prevalence by genre and by platform (chi-square test of
   independence + Cramer's V effect size + Benjamini-Hochberg FDR correction
   within each family of tests, since this runs one test per category).
3. Typology cluster membership (from §6/07) by genre and by platform, same
   test.

platform_1 caveat, repeated because it matters every time it comes up: all
89 platform_1 reviews were written by one reviewer (see ANALYSIS_PLAN.md
§0). Any platform_1-specific pattern below is that person's individual style,
not evidence about the platform.

Usage: python src/06a_genre_platform.py
Writes results/gold_genre_platform_descriptives.csv,
results/06_genre_platform/gold_platform_date_ranges.csv, results/06_genre_platform/gold_genre_prevalence.csv,
results/06_genre_platform/gold_platform_prevalence.csv, results/06_genre_platform/gold_typology_by_genre.csv,
results/06_genre_platform/gold_typology_by_platform.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS, SPARSE_GOLD_LABELS  # noqa: E402
from stats_utils import (  # noqa: E402
    benjamini_hochberg, cramers_v,
    prevalence_by_group as shared_prevalence_by_group,
    typology_by_group as shared_typology_by_group,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
TESTED_LABELS = [l for l in MAIN_LABELS_USED + SUB_LABELS if l not in SPARSE_GOLD_LABELS]


def descriptives(df, group_col):
    rows = []
    for g, sub in df.groupby(group_col):
        rating = pd.to_numeric(sub["rating"], errors="coerce")
        rows.append({
            group_col: g, "n_reviews": len(sub),
            "mean_num_sentences": round(sub["num_sentences"].mean(), 1),
            "median_num_sentences": sub["num_sentences"].median(),
            "mean_num_words": round(sub["review_num_words"].mean(), 1),
            "median_num_words": sub["review_num_words"].median(),
            "mean_rating": round(rating.mean(), 2) if rating.notna().any() else float("nan"),
            "n_with_rating": int(rating.notna().sum()),
        })
    return pd.DataFrame(rows)


def confound_tests(df):
    rows = []
    rating = pd.to_numeric(df["rating"], errors="coerce")
    for group_col in ["nur_genre", "source_id"]:
        groups_len = [g["num_sentences"].to_numpy() for _, g in df.groupby(group_col)]
        h_len, p_len = stats.kruskal(*groups_len)
        rows.append({"outcome": "num_sentences", "grouped_by": group_col,
                      "test": "Kruskal-Wallis", "statistic": round(h_len, 2), "p_value": p_len})

        rating_df = df.assign(rating=rating).dropna(subset=["rating"])
        groups_rating = [g["rating"].to_numpy() for _, g in rating_df.groupby(group_col)]
        groups_rating = [g for g in groups_rating if len(g) > 0]
        h_rat, p_rat = stats.kruskal(*groups_rating)
        rows.append({"outcome": "rating", "grouped_by": group_col,
                      "test": "Kruskal-Wallis", "statistic": round(h_rat, 2), "p_value": p_rat})
    return pd.DataFrame(rows)


def date_ranges(df):
    rows = []
    for platform, sub in df.groupby("source_id"):
        dates = sub["review_date"].dropna()
        dates = dates[dates != ""]
        rows.append({
            "source_id": platform, "n_reviews": len(sub), "n_with_date": len(dates),
            "min_date": dates.min() if len(dates) else "", "max_date": dates.max() if len(dates) else "",
        })
    return pd.DataFrame(rows)




def length_stratified_check(df, label, group_col="source_id"):
    """Diagnostic for the length confound: does a group difference in prevalence
    survive within length terciles, or does it disappear once length is held
    roughly constant?"""
    d = df.copy()
    d["length_tercile"] = pd.qcut(d["num_sentences"], q=3, labels=["short", "medium", "long"])
    d["present"] = d[f"count__{label}"] > 0
    out = d.groupby(["length_tercile", group_col], observed=True)["present"].agg(["mean", "count"]).reset_index()
    out.columns = ["length_tercile", group_col, "prevalence", "n"]
    out["label"] = label
    return out


def rating_by_genre_within_platform(df):
    """Checks whether Literary_fiction's markedly lower average rating (see
    descriptives) holds within each platform separately, or is an artifact of
    Literary_fiction being concentrated on lower-average-rating platforms."""
    d = df.copy()
    d["rating"] = pd.to_numeric(d["rating"], errors="coerce")
    d["is_literary_fiction"] = d["nur_genre"] == "Literary_fiction"
    out = d.dropna(subset=["rating"]).groupby(
        ["source_id", "is_literary_fiction"], observed=True
    )["rating"].agg(["mean", "count"]).reset_index()
    out.columns = ["source_id", "is_literary_fiction", "mean_rating", "n"]
    return out


def main():
    df = pd.read_csv(RESULTS / "00_data_prep/gold_review_features.csv")
    clusters = pd.read_csv(RESULTS / "05_typology/gold_typology_clusters.csv")

    genre_desc = descriptives(df, "nur_genre")
    platform_desc = descriptives(df, "source_id")
    genre_desc.to_csv(RESULTS / "06_genre_platform/gold_genre_platform_descriptives_genre.csv", index=False)
    platform_desc.to_csv(RESULTS / "06_genre_platform/gold_genre_platform_descriptives_platform.csv", index=False)

    confounds = confound_tests(df)
    confounds.to_csv(RESULTS / "06_genre_platform/gold_genre_platform_confounds.csv", index=False)

    dates = date_ranges(df)
    dates.to_csv(RESULTS / "06_genre_platform/gold_platform_date_ranges.csv", index=False)

    genre_prev = shared_prevalence_by_group(df, "nur_genre", TESTED_LABELS)
    genre_prev.to_csv(RESULTS / "06_genre_platform/gold_genre_prevalence.csv", index=False)

    platform_prev = shared_prevalence_by_group(df, "source_id", TESTED_LABELS)
    platform_prev.to_csv(RESULTS / "06_genre_platform/gold_platform_prevalence.csv", index=False)

    litfic_rating = rating_by_genre_within_platform(df)
    litfic_rating.to_csv(RESULTS / "06_genre_platform/gold_litfic_rating_by_platform.csv", index=False)
    print("\n=== rating: Literary_fiction vs. rest, within each platform ===")
    print(litfic_rating.to_string(index=False))

    length_check = length_stratified_check(df, "Content--Narrative", "source_id")
    length_check.to_csv(RESULTS / "06_genre_platform/gold_platform_length_stratified_check.csv", index=False)
    print("\n=== Content--Narrative prevalence by platform, within length tercile ===")
    print(length_check.to_string(index=False))

    genre_table, genre_stats = shared_typology_by_group(clusters, "nur_genre")
    genre_table.to_csv(RESULTS / "06_genre_platform/gold_typology_by_genre.csv")
    platform_table, platform_stats = shared_typology_by_group(clusters, "source_id")
    platform_table.to_csv(RESULTS / "06_genre_platform/gold_typology_by_platform.csv")

    print("\n=== descriptives by genre ===")
    print(genre_desc.to_string(index=False))
    print("\n=== descriptives by platform ===")
    print(platform_desc.to_string(index=False))
    print("\n=== confound tests (length, rating ~ genre/platform) ===")
    print(confounds.to_string(index=False))
    print("\n=== platform date ranges ===")
    print(dates.to_string(index=False))
    print("\n=== prevalence x genre: top associations (by Cramer's V) ===")
    print(genre_prev[["label", "chi2", "p_value", "q_value_bh", "cramers_v", "min_expected_cell"]].head(10).to_string(index=False))
    print("\n=== prevalence x platform: top associations (by Cramer's V) ===")
    print(platform_prev[["label", "chi2", "p_value", "q_value_bh", "cramers_v", "min_expected_cell"]].head(10).to_string(index=False))

    print(f"\nwrote results to {RESULTS} (gold_genre_platform_*, gold_genre_prevalence.csv, "
          f"gold_platform_prevalence.csv, gold_typology_by_genre.csv, gold_typology_by_platform.csv)")


if __name__ == "__main__":
    main()

"""Genre and platform effects on the silver set (ANALYSIS_PLAN.md §7, Phase
C), raw (uncorrected) -- 466,976 reviews, 10 fiction genres x 8 platforms (vs.
gold's 4) -- real cell sizes this time (smallest genre x platform cells are in
the hundreds, not zero, unlike gold's `Romance` x `platform_3`), so this can
go further than gold's version could: full genre x platform stratification
becomes viable, not just marginal genre and marginal platform effects.

Same three pieces as gold's 06a_genre_platform.py (descriptives/confounds,
prevalence by genre/platform, typology cluster by genre/platform), reusing
the same src/common/stats_utils.py helpers so results are directly comparable
in method, not just in scale.

`platform_1`'s single-placeholder-user_id issue (ANALYSIS_PLAN.md §0) is
irrelevant here -- this script doesn't touch `user_id` at all.

Usage: python src/06c_genre_platform.py
Writes results/silver_genre_platform_descriptives_{genre,platform}.csv,
results/06_genre_platform/silver_genre_platform_confounds.csv, results/06_genre_platform/silver_platform_date_ranges.csv,
results/06_genre_platform/silver_genre_prevalence.csv, results/06_genre_platform/silver_platform_prevalence.csv,
results/06_genre_platform/silver_typology_by_genre.csv, results/06_genre_platform/silver_typology_by_platform.csv,
results/06_genre_platform/silver_litfic_rating_by_platform.csv
"""
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS, SPARSE_GOLD_LABELS, WEAK_MODEL_LABELS  # noqa: E402
from stats_utils import prevalence_by_group, typology_by_group  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
TESTED_LABELS = [l for l in MAIN_LABELS_USED + SUB_LABELS if l not in WEAK_MODEL_LABELS and l not in SPARSE_GOLD_LABELS]


def descriptives(df, group_col):
    rows = []
    for g, sub in df.groupby(group_col):
        rating = sub["rating"]
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
    for group_col in ["nur_genre", "source_id"]:
        groups_len = [g["num_sentences"].to_numpy() for _, g in df.groupby(group_col)]
        h_len, p_len = stats.kruskal(*groups_len)
        rows.append({"outcome": "num_sentences", "grouped_by": group_col,
                      "test": "Kruskal-Wallis", "statistic": round(h_len, 2), "p_value": p_len})

        rating_df = df.dropna(subset=["rating"])
        groups_rating = [g["rating"].to_numpy() for _, g in rating_df.groupby(group_col) if len(g) > 0]
        h_rat, p_rat = stats.kruskal(*groups_rating)
        rows.append({"outcome": "rating", "grouped_by": group_col,
                      "test": "Kruskal-Wallis", "statistic": round(h_rat, 2), "p_value": p_rat})
    return pd.DataFrame(rows)


def date_ranges(df):
    rows = []
    for platform, sub in df.groupby("source_id"):
        dates = sub["review_date"]
        dates = dates[dates != ""]
        rows.append({
            "source_id": platform, "n_reviews": len(sub), "n_with_date": len(dates),
            "min_date": dates.min() if len(dates) else "", "max_date": dates.max() if len(dates) else "",
        })
    return pd.DataFrame(rows)


def rating_by_genre_within_platform(df):
    d = df.copy()
    d["is_literary_fiction"] = d["nur_genre"] == "Literary_fiction"
    out = d.dropna(subset=["rating"]).groupby(
        ["source_id", "is_literary_fiction"], observed=True
    )["rating"].agg(["mean", "count"]).reset_index()
    out.columns = ["source_id", "is_literary_fiction", "mean_rating", "n"]
    return out


def main():
    df = pd.read_parquet(RESULTS / "00_data_prep/silver_review_features.parquet")
    clusters = pd.read_csv(RESULTS / "05_typology/silver_typology_clusters.csv")
    print(f"n reviews: {len(df)}")

    genre_desc = descriptives(df, "nur_genre")
    platform_desc = descriptives(df, "source_id")
    genre_desc.to_csv(RESULTS / "06_genre_platform/silver_genre_platform_descriptives_genre.csv", index=False)
    platform_desc.to_csv(RESULTS / "06_genre_platform/silver_genre_platform_descriptives_platform.csv", index=False)

    confounds = confound_tests(df)
    confounds.to_csv(RESULTS / "06_genre_platform/silver_genre_platform_confounds.csv", index=False)

    dates = date_ranges(df)
    dates.to_csv(RESULTS / "06_genre_platform/silver_platform_date_ranges.csv", index=False)

    print("computing prevalence by genre (chi-square)...")
    genre_prev = prevalence_by_group(df, "nur_genre", TESTED_LABELS)
    genre_prev.to_csv(RESULTS / "06_genre_platform/silver_genre_prevalence.csv", index=False)

    print("computing prevalence by platform (chi-square)...")
    platform_prev = prevalence_by_group(df, "source_id", TESTED_LABELS)
    platform_prev.to_csv(RESULTS / "06_genre_platform/silver_platform_prevalence.csv", index=False)

    litfic_rating = rating_by_genre_within_platform(df)
    litfic_rating.to_csv(RESULTS / "06_genre_platform/silver_litfic_rating_by_platform.csv", index=False)

    genre_table, genre_stats = typology_by_group(clusters, "nur_genre")
    genre_table.to_csv(RESULTS / "06_genre_platform/silver_typology_by_genre.csv")
    platform_table, platform_stats = typology_by_group(clusters, "source_id")
    platform_table.to_csv(RESULTS / "06_genre_platform/silver_typology_by_platform.csv")

    print("\n=== descriptives by genre ===")
    print(genre_desc.to_string(index=False))
    print("\n=== descriptives by platform ===")
    print(platform_desc.to_string(index=False))
    print("\n=== confound tests ===")
    print(confounds.to_string(index=False))
    print("\n=== platform date ranges ===")
    print(dates.to_string(index=False))
    print("\n=== rating: Literary_fiction vs. rest, within each platform ===")
    print(litfic_rating.to_string(index=False))
    print("\n=== prevalence x genre: top by Cramer's V ===")
    print(genre_prev[["label", "chi2", "p_value", "q_value_bh", "cramers_v", "min_expected_cell"]].head(10).to_string(index=False))
    print("\n=== prevalence x platform: top by Cramer's V ===")
    print(platform_prev[["label", "chi2", "p_value", "q_value_bh", "cramers_v", "min_expected_cell"]].head(10).to_string(index=False))

    print(f"\nwrote results to {RESULTS} (silver_genre_platform_*, silver_genre_prevalence.csv, "
          f"silver_platform_prevalence.csv, silver_typology_by_genre.csv, silver_typology_by_platform.csv, "
          f"silver_litfic_rating_by_platform.csv)")


if __name__ == "__main__":
    main()

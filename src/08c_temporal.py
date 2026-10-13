"""Temporality on the silver set (ANALYSIS_PLAN.md §10, Phase C), raw
(uncorrected) -- 466,973 reviews, same platform-date confound as gold
(`platform_1` still has zero dated reviews) plus 4 reviews dated 1970 (an
obvious placeholder/default timestamp predating every platform by decades --
excluded, not treated as data). Same platform-adjusted design as gold's
08a_temporal.py: every trend reported raw (year alone) and adjusted (year +
platform), only the adjusted version trusted as a genuine temporal claim.

Logistic/OLS regression on ~440K rows is still fast (IRLS scales fine), no
restructuring needed versus the gold version beyond the data source and label
list.

Second confound fix (ANALYSIS_PLAN.md TODO, 2026-08-19), same as gold's:
review length grew substantially over the period, so a review-level "contains
>=1 sentence of this label" trend can rise even when the true per-sentence
rate is flat, just because longer reviews have more chances to contain one
instance. Every category trend is now also fit at the sentence level (~5.2M
rows after date filtering -- still fast, logistic IRLS scales linearly), which
has no such mechanism to confound it; a review-level trend that vanishes at
the sentence level was mostly a length artifact, not a real shift.

Usage: python src/08c_temporal.py
Writes results/08_temporal/silver_temporal_category_trends.csv, silver_temporal_rating_length.csv,
results/08_temporal/silver_period_prevalence.csv, results/08_temporal/silver_typology_by_period.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS, SPARSE_GOLD_LABELS, WEAK_MODEL_LABELS  # noqa: E402
from stats_utils import benjamini_hochberg, prevalence_by_group, typology_by_group  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
TESTED_LABELS = [l for l in MAIN_LABELS_USED + SUB_LABELS if l not in SPARSE_GOLD_LABELS and l not in WEAK_MODEL_LABELS]

PERIOD_BINS = [1999, 2006, 2011, 2014, 2017, 2023]
PERIOD_LABELS = ["1999-2005", "2006-2010", "2011-2013", "2014-2016", "2017-2022"]


def load_dated():
    df = pd.read_parquet(RESULTS / "00_data_prep/silver_review_features.parquet")
    n_total = len(df)
    dated = df[df["review_date"] != ""].copy()
    dated["year"] = pd.to_datetime(dated["review_date"]).dt.year
    n_anomalous = (dated["year"] < 1999).sum()
    dated = dated[dated["year"] >= 1999]
    dated["year_c"] = dated["year"] - dated["year"].mean()
    dated["period"] = pd.cut(dated["year"], bins=PERIOD_BINS, labels=PERIOD_LABELS, right=False)
    n_p1_no_date = ((df["source_id"] == "platform_1") & (df["review_date"] == "")).sum()
    n_p1 = (df["source_id"] == "platform_1").sum()
    print(f"{len(dated)}/{n_total} reviews have a valid date "
          f"({n_anomalous} excluded as pre-1999 placeholder dates; "
          f"platform_1: {n_p1_no_date}/{n_p1} missing -- effectively all of it)")
    return dated


def category_trends(df):
    rows = []
    for label in TESTED_LABELS:
        d = df.assign(present=(df[f"count__{label}"] > 0).astype(int))
        raw = smf.logit("present ~ year_c", data=d).fit(disp=0)
        adj = smf.logit("present ~ year_c + C(source_id)", data=d).fit(disp=0)
        rows.append({
            "label": label,
            "raw_year_coef": round(raw.params["year_c"], 4),
            "raw_p_value": raw.pvalues["year_c"],
            "adjusted_year_coef": round(adj.params["year_c"], 4),
            "adjusted_p_value": adj.pvalues["year_c"],
        })
    out = pd.DataFrame(rows)
    out["raw_q_bh"] = benjamini_hochberg(out["raw_p_value"].to_numpy())
    out["adjusted_q_bh"] = benjamini_hochberg(out["adjusted_p_value"].to_numpy())
    return out.sort_values("adjusted_p_value")


def sentence_level_category_trends(dated, labels):
    sentence_df = pd.read_parquet(RESULTS / "00_data_prep/silver_sentence_sequences.parquet")
    year_map = dated.set_index("review_id")["year_c"]
    platform_map = dated.set_index("review_id")["source_id"]
    sent = sentence_df[sentence_df["review_id"].isin(year_map.index)].copy()
    sent["year_c"] = sent["review_id"].map(year_map)
    sent["source_id"] = sent["review_id"].map(platform_map)

    rows = []
    for label in labels:
        d = sent.assign(present=sent[label])
        raw = smf.logit("present ~ year_c", data=d).fit(disp=0)
        adj = smf.logit("present ~ year_c + C(source_id)", data=d).fit(disp=0)
        rows.append({
            "label": label,
            "sentence_raw_year_coef": round(raw.params["year_c"], 4),
            "sentence_raw_p_value": raw.pvalues["year_c"],
            "sentence_adjusted_year_coef": round(adj.params["year_c"], 4),
            "sentence_adjusted_p_value": adj.pvalues["year_c"],
        })
    out = pd.DataFrame(rows)
    out["sentence_raw_q_bh"] = benjamini_hochberg(out["sentence_raw_p_value"].to_numpy())
    out["sentence_adjusted_q_bh"] = benjamini_hochberg(out["sentence_adjusted_p_value"].to_numpy())
    return out


def rating_and_length_trends(df):
    rows = []
    for outcome in ["rating", "num_sentences"]:
        sub = df.dropna(subset=[outcome])
        raw = smf.ols(f"{outcome} ~ year_c", data=sub).fit()
        adj = smf.ols(f"{outcome} ~ year_c + C(source_id)", data=sub).fit()
        rho, rho_p = stats.spearmanr(sub["year"], sub[outcome])
        rows.append({
            "outcome": outcome, "n": len(sub),
            "spearman_rho": round(rho, 4), "spearman_p": rho_p,
            "raw_year_coef": round(raw.params["year_c"], 4), "raw_p_value": raw.pvalues["year_c"],
            "adjusted_year_coef": round(adj.params["year_c"], 4), "adjusted_p_value": adj.pvalues["year_c"],
        })
    return pd.DataFrame(rows)


def main():
    dated = load_dated()

    print("fitting category trend models (raw + platform-adjusted, review-level)...")
    trends = category_trends(dated)
    print("fitting category trend models (sentence-level -- controls for length by construction)...")
    sentence_trends = sentence_level_category_trends(dated, TESTED_LABELS)
    trends = trends.merge(sentence_trends, on="label").sort_values("adjusted_p_value")
    trends.to_csv(RESULTS / "08_temporal/silver_temporal_category_trends.csv", index=False)

    rating_length = rating_and_length_trends(dated)
    rating_length.to_csv(RESULTS / "08_temporal/silver_temporal_rating_length.csv", index=False)

    period_prev = prevalence_by_group(dated, "period", TESTED_LABELS)
    period_prev.to_csv(RESULTS / "08_temporal/silver_period_prevalence.csv", index=False)

    clusters = pd.read_csv(RESULTS / "05_typology/silver_typology_clusters.csv")
    clusters_dated = clusters.merge(dated[["review_id", "period"]], on="review_id", how="inner")
    period_table, period_stats = typology_by_group(clusters_dated, "period")
    period_table.to_csv(RESULTS / "08_temporal/silver_typology_by_period.csv")

    print("\n=== category prevalence trend over time, review-level (raw vs. platform-adjusted) ===")
    show = trends[["label", "raw_year_coef", "raw_p_value", "raw_q_bh",
                    "adjusted_year_coef", "adjusted_p_value", "adjusted_q_bh"]]
    print(show.to_string(index=False))

    print("\nsentence-level: 'this sentence carries the label' -- no review-length confound possible")
    show_sent = trends[["label", "sentence_raw_year_coef", "sentence_raw_q_bh",
                         "sentence_adjusted_year_coef", "sentence_adjusted_q_bh"]]
    print(show_sent.to_string(index=False))

    print("\nsurvives platform-adjustment at q<0.05, review-level:")
    survives = trends[trends["adjusted_q_bh"] < 0.05]
    print(survives[["label", "adjusted_year_coef", "adjusted_q_bh"]].to_string(index=False) if len(survives) else "  (none)")

    print("\n...of those, which ALSO survive at the sentence level (q<0.05, same direction --"
          " these are the trends not explained by review length growth):")
    both = survives[survives["sentence_adjusted_q_bh"] < 0.05]
    both = both[np.sign(both["adjusted_year_coef"]) == np.sign(both["sentence_adjusted_year_coef"])]
    print(both[["label", "adjusted_year_coef", "sentence_adjusted_year_coef"]].to_string(index=False) if len(both) else "  (none)")

    review_only = survives[~survives["label"].isin(both["label"])]
    print("\n...review-level trend that does NOT survive at the sentence level "
          "(consistent with a length-driven artifact, not a real per-sentence shift):")
    print(review_only[["label", "adjusted_year_coef", "sentence_adjusted_year_coef", "sentence_adjusted_q_bh"]].to_string(index=False)
          if len(review_only) else "  (none)")

    print("\n=== rating & length over time ===")
    print(rating_length.to_string(index=False))

    print(f"\n=== typology cluster x period ===")
    print(f"chi2={period_stats['chi2']:.2f}, p={period_stats['p_value']:.4g}, "
          f"Cramer's V={period_stats['cramers_v']:.3f}, min expected cell={period_stats['min_expected_cell']:.2f}")
    print(period_table)

    print(f"\nwrote results to {RESULTS} (silver_temporal_*, silver_period_prevalence.csv, "
          f"silver_typology_by_period.csv)")


if __name__ == "__main__":
    main()

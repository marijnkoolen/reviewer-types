"""Temporality (ANALYSIS_PLAN.md §10), gold set only.

platform_1 has zero dated reviews (all 89 -- confirmed below), so it drops
out of this entire section automatically once reviews without a date are
excluded; 1,063/1,152 reviews remain.

The central problem this section has to deal with before anything else:
platform and date are badly confounded here. platform_5 supplies 73-100% of
reviews before 2012 and vanishes entirely after 2016; platform_2 is the only
source left by 2021-2022. So a raw "category X over time" trend risks just
re-describing "platform composition changed over time," not a real shift in
how people review. Every trend below is reported two ways: raw (year alone)
and platform-adjusted (year + platform as covariates in a logistic/linear
model) -- if a trend survives adjustment, that's real evidence of a temporal
effect net of the platform-mix confound; if it doesn't, the raw trend was
mostly platform composition.

Second confound, found later (see ANALYSIS_PLAN.md TODO, 2026-08-19) and
fixed here rather than just caveated: the category-trend regression's outcome
was review-level "contains >=1 sentence of this label," and review length
grew substantially over 1999-2022 -- so a category whose true per-sentence
rate never changed could still show a rising review-level trend purely
because longer reviews have more chances to contain at least one instance,
independent of platform. The fix is the same in spirit as the occurrence
unit-of-analysis correction (Phase C, silver occurrence): report the trend at
the SENTENCE level too (does an individual sentence's probability of carrying
the label change over time), which has no "more chances per review" mechanism
to confound it. Both levels are reported side by side below -- a coefficient
that only shows up at the review level and vanishes at the sentence level was
mostly a length artifact, not a real shift in emphasis.

Usage: python src/08a_temporal.py
Writes results/08_temporal/gold_temporal_category_trends.csv, gold_temporal_rating_length.csv,
results/08_temporal/gold_period_prevalence.csv, results/08_temporal/gold_typology_by_period.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS, SPARSE_GOLD_LABELS  # noqa: E402
from stats_utils import benjamini_hochberg, prevalence_by_group, typology_by_group  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
TESTED_LABELS = [l for l in MAIN_LABELS_USED + SUB_LABELS if l not in SPARSE_GOLD_LABELS]

PERIOD_BINS = [1999, 2006, 2011, 2014, 2017, 2023]
PERIOD_LABELS = ["1999-2005", "2006-2010", "2011-2013", "2014-2016", "2017-2022"]


def load_dated():
    df = pd.read_csv(RESULTS / "00_data_prep/gold_review_features.csv")
    df["review_date"] = df["review_date"].fillna("")
    n_total = len(df)
    dated = df[df["review_date"] != ""].copy()
    dated["year"] = pd.to_datetime(dated["review_date"]).dt.year
    dated["year_c"] = dated["year"] - dated["year"].mean()  # centered, for stabler fits
    dated["period"] = pd.cut(dated["year"], bins=PERIOD_BINS, labels=PERIOD_LABELS, right=False)
    n_p1_no_date = ((df["source_id"] == "platform_1") & (df["review_date"] == "")).sum()
    n_p1 = (df["source_id"] == "platform_1").sum()
    print(f"{len(dated)}/{n_total} reviews have a date "
          f"(platform_1: {n_p1_no_date}/{n_p1} missing -- effectively all of it)")
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
    """Same trend, computed on the sentence as the unit instead of the review,
    so a category whose per-sentence rate is flat can't show a spurious trend
    just because reviews got longer (see module docstring)."""
    sentence_df = pd.read_csv(RESULTS / "00_data_prep/gold_sentence_sequences.csv")
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
    d = df.copy()
    d["rating"] = pd.to_numeric(d["rating"], errors="coerce")
    rows = []
    for outcome in ["rating", "num_sentences"]:
        sub = d.dropna(subset=[outcome])
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

    trends = category_trends(dated)
    sentence_trends = sentence_level_category_trends(dated, TESTED_LABELS)
    trends = trends.merge(sentence_trends, on="label").sort_values("adjusted_p_value")
    trends.to_csv(RESULTS / "08_temporal/gold_temporal_category_trends.csv", index=False)

    rating_length = rating_and_length_trends(dated)
    rating_length.to_csv(RESULTS / "08_temporal/gold_temporal_rating_length.csv", index=False)

    period_prev = prevalence_by_group(dated, "period", TESTED_LABELS)
    period_prev.to_csv(RESULTS / "08_temporal/gold_period_prevalence.csv", index=False)

    clusters = pd.read_csv(RESULTS / "05_typology/gold_typology_clusters.csv")
    clusters_dated = clusters.merge(dated[["review_id", "period"]], on="review_id", how="inner")
    period_table, period_stats = typology_by_group(clusters_dated, "period")
    period_table.to_csv(RESULTS / "08_temporal/gold_typology_by_period.csv")

    print("\n=== category prevalence trend over time (logistic regression, year centered) ===")
    print("review-level: 'review contains >=1 sentence of this label' -- raw = year alone; "
          "adjusted = year + platform")
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
    if len(both):
        print(both[["label", "adjusted_year_coef", "sentence_adjusted_year_coef"]].to_string(index=False))
    else:
        print("  (none)")

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

    print(f"\nwrote results to {RESULTS} (gold_temporal_*, gold_period_prevalence.csv, "
          f"gold_typology_by_period.csv)")


if __name__ == "__main__":
    main()

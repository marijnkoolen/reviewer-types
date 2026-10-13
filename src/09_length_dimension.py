"""Review length as a substantive dimension (ANALYSIS_PLAN.md TODO item 2),
gold + silver via a `source` argument, same pattern as
11_sequence_cluster_profile.py.

Motivation (chat, 2026-08-19): item 1 fixed a real measurement artifact --
review-level "contains >=1 sentence of this label" mechanically inflates with
more sentences -- but "length artifact" was doing too much work as a label for
that fix. Length itself (how much a reviewer chose to say) is plausibly
meaningful, not noise, and until now has only ever appeared as a nuisance
covariate to net out (temporal's platform-adjustment, item 1's sentence-level
reweighting). This script treats it as a first-class variable instead, three
ways:

1. Composition vs. length, cross-sectionally: does each category's *share* of
   a review (prop__label -- the length-neutral proportion the typology
   clustering is itself built on) shrink or grow as reviews get longer?
   Complements, and is distinct from, item 1's temporal trends: this asks "do
   long and short reviews look different," not "has the corpus changed over
   time." Also completes the presence/rate/share triangle from item 1 by
   re-running the year-trend regression a third way, with prop__label as the
   (OLS, unweighted, one row per review) outcome instead of review-level
   presence (logistic) or pooled sentence-level rate (logistic, length-weighted).
2. Do long reviews cluster differently? Cross-tabs the existing compositional
   and sequence cluster assignments (05a_typology_clustering.py /
   05c_typology_clustering.py) against a length quartile, chi-square + Cramer's V,
   same pattern as the genre/platform-by-cluster tests.
3. Sentence density and specificity: two new per-review measures never
   computed anywhere else -- density (mean number of simultaneously active
   labels per sentence, main+sub together) and specificity (the fraction of
   *labeled* sentences that carry at least one sub-category, vs. main-only).
   Correlated with length and year, and profiled per typology cluster.
4. Sentence length itself, gold-only (chat, 2026-09-22): does review length's
   negative correlation with density (item 3) come from longer reviews using
   shorter, more clipped sentences? Testable directly since gold's sentence
   table carries character offsets (`sent_offset`/`sent_end`) -- no
   reparsing needed. Silver's raw predictions file has no sentence text or
   offsets at all, so this part is gold-only, a hard data limitation, not a
   scoping choice. Two correlations: mean sentence length vs. review length
   (does a longer review's average sentence get shorter?), and sentence
   length vs. that sentence's own density (do longer sentences carry more
   simultaneous labels, mechanically?).

Density/specificity use each source's own available label set (gold: all 20
labels; silver: 16, WEAK_MODEL_LABELS dropped) -- not directly comparable
gold-to-silver in absolute terms as a result (silver has 4 fewer sub-category
dimensions to be "specific" about), a difference already baked into every
other silver composition measure in this project, not a new asymmetry
introduced here.

Usage: python src/09_length_dimension.py [gold|silver]  (default: gold)
Requires {source}_temporal_category_trends.csv (08a_temporal.py / 08c_temporal.py)
to already exist, for the three-way comparison.
Writes results/{source}_length_share_correlation.csv,
  {source}_length_proportion_year_trends.csv,
  {source}_length_temporal_three_way_comparison.csv,
  {source}_length_quartile_by_composition_cluster.csv,
  {source}_length_quartile_by_sequence_cluster.csv,
  {source}_length_quartile_typology_stats.csv,
  {source}_density_specificity_by_review.csv,
  {source}_density_specificity_correlations.csv,
  {source}_density_specificity_by_composition_cluster.csv,
  {source}_density_specificity_by_sequence_cluster.csv
Gold only, additionally writes gold_sentence_length_correlations.csv,
  gold_sentence_length_by_review.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS, SPARSE_GOLD_LABELS, WEAK_MODEL_LABELS  # noqa: E402
from stats_utils import benjamini_hochberg, typology_by_group  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "gold"
assert SOURCE in ("gold", "silver")

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
EXCLUDE = set(SPARSE_GOLD_LABELS) | (set(WEAK_MODEL_LABELS) if SOURCE == "silver" else set())
TESTED_LABELS = [l for l in MAIN_LABELS_USED + SUB_LABELS if l not in EXCLUDE]

QUARTILE_LABELS = ["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"]


def load(name, subdir):
    if SOURCE == "gold":
        return pd.read_csv(RESULTS / subdir / f"gold_{name}.csv")
    path_csv = RESULTS / subdir / f"silver_{name}.csv"
    path_parquet = RESULTS / subdir / f"silver_{name}.parquet"
    return pd.read_csv(path_csv) if path_csv.exists() else pd.read_parquet(path_parquet)


def load_dated_years(review_features):
    df = review_features.copy()
    df["review_date"] = df["review_date"].fillna("") if df["review_date"].dtype == object else df["review_date"]
    dated = df[df["review_date"] != ""].copy()
    dated["year"] = pd.to_datetime(dated["review_date"]).dt.year
    if SOURCE == "silver":
        dated = dated[dated["year"] >= 1999]  # 4 pre-1999 placeholder dates, see 08c_temporal.py
    dated["year_c"] = dated["year"] - dated["year"].mean()
    return dated


# --- sub-analysis 1: composition vs. length, cross-sectionally ------------

def length_share_correlation(df):
    rows = []
    for label in TESTED_LABELS:
        rho, rho_p = stats.spearmanr(df["num_sentences"], df[f"prop__{label}"])
        d = df.assign(share=df[f"prop__{label}"])
        ols = smf.ols("share ~ num_sentences", data=d).fit()
        rows.append({
            "label": label,
            "spearman_rho": round(rho, 4), "spearman_p": rho_p,
            "ols_coef_per_sentence": round(ols.params["num_sentences"], 6),
            "ols_p_value": ols.pvalues["num_sentences"],
            "ols_r2": round(ols.rsquared, 4),
        })
    out = pd.DataFrame(rows)
    out["spearman_q_bh"] = benjamini_hochberg(out["spearman_p"].to_numpy())
    out["ols_q_bh"] = benjamini_hochberg(out["ols_p_value"].to_numpy())
    return out.sort_values("spearman_rho")


def length_proportion_year_trends(dated):
    rows = []
    for label in TESTED_LABELS:
        d = dated.assign(share=dated[f"prop__{label}"])
        raw = smf.ols("share ~ year_c", data=d).fit()
        adj = smf.ols("share ~ year_c + C(source_id)", data=d).fit()
        rows.append({
            "label": label,
            "proportion_raw_year_coef": round(raw.params["year_c"], 6),
            "proportion_raw_p_value": raw.pvalues["year_c"],
            "proportion_adjusted_year_coef": round(adj.params["year_c"], 6),
            "proportion_adjusted_p_value": adj.pvalues["year_c"],
        })
    out = pd.DataFrame(rows)
    out["proportion_raw_q_bh"] = benjamini_hochberg(out["proportion_raw_p_value"].to_numpy())
    out["proportion_adjusted_q_bh"] = benjamini_hochberg(out["proportion_adjusted_p_value"].to_numpy())
    return out.sort_values("proportion_adjusted_p_value")


# --- sub-analysis 2: do long reviews cluster differently? ------------------

def length_quartile_clusters(clusters):
    clusters = clusters.copy()
    clusters["length_quartile"] = pd.qcut(clusters["num_sentences"], 4, labels=QUARTILE_LABELS, duplicates="drop")

    table_comp, stats_comp = typology_by_group(clusters, "length_quartile", "cluster_composition")
    table_seq, stats_seq = typology_by_group(clusters, "length_quartile", "cluster_sequence")

    stats_out = pd.DataFrame([
        {"cluster_type": "composition", **stats_comp},
        {"cluster_type": "sequence", **stats_seq},
    ])
    return clusters, table_comp, table_seq, stats_out


# --- sub-analysis 3: sentence density and specificity ----------------------

def density_specificity_per_review(sentence_df):
    sub_cols = [l for l in SUB_LABELS if l in sentence_df.columns]
    all_cols = MAIN_LABELS_USED + sub_cols
    s = sentence_df.copy()
    s["_density"] = s[all_cols].sum(axis=1)
    s["_is_labeled"] = s["_density"] > 0
    s["_has_sub"] = s[sub_cols].sum(axis=1) > 0

    agg = s.groupby("review_id").agg(
        mean_density=("_density", "mean"),
        n_sentences=("_density", "size"),
        n_labeled_sentences=("_is_labeled", "sum"),
        n_specific_sentences=("_has_sub", "sum"),
    ).reset_index()
    agg["specificity"] = agg["n_specific_sentences"] / agg["n_labeled_sentences"]
    return agg


def density_specificity_correlations(df, dated_ids_to_year):
    rows = []
    for measure in ["mean_density", "specificity"]:
        sub = df.dropna(subset=[measure])
        rho_len, p_len = stats.spearmanr(sub["num_sentences"], sub[measure])
        ols_len = smf.ols(f"{measure} ~ num_sentences", data=sub).fit()
        rows.append({
            "measure": measure, "correlated_with": "num_sentences", "n": len(sub),
            "spearman_rho": round(rho_len, 4), "spearman_p": p_len,
            "ols_coef": round(ols_len.params["num_sentences"], 6), "ols_p_value": ols_len.pvalues["num_sentences"],
        })

        sub_y = sub[sub["review_id"].isin(dated_ids_to_year.index)].copy()
        sub_y["year_c"] = sub_y["review_id"].map(dated_ids_to_year)
        rho_y, p_y = stats.spearmanr(sub_y["year_c"], sub_y[measure])
        ols_y = smf.ols(f"{measure} ~ year_c", data=sub_y).fit()
        rows.append({
            "measure": measure, "correlated_with": "year", "n": len(sub_y),
            "spearman_rho": round(rho_y, 4), "spearman_p": p_y,
            "ols_coef": round(ols_y.params["year_c"], 6), "ols_p_value": ols_y.pvalues["year_c"],
        })
    return pd.DataFrame(rows)


def sentence_length_analysis(sentence_df, review_features):
    """Gold-only: does review length's negative correlation with density
    come from shorter sentences in longer reviews? Two correlations, both
    Spearman since sentence/review length are themselves right-skewed."""
    s = sentence_df.copy()
    s["sent_chars"] = s["sent_end"] - s["sent_offset"]
    sub_cols = [l for l in SUB_LABELS if l in s.columns]
    s["_density"] = s[MAIN_LABELS_USED + sub_cols].sum(axis=1)

    per_review = s.groupby("review_id").agg(
        mean_sent_chars=("sent_chars", "mean"), num_sentences=("sent_chars", "size")
    ).reset_index()

    rows = []
    rho1, p1 = stats.spearmanr(per_review["num_sentences"], per_review["mean_sent_chars"])
    rows.append({"measure": "mean_sent_chars", "correlated_with": "review num_sentences",
                  "n": len(per_review), "spearman_rho": round(rho1, 4), "spearman_p": p1})

    rho2, p2 = stats.spearmanr(s["sent_chars"], s["_density"])
    rows.append({"measure": "sent_chars", "correlated_with": "own-sentence density",
                  "n": len(s), "spearman_rho": round(rho2, 4), "spearman_p": p2})

    return pd.DataFrame(rows), per_review


def density_specificity_by_cluster(df, clusters, cluster_col):
    merged = df.merge(clusters[["review_id", cluster_col]], on="review_id")
    out = merged.groupby(cluster_col).agg(
        n_reviews=("review_id", "size"),
        mean_density=("mean_density", "mean"),
        mean_specificity=("specificity", lambda x: x.dropna().mean()),
    ).reset_index()
    return out


def main():
    review_features = load("review_features", "00_data_prep")
    clusters = load("typology_clusters", "05_typology")
    sentence_df = load("sentence_sequences", "00_data_prep")
    dated = load_dated_years(review_features)

    print(f"=== {SOURCE}: {len(review_features)} reviews, {len(sentence_df)} sentences, "
          f"{len(dated)} dated ===")

    # 1. composition vs. length
    share_corr = length_share_correlation(review_features)
    share_corr.to_csv(RESULTS / "09_length_dimension" / f"{SOURCE}_length_share_correlation.csv", index=False)
    print(f"\n--- share (prop__label) vs. num_sentences, cross-sectional ---")
    print(share_corr[["label", "spearman_rho", "spearman_q_bh", "ols_coef_per_sentence", "ols_q_bh"]]
          .to_string(index=False))

    prop_trends = length_proportion_year_trends(dated)
    prop_trends.to_csv(RESULTS / "09_length_dimension" / f"{SOURCE}_length_proportion_year_trends.csv", index=False)
    print(f"\n--- third way: share (prop__label) trend over time, raw vs. platform-adjusted ---")
    print(prop_trends[["label", "proportion_adjusted_year_coef", "proportion_adjusted_q_bh"]]
          .to_string(index=False))

    # three-way comparison: review-level presence, sentence-level rate (item 1),
    # and now share/proportion (this item) -- do all three agree on direction?
    existing_trends = pd.read_csv(RESULTS / "08_temporal" / f"{SOURCE}_temporal_category_trends.csv")
    three_way = existing_trends.merge(prop_trends, on="label")
    three_way["agree_review_sentence"] = (np.sign(three_way["adjusted_year_coef"])
                                           == np.sign(three_way["sentence_adjusted_year_coef"]))
    three_way["agree_all_three"] = (three_way["agree_review_sentence"]
                                     & (np.sign(three_way["sentence_adjusted_year_coef"])
                                        == np.sign(three_way["proportion_adjusted_year_coef"])))
    three_way.to_csv(RESULTS / "09_length_dimension" / f"{SOURCE}_length_temporal_three_way_comparison.csv", index=False)
    print(f"\n--- three-way comparison: review-level presence vs. sentence-level rate vs. share/proportion ---")
    show3 = three_way[["label", "adjusted_year_coef", "sentence_adjusted_year_coef",
                        "proportion_adjusted_year_coef", "agree_all_three"]]
    print(show3.to_string(index=False))

    # 2. do long reviews cluster differently?
    clusters_q, table_comp, table_seq, quartile_stats = length_quartile_clusters(clusters)
    table_comp.to_csv(RESULTS / "09_length_dimension" / f"{SOURCE}_length_quartile_by_composition_cluster.csv")
    table_seq.to_csv(RESULTS / "09_length_dimension" / f"{SOURCE}_length_quartile_by_sequence_cluster.csv")
    quartile_stats.to_csv(RESULTS / "09_length_dimension" / f"{SOURCE}_length_quartile_typology_stats.csv", index=False)
    print(f"\n--- length quartile x typology cluster ---")
    print(quartile_stats.to_string(index=False))

    # 3. sentence density and specificity
    dens_spec = density_specificity_per_review(sentence_df)
    dens_spec = dens_spec.merge(review_features[["review_id", "num_sentences"]], on="review_id")
    dens_spec.to_csv(RESULTS / "09_length_dimension" / f"{SOURCE}_density_specificity_by_review.csv", index=False)

    year_map = dated.set_index("review_id")["year_c"]
    dens_spec_corr = density_specificity_correlations(dens_spec, year_map)
    dens_spec_corr.to_csv(RESULTS / "09_length_dimension" / f"{SOURCE}_density_specificity_correlations.csv", index=False)
    print(f"\n--- density / specificity correlations ---")
    print(dens_spec_corr.to_string(index=False))

    dens_spec_by_comp = density_specificity_by_cluster(dens_spec, clusters, "cluster_composition")
    dens_spec_by_comp.to_csv(RESULTS / "09_length_dimension" / f"{SOURCE}_density_specificity_by_composition_cluster.csv", index=False)
    dens_spec_by_seq = density_specificity_by_cluster(dens_spec, clusters, "cluster_sequence")
    dens_spec_by_seq.to_csv(RESULTS / "09_length_dimension" / f"{SOURCE}_density_specificity_by_sequence_cluster.csv", index=False)
    print(f"\n--- mean density/specificity by compositional cluster ---")
    print(dens_spec_by_comp.to_string(index=False))
    print(f"\n--- mean density/specificity by sequence cluster ---")
    print(dens_spec_by_seq.to_string(index=False))

    # 4. sentence length itself, gold-only (no sentence text/offsets in silver)
    if SOURCE == "gold":
        sent_len_corr, sent_len_by_review = sentence_length_analysis(sentence_df, review_features)
        sent_len_corr.to_csv(RESULTS / "09_length_dimension/gold_sentence_length_correlations.csv", index=False)
        sent_len_by_review.to_csv(RESULTS / "09_length_dimension/gold_sentence_length_by_review.csv", index=False)
        print(f"\n--- sentence length vs. review length and density (gold-only) ---")
        print(sent_len_corr.to_string(index=False))

    print(f"\nwrote results to {RESULTS} ({SOURCE}_length_*, {SOURCE}_density_specificity_*)")


if __name__ == "__main__":
    main()

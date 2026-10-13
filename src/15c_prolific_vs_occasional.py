"""Do prolific reviewers differ from occasional reviewers, and how?
(ANALYSIS_PLAN.md item 8). Silver-only -- gold has no reviewer with >=10
reviews (excl. platform_1, max 8), so there's no gold counterpart the way
items 1/2/5/7 had.

Items 3/4/6 established that prolific reviewers (>=10 reviews) each have an
internally consistent personal "signature." This asks a different question:
does that signature differ *systematically* from how occasional reviewers
(1-2 reviews) write, and is it a stable trait visible from a reviewer's very
first review (a selection effect -- people prone to writing that way are the
ones who keep reviewing) or something that develops as a reviewer
accumulates experience (a within-person habituation effect)?

Five comparisons, all pooling REVIEWS (not reviewers) within each group so
sample size reflects how much material each group actually contributes, not
how many people are in it:
1. Compositional/sequential profile: prevalence_by_group + typology_by_group,
   same machinery as the genre/platform sections.
2. Length, density (item 2), specificity (item 2): Mann-Whitney U + Cohen's d.
3. Rating: Mann-Whitney U + Cohen's d.
4. Genre/platform concentration: does the prolific group's reviews cluster
   into particular genres/platforms more than the occasional group's do --
   crosstab(group, genre/platform), chi-square + Cramer's V.
5. Selection vs. development, the sharpest test: split each prolific
   reviewer's own dated history into their first N_EARLY_LATE reviews and
   their most recent N_EARLY_LATE (requires >=10 total so the two windows
   never overlap). (a) early reviews vs. occasional reviews directly --
   evidence of a trait/selection effect if these already differ; (b) early
   vs. late WITHIN THE SAME REVIEWER, paired (Wilcoxon signed-rank on the
   per-reviewer early->late difference, not a pooled comparison) -- evidence
   of genuine development over a career if these differ. Both needed
   together: "prolific" is itself a selection effect (having stayed
   engaged), so neither comparison alone can separate who-becomes-prolific
   from what-becoming-prolific-does.

At this scale (hundreds of thousands of reviews on one side), p-values are
close to guaranteed for any real effect -- effect sizes (Cramer's V, Cohen's
d, median differences) are the numbers to read, p/q-values for completeness.

Usage: python src/15c_prolific_vs_occasional.py
Writes results/silver_prolific_vs_occasional_{composition,cluster,scalar,genre,platform}.csv,
  results/silver_prolific_early_vs_occasional.csv,
  results/silver_prolific_early_vs_late_paired.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS, SPARSE_GOLD_LABELS, WEAK_MODEL_LABELS  # noqa: E402
from stats_utils import benjamini_hochberg, prevalence_by_group, typology_by_group, cramers_v  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
TESTED_LABELS = [l for l in MAIN_LABELS_USED + SUB_LABELS if l not in SPARSE_GOLD_LABELS and l not in WEAK_MODEL_LABELS]

MIN_PROLIFIC = 10
MAX_OCCASIONAL = 2
N_EARLY_LATE = 3


def load():
    review = pd.read_parquet(RESULTS / "00_data_prep/silver_review_features.parquet")
    review = review[review["source_id"] != "platform_1"].reset_index(drop=True)

    counts = review["user_id"].value_counts()
    prolific_ids = set(counts[counts >= MIN_PROLIFIC].index)
    occasional_ids = set(counts[(counts >= 1) & (counts <= MAX_OCCASIONAL)].index)

    review["group"] = np.where(review["user_id"].isin(prolific_ids), "prolific",
                        np.where(review["user_id"].isin(occasional_ids), "occasional", "other"))

    dens_spec = pd.read_csv(RESULTS / "09_length_dimension/silver_density_specificity_by_review.csv")[
        ["review_id", "mean_density", "specificity"]]
    review = review.merge(dens_spec, on="review_id", how="left")

    clusters = pd.read_csv(RESULTS / "05_typology/silver_typology_clusters.csv")[
        ["review_id", "cluster_composition", "cluster_sequence"]]
    review = review.merge(clusters, on="review_id", how="left")

    return review, prolific_ids, occasional_ids


def mann_whitney_effect(a, b, name):
    a, b = a.dropna(), b.dropna()
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    pooled_std = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    cohens_d = (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else float("nan")
    return {
        "measure": name, "n_prolific": len(a), "n_occasional": len(b),
        "mean_prolific": a.mean(), "mean_occasional": b.mean(),
        "median_prolific": a.median(), "median_occasional": b.median(),
        "cohens_d": cohens_d, "mann_whitney_p": p,
    }


def scalar_comparison(df):
    prol = df[df["group"] == "prolific"]
    occ = df[df["group"] == "occasional"]
    rows = [
        mann_whitney_effect(prol["num_sentences"], occ["num_sentences"], "num_sentences"),
        mann_whitney_effect(prol["mean_density"], occ["mean_density"], "mean_density"),
        mann_whitney_effect(prol["specificity"], occ["specificity"], "specificity"),
        mann_whitney_effect(pd.to_numeric(prol["rating"], errors="coerce"),
                             pd.to_numeric(occ["rating"], errors="coerce"), "rating"),
    ]
    out = pd.DataFrame(rows)
    out["q_bh"] = benjamini_hochberg(out["mann_whitney_p"].to_numpy())
    return out


def group_concentration(df, col):
    sub = df[df["group"].isin(["prolific", "occasional"])]
    table = pd.crosstab(sub["group"], sub[col])
    chi2, p, v, min_exp = cramers_v(table)
    row_pct = (table.T / table.sum(axis=1)).T * 100
    return table, row_pct, {"chi2": chi2, "p_value": p, "cramers_v": v, "min_expected_cell": min_exp}


def early_late_splits(review, prolific_ids):
    dated = review[(review["group"] == "prolific") & (review["review_date"] != "")].copy()
    dated["review_date"] = pd.to_datetime(dated["review_date"])
    dated = dated.sort_values(["user_id", "review_date"])

    counts = dated.groupby("user_id").size()
    eligible = counts[counts >= 2 * N_EARLY_LATE].index  # guarantees early/late never overlap
    dated = dated[dated["user_id"].isin(eligible)]

    dated["rank"] = dated.groupby("user_id").cumcount() + 1
    dated["rank_from_end"] = dated.groupby("user_id").cumcount(ascending=False) + 1
    early = dated[dated["rank"] <= N_EARLY_LATE].copy()
    late = dated[dated["rank_from_end"] <= N_EARLY_LATE].copy()
    print(f"early/late split: {len(eligible)} prolific reviewers with >={2 * N_EARLY_LATE} dated reviews "
          f"({len(early)} early reviews, {len(late)} late reviews)")
    return early, late


def paired_test(early_means, late_means, name):
    diff = (late_means - early_means).dropna()
    if len(diff) < 10:
        return None
    stat, p = stats.wilcoxon(diff)
    return {
        "measure": name, "n_reviewers": len(diff),
        "mean_early": early_means.mean(), "mean_late": late_means.mean(),
        "mean_paired_diff": diff.mean(), "median_paired_diff": diff.median(),
        "wilcoxon_p": p,
    }


def early_vs_late_comparison(early, late):
    rows = []
    for label in TESTED_LABELS:
        e = early.groupby("user_id")[f"prop__{label}"].mean()
        l = late.groupby("user_id")[f"prop__{label}"].mean()
        e, l = e.align(l, join="inner")
        r = paired_test(e, l, label)
        if r:
            rows.append(r)
    out = pd.DataFrame(rows)
    out["q_bh"] = benjamini_hochberg(out["wilcoxon_p"].to_numpy())

    scalar_rows = []
    for col, name in [("num_sentences", "num_sentences"), ("mean_density", "mean_density"),
                       ("specificity", "specificity")]:
        e = early.groupby("user_id")[col].mean()
        l = late.groupby("user_id")[col].mean()
        e, l = e.align(l, join="inner")
        r = paired_test(e, l, name)
        if r:
            scalar_rows.append(r)
    rating_e = pd.to_numeric(early["rating"], errors="coerce")
    rating_l = pd.to_numeric(late["rating"], errors="coerce")
    e = rating_e.groupby(early["user_id"]).mean()
    l = rating_l.groupby(late["user_id"]).mean()
    e, l = e.align(l, join="inner")
    r = paired_test(e, l, "rating")
    if r:
        scalar_rows.append(r)
    scalar_out = pd.DataFrame(scalar_rows)
    scalar_out["q_bh"] = benjamini_hochberg(scalar_out["wilcoxon_p"].to_numpy())
    return out.sort_values("wilcoxon_p"), scalar_out.sort_values("wilcoxon_p")


def main():
    review, prolific_ids, occasional_ids = load()
    print(f"{len(prolific_ids)} prolific reviewers ({(review['group'] == 'prolific').sum()} reviews), "
          f"{len(occasional_ids)} occasional reviewers ({(review['group'] == 'occasional').sum()} reviews)")

    # 1. compositional/sequential profile
    grouped_for_prevalence = review[review["group"].isin(["prolific", "occasional"])]
    comp_prevalence = prevalence_by_group(grouped_for_prevalence, "group", TESTED_LABELS)
    comp_prevalence.to_csv(RESULTS / "15_prolific_vs_occasional/silver_prolific_vs_occasional_composition.csv", index=False)
    comp_table, comp_stats = typology_by_group(grouped_for_prevalence, "group", "cluster_composition")
    seq_table, seq_stats = typology_by_group(grouped_for_prevalence, "group", "cluster_sequence")
    pd.DataFrame([{"cluster_type": "composition", **comp_stats}, {"cluster_type": "sequence", **seq_stats}]) \
        .to_csv(RESULTS / "15_prolific_vs_occasional/silver_prolific_vs_occasional_cluster_stats.csv", index=False)
    comp_table.to_csv(RESULTS / "15_prolific_vs_occasional/silver_prolific_vs_occasional_composition_cluster_crosstab.csv")
    seq_table.to_csv(RESULTS / "15_prolific_vs_occasional/silver_prolific_vs_occasional_sequence_cluster_crosstab.csv")
    print("\n=== compositional cluster membership, prolific vs. occasional ===")
    print(comp_stats)
    print("\n=== sequence cluster membership, prolific vs. occasional ===")
    print(seq_stats)

    # 2 & 3. length/density/specificity/rating
    scalar = scalar_comparison(review)
    scalar.to_csv(RESULTS / "15_prolific_vs_occasional/silver_prolific_vs_occasional_scalar.csv", index=False)
    print("\n=== length, density, specificity, rating: prolific vs. occasional ===")
    print(scalar.to_string(index=False))

    # 4. genre/platform concentration
    genre_table, genre_pct, genre_stats = group_concentration(review, "nur_genre")
    genre_pct.to_csv(RESULTS / "15_prolific_vs_occasional/silver_prolific_vs_occasional_genre.csv")
    platform_table, platform_pct, platform_stats = group_concentration(review, "source_id")
    platform_pct.to_csv(RESULTS / "15_prolific_vs_occasional/silver_prolific_vs_occasional_platform.csv")
    print(f"\n=== genre concentration: chi2={genre_stats['chi2']:.1f}, p={genre_stats['p_value']:.4g}, "
          f"Cramer's V={genre_stats['cramers_v']:.4f} ===")
    print(genre_pct.round(1))
    print(f"\n=== platform concentration: chi2={platform_stats['chi2']:.1f}, p={platform_stats['p_value']:.4g}, "
          f"Cramer's V={platform_stats['cramers_v']:.4f} ===")
    print(platform_pct.round(1))

    # 5. selection vs. development
    early, late = early_late_splits(review, prolific_ids)
    occasional = review[review["group"] == "occasional"]

    early_vs_occ_comp = prevalence_by_group(pd.concat([early.assign(group="prolific_early"), occasional]),
                                             "group", TESTED_LABELS)
    early_vs_occ_comp.to_csv(RESULTS / "15_prolific_vs_occasional/silver_prolific_early_vs_occasional_composition.csv", index=False)
    early_vs_occ_scalar = scalar_comparison(pd.concat([early.assign(group="prolific"), occasional]))
    early_vs_occ_scalar.to_csv(RESULTS / "15_prolific_vs_occasional/silver_prolific_early_vs_occasional_scalar.csv", index=False)
    print("\n=== prolific reviewers' EARLY reviews vs. occasional reviewers (selection-effect test) ===")
    print(early_vs_occ_scalar.to_string(index=False))

    comp_diff, scalar_diff = early_vs_late_comparison(early, late)
    comp_diff.to_csv(RESULTS / "15_prolific_vs_occasional/silver_prolific_early_vs_late_paired_composition.csv", index=False)
    scalar_diff.to_csv(RESULTS / "15_prolific_vs_occasional/silver_prolific_early_vs_late_paired_scalar.csv", index=False)
    print("\n=== prolific reviewers' EARLY vs. LATE reviews, paired within-reviewer (development-effect test) ===")
    print("composition (top 10 by q_bh):")
    print(comp_diff.head(10).to_string(index=False))
    print("\nscalar measures:")
    print(scalar_diff.to_string(index=False))

    print(f"\nwrote results to {RESULTS} (silver_prolific_{{vs_occasional,early_vs_occasional,early_vs_late}}*.csv)")


if __name__ == "__main__":
    main()

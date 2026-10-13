"""Zone discovery (ANALYSIS_PLAN.md item 13, part B3), gold. Completes the
contiguity/order follow-ups begun in 31_ (A1/A1b/B1) and 32_ (A2/A3/B2).

REFRAMED after B2's result (plan note, 2026-08-31). The original spec was:
build a zone-affinity matrix from B2 and cluster it. B2 found that **no
category pair merges into a contiguity zone above chance** -- several are
significantly separated -- so clustering zone affinity would be clustering
noise around a null result. Instead:

  1. Cluster on the PRECEDENCE structure (B1), which does show clean
     grouping and, unlike the contiguity measures, covers all 21 pairs.
     Distance d(A,B) = 2*|P(A before B) - 0.5|: 0 when the two have no
     reliable order (same positional zone), 1 when one always precedes the
     other. Average-linkage hierarchical clustering, k chosen by silhouette.
  2. CHARACTERIZE each discovered zone with the contiguity measures rather
     than deriving zones from them -- are the categories sharing a
     positional zone internally merged into one stretch, or internally
     separated? B2 predicts separated; this tests that per zone.

The `Recommendations` coverage gap is handled explicitly rather than
silently: it has full precedence coverage (all 6 of its pairs) so it
clusters normally, but 4 of its 6 contiguity pairs are unestimable (it is
usually a single sentence, below the k>=2 bar), so its zone cannot be
characterized on the contiguity side. Reported, not dropped.

Usage: python src/21a_zone_discovery.py [gold|silver]  (default: gold)
Writes results/{source}_zone_membership.csv, results/{source}_zone_profile.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, WEAK_MODEL_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "gold"
assert SOURCE in ("gold", "silver")

# gold's precedence/interleaving/affinity come from the unvectorized 19a_/20a_
# (permutation loop finishes fine at gold's scale); silver's come from the
# vectorized 23_zone_analysis_scaled.py instead (same schemas, per its own docstring).
PRECEDENCE_DIR = {"gold": "19_review_contiguity_precedence", "silver": "23_zone_analysis_scaled"}
ZONE_MEASURE_DIR = {"gold": "20_contiguity_zones", "silver": "23_zone_analysis_scaled"}

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
if SOURCE == "silver":
    MAIN_LABELS_USED = [m for m in MAIN_LABELS_USED if m not in WEAK_MODEL_LABELS]

# "Reliably ordered" is judged by EFFECT SIZE, not significance (2026-08-31).
# At gold's n the two coincide; at silver's they do not, and significance
# becomes useless for this purpose: 20 of 21 pairs clear q<0.05, including
# Content/Other_works at P=0.5056 (significant only because n=105,267), which
# would put nearly every category in its own zone and dissolve the structure.
# Gold's significance test was in effect already applying a threshold in this
# region -- its significant pairs bottom out at |P-0.5| = 0.0602 and its
# non-significant ones top out at 0.0542 -- so any cut in (0.054, 0.060)
# reproduces gold's classification exactly while remaining meaningful at
# silver scale. 0.055 is used for both sources so the criterion means the
# same thing in each.
ORDER_EFFECT_THRESHOLD = 0.055


def is_ordered(df):
    return (df["p_a_before_b"] - 0.5).abs() >= ORDER_EFFECT_THRESHOLD


def precedence_distance():
    """d = 2*|P(A before B) - 0.5|. Categories with no reliable order sit at
    distance ~0; a category that always precedes another sits at ~1."""
    prec = pd.read_csv(RESULTS / PRECEDENCE_DIR[SOURCE] / f"{SOURCE}_pairwise_precedence.csv")
    labels = [l for l in MAIN_LABELS_USED
              if l in set(prec["label_a"]) | set(prec["label_b"])]
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    D = np.zeros((n, n))
    covered = np.zeros((n, n), dtype=bool)
    np.fill_diagonal(covered, True)
    for _, r in prec.iterrows():
        i, j = idx[r["label_a"]], idx[r["label_b"]]
        d = 2 * abs(r["p_a_before_b"] - 0.5)
        D[i, j] = D[j, i] = d
        covered[i, j] = covered[j, i] = True
    if not covered.all():
        missing = [(labels[i], labels[j]) for i in range(n) for j in range(i + 1, n) if not covered[i, j]]
        raise ValueError(f"precedence matrix incomplete, cannot cluster: {missing}")
    return labels, D, prec


def choose_k(D, labels, prec):
    """k is chosen by a CONSISTENCY criterion, not silhouette.

    Silhouette is reported but is actively misleading on this distance
    matrix: it rewards isolating the single most distant point, so it picks
    k=2 ({Recommendations} vs. everything else) -- a partition that leaves 8
    significantly-ordered pairs sitting inside one "zone", which is exactly
    what a zone is supposed to exclude. The criterion that matches the
    construct: a good zone partition has few within-zone pairs that ARE
    reliably ordered (by effect size) (those zones should split) and few between-zone
    pairs that are NOT (those zones should merge). Ties break on silhouette.
    """
    Z = linkage(squareform(D, checks=False), method="average")
    idx = {l: i for i, l in enumerate(labels)}
    rows = []
    for k in range(2, len(labels)):
        lab = fcluster(Z, k, criterion="maxclust")
        if len(np.unique(lab)) < 2:
            continue
        zmap = {l: lab[idx[l]] for l in labels}
        p = prec.copy()
        same = p["label_a"].map(zmap) == p["label_b"].map(zmap)
        ordered = is_ordered(p)
        within_ordered = int((same & ordered).sum())
        between_unordered = int((~same & ~ordered).sum())
        rows.append({
            "k": k, "silhouette": round(float(silhouette_score(D, lab, metric="precomputed")), 4),
            "within_zone_but_ordered": within_ordered,
            "between_zone_but_unordered": between_unordered,
            "consistency_errors": within_ordered + between_unordered,
            "_labels": lab,
        })
    tab = pd.DataFrame(rows)
    best_row = tab.sort_values(["consistency_errors", "silhouette"],
                               ascending=[True, False]).iloc[0]
    return Z, tab.drop(columns="_labels"), (int(best_row["k"]), best_row["_labels"], tab)


def zone_contiguity_profile(labels, membership):
    """Characterize each zone with the contiguity measures: are categories
    sharing a positional zone internally merged or internally separated?"""
    inter = pd.read_csv(RESULTS / ZONE_MEASURE_DIR[SOURCE] / f"{SOURCE}_pair_interleaving.csv")
    zone = pd.read_csv(RESULTS / ZONE_MEASURE_DIR[SOURCE] / f"{SOURCE}_pair_zone_affinity.csv")
    zmap = dict(zip(membership["label"], membership["zone"]))

    def tag(df):
        d = df.copy()
        d["za"] = d["label_a"].map(zmap)
        d["zb"] = d["label_b"].map(zmap)
        d["same_zone"] = d["za"] == d["zb"]
        return d

    inter, zone = tag(inter), tag(zone)
    rows = []
    for z in sorted(membership["zone"].unique()):
        members = membership.loc[membership["zone"] == z, "label"].tolist()
        n_pairs_possible = len(members) * (len(members) - 1) // 2
        wi = inter[(inter["za"] == z) & inter["same_zone"]]
        wz = zone[(zone["za"] == z) & zone["same_zone"]]
        rows.append({
            "zone": z, "n_categories": len(members), "categories": ", ".join(members),
            "within_zone_pairs_possible": n_pairs_possible,
            "within_zone_pairs_estimable": len(wz),
            "mean_separation_excess": round(float(wi["mean_excess"].mean()), 4) if len(wi) else np.nan,
            "mean_zone_affinity_excess": round(float(wz["mean_excess"].mean()), 4) if len(wz) else np.nan,
        })
    prof = pd.DataFrame(rows)

    between_i = inter[~inter["same_zone"]]["mean_excess"].mean()
    between_z = zone[~zone["same_zone"]]["mean_excess"].mean()
    return prof, float(between_i), float(between_z), inter, zone


def main():
    labels, D, prec = precedence_distance()
    print(f"{SOURCE}: clustering {len(labels)} categories on precedence distance "
          f"({len(prec)} pairs, full coverage)")

    Z, sil, best = choose_k(D, labels, prec)
    print("\n=== k selection ===")
    print(sil.to_string(index=False))
    k, lab, tab = best
    print(f"\nchosen k={k} by consistency (fewest ordered-within / unordered-between pairs).")
    print("NOTE: silhouette would pick k=2, which is degenerate here -- it isolates the single "
          "most distant category (Recommendations) and leaves 8 significantly-ordered pairs "
          "inside one zone. Silhouette measures separation, not respect for the ordering structure.")
    runner = tab[tab["k"] != k].sort_values(["consistency_errors", "silhouette"],
                                             ascending=[True, False]).iloc[0]
    print(f"runner-up k={int(runner['k'])} ({int(runner['consistency_errors'])} errors vs "
          f"{int(tab[tab['k']==k]['consistency_errors'].iloc[0])}) -- see the report for why the "
          f"margin between them is thin.")

    # order zones by mean within-review position so zone 1 = earliest
    pos = pd.read_csv(RESULTS / "04_order" / f"{SOURCE}_order_position.csv")
    pos = pos[pos["level"] == "main"].set_index("label")["mean_rel_position"]
    membership = pd.DataFrame({"label": labels, "raw_cluster": lab})
    membership["mean_rel_position"] = membership["label"].map(pos)
    order = (membership.groupby("raw_cluster")["mean_rel_position"].mean()
             .sort_values().reset_index().assign(zone=lambda d: np.arange(1, len(d) + 1)))
    membership = membership.merge(order[["raw_cluster", "zone"]], on="raw_cluster")
    membership = membership.sort_values(["zone", "mean_rel_position"]).drop(columns="raw_cluster")
    membership.to_csv(RESULTS / "21_zone_discovery" / f"{SOURCE}_zone_membership.csv", index=False)

    print("\n=== discovered zones (ordered by mean within-review position) ===")
    print(membership.to_string(index=False))

    # consistency check against B1's raw significance pattern
    zmap = dict(zip(membership["label"], membership["zone"]))
    p = prec.copy()
    p["same_zone"] = p["label_a"].map(zmap) == p["label_b"].map(zmap)
    p["ordered"] = is_ordered(p)
    tab = pd.crosstab(p["same_zone"], p["ordered"])
    print("\n=== consistency: does the clustering match B1's significance pattern? ===")
    print(tab.to_string())
    within_ordered = int(tab.loc[True, True]) if (True in tab.index and True in tab.columns) else 0
    between_unordered = int(tab.loc[False, False]) if (False in tab.index and False in tab.columns) else 0
    print(f"within-zone pairs that ARE significantly ordered: {within_ordered} "
          f"(ideally 0 -- these are zones that could arguably split)")
    print(f"between-zone pairs that are NOT significantly ordered: {between_unordered} "
          f"(ideally 0 -- these are zones that could arguably merge)")

    prof, between_i, between_z, inter, zone = zone_contiguity_profile(labels, membership)
    prof.to_csv(RESULTS / "21_zone_discovery" / f"{SOURCE}_zone_profile.csv", index=False)
    print("\n=== zone profile: are positional zones internally merged, or internally separated? ===")
    print(prof.to_string(index=False))
    print(f"\nbetween-zone mean separation excess:   {between_i:+.4f}")
    print(f"between-zone mean zone-affinity excess: {between_z:+.4f}")
    print("(B2 predicts within-zone affinity is NOT positive -- positional zones are not merged passages)")

    missing = prof[prof["within_zone_pairs_estimable"] < prof["within_zone_pairs_possible"]]
    if len(missing):
        print("\nCOVERAGE GAP (stated, not silently dropped):")
        for _, r in missing.iterrows():
            print(f"  zone {r['zone']} ({r['categories']}): "
                  f"{r['within_zone_pairs_estimable']}/{r['within_zone_pairs_possible']} "
                  f"within-zone pairs estimable on the contiguity side")

    print(f"\nwrote results to {RESULTS} ({SOURCE}_zone_membership.csv, {SOURCE}_zone_profile.csv)")


if __name__ == "__main__":
    main()

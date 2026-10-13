"""Contiguity follow-ups, parts A2, A3 and B2 (ANALYSIS_PLAN.md item 13),
gold. Continues 19a_review_contiguity_precedence.py (A1/A1b/B1).

A2. CONFIGURATION-BOUNDARY score (whole-review, non-pairwise). Count
    positions where the active label-SET changes between sentence i and i+1,
    normalized against the minimum possible given the review's own distinct
    configurations (m-1, perfectly blocked) and the maximum (n-1, every
    sentence differs): 1 - (B - (m-1)) / ((n-1) - (m-1)). Legitimate despite
    66 distinct label-sets occurring corpus-wide, because it only ever tests
    ADJACENT configurations for equality -- it never treats sets as a
    modelling alphabet. Measures something distinct from A1: a review can
    have every label individually well-blocked (high A1) while staggered
    block boundaries still produce many configuration changes (lower A2).

A3. PAIRWISE INTERLEAVING (pairwise-decomposed). For each pair of labels both
    with k>=2, walk the review keeping only sentences carrying exactly one of
    them (A-only / B-only) and count switches: 1 switch = cleanly separated
    stretches, many = A->B->A alternation. Sentences carrying BOTH are
    skipped rather than forced to one side -- they are genuinely compatible
    with either, and assigning them arbitrarily would manufacture or hide
    switches. Reported as `separation` = 1 - interleaving, so higher means
    more block-like, matching A1/A2's direction.

B2. UNION ("category pair") CLUMPINESS -> zone affinity. Treat A∪B as one
    pseudo-label and compute its clumpiness. Distinct from §4's transition
    PMI, which is a local i->i+1 measure; this asks whether two BLOCKS sit
    together as one stretch. Two statistics, because they answer different
    questions:
      - `union_excess`: union clumpiness minus its own permutation null --
        "is A∪B contiguous beyond chance at all?"
      - `zone_affinity`: union clumpiness minus the k-weighted mean of A's
        and B's INDIVIDUAL clumpiness -- "does merging them ADD contiguity?"
        Strongly positive when two individually-scattered labels together
        fill one stretch (A->B->A->B is a zone even though neither label is
        blocked); negative when two well-blocked labels sit far apart.

All three use ONE joint sentence-order permutation per review (shared across
measures), which preserves every label's count and all co-occurrence
structure. Per A1b's convention, the headline for every measure is
excess-over-own-null, never the raw score: these are all count-based indices
whose chance baseline moves with n and k, so raw values are not comparable
across reviews of different lengths.

Usage: python src/20a_contiguity_zones.py [gold|silver]  (default: gold)
Writes results/{source}_review_boundary_interleaving.csv,
  results/{source}_pair_interleaving.csv,
  results/{source}_pair_zone_affinity.csv
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, WEAK_MODEL_LABELS  # noqa: E402
from stats_utils import benjamini_hochberg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "gold"
assert SOURCE in ("gold", "silver")

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
if SOURCE == "silver":
    MAIN_LABELS_USED = [m for m in MAIN_LABELS_USED if m not in WEAK_MODEL_LABELS]
PAIRS = list(combinations(range(len(MAIN_LABELS_USED)), 2))
PAIR_NAMES = [(MAIN_LABELS_USED[i], MAIN_LABELS_USED[j]) for i, j in PAIRS]

N_PERM = 1000
RNG_SEED = 0


def load_sentences():
    if SOURCE == "gold":
        return pd.read_csv(RESULTS / "00_data_prep/gold_sentence_sequences.csv")
    return pd.read_parquet(RESULTS / "00_data_prep/silver_sentence_sequences.parquet")


def count_runs(M):
    """Runs of 1s down axis -2 of a (..., n, L) binary array -> (..., L)."""
    first = M[..., 0, :]
    trans = ((M[..., 1:, :] == 1) & (M[..., :-1, :] == 0)).sum(axis=-2)
    return first + trans


def clumpiness(runs, k):
    return np.where(k > 1, 1.0 - (runs - 1) / np.maximum(k - 1, 1), np.nan)


def boundary_score(config_seq, m, n):
    """A2 for one ordering. config_seq: (..., n) integer codes."""
    b = (config_seq[..., 1:] != config_seq[..., :-1]).sum(axis=-1)
    denom = (n - 1) - (m - 1)
    return np.where(denom > 0, 1.0 - (b - (m - 1)) / np.maximum(denom, 1), np.nan)


def interleaving_separation(markers_sel):
    """A3 for one or many orderings. markers_sel: (..., n_sel) of +1/-1,
    already restricted to A-only/B-only sentences. Returns `separation`."""
    n_sel = markers_sel.shape[-1]
    if n_sel < 2:
        return np.full(markers_sel.shape[:-1], np.nan)
    switches = (markers_sel[..., 1:] != markers_sel[..., :-1]).sum(axis=-1)
    denom = n_sel - 2
    interleaving = np.where(denom > 0, (switches - 1) / max(denom, 1), 0.0)
    return 1.0 - interleaving


def analyse(sent):
    rng = np.random.default_rng(RNG_SEED)
    sent = sent.sort_values(["review_id", "sent_num"])
    label_arr = sent[MAIN_LABELS_USED].to_numpy(dtype=np.int8)
    review_ids = sent["review_id"].to_numpy()
    # integer code per sentence for its full label configuration (A2)
    weights = (1 << np.arange(len(MAIN_LABELS_USED))).astype(np.int64)
    config_all = (label_arr.astype(np.int64) * weights).sum(axis=1)

    review_rows = []
    pair_inter_rows = []
    pair_zone_rows = []

    for rid, idx in pd.Series(np.arange(len(sent))).groupby(review_ids, sort=False).indices.items():
        M = label_arr[idx]
        n = M.shape[0]
        if n < 3:
            continue
        k = M.sum(axis=0)
        cfg = config_all[idx]
        m = len(np.unique(cfg))
        perm = np.argsort(rng.random((N_PERM, n)), axis=1)

        # --- A2: configuration boundaries ---
        obs_b = boundary_score(cfg, m, n)
        null_b = boundary_score(cfg[perm], m, n)
        a2_defined = np.isfinite(obs_b)
        row = {"review_id": rid, "n_sentences": n, "n_configurations": m}
        if a2_defined:
            row.update({
                "boundary_score": round(float(obs_b), 4),
                "boundary_null": round(float(np.nanmean(null_b)), 4),
                "boundary_excess": round(float(obs_b - np.nanmean(null_b)), 4),
            })
        else:
            # m == n: every sentence has a distinct configuration, so the number of
            # boundaries is forced to n-1 and the score carries no information.
            row.update({"boundary_score": np.nan, "boundary_null": np.nan, "boundary_excess": np.nan})

        # --- A3 + B2 over label pairs ---
        seps_obs, seps_null = [], []
        for (ia, ib), (na, nb) in zip(PAIRS, PAIR_NAMES):
            if k[ia] < 2 or k[ib] < 2:
                continue
            a, b = M[:, ia].astype(bool), M[:, ib].astype(bool)

            # B2: union clumpiness and zone affinity
            union = (a | b).astype(np.int8)
            k_u = int(union.sum())
            kab = k[[ia, ib]].astype(float)
            obs_uc = float(clumpiness(count_runs(union[:, None]), np.array([k_u]))[0])
            null_uc = clumpiness(count_runs(union[perm][:, :, None]),
                                 np.array([k_u]))[:, 0]
            ind = clumpiness(count_runs(M[:, [ia, ib]]), kab)
            ind_weighted = float(np.average(ind, weights=kab))
            # zone_affinity needs its OWN null too: merging any two labels raises
            # clumpiness mechanically (bigger k, fewer possible runs), so a positive
            # raw affinity means nothing on its own. The null is the same quantity
            # computed on permuted sentence order -- per A1b's standing convention
            # that every count-based index is reported against its own baseline.
            null_ind = clumpiness(count_runs(M[:, [ia, ib]][perm]), kab)   # (B, 2)
            null_ind_weighted = np.average(null_ind, weights=kab, axis=1)  # (B,)
            null_affinity = null_uc - null_ind_weighted
            obs_affinity = obs_uc - ind_weighted
            pair_zone_rows.append({
                "review_id": rid, "label_a": na, "label_b": nb,
                "k_a": int(k[ia]), "k_b": int(k[ib]), "k_union": k_u,
                "union_clumpiness": round(obs_uc, 4),
                "union_null": round(float(null_uc.mean()), 4),
                "union_excess": round(obs_uc - float(null_uc.mean()), 4),
                "individual_clumpiness": round(ind_weighted, 4),
                "zone_affinity": round(obs_affinity, 4),
                "zone_affinity_null": round(float(np.nanmean(null_affinity)), 4),
                "zone_affinity_excess": round(obs_affinity - float(np.nanmean(null_affinity)), 4),
            })

            # A3: interleaving over A-only / B-only sentences
            a_only, b_only = a & ~b, b & ~a
            sel = a_only | b_only
            n_sel = int(sel.sum())
            if n_sel < 2 or not a_only.any() or not b_only.any():
                continue
            markers = np.where(a_only, 1, np.where(b_only, -1, 0)).astype(np.int8)
            obs_sep = float(interleaving_separation(markers[sel]))
            mp = markers[perm]                             # (B, n)
            sub = mp[mp != 0].reshape(N_PERM, n_sel)       # count is invariant under permutation
            null_sep = interleaving_separation(sub)
            seps_obs.append(obs_sep)
            seps_null.append(float(null_sep.mean()))
            pair_inter_rows.append({
                "review_id": rid, "label_a": na, "label_b": nb,
                "n_a_only": int(a_only.sum()), "n_b_only": int(b_only.sum()),
                "separation": round(obs_sep, 4),
                "separation_null": round(float(null_sep.mean()), 4),
                "separation_excess": round(obs_sep - float(null_sep.mean()), 4),
            })

        if seps_obs:
            row.update({
                "n_pairs": len(seps_obs),
                "mean_separation": round(float(np.mean(seps_obs)), 4),
                "separation_null": round(float(np.mean(seps_null)), 4),
                "separation_excess": round(float(np.mean(seps_obs) - np.mean(seps_null)), 4),
            })
        else:
            row.update({"n_pairs": 0, "mean_separation": np.nan,
                        "separation_null": np.nan, "separation_excess": np.nan})
        review_rows.append(row)

    return (pd.DataFrame(review_rows), pd.DataFrame(pair_inter_rows), pd.DataFrame(pair_zone_rows))


def aggregate_pairs(df, value_col, null_col):
    """Per label pair across reviews: mean excess + a paired t-test of
    observed vs. its own permutation null, BH-corrected over the pairs."""
    rows = []
    for (a, b), sub in df.groupby(["label_a", "label_b"]):
        if len(sub) < 10:
            continue
        diff = sub[value_col] - sub[null_col]
        t, p = stats.ttest_1samp(diff, 0.0)
        rows.append({
            "label_a": a, "label_b": b, "n_reviews": len(sub),
            f"mean_{value_col}": round(float(sub[value_col].mean()), 4),
            f"mean_{null_col}": round(float(sub[null_col].mean()), 4),
            "mean_excess": round(float(diff.mean()), 4),
            "t_stat": round(float(t), 2), "p_value": p,
        })
    out = pd.DataFrame(rows)
    out["q_bh"] = benjamini_hochberg(out["p_value"].to_numpy())
    return out.sort_values("mean_excess", ascending=False)


def main():
    sent = load_sentences()
    print(f"{SOURCE}: {len(sent)} sentences, {sent['review_id'].nunique()} reviews, "
          f"{len(MAIN_LABELS_USED)} main labels, {len(PAIRS)} pairs")

    review, pair_inter, pair_zone = analyse(sent)

    print("\n=== A2: configuration-boundary score (whole-review) ===")
    ok = review["boundary_excess"].notna()
    print(f"{int(ok.sum())} reviews scored; {int((~ok).sum())} undefined (m == n: every sentence a "
          f"distinct configuration, so boundaries are forced)")
    print(f"mean boundary score {review.loc[ok,'boundary_score'].mean():.4f} "
          f"vs. null {review.loc[ok,'boundary_null'].mean():.4f} "
          f"(excess {review.loc[ok,'boundary_excess'].mean():+.4f})")

    a1 = pd.read_csv(RESULTS / "19_review_contiguity_precedence" / f"{SOURCE}_review_contiguity.csv")
    merged = review.merge(a1[["review_id", "excess_over_chance"]], on="review_id", how="inner")
    both = merged["boundary_excess"].notna() & merged["excess_over_chance"].notna()
    rho, p = stats.spearmanr(merged.loc[both, "boundary_excess"], merged.loc[both, "excess_over_chance"])
    print(f"correlation with A1's excess (n={int(both.sum())}): spearman rho={rho:+.3f} (p={p:.2g}) "
          f"-- related but not redundant, as expected")

    print("\n=== A3: pairwise interleaving, aggregated to the review ===")
    ok3 = review["separation_excess"].notna()
    print(f"{int(ok3.sum())} reviews with >=1 eligible pair")
    print(f"mean separation {review.loc[ok3,'mean_separation'].mean():.4f} "
          f"vs. null {review.loc[ok3,'separation_null'].mean():.4f} "
          f"(excess {review.loc[ok3,'separation_excess'].mean():+.4f})")
    review.to_csv(RESULTS / "20_contiguity_zones" / f"{SOURCE}_review_boundary_interleaving.csv", index=False)

    print("\n=== A3 by label pair: which pairs stay separated vs. interleave? ===")
    inter_agg = aggregate_pairs(pair_inter, "separation", "separation_null")
    inter_agg.to_csv(RESULTS / "20_contiguity_zones" / f"{SOURCE}_pair_interleaving.csv", index=False)
    print(inter_agg[["label_a", "label_b", "n_reviews", "mean_separation",
                      "mean_separation_null", "mean_excess", "q_bh"]].to_string(index=False))

    print("\n=== B2: zone affinity (does merging the pair ADD contiguity, beyond chance?) ===")
    zone_agg = aggregate_pairs(pair_zone, "zone_affinity", "zone_affinity_null")
    extra = pair_zone.groupby(["label_a", "label_b"]).agg(
        mean_union_clumpiness=("union_clumpiness", "mean"),
        mean_individual=("individual_clumpiness", "mean")).reset_index()
    zone_agg = zone_agg.merge(extra, on=["label_a", "label_b"]).sort_values(
        "mean_excess", ascending=False)
    zone_agg.to_csv(RESULTS / "20_contiguity_zones" / f"{SOURCE}_pair_zone_affinity.csv", index=False)
    print(zone_agg[["label_a", "label_b", "n_reviews", "mean_union_clumpiness",
                     "mean_individual", "mean_zone_affinity", "mean_zone_affinity_null",
                     "mean_excess", "q_bh"]].to_string(index=False))

    print(f"\nwrote results to {RESULTS} ({SOURCE}_review_boundary_interleaving.csv, "
          f"{SOURCE}_pair_interleaving.csv, {SOURCE}_pair_zone_affinity.csv)")


if __name__ == "__main__":
    main()

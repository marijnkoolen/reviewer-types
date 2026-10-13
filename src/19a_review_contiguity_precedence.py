"""Contiguity follow-ups, parts A1 and B1 (ANALYSIS_PLAN.md item 13), gold
first per the plan's own build order -- validate the measures at gold scale
before doing the silver vectorization work.

§3/§4's contiguity analysis is per-LABEL, aggregated across reviews ("does
category X form one block, corpus-wide?"). These two measures change the
unit:

A1. REVIEW-level contiguity: is an individual review block-structured, or
    does it alternate between categories? Per review, the k-weighted mean of
    its own per-label clumpiness, over labels with k>=2 only (a label
    appearing once is trivially one run -- no information about
    organization, and including it would inflate every score toward 1).
    Restricted to reviews with >=2 such labels, since "does this review
    alternate BETWEEN categories" is undefined with only one.

B1. Within-review PRECEDENCE: among reviews containing both A and B, how
    often does A's mean position precede B's? §4 reports each label's
    MARGINAL mean position, which cannot answer a pairwise question -- two
    labels can have similar marginal means while reliably ordering one way
    within any single review that has both. Precedence is paired by
    construction, so it controls for review-specific composition.

Main labels only (7, excl. 'None'), per the plan: sub-labels are nested in
their parent's sentences, so including both double-counts the same block
structure. Unlike §3's per-(n,k) null, A1's null permutes SENTENCE ORDER
within the review -- one joint shuffle for all labels at once, preserving
every label's k AND the co-occurrence structure between labels (31.7% of
gold sentences carry >=2 main labels, so labels are far from independent;
combining independent per-label nulls would get the variance wrong).

B1 uses k>=1, not k>=2: a single sentence still has a well-defined position,
so precedence is meaningful where contiguity would not be.

Usage: python src/19a_review_contiguity_precedence.py [gold|silver]  (default: gold)
Writes results/{source}_review_contiguity.csv,
  results/{source}_pairwise_precedence.csv
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

MIN_ELIGIBLE_LABELS = 2   # need >=2 multi-sentence labels for "alternates between categories"
N_PERM = 1000
RNG_SEED = 0


def load_sentences():
    if SOURCE == "gold":
        return pd.read_csv(RESULTS / "00_data_prep/gold_sentence_sequences.csv")
    return pd.read_parquet(RESULTS / "00_data_prep/silver_sentence_sequences.parquet")


def clumpiness_from_runs(runs, k):
    """1 - (runs-1)/(k-1), the same index §3 defines. k==1 is trivially one
    run (clumpiness 1.0) and is filtered out by the caller, not scored here."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(k > 1, 1.0 - (runs - 1) / np.maximum(k - 1, 1), np.nan)


def count_runs_axis(M):
    """Runs of 1s down axis -2 of a (..., n, L) binary array -> (..., L)."""
    first = M[..., 0, :]
    trans = ((M[..., 1:, :] == 1) & (M[..., :-1, :] == 0)).sum(axis=-2)
    return first + trans


def review_contiguity(sent):
    """A1: per-review k-weighted mean clumpiness + joint sentence-order
    permutation null."""
    rng = np.random.default_rng(RNG_SEED)
    sent = sent.sort_values(["review_id", "sent_num"])
    label_arr = sent[MAIN_LABELS_USED].to_numpy(dtype=np.int8)
    review_ids = sent["review_id"].to_numpy()

    rows = []
    n_ineligible = 0
    for rid, idx in pd.Series(np.arange(len(sent))).groupby(review_ids, sort=False).indices.items():
        M = label_arr[idx]                      # (n, L)
        n = M.shape[0]
        k = M.sum(axis=0)                       # (L,)
        eligible = k >= 2
        if eligible.sum() < MIN_ELIGIBLE_LABELS or n < 3:
            n_ineligible += 1
            continue

        Me = M[:, eligible]
        ke = k[eligible].astype(float)
        obs_runs = count_runs_axis(Me)
        obs_clump = clumpiness_from_runs(obs_runs, ke)
        observed = float(np.average(obs_clump, weights=ke))

        # joint null: shuffle SENTENCE ORDER (one permutation applies to all
        # labels at once), preserving each k and all co-occurrence structure
        perm = np.argsort(rng.random((N_PERM, n)), axis=1)       # (B, n)
        Mp = Me[perm]                                            # (B, n, L_e)
        null_runs = count_runs_axis(Mp)                          # (B, L_e)
        null_clump = clumpiness_from_runs(null_runs, ke)
        null_scores = (null_clump * ke).sum(axis=1) / ke.sum()

        null_mean, null_std = null_scores.mean(), null_scores.std()
        # A degenerate null (std==0) means every permutation of this review scores
        # identically -- only happens for very short reviews, where the observed
        # score carries no information about organization at all. Flagged, not
        # silently NaN'd, and excluded from z-based summaries by the caller.
        degenerate = null_std == 0
        z = np.nan if degenerate else (observed - null_mean) / null_std
        p = (1 + np.sum(null_scores >= observed)) / (N_PERM + 1)

        rows.append({
            "review_id": rid, "n_sentences": n,
            "n_eligible_labels": int(eligible.sum()),
            "total_eligible_sentences": int(ke.sum()),
            "review_clumpiness": round(observed, 4),
            "null_mean": round(float(null_mean), 4),
            # A1b (chat, 2026-08-31): the headline measure. Raw clumpiness is NOT
            # comparable across review lengths -- the chance baseline itself falls
            # steeply with n (a 2-sentence category is adjacent by luck far more
            # often in a 5-sentence review than a 30-sentence one), so a flat raw
            # score across lengths actually hides a rising excess. Reporting
            # observed-minus-own-null makes reviews of different lengths comparable.
            "excess_over_chance": round(observed - float(null_mean), 4),
            "z_score": round(float(z), 3) if not degenerate else np.nan,
            "degenerate_null": bool(degenerate),
            "p_value": p,
        })

    out = pd.DataFrame(rows)
    out["q_bh"] = benjamini_hochberg(out["p_value"].to_numpy())
    return out, n_ineligible


def length_stratified(contig):
    """A1b: the same numbers split by review-length quartile. The point is the
    contrast between the raw column (near-flat) and the excess column (rising)
    -- see `excess_over_chance`'s note in review_contiguity()."""
    d = contig.copy()
    d["length_quartile"] = pd.qcut(d["n_sentences"], 4,
                                    labels=["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"])
    out = d.groupby("length_quartile", observed=True).agg(
        n_reviews=("review_id", "size"),
        median_sentences=("n_sentences", "median"),
        mean_raw_clumpiness=("review_clumpiness", "mean"),
        mean_chance_baseline=("null_mean", "mean"),
        mean_excess_over_chance=("excess_over_chance", "mean"),
        mean_z=("z_score", "mean"),
        pct_significant=("q_bh", lambda s: 100 * (s < 0.05).mean()),
    ).round(3).reset_index()
    return out


def validate_against_detail(sent):
    """Cross-check A1's per-label clumpiness against the already-published
    per-review x label table (§3) -- confirms this reproduces the established
    measure rather than a subtly different one. Gold only (silver's script
    doesn't persist per-review detail yet -- see item 13's plan note)."""
    path = RESULTS / "03_contiguity/gold_contiguity_detail.csv"
    if SOURCE != "gold" or not path.exists():
        return None
    detail = pd.read_csv(path)
    detail = detail[detail["label"].isin(MAIN_LABELS_USED) & (detail["k"] >= 2)]

    sent = sent.sort_values(["review_id", "sent_num"])
    label_arr = sent[MAIN_LABELS_USED].to_numpy(dtype=np.int8)
    review_ids = sent["review_id"].to_numpy()
    recomputed = []
    for rid, idx in pd.Series(np.arange(len(sent))).groupby(review_ids, sort=False).indices.items():
        M = label_arr[idx]
        k = M.sum(axis=0)
        runs = count_runs_axis(M)
        for j, label in enumerate(MAIN_LABELS_USED):
            if k[j] >= 2:
                recomputed.append({"review_id": rid, "label": label,
                                    "clumpiness_recomputed": clumpiness_from_runs(runs[j], float(k[j]))})
    rec = pd.DataFrame(recomputed)
    merged = detail.merge(rec, on=["review_id", "label"])
    merged["diff"] = (merged["clumpiness"] - merged["clumpiness_recomputed"]).abs()
    return merged


def pairwise_precedence(sent):
    """B1: among reviews containing both A and B, how often does A's mean
    position precede B's? Paired within review, so review-specific
    composition cancels out."""
    present = sent.groupby("review_id")[MAIN_LABELS_USED].sum()
    pos = {}
    for label in MAIN_LABELS_USED:
        active = sent[sent[label] == 1]
        pos[label] = active.groupby("review_id")["rel_position"].mean()
    pos = pd.DataFrame(pos)

    rows = []
    for a, b in combinations(MAIN_LABELS_USED, 2):
        both = (present[a] >= 1) & (present[b] >= 1)
        sub = pos.loc[both[both].index, [a, b]].dropna()
        if len(sub) < 10:
            continue
        diff = sub[b] - sub[a]              # >0 means A precedes B
        n_tie = int((diff == 0).sum())
        nz = diff[diff != 0]
        n_before = int((nz > 0).sum())
        n_total = len(nz)
        if n_total < 10:
            continue
        bt = stats.binomtest(n_before, n_total, 0.5)
        rows.append({
            "label_a": a, "label_b": b,
            "n_reviews_both": int(both.sum()), "n_tested": n_total, "n_ties": n_tie,
            "p_a_before_b": round(n_before / n_total, 4),
            "mean_position_gap": round(float(nz.mean()), 4),
            "p_value": bt.pvalue,
        })

    out = pd.DataFrame(rows)
    out["q_bh"] = benjamini_hochberg(out["p_value"].to_numpy())
    out["direction"] = np.where(out["p_a_before_b"] > 0.5, out["label_a"] + " before " + out["label_b"],
                                 out["label_b"] + " before " + out["label_a"])
    out["consistency"] = np.maximum(out["p_a_before_b"], 1 - out["p_a_before_b"]).round(4)
    return out.sort_values("consistency", ascending=False)


def main():
    sent = load_sentences()
    print(f"{SOURCE}: {len(sent)} sentences, {sent['review_id'].nunique()} reviews, "
          f"{len(MAIN_LABELS_USED)} main labels")

    print("\n=== A1: review-level contiguity ===")
    contig, n_ineligible = review_contiguity(sent)
    contig.to_csv(RESULTS / "19_review_contiguity_precedence" / f"{SOURCE}_review_contiguity.csv", index=False)
    print(f"{len(contig)} eligible reviews (>= {MIN_ELIGIBLE_LABELS} labels with k>=2, n>=3); "
          f"{n_ineligible} ineligible")
    print(f"mean review clumpiness: {contig['review_clumpiness'].mean():.4f} "
          f"(null mean {contig['null_mean'].mean():.4f}, "
          f"excess {contig['excess_over_chance'].mean():.4f})")
    print(f"mean z-score: {contig['z_score'].mean():.3f} "
          f"({int(contig['degenerate_null'].sum())} reviews excluded: degenerate null)")
    print(f"reviews significantly MORE blocked than chance (q<0.05): "
          f"{((contig['q_bh'] < 0.05) & (contig['z_score'] > 0)).sum()}/{len(contig)}")
    print(f"reviews significantly LESS blocked (more alternating) than chance (q<0.05): "
          f"{((contig['q_bh'] < 0.05) & (contig['z_score'] < 0)).sum()}/{len(contig)}")

    # A1b: raw clumpiness is not comparable across lengths -- report the
    # length-stratified breakdown, which is what makes the pattern legible.
    strat = length_stratified(contig)
    strat.to_csv(RESULTS / "22_review_contiguity_scaled" / f"{SOURCE}_review_contiguity_by_length.csv", index=False)
    print("\n--- A1b: by review-length quartile (the chance baseline itself moves with length) ---")
    print(strat.to_string(index=False))
    rho, p = stats.spearmanr(contig["n_sentences"], contig["excess_over_chance"])
    rho_raw, p_raw = stats.spearmanr(contig["n_sentences"], contig["review_clumpiness"])
    rho_null, p_null = stats.spearmanr(contig["n_sentences"], contig["null_mean"])
    print(f"\nspearman(n_sentences, raw clumpiness)   = {rho_raw:+.3f} (p={p_raw:.2g})  <- looks flat")
    print(f"spearman(n_sentences, chance baseline)  = {rho_null:+.3f} (p={p_null:.2g})  <- falls steeply")
    print(f"spearman(n_sentences, EXCESS)           = {rho:+.3f} (p={p:.2g})  <- the real pattern")

    val = validate_against_detail(sent)
    if val is not None:
        print(f"\nvalidation vs. published §3 per-review detail: {len(val)} review x label rows, "
              f"max |diff| = {val['diff'].max():.6f}")

    print("\n=== B1: within-review pairwise precedence ===")
    prec = pairwise_precedence(sent)
    prec.to_csv(RESULTS / "19_review_contiguity_precedence" / f"{SOURCE}_pairwise_precedence.csv", index=False)
    show = prec[["label_a", "label_b", "n_tested", "p_a_before_b", "mean_position_gap",
                  "consistency", "direction", "q_bh"]]
    print(show.to_string(index=False))
    print(f"\npairs with a significant ordering (q<0.05): {(prec['q_bh'] < 0.05).sum()}/{len(prec)}")

    print(f"\nwrote results to {RESULTS} ({SOURCE}_review_contiguity.csv, {SOURCE}_pairwise_precedence.csv)")


if __name__ == "__main__":
    main()

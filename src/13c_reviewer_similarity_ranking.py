"""Prolific-reviewer tendency analysis, part (b): per-reviewer similarity
ranking (ANALYSIS_PLAN.md item 6). Natural follow-on to item 5, which built
the silver-scale reviewer-consistency baseline this reuses, and a different
operationalization of item 3's question: item 3 asked "how often does a
reviewer land in their single most-common typology cluster" (a categorical,
modal-share measure); this asks "how similar are a reviewer's own reviews to
each other" directly, via mean pairwise cosine similarity on the same
continuous feature vectors item 5's aggregate reviewer test used -- for both
the compositional vectors (prop__label) AND, new here, per-review sequence
("positional") vectors, the same onset (first-occurrence) position-per-label
features 05c_typology_clustering.py's sequence clustering was fit on and
11_sequence_cluster_profile.py already knows how to reconstruct.

Method: mean pairwise cosine similarity per reviewer (block-diagonal,
capped at 100 reviews/reviewer like item 5), z-tested against item 5's
corpus-wide random-pairs baseline (composition: reused directly from
results/07_book_author_similarity/silver_reviewer_similarity_test.csv; sequence: computed fresh here,
no prior baseline existed for it).

First attempt at the per-reviewer SE used std(pairwise similarities)/sqrt(
n_pairs) -- cheap, and what the plan's "cheaper than 5,782 individual
permutation tests" language suggested. The plan also explicitly called for
spot-checking that approximation against real permutation before trusting
it broadly, and the spot-check caught a real problem: pairwise similarities
within one reviewer's reviews aren't independent (each review appears in
n-1 pairs, not one), so std/sqrt(n_pairs) badly understates the true SE.
Concretely, for reviewers sitting right at the naive method's q=0.05 line,
a real permutation test put their p-values at 0.25-0.4, not ~0.05 -- the
naive version would have overstated significance for a large share of the
5,782 reviewers, not just at the margin. Fixed, not just caveated: the SE
below is a within-reviewer cluster bootstrap instead (resample that
reviewer's own n reviews with replacement, recompute mean similarity over
pairs whose two members came from *different* original reviews each time,
repeat 500x) -- the correct unit to resample when pairs share members,
same principle `cluster_bootstrap_weighted_mean` already uses at the
aggregate level in similarity.py, applied here per-reviewer instead of
per-group-of-groups. Both extreme and borderline cases are spot-checked
against real permutation below to confirm the fix, not just the original
naive version.

Same prolific population as items 3/4 (>=10 reviews, excl. platform_1) --
loaded directly from results/12_prolific_reviewer_tendency/silver_prolific_reviewer_tendency.csv rather
than re-filtered, so it's guaranteed to be the identical 5,782 reviewers.
Same BH-FDR treatment as item 4 (separate families: compositional,
sequential), not item 3's mistake of no correction at all.

Usage: python src/13c_reviewer_similarity_ranking.py
Requires results/12_prolific_reviewer_tendency/silver_prolific_reviewer_tendency.csv (12c_..._tendency.py),
  results/07_book_author_similarity/silver_reviewer_similarity_test.csv (08c_..._similarity.py),
  results/04_order/silver_order_onset.csv (04c_order.py)
Writes results/13_reviewer_similarity_ranking/silver_reviewer_similarity_ranking.csv,
  results/13_reviewer_similarity_ranking/silver_reviewer_similarity_ranking_permutation_check.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS, WEAK_MODEL_LABELS  # noqa: E402
from similarity import random_pairs_baseline, cosine_distance_matrix  # noqa: E402
from stats_utils import benjamini_hochberg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
FEATURE_LABELS = [l for l in MAIN_LABELS_USED + SUB_LABELS if l not in WEAK_MODEL_LABELS]
CAP = 100
N_BASELINE_SAMPLE = 1_000_000
N_BOOT_SE = 500
N_PERM = 2000
N_SPOTCHECK = 3
RNG_SEED = 0


def load_prolific_ids():
    tendency = pd.read_csv(RESULTS / "12_prolific_reviewer_tendency/silver_prolific_reviewer_tendency.csv")
    return set(tendency["user_id"]), tendency


def build_eligible_pool(review_features):
    """Same eligible pool item 5's reviewer test used (excl. platform_1,
    users with >=2 reviews) -- reconstructed here, not just the prolific
    (>=10) subset, so the baseline and permutation null below are computed
    on the same population item 5's reused composition baseline represents,
    not a narrower one that would make composition/sequence baselines and
    the permutation spot-check subtly incomparable to each other."""
    d = review_features[review_features["source_id"] != "platform_1"].reset_index(drop=True)
    counts = d["user_id"].value_counts()
    multi_ids = counts[counts >= 2].index
    return d[d["user_id"].isin(multi_ids)].reset_index(drop=True)


def build_composition_matrix(review_features):
    X = review_features[[f"prop__{l}" for l in FEATURE_LABELS]].to_numpy()
    nonzero = X.sum(axis=1) > 0
    return review_features.loc[nonzero].reset_index(drop=True), X[nonzero]


def build_sequence_matrix(review_features, sentence_df):
    """Per-review onset (first-occurrence) position of each main label's own
    sentences, corpus onset-marginal-imputed where a review lacks that label
    -- identical construction to 11_sequence_cluster_profile.py's
    compute_seqpos, just inlined here to avoid importing a script by
    numeric-prefixed filename."""
    onset_marginal = pd.read_csv(RESULTS / "04_order/silver_order_onset.csv").set_index("label")["first_mean_position"]
    review_ids = review_features["review_id"].to_numpy()
    out = np.empty((len(review_ids), len(MAIN_LABELS_USED)))
    for j, label in enumerate(MAIN_LABELS_USED):
        sub = sentence_df[sentence_df[label] == 1].groupby("review_id")["rel_position"].min()
        marginal = onset_marginal.get(label, 0.5)
        out[:, j] = pd.Series(review_ids).map(sub).fillna(marginal).to_numpy()
    return review_features, out


def reviewer_similarity_bootstrap(df, X, group_col, prolific_ids, min_size=10, cap=CAP,
                                   n_boot=N_BOOT_SE, seed=RNG_SEED):
    """Per prolific reviewer: mean pairwise cosine similarity (exact under
    the cap, an unbiased random-subsample estimate over it) plus a within-
    reviewer cluster bootstrap SE -- resample that reviewer's own n reviews
    WITH replacement, recompute the mean similarity over pairs whose two
    slots landed on different original reviews (a slot re-drawing the same
    review as another slot contributes no real pairwise information and is
    excluded, same as a review can't be "similar to itself"), repeat
    n_boot times, take the std of that distribution. Vectorized across all
    n_boot resamples via fancy indexing on the reviewer's own (<=cap x cap)
    similarity matrix, so cost stays trivial even for n_boot=500."""
    rng = np.random.default_rng(seed)
    group_col_values = df[group_col].to_numpy()
    rows = []
    for g, idx in pd.Series(np.arange(len(df))).groupby(group_col_values).indices.items():
        if g not in prolific_ids:
            continue
        n_true = len(idx)
        if n_true < min_size:
            continue
        if n_true > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        n = len(idx)
        Xg = X[idx]
        S = 1.0 - cosine_distance_matrix(Xg)
        iu = np.triu_indices(n, k=1)
        mean_similarity = S[iu].mean()

        boot_idx = rng.integers(0, n, size=(n_boot, n))
        S_b = S[boot_idx[:, :, None], boot_idx[:, None, :]]
        valid = boot_idx[:, :, None] != boot_idx[:, None, :]
        denom = valid.sum(axis=(1, 2))
        boot_means = (S_b * valid).sum(axis=(1, 2)) / denom
        bootstrap_se = boot_means.std()

        rows.append({group_col: g, "n": n, "n_pairs": len(iu[0]),
                      "mean_similarity": mean_similarity, "bootstrap_se": bootstrap_se})
    return pd.DataFrame(rows)


def zscore_test(per_group, baseline_mean, baseline_se):
    d = per_group.copy()
    se_combined = np.sqrt(d["bootstrap_se"] ** 2 + baseline_se ** 2)
    d["z"] = np.where(se_combined > 0, (d["mean_similarity"] - baseline_mean) / se_combined, np.nan)
    d["p_value"] = stats.norm.sf(d["z"])  # one-sided: "unusually high" only
    d["q_bh"] = benjamini_hochberg(d["p_value"].to_numpy())
    return d


def permutation_spot_check(df, X, user_id, n, seed):
    """Real permutation test for one reviewer: draw N_PERM random n-sized
    samples (without replacement) from the full eligible pool, compute mean
    pairwise similarity for each, and compare the reviewer's actual mean
    against that empirical null -- the ground truth the z-score approximates."""
    rng = np.random.default_rng(seed)
    n_total = len(X)
    null_means = np.empty(N_PERM)
    for i in range(N_PERM):
        idx = rng.choice(n_total, size=n, replace=False)
        D = cosine_distance_matrix(X[idx])
        sim = 1.0 - D
        iu = np.triu_indices(n, k=1)
        null_means[i] = sim[iu].mean()

    own_idx = np.where(df["user_id"].to_numpy() == user_id)[0]
    D_own = cosine_distance_matrix(X[own_idx])
    sim_own = 1.0 - D_own
    iu = np.triu_indices(len(own_idx), k=1)
    observed = sim_own[iu].mean()

    perm_p = (1 + np.sum(null_means >= observed)) / (N_PERM + 1)
    return observed, null_means.mean(), null_means.std(), perm_p


def spot_check(label, per_group_z, df, X, mode="extreme"):
    """mode='extreme': the most extreme z-scores, where the z-test and a real
    permutation should trivially agree (both near the floor) -- confirms no
    gross false positives, but not much else. mode='borderline': closest to
    q=0.05, which is where the per-reviewer SE's independence approximation
    (flagged in the module docstring) would actually matter for a real
    significant/not-significant decision -- the sharper test of the two."""
    if mode == "extreme":
        top = per_group_z.sort_values("z", ascending=False).head(N_SPOTCHECK)
    else:
        top = per_group_z.assign(_dist=(per_group_z["q_bh"] - 0.05).abs()).sort_values("_dist").head(N_SPOTCHECK)
    rows = []
    for _, row in top.iterrows():
        observed, null_mean, null_std, perm_p = permutation_spot_check(df, X, row["user_id"], int(row["n"]), RNG_SEED)
        rows.append({
            "family": label, "mode": mode, "user_id": row["user_id"], "n": int(row["n"]),
            "z_score_p_value": row["p_value"], "z_score_q_bh": row["q_bh"],
            "permutation_observed_mean": observed, "permutation_null_mean": null_mean,
            "permutation_p_value": perm_p,
        })
    return pd.DataFrame(rows)


def main():
    prolific_ids, tendency = load_prolific_ids()
    print(f"{len(prolific_ids)} prolific reviewers (loaded from silver_prolific_reviewer_tendency.csv)")

    review_features = pd.read_parquet(RESULTS / "00_data_prep/silver_review_features.parquet")
    pool = build_eligible_pool(review_features)
    sentence_df = pd.read_parquet(RESULTS / "00_data_prep/silver_sentence_sequences.parquet")
    print(f"eligible pool (excl. platform_1, >=2 reviews/user): {len(pool)} reviews, "
          f"{pool['user_id'].nunique()} users")

    # --- composition ---
    comp_df, X_comp = build_composition_matrix(pool)
    comp_pg = reviewer_similarity_bootstrap(comp_df, X_comp, "user_id", prolific_ids, min_size=10, cap=CAP)
    comp_baseline = pd.read_csv(RESULTS / "07_book_author_similarity/silver_reviewer_similarity_test.csv").iloc[0]
    comp_z = zscore_test(comp_pg, comp_baseline["baseline_mean_similarity"], comp_baseline["baseline_se"])
    comp_z = comp_z.rename(columns={c: f"comp_{c}" for c in comp_z.columns if c != "user_id"})
    print(f"\ncomposition: {len(comp_z)} prolific reviewers matched (after all-zero-vector drop from pool), "
          f"baseline mean={comp_baseline['baseline_mean_similarity']:.4f} (reused from item 5)")

    # --- sequence ---
    seq_df, X_seq = build_sequence_matrix(pool, sentence_df)
    seq_pg = reviewer_similarity_bootstrap(seq_df, X_seq, "user_id", prolific_ids, min_size=10, cap=CAP)
    seq_baseline = random_pairs_baseline(X_seq, N_BASELINE_SAMPLE, RNG_SEED)
    seq_z = zscore_test(seq_pg, seq_baseline["mean"], seq_baseline["se"])
    seq_z = seq_z.rename(columns={c: f"seq_{c}" for c in seq_z.columns if c != "user_id"})
    print(f"sequence: {len(seq_z)} prolific reviewers matched, baseline mean={seq_baseline['mean']:.4f} "
          f"(fresh, no prior baseline existed for the sequence feature space)")

    out = comp_z.merge(seq_z, on="user_id")
    out = out.merge(tendency[["user_id", "cluster_composition_share", "cluster_composition_q_bh",
                               "cluster_sequence_share", "cluster_sequence_q_bh"]], on="user_id", how="left")
    out.to_csv(RESULTS / "13_reviewer_similarity_ranking/silver_reviewer_similarity_ranking.csv", index=False)

    n_sig_comp = (out["comp_q_bh"] < 0.05).sum()
    n_sig_seq = (out["seq_q_bh"] < 0.05).sum()
    n_sig_both = ((out["comp_q_bh"] < 0.05) & (out["seq_q_bh"] < 0.05)).sum()
    print(f"\nsignificantly (q<0.05) high compositional similarity: {n_sig_comp}/{len(out)}")
    print(f"significantly (q<0.05) high sequential similarity: {n_sig_seq}/{len(out)}")
    print(f"significant on both: {n_sig_both}/{len(out)}")

    corr_cont = out[["comp_mean_similarity", "seq_mean_similarity"]].corr().iloc[0, 1]
    print(f"\ncorrelation between compositional and sequential mean-similarity (continuous method): {corr_cont:.3f}")

    tri_corr = out[["comp_mean_similarity", "cluster_composition_share"]].dropna().corr().iloc[0, 1]
    print(f"cross-method triangulation: corr(this item's comp. similarity, item 3's comp. modal share) = {tri_corr:.3f}")

    comp_z_r = comp_z.rename(columns={"comp_z": "z", "comp_p_value": "p_value", "comp_q_bh": "q_bh", "comp_n": "n"})
    seq_z_r = seq_z.rename(columns={"seq_z": "z", "seq_p_value": "p_value", "seq_q_bh": "q_bh", "seq_n": "n"})

    print(f"\n--- permutation spot-check: top {N_SPOTCHECK} most extreme by z-score per family "
          f"(sanity check -- both methods should trivially agree near the floor) ---")
    check_comp_ext = spot_check("composition", comp_z_r, comp_df, X_comp, mode="extreme")
    check_seq_ext = spot_check("sequence", seq_z_r, seq_df, X_seq, mode="extreme")

    print(f"\n--- permutation spot-check: {N_SPOTCHECK} closest to q=0.05 per family "
          f"(sharper test -- this is where the SE approximation could actually flip a call) ---")
    check_comp_bord = spot_check("composition", comp_z_r, comp_df, X_comp, mode="borderline")
    check_seq_bord = spot_check("sequence", seq_z_r, seq_df, X_seq, mode="borderline")

    check = pd.concat([check_comp_ext, check_seq_ext, check_comp_bord, check_seq_bord], ignore_index=True)
    check.to_csv(RESULTS / "13_reviewer_similarity_ranking/silver_reviewer_similarity_ranking_permutation_check.csv", index=False)
    print(check.to_string(index=False))

    print(f"\nwrote results to {RESULTS} (silver_reviewer_similarity_ranking*.csv)")


if __name__ == "__main__":
    main()

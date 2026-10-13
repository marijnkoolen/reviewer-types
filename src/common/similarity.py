"""Shared machinery for book/author/reviewer similarity and variance-partitioning
(ANALYSIS_PLAN.md §8-9): cosine-distance review vectors, pairwise within-group
similarity permutation tests, and one-way PERMANOVA (McArdle & Anderson 2001)
via a distance matrix, since scikit-learn has no PERMANOVA and pulling in
scikit-bio for one routine isn't worth the dependency.

The functions above (`pairwise_group_similarity_test`, `oneway_permanova`) form
one dense N x N distance matrix -- fine at gold scale (a few hundred rows in
the relevant subset) but not at silver scale, where the "multi-review" subset
for books/authors/reviewers turns out to BE most of the corpus (silver's
largest single author has 54,889 reviews) -- a dense matrix over that many
rows is many GB before you've computed anything. The functions below
(`per_group_similarity` onward) answer the same question without ever forming
that matrix: within-group similarity only needs block-diagonal blocks (one
small matrix per group, size-capped so one pathological group can't dominate
compute), and the random-pairs baseline only needs a large random SAMPLE of
pairs, not an exhaustive enumeration of all ~10^11 of them.
"""
import numpy as np
import pandas as pd


def cosine_distance_matrix(X):
    """1 - cosine similarity, pairwise. Rows with all-zero vectors are invalid
    (cosine undefined) and must be excluded by the caller before this point."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero vectors present -- exclude all-zero rows first")
    Xn = X / norms
    sim = Xn @ Xn.T
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def pairwise_group_similarity_test(D, group_ids, pair_filter=None, n_perm=999, seed=0):
    """Mean cosine similarity of same-group pairs vs. a label-permutation null.

    group_ids: array of group labels, one per row of D (same order).
    pair_filter: optional boolean matrix (n x n), True where a pair is eligible
      to be counted at all (e.g. work_id differs, for the "same author, different
      book" version) -- evaluated on the TRUE grouping/identity, held fixed
      across permutations, so the permutation only shuffles the factor being
      tested, not the eligibility of a pair to be compared in the first place.
    """
    n = len(group_ids)
    sim = 1.0 - D
    rng = np.random.default_rng(seed)

    same_group = group_ids[:, None] == group_ids[None, :]
    iu = np.triu_indices(n, k=1)
    eligible = np.ones((n, n), dtype=bool) if pair_filter is None else pair_filter

    def stat(labels):
        sg = labels[:, None] == labels[None, :]
        mask = sg & eligible
        vals = sim[iu][mask[iu]]
        return vals.mean() if len(vals) else float("nan"), len(vals)

    observed, n_pairs = stat(group_ids)

    # random-pairs baseline: mean similarity over all eligible pairs regardless of group
    baseline_vals = sim[iu][eligible[iu]]
    baseline_mean = baseline_vals.mean()

    null_stats = np.empty(n_perm)
    for p in range(n_perm):
        perm_labels = rng.permutation(group_ids)
        null_stats[p], _ = stat(perm_labels)
    null_stats = null_stats[~np.isnan(null_stats)]

    p_value = (1 + np.sum(null_stats >= observed)) / (len(null_stats) + 1)
    return {
        "n_pairs": n_pairs,
        "observed_mean_similarity": observed,
        "baseline_mean_similarity": baseline_mean,
        "null_mean": null_stats.mean(),
        "null_std": null_stats.std(),
        "z_score": (observed - null_stats.mean()) / null_stats.std() if null_stats.std() > 0 else float("nan"),
        "p_value": p_value,
    }


def oneway_permanova(D, group_ids, n_perm=999, seed=0):
    """One-way PERMANOVA: R^2 and pseudo-F for how much a single grouping
    variable explains of the total variance in a distance matrix, with a
    permutation p-value. Singleton groups are valid (contribute 0 to
    within-group SS) and don't need to be filtered out beforehand."""
    n = D.shape[0]
    D2 = D ** 2
    ss_total = D2.sum() / (2 * n)

    groups, group_idx = np.unique(group_ids, return_inverse=True)
    k = len(groups)
    Z = np.zeros((n, k))
    Z[np.arange(n), group_idx] = 1.0
    sizes = Z.sum(axis=0)

    def ss_within(labels):
        _, idx = np.unique(labels, return_inverse=True)
        k_ = idx.max() + 1
        Zp = np.zeros((n, k_))
        Zp[np.arange(n), idx] = 1.0
        sizes_p = Zp.sum(axis=0)
        A = D2 @ Zp
        diagvals = (Zp * A).sum(axis=0)
        return 0.5 * np.sum(diagvals / sizes_p)

    ss_w = ss_within(group_ids)
    ss_a = ss_total - ss_w
    r2 = ss_a / ss_total
    df_a, df_w = k - 1, n - k
    pseudo_f = (ss_a / df_a) / (ss_w / df_w) if df_w > 0 else float("nan")

    rng = np.random.default_rng(seed)
    null_f = np.empty(n_perm)
    for p in range(n_perm):
        perm_labels = rng.permutation(group_ids)
        ss_w_p = ss_within(perm_labels)
        ss_a_p = ss_total - ss_w_p
        null_f[p] = (ss_a_p / df_a) / (ss_w_p / df_w) if df_w > 0 else float("nan")

    p_value = (1 + np.sum(null_f >= pseudo_f)) / (n_perm + 1)
    return {
        "n": n, "k_groups": k, "r2": r2, "pseudo_f": pseudo_f,
        "df_between": df_a, "df_within": df_w, "p_value": p_value,
    }


def per_group_similarity(df, X, group_col, min_size=2, cap=100, pair_filter_col=None, seed=0):
    """Mean within-group cosine similarity, one small block-diagonal matrix per
    group instead of one dense matrix over everything. Groups larger than `cap`
    are randomly subsampled down to `cap` members -- exact for groups at or
    under the cap, an unbiased (if noisier) estimate for larger ones, and
    bounds the cost of any single pathologically large group (a cap of 100
    keeps every group's own distance matrix under 100x100, trivial, regardless
    of how large the group truly is).

    pair_filter_col: optional column name; pairs sharing the same value on it
    are excluded from that group's own similarity (e.g. 'work_id', for the
    same-author-different-book test -- excludes same-book pairs from counting
    toward an author's "similarity to themselves").

    Returns a per-group DataFrame: group_col, n (post-cap), n_true (pre-cap),
    n_pairs, mean_similarity, std_similarity (population std of that group's
    own pairwise similarities -- for a per-group z-test against an external
    baseline, see `scalable_group_similarity_test` for the aggregate version
    and ANALYSIS_PLAN.md item 6 for the per-group one).
    """
    rng = np.random.default_rng(seed)
    group_col_values = df[group_col].to_numpy()
    filter_values = df[pair_filter_col].to_numpy() if pair_filter_col else None

    rows = []
    for g, idx in pd.Series(np.arange(len(df))).groupby(group_col_values).indices.items():
        n_true = len(idx)
        if n_true < min_size:
            continue
        if n_true > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        Xg = X[idx]
        D = cosine_distance_matrix(Xg)
        sim = 1.0 - D
        n = len(idx)
        iu = np.triu_indices(n, k=1)
        pair_sims = sim[iu]
        if pair_filter_col is not None:
            vals = filter_values[idx]
            eligible = (vals[:, None] != vals[None, :])[iu]
            pair_sims = pair_sims[eligible]
        if len(pair_sims) == 0:
            continue
        rows.append({group_col: g, "n": n, "n_true": n_true,
                      "n_pairs": len(pair_sims), "mean_similarity": pair_sims.mean(),
                      "std_similarity": pair_sims.std()})
    return pd.DataFrame(rows)


def cluster_bootstrap_weighted_mean(per_group_df, n_boot=2000, seed=0):
    """Bootstrap CI for the pair-weighted mean of `mean_similarity` across
    groups, resampling GROUPS with replacement (a cluster bootstrap -- the
    right unit to resample when observations within a group aren't
    independent of each other, which within-group similarity pairs never
    are). Cheap: each replicate is just a re-weighted average over the
    already-computed per-group means, no new distance computation."""
    rng = np.random.default_rng(seed)
    means = per_group_df["mean_similarity"].to_numpy()
    weights = per_group_df["n_pairs"].to_numpy()
    n_groups = len(per_group_df)
    estimate = np.average(means, weights=weights)

    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_groups, size=n_groups)
        boot[b] = np.average(means[idx], weights=weights[idx])
    return {
        "estimate": estimate, "boot_std": boot.std(),
        "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5),
        "n_groups": n_groups, "n_pairs_total": int(weights.sum()),
    }


def random_pairs_baseline(X, n_sample=1_000_000, seed=0, exclude_same_col=None, col_values=None):
    """Mean cosine similarity over a large random SAMPLE of pairs (not every
    pair -- infeasible at silver scale), as the random-pairs reference point.
    A million-pair sample gives a standard error small enough (~1e-4 scale for
    cosine similarities in [0,1]) to be a precise baseline without ever
    enumerating the ~10^11 true pairs in the full corpus."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / norms

    i = rng.integers(0, n, size=n_sample)
    j = rng.integers(0, n, size=n_sample)
    valid = i != j
    if exclude_same_col is not None:
        valid &= col_values[i] != col_values[j]
    i, j = i[valid], j[valid]

    sims = (Xn[i] * Xn[j]).sum(axis=1)
    sims = np.clip(sims, -1.0, 1.0)
    return {"mean": sims.mean(), "std": sims.std(), "n": len(sims), "se": sims.std() / np.sqrt(len(sims))}


def scalable_group_similarity_test(df, X, group_col, min_size=2, cap=100, pair_filter_col=None,
                                    n_boot=2000, n_baseline_sample=1_000_000, seed=0):
    """Silver-scale replacement for `pairwise_group_similarity_test`: same
    question (are same-group review pairs more similar than random pairs?),
    answered via block-diagonal within-group computation + a cluster
    bootstrap CI, compared against a sampled-pairs baseline via a z-test --
    instead of a dense N x N matrix and a full label-permutation, both
    infeasible once the "multi-review" subset is most of the corpus."""
    per_group = per_group_similarity(df, X, group_col, min_size, cap, pair_filter_col, seed)
    boot = cluster_bootstrap_weighted_mean(per_group, n_boot, seed)

    filter_vals = df[pair_filter_col].to_numpy() if pair_filter_col else None
    baseline = random_pairs_baseline(X, n_baseline_sample, seed, pair_filter_col, filter_vals)

    se_combined = np.sqrt(boot["boot_std"] ** 2 + baseline["se"] ** 2)
    z = (boot["estimate"] - baseline["mean"]) / se_combined if se_combined > 0 else float("nan")
    from scipy import stats as _stats
    p_value = 2 * _stats.norm.sf(abs(z))

    return {
        "n_groups": boot["n_groups"], "n_pairs": boot["n_pairs_total"],
        "observed_mean_similarity": boot["estimate"],
        "observed_ci_lo": boot["ci_lo"], "observed_ci_hi": boot["ci_hi"],
        "baseline_mean_similarity": baseline["mean"], "baseline_se": baseline["se"],
        "z_score": z, "p_value": p_value,
    }, per_group

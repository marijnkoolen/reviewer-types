"""P3: matched random-pairs baselines for the book / author / reviewer similarity
tests of Phase C section 7.

Section 7 compares within-group pairs (same book / same author different book /
same reviewer) against random pairs from the WHOLE corpus, so each gap also
contains whatever the group shares by construction: genre, platform, review
length. Here the baseline pairs are drawn from the same strata as the within-group
pairs (but from a different group), and the gap is re-estimated.

Matching scheme (documented, applied identically to all tests)
  * Each review gets a stratum cell (genre; platform; length quartile of
    num_sentences, cut at 3/6/14 sentences -> bins <=3, 4-6, 7-14, >14, which
    hold 27/~23/~26/~24% of reviews because of ties; the corpus has many short
    reviews with identical lengths, so exact quartiles do not exist).
  * A pair's stratum key is the UNORDERED pair of its two reviews' cells (so
    for length, "(Q1,Q3)" == "(Q3,Q1)"; for genre in the author / reviewer tests
    the key is the pair's genre combination).
  * Baseline pairs: for every stratum key occurring among the within-group
    pairs, draw pairs (one review from each cell, different group), n
    proportional to that key's within-group pair count (floor 30 draws/key,
    >= 1M draws in total per matched baseline), then average the per-key
    means with the within-group pair counts as weights. => the baseline has
    exactly the within-group pairs' stratum distribution.
  * Keys with <10 drawable different-group pairs are dropped, and the
    within-group pairs in them are dropped too (reported as "dropped").
  * Same vectors, same cosine, same cap (100/group) as section 7; compositional
    = prop__ vector (section 7), sequence = onset (first-occurrence) position per
    main label, corpus-onset-marginal imputed (as in 13c).
  * CIs: cluster bootstrap over groups (1000 reps); the matched baseline's
    group-dependent weights are recomputed in every replicate and per-key
    sampling noise (se) is added, so the gap CI reflects both.

Usage: venv/bin/python src/27c_similarity_matched_baselines.py
Writes results/27_similarity_matched_baselines/{matched_similarity_results.csv,
validation_vs_section7.csv, same_book_shared_reviewer.csv, tables.md}
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS, WEAK_MODEL_LABELS  # noqa: E402
from similarity import (within_group_pairs, pair_cosine, pair_keys, sample_keys,  # noqa: E402
                        random_pairs_baseline)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "27_similarity_matched_baselines"
OUT.mkdir(exist_ok=True)

MAIN_USED = [m for m in MAIN_LABELS if m != "None"]
FEATURE_LABELS = [l for l in MAIN_USED + SUB_LABELS if l not in WEAK_MODEL_LABELS]
CAP = 100
N_BASE = 1_000_000
MIN_PER_KEY = 30
N_BOOT = 1000
CHUNK = 100
SEED_PAIRS, SEED_SAMPLE, SEED_BOOT = 0, 1, 2
VECS = ["compositional", "sequence"]


def load():
    df = pd.read_parquet(RESULTS / "00_data_prep/silver_review_features.parquet")
    X = df[[f"prop__{l}" for l in FEATURE_LABELS]].to_numpy()
    nz = X.sum(axis=1) > 0
    print(f"dropping {(~nz).sum()} all-zero compositional rows")
    df, X = df.loc[nz].reset_index(drop=True), X[nz]
    sent = pd.read_parquet(RESULTS / "00_data_prep/silver_sentence_sequences.parquet")
    marg = pd.read_csv(RESULTS / "04_order/silver_order_onset.csv").set_index("label")["first_mean_position"]
    rid = pd.Series(df["review_id"].to_numpy())
    S = np.empty((len(df), len(MAIN_USED)))
    for j, l in enumerate(MAIN_USED):  # identical construction to 13c build_sequence_matrix
        sub = sent[sent[l] == 1].groupby("review_id")["rel_position"].min()
        S[:, j] = rid.map(sub).fillna(marg.get(l, 0.5)).to_numpy()
    df["lenq"] = np.digitize(df["num_sentences"].to_numpy(), [3.5, 6.5, 14.5])  # <=3,4-6,7-14,>14
    df["g_code"] = pd.factorize(df["nur_genre"])[0]
    df["p_code"] = pd.factorize(df["source_id"])[0]
    return df, X, S


def keysets_for(df, kind):
    g, p, q = df["g_code"].to_numpy(), df["p_code"].to_numpy(), df["lenq"].to_numpy()
    NG, NP = g.max() + 1, p.max() + 1
    z = np.zeros(len(df), dtype=np.int64)
    if kind == "book" or kind == "author":
        gp = g * NP + p
        return {"unmatched": z, "genre": g.astype(np.int64), "genre+platform": gp.astype(np.int64),
                "genre+platform+length": gp * 4 + q}
    pg = p * NG + g
    return {"unmatched": z, "platform": p.astype(np.int64), "platform+genre": pg.astype(np.int64),
            "platform+genre+length": pg * 4 + q}


def run_cell(pairs, sims, mask, cell, grp_codes, seed_boot=SEED_BOOT):
    """One (pair-subset, keyset) cell. Returns dict per vector."""
    g, i, j = pairs[0][mask], pairs[1][mask], pairs[2][mask]
    sv = {v: sims[v][mask] for v in VECS}
    klo, khi, inv = pair_keys(cell, i, j)
    K = len(klo)
    w = np.bincount(inv, minlength=K).astype(float)
    key_n = np.maximum(MIN_PER_KEY, np.round(N_BASE * w / w.sum())).astype(int)
    if K == 1:
        key_n = np.array([N_BASE])
    mean, se, n_got = sample_keys(CELL_X, cell, grp_codes, klo, khi, key_n, seed=SEED_SAMPLE)
    valid = ~np.isnan(mean[0])
    pv = valid[inv]
    gu, ginv_all = np.unique(g, return_inverse=True)
    G = len(gu)
    ginv, inv_v = ginv_all[pv], inv[pv]
    n_g = np.bincount(ginv, minlength=G).astype(float)
    C = sparse.csr_matrix((np.ones(len(ginv)), (ginv, inv_v)), shape=(G, K))
    S_g = {v: np.bincount(ginv, weights=sv[v][pv], minlength=G) for v in VECS}
    wv = np.bincount(inv_v, minlength=K).astype(float)
    mfill = np.where(valid, mean, 0.0)
    sefill = np.where(valid, se, 0.0)
    out = {"n_groups": int((n_g > 0).sum()), "n_pairs": int(pv.sum()), "n_pairs_before": len(pv),
           "n_keys": K, "n_keys_invalid": int((~valid).sum()), "base_n": int(n_got.sum())}
    res = {}
    for vi, v in enumerate(VECS):
        res[v] = {"within": S_g[v].sum() / n_g.sum(),
                  "baseline": (wv * mfill[vi]).sum() / wv.sum()}
    # cluster bootstrap over groups
    rng = np.random.default_rng(seed_boot)
    boots = {v: {"within": [], "baseline": []} for v in VECS}
    done = 0
    while done < N_BOOT:
        b = min(CHUNK, N_BOOT - done)
        M = np.stack([np.bincount(rng.integers(0, G, G), minlength=G) for _ in range(b)], axis=1).astype(float)
        den = n_g @ M
        Wb = C.T @ M  # K x b
        nz_rng = np.random.default_rng(seed_boot + 1000 + done)
        for vi, v in enumerate(VECS):
            boots[v]["within"].append((S_g[v] @ M) / den)
            mb = mfill[vi][:, None] + sefill[vi][:, None] * nz_rng.standard_normal((K, b))
            boots[v]["baseline"].append((Wb * mb).sum(0) / Wb.sum(0))
        done += b
    for v in VECS:
        wb = np.concatenate(boots[v]["within"]); bb = np.concatenate(boots[v]["baseline"])
        res[v]["wb"], res[v]["bb"] = wb, bb
    out["res"] = res
    return out


def summarize(cell_out, unm_out, test, subset, keyset):
    rows = []
    for v in VECS:
        r, u = cell_out["res"][v], unm_out["res"][v]
        gap = r["within"] - r["baseline"]
        gap_u = u["within"] - u["baseline"]
        gb = r["wb"] - r["bb"]
        gbu = u["wb"] - u["bb"]
        share = gb / gbu
        q = lambda a: np.percentile(a, [2.5, 97.5])
        rows.append(dict(test=test, subset=subset, vector=v, baseline_type=keyset,
                         n_groups=cell_out["n_groups"], n_pairs=cell_out["n_pairs"],
                         pairs_dropped_share=1 - cell_out["n_pairs"] / cell_out["n_pairs_before"],
                         n_keys=cell_out["n_keys"], n_keys_invalid=cell_out["n_keys_invalid"],
                         within=r["within"], within_lo=q(r["wb"])[0], within_hi=q(r["wb"])[1],
                         baseline=r["baseline"], baseline_lo=q(r["bb"])[0], baseline_hi=q(r["bb"])[1],
                         gap=gap, gap_lo=q(gb)[0], gap_hi=q(gb)[1],
                         unmatched_gap=gap_u,
                         pct_gap_surviving=100 * gap / gap_u,
                         pct_lo=100 * q(share)[0], pct_hi=100 * q(share)[1]))
    return rows


def run_test(name, pool, Xc, Xs, pairs, subsets, kind, grp_col, rows):
    global CELL_X
    CELL_X = [Xc, Xs]
    sims = {"compositional": pair_cosine(Xc, pairs[1], pairs[2]),
            "sequence": pair_cosine(Xs, pairs[1], pairs[2])}
    grp_codes = pd.factorize(pool[grp_col])[0]
    ks = keysets_for(pool, kind)
    for sname, mask in subsets.items():
        t0 = time.time()
        outs = {k: run_cell(pairs, sims, mask, c, grp_codes) for k, c in ks.items()}
        for k in ks:
            rows += summarize(outs[k], outs["unmatched"], name, sname, k)
        print(f"  [{name}/{sname}] {mask.sum()} pairs, {time.time()-t0:.0f}s", flush=True)
        for r in rows[-len(ks) * 2:]:
            if r["vector"] == "compositional":
                print(f"     {r['baseline_type']:24s} within={r['within']:.4f} base={r['baseline']:.4f} "
                      f"gap={r['gap']:.4f} surv={r['pct_gap_surviving']:.0f}%")


def pool_of(df, X, S, group_col, extra_mask=None):
    m = np.ones(len(df), bool) if extra_mask is None else extra_mask
    cnt = df.loc[m, group_col].value_counts()
    keep = m & df[group_col].isin(cnt[cnt >= 2].index).to_numpy()
    return df.loc[keep].reset_index(drop=True), X[keep], S[keep]


def main():
    t_start = time.time()
    df, X, S = load()
    rows, valrows = [], []
    sec7 = {k: pd.read_csv(RESULTS / f"07_book_author_similarity/silver_{k}_similarity_test.csv").iloc[0]
            for k in ["book", "author", "reviewer"]}

    # ---------------- (a) book, (e) book without shared reviewers
    pool, Xc, Xs = pool_of(df, X, S, "work_id")
    pairs = within_group_pairs(pool, "work_id", cap=CAP, seed=SEED_PAIRS)
    i, j = pairs[1], pairs[2]
    u = pool["user_id"].to_numpy(); sid = pool["source_id"].to_numpy()
    shared = (u[i] == u[j]) & (sid[i] != "platform_1")
    p1 = (sid[i] == "platform_1") | (sid[j] == "platform_1")
    pd.DataFrame([{"n_same_book_pairs": len(i), "pairs_sharing_reviewer_excl_platform_1": int(shared.sum()),
                   "share": shared.mean(), "pairs_involving_platform_1": int(p1.sum())}]
                 ).to_csv(OUT / "same_book_shared_reviewer.csv", index=False)
    print(f"book: {len(i)} pairs, {shared.sum()} share a (non-platform_1) reviewer", flush=True)
    allp = np.ones(len(i), bool)
    run_test("book", pool, Xc, Xs, pairs,
             {"all": allp, "e_drop_shared_reviewer": ~shared, "e_drop_shared_reviewer_and_platform_1": ~shared & ~p1},
             "book", "work_id", rows)
    b = within_group_pairs  # noqa
    valrows += validate("book", pool, Xc, pairs, sec7["book"], None)

    # ---------------- (b) author, different books
    pool, Xc, Xs = pool_of(df, X, S, "author_id")
    pairs = within_group_pairs(pool, "author_id", cap=CAP, pair_filter_col="work_id", seed=SEED_PAIRS)
    run_test("author_diff_book", pool, Xc, Xs, pairs, {"all": np.ones(len(pairs[0]), bool)},
             "author", "author_id", rows)
    valrows += validate("author", pool, Xc, pairs, sec7["author"], "work_id")

    # ---------------- (c, d) reviewer
    nop1 = (df["source_id"] != "platform_1").to_numpy()
    pool, Xc, Xs = pool_of(df, X, S, "user_id", nop1)
    pairs = within_group_pairs(pool, "user_id", cap=CAP, seed=SEED_PAIRS)
    i, j = pairs[1], pairs[2]
    gen = pool["g_code"].to_numpy(); au = pool["author_id"].to_numpy()
    sg = gen[i] == gen[j]; da = au[i] != au[j]
    subsets = {"all": np.ones(len(i), bool), "d_same_genre": sg, "d_diff_genre": ~sg,
               "d_diff_author": da, "d_diff_author_same_genre": da & sg, "d_diff_author_diff_genre": da & ~sg}
    run_test("reviewer", pool, Xc, Xs, pairs, subsets, "reviewer", "user_id", rows)
    valrows += validate("reviewer", pool, Xc, pairs, sec7["reviewer"], None)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "matched_similarity_results.csv", index=False)
    val = pd.DataFrame(valrows)
    val.to_csv(OUT / "validation_vs_section7.csv", index=False)
    print(val.to_string())
    write_tables(res)
    print(f"total runtime {time.time()-t_start:.0f}s")


def validate(kind, pool, Xc, pairs, s7, filter_col):
    g, i, j = pairs
    sim = pair_cosine(Xc, i, j)
    n_g = np.bincount(g); S_g = np.bincount(g, weights=sim)
    within = S_g.sum() / n_g.sum()
    col = pool[filter_col].to_numpy() if filter_col else None
    base = random_pairs_baseline(Xc, 1_000_000, 0, filter_col, col)
    return [dict(test=kind, quantity="within mean", this_run=within, section7=s7["observed_mean_similarity"]),
            dict(test=kind, quantity="unmatched baseline (random_pairs_baseline, seed 0)", this_run=base["mean"],
                 section7=s7["baseline_mean_similarity"]),
            dict(test=kind, quantity="n groups", this_run=len(n_g), section7=s7["n_groups"]),
            dict(test=kind, quantity="n pairs", this_run=len(g), section7=s7["n_pairs"])]


def write_tables(res):
    L = []
    for v in VECS:
        L.append(f"\n### {v} vector\n")
        L.append("| test | pair subset | baseline | n pairs (dropped) | within [95% CI] | baseline | gap [95% CI] | % of unmatched gap surviving [95% CI] |")
        L.append("|---|---|---|---|---|---|---|---|")
        for _, r in res[res.vector == v].iterrows():
            drop = f" ({100*r.pairs_dropped_share:.1f}%)" if r.pairs_dropped_share > 0.0005 else ""
            surv = "100 (ref.)" if r.baseline_type == "unmatched" else f"{r.pct_gap_surviving:.0f} [{r.pct_lo:.0f}, {r.pct_hi:.0f}]"
            L.append(f"| {r.test} | {r['subset']} | {r.baseline_type} | {r.n_pairs:,}{drop} | "
                     f"{r.within:.4f} [{r.within_lo:.4f}, {r.within_hi:.4f}] | {r.baseline:.4f} | "
                     f"{r.gap:.4f} [{r.gap_lo:.4f}, {r.gap_hi:.4f}] | {surv} |")
    (OUT / "tables.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()

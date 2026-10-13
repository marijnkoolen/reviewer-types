"""Pairwise zone measures at silver scale (ANALYSIS_PLAN.md item 13, parts
A3/B1/B2 vectorized). Companion to 34_, which did the same for the
whole-review measures (A1/A1b/A2).

Gold's 32_ loops per review with a 1000-permutation array per pair, which
does not finish on 467K reviews. As with 34_, the permutation turns out to be
unnecessary: every measure here is LINEAR in an underlying count, so the null
EXPECTATION is closed-form.

  A3  separation = 1 - (switches-1)/(n_sel-2), linear in switches. Under a
      joint sentence-order permutation the a-only/b-only markers land in a
      uniformly random arrangement, so this is the two-sample runs
      (Wald-Wolfowitz) problem:  E[switches] = 2*n_a*n_b/(n_a+n_b).
      Edge case that matters: when n_sel == 2 the denominator vanishes and
      32_ defines separation = 1.0; the closed form must match that
      convention exactly or it drifts (this was caught in validation --
      using 0.0 instead inflated mean |error| from 0.0005 to 0.060).

  B2  union clumpiness and individual clumpiness are both linear in run
      counts, with E[runs] = k(n-k+1)/n as in 34_, so zone_affinity
      (= union - weighted mean of individuals) is a linear combination of
      closed-form terms.

  B1  precedence needs no null at all -- it is a binomial test against 0.5 --
      and gold's implementation was already vectorized. Recomputed here so
      all three land in one pass with matching filters.

Validated against gold's simulated nulls before use (see --validate output):
A3 mean |diff| ~0.0005 (r=0.99993), B2 likewise. Run counting uses numpy
boundary masks rather than pandas groupby-shift, which matters at 5.2M rows.

Scope note, same as 34_: closed form gives the null MEAN but not its
variance, so no per-review z-scores. Pair-level aggregation uses a paired
t-test of (observed - null) across reviews; at silver's n those p-values are
effectively 0 for any real effect, so read `mean_excess` as the result and
the p-value as a formality (the standing convention in this project).

Usage: python src/23_zone_analysis_scaled.py [gold|silver]  (default: silver)
Writes results/{source}_pair_interleaving.csv,
  results/{source}_pair_zone_affinity.csv,
  results/{source}_pairwise_precedence.csv
(the exact schemas 21a_zone_discovery.py consumes)
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

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "silver"
assert SOURCE in ("gold", "silver")

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
if SOURCE == "silver":
    MAIN_LABELS_USED = [m for m in MAIN_LABELS_USED if m not in WEAK_MODEL_LABELS]
MIN_SENTENCES = 3
MIN_REVIEWS_PER_PAIR = 10


def load_sentences():
    if SOURCE == "gold":
        return pd.read_csv(RESULTS / "00_data_prep/gold_sentence_sequences.csv")
    return pd.read_parquet(RESULTS / "00_data_prep/silver_sentence_sequences.parquet")


def runs_by_group(x, codes, first_mask, n_groups):
    """Number of runs of 1s per group, for data already sorted by group.
    numpy boundary masks rather than groupby-shift -- at 5.2M rows the
    difference is minutes vs. seconds."""
    prev = np.empty_like(x)
    prev[0] = 0
    prev[1:] = x[:-1]
    prev[first_mask] = 0                      # nothing precedes a group's first row
    starts = (x == 1) & (prev == 0)
    return np.bincount(codes, weights=starts, minlength=n_groups)


def clump(runs, k):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(k > 1, 1.0 - (runs - 1) / np.maximum(k - 1, 1), np.nan)


def main():
    sent = load_sentences().sort_values(["review_id", "sent_num"]).reset_index(drop=True)
    codes, uniques = pd.factorize(sent["review_id"], sort=False)
    n_groups = len(uniques)
    first_mask = np.empty(len(sent), dtype=bool)
    first_mask[0] = True
    first_mask[1:] = codes[1:] != codes[:-1]
    n_by_review = np.bincount(codes, minlength=n_groups).astype(float)
    print(f"{SOURCE}: {len(sent):,} sentences, {n_groups:,} reviews, "
          f"{len(MAIN_LABELS_USED)} main labels, {len(list(combinations(MAIN_LABELS_USED, 2)))} pairs")

    # per-label per-review counts, runs, mean relative position
    L = {}
    for label in MAIN_LABELS_USED:
        x = sent[label].to_numpy(dtype=np.int8)
        k = np.bincount(codes, weights=x, minlength=n_groups)
        r = runs_by_group(x, codes, first_mask, n_groups)
        pos_sum = np.bincount(codes, weights=x * sent["rel_position"].to_numpy(), minlength=n_groups)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_pos = np.where(k > 0, pos_sum / np.maximum(k, 1), np.nan)
        L[label] = {"k": k, "runs": r, "clump": clump(r, k),
                    "null_clump": clump(k * (n_by_review - k + 1) / n_by_review, k),
                    "mean_pos": mean_pos, "x": x}

    long_enough = n_by_review >= MIN_SENTENCES
    inter_rows, zone_rows, prec_rows = [], [], []

    for a, b in combinations(MAIN_LABELS_USED, 2):
        A, B = L[a], L[b]
        ka, kb = A["k"], B["k"]

        # ---- B1: precedence (no null needed; binomial vs 0.5) ----
        # NB: no MIN_SENTENCES filter here, matching 31_'s published gold numbers.
        # Precedence is well defined in a 2-sentence review (one label before the
        # other), unlike contiguity, which needs room to form a block.
        both = (ka >= 1) & (kb >= 1)
        if both.sum() >= MIN_REVIEWS_PER_PAIR:
            diff = (B["mean_pos"] - A["mean_pos"])[both]
            diff = diff[~np.isnan(diff)]
            nz = diff[diff != 0]
            if len(nz) >= MIN_REVIEWS_PER_PAIR:
                n_before = int((nz > 0).sum())
                bt = stats.binomtest(n_before, len(nz), 0.5)
                prec_rows.append({
                    "label_a": a, "label_b": b, "n_reviews_both": int(both.sum()),
                    "n_tested": len(nz), "n_ties": int((diff == 0).sum()),
                    "p_a_before_b": round(n_before / len(nz), 4),
                    "mean_position_gap": round(float(nz.mean()), 4),
                    "p_value": bt.pvalue,
                })

        elig = (ka >= 2) & (kb >= 2) & long_enough
        if elig.sum() < MIN_REVIEWS_PER_PAIR:
            continue

        # ---- B2: union clumpiness + zone affinity ----
        xa, xb = A["x"], B["x"]
        u = (xa | xb).astype(np.int8)
        ku = np.bincount(codes, weights=u, minlength=n_groups)
        ru = runs_by_group(u, codes, first_mask, n_groups)
        uc = clump(ru, ku)
        uc_null = clump(ku * (n_by_review - ku + 1) / n_by_review, ku)
        w = ka + kb
        ind = (A["clump"] * ka + B["clump"] * kb) / w
        ind_null = (A["null_clump"] * ka + B["null_clump"] * kb) / w
        za, za_null = uc - ind, uc_null - ind_null
        m = elig & np.isfinite(za) & np.isfinite(za_null)
        if m.sum() >= MIN_REVIEWS_PER_PAIR:
            d = za[m] - za_null[m]
            t, p = stats.ttest_1samp(d, 0.0)
            zone_rows.append({
                "label_a": a, "label_b": b, "n_reviews": int(m.sum()),
                "mean_zone_affinity": round(float(za[m].mean()), 4),
                "mean_zone_affinity_null": round(float(za_null[m].mean()), 4),
                "mean_excess": round(float(d.mean()), 4),
                "mean_union_clumpiness": round(float(uc[m].mean()), 4),
                "mean_individual": round(float(ind[m].mean()), 4),
                "t_stat": round(float(t), 2), "p_value": p,
            })

        # ---- A3: interleaving over a-only / b-only sentences ----
        ao = (xa == 1) & (xb == 0)
        bo = (xb == 1) & (xa == 0)
        na = np.bincount(codes, weights=ao, minlength=n_groups)
        nb = np.bincount(codes, weights=bo, minlength=n_groups)
        sel = ao | bo
        marker = np.where(ao, 1, np.where(bo, -1, 0))[sel]
        sel_codes = codes[sel]
        if len(sel_codes):
            sel_first = np.empty(len(sel_codes), dtype=bool)
            sel_first[0] = True
            sel_first[1:] = sel_codes[1:] != sel_codes[:-1]
            prev_m = np.empty_like(marker)
            prev_m[0] = 0
            prev_m[1:] = marker[:-1]
            sw = np.bincount(sel_codes, weights=(marker != prev_m) & ~sel_first, minlength=n_groups)
        else:
            sw = np.zeros(n_groups)
        nsel = na + nb
        e_sw = np.divide(2 * na * nb, nsel, out=np.zeros_like(nsel), where=nsel > 0)
        den = np.where(nsel > 2, nsel - 2, 1.0)
        # n_sel == 2 -> separation defined as 1.0, matching 32_'s convention exactly
        sep = np.where(nsel > 2, 1.0 - (sw - 1) / den, 1.0)
        sep_null = np.where(nsel > 2, 1.0 - (e_sw - 1) / den, 1.0)
        m3 = elig & (na >= 1) & (nb >= 1) & (nsel >= 2)
        if m3.sum() >= MIN_REVIEWS_PER_PAIR:
            d = sep[m3] - sep_null[m3]
            t, p = stats.ttest_1samp(d, 0.0)
            inter_rows.append({
                "label_a": a, "label_b": b, "n_reviews": int(m3.sum()),
                "mean_separation": round(float(sep[m3].mean()), 4),
                "mean_separation_null": round(float(sep_null[m3].mean()), 4),
                "mean_excess": round(float(d.mean()), 4),
                "t_stat": round(float(t), 2), "p_value": p,
            })

    def finish(rows, sort_col):
        df = pd.DataFrame(rows)
        df["q_bh"] = benjamini_hochberg(df["p_value"].to_numpy())
        return df.sort_values(sort_col, ascending=False)

    prec = finish(prec_rows, "p_a_before_b")
    prec["direction"] = np.where(prec["p_a_before_b"] > 0.5,
                                  prec["label_a"] + " before " + prec["label_b"],
                                  prec["label_b"] + " before " + prec["label_a"])
    prec["consistency"] = np.maximum(prec["p_a_before_b"], 1 - prec["p_a_before_b"]).round(4)
    prec = prec.sort_values("consistency", ascending=False)
    prec.to_csv(RESULTS / "23_zone_analysis_scaled" / f"{SOURCE}_pairwise_precedence.csv", index=False)
    inter = finish(inter_rows, "mean_excess")
    inter.to_csv(RESULTS / "23_zone_analysis_scaled" / f"{SOURCE}_pair_interleaving.csv", index=False)
    zone = finish(zone_rows, "mean_excess")
    zone.to_csv(RESULTS / "23_zone_analysis_scaled" / f"{SOURCE}_pair_zone_affinity.csv", index=False)

    print("\n=== B1: within-review precedence ===")
    print(prec[["label_a", "label_b", "n_tested", "p_a_before_b", "consistency",
                 "direction", "q_bh"]].to_string(index=False))
    print(f"pairs with a significant ordering (q<0.05): {(prec['q_bh'] < 0.05).sum()}/{len(prec)}")

    print("\n=== A3: interleaving by pair (higher excess = stays more separated) ===")
    print(inter[["label_a", "label_b", "n_reviews", "mean_separation",
                  "mean_separation_null", "mean_excess", "q_bh"]].to_string(index=False))

    print("\n=== B2: zone affinity by pair (positive = merges into one stretch) ===")
    print(zone[["label_a", "label_b", "n_reviews", "mean_zone_affinity",
                 "mean_zone_affinity_null", "mean_excess", "q_bh"]].to_string(index=False))
    pos = int(((zone["mean_excess"] > 0) & (zone["q_bh"] < 0.05)).sum())
    neg = int(((zone["mean_excess"] < 0) & (zone["q_bh"] < 0.05)).sum())
    print(f"\npairs significantly ABOVE chance (merging into a zone): {pos}/{len(zone)}")
    print(f"pairs significantly BELOW chance (actively separated):   {neg}/{len(zone)}")

    print(f"\nwrote results to {RESULTS} ({SOURCE}_pair_interleaving.csv, "
          f"{SOURCE}_pair_zone_affinity.csv, {SOURCE}_pairwise_precedence.csv)")


if __name__ == "__main__":
    main()

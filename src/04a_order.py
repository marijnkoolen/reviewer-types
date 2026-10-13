"""Order: where in a review a category tends to appear, and what tends to
follow what (ANALYSIS_PLAN.md §5, gold set only).

Positional density: for each label, the distribution of relative sentence
position (0 = first sentence, 1 = last) across every sentence carrying that
label, tested against the uninformative expectation (mean position = 0.5,
i.e. no systematic preference for early/late).

Three refinements on top of the plain pooled mean (chat, 2026-09-22), because
a label's multiplicity per review (how many of its own sentences a review
tends to carry) distorts both the pooled mean's magnitude and the naive CI's
width, and review length itself may confound the whole comparison:

1. Non-independence in the significance test. The pooled array mixes
   multiple, correlated sentences from the same review (e.g. Content
   averages 7.1 sentences/review) as if they were independent draws. A
   sentence-level bootstrap pseudoreplicates that correlation and understates
   the CI for high-multiplicity labels. Fixed by a cluster bootstrap:
   resample review_ids (not sentences) with replacement, so a label's
   between-review variability -- not its within-review sentence count --
   sets the CI width.
2. Tail mass and onset, not just the mean. A label that recurs many times
   per review has its own instances averaged toward the center of wherever
   they span, which pulls the *pooled* mean toward 0.5 regardless of how
   early/late the label's usage actually is -- a low-multiplicity label
   (one mention per review) carries no such internal averaging and can sit
   at a much more extreme mean for the same underlying effect size. Two
   statistics that aren't compressed this way: the fraction of a label's
   sentences in the first/last fifth of the review (frac_front/back_fifth),
   and the position of each review's *first* and *last* sentence carrying
   the label (gold_order_onset.csv) -- exactly one value per review
   regardless of recurrence, so directly comparable across labels.
3. Length stratification. Gold review length is heavily skewed (median 6
   sentences; 16.6% have <=2, 29% have 3-5, only 14% have >=20) and
   rel_position's granularity differs sharply across that range (a 2-sentence
   review only has positions {0, 1}). Positional density is recomputed within
   length quartiles (gold_order_position_by_length.csv) as a robustness check
   on whether a label's early/late direction holds across review lengths or
   is driven by one length regime -- same "length as a substantive
   dimension, not just noise" stance as 09_length_dimension.py.

Transition structure: restricted to the 7 main categories (excluding the
unused 'None'). Sub-categories are left out here because mixing 20 labels
into one transition matrix mostly just re-encodes the co-occurrence structure
from §3 (a sub-category transition is dominated by its parent's) and would
make the matrix far less readable; the main-category level is what actually
answers "what kind of sentence tends to follow what kind." For each pair of
consecutive sentences and each pair of their active main labels, one
transition is counted (self-transitions included -- a category persisting
into the next sentence is a real and expected pattern given §4's contiguity
finding). Reported both as row-normalized transition probabilities and as a
PMI-style log2(observed / expected-under-marginal-independence) matrix, since
raw probabilities are dominated by how common a label is overall rather than
by genuine sequential structure.

Sequence clustering (e.g. optimal matching for recurring review "arcs") is
deferred to the typology-clustering step (§6), where it's compared directly
against the compositional clustering rather than run in isolation here.

Usage: python src/04a_order.py
Writes results/04_order/gold_order_position.csv (pooled positional density, cluster
bootstrap CI, tail-mass columns), results/04_order/gold_order_onset.csv (per-review
first/last occurrence position), results/04_order/gold_order_position_by_length.csv
(positional density within review-length quartiles), results/
gold_order_transitions.csv, and results/04_order/gold_order_transition_matrix.csv
(wide, for heatmap plotting).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SENTENCE_TABLE = ROOT / "results" / "00_data_prep/gold_sentence_sequences.csv"
POSITION_OUT = ROOT / "results" / "04_order/gold_order_position.csv"
ONSET_OUT = ROOT / "results" / "04_order/gold_order_onset.csv"
LENGTH_STRATIFIED_OUT = ROOT / "results" / "04_order/gold_order_position_by_length.csv"
TRANSITIONS_OUT = ROOT / "results" / "04_order/gold_order_transitions.csv"
TRANSITION_MATRIX_OUT = ROOT / "results" / "04_order/gold_order_transition_matrix.csv"

FEATURE_LABELS = [m for m in MAIN_LABELS if m != "None"] + SUB_LABELS
MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
N_BOOTSTRAP = 2000
RNG_SEED = 0
FRONT_BACK_THRESHOLD = 0.2  # "front fifth" / "back fifth" of the review
LENGTH_QUARTILE_LABELS = ["Q1 (shortest)", "Q2", "Q3", "Q4 (longest)"]


def cluster_bootstrap_ci(group_sums, group_sizes, rng, n_bootstrap=N_BOOTSTRAP):
    """CI on the pooled mean, resampling review-groups (not sentences) with
    replacement -- keeps sentences from the same review from being counted as
    independent evidence."""
    n_groups = len(group_sums)
    idx = rng.integers(0, n_groups, size=(n_bootstrap, n_groups))
    boot_means = group_sums[idx].sum(axis=1) / group_sizes[idx].sum(axis=1)
    return np.percentile(boot_means, [2.5, 97.5])


def bootstrap_mean_ci(values, rng, n_bootstrap=N_BOOTSTRAP):
    """Plain bootstrap for statistics that are already one-value-per-review
    (e.g. onset position), so sentence-level clustering doesn't apply."""
    n = len(values)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = values[idx].mean(axis=1)
    return np.percentile(boot_means, [2.5, 97.5])


def positional_density(df):
    rng = np.random.default_rng(RNG_SEED)
    multi_sentence = df[df["num_sentences"] > 1]
    rows = []
    for label in FEATURE_LABELS:
        active = multi_sentence.loc[multi_sentence[label] == 1]
        n = len(active)
        if n == 0:
            continue
        positions = active["rel_position"].to_numpy()
        grouped = active.groupby("review_id")["rel_position"].agg(["sum", "count"])
        mean_pos = grouped["sum"].sum() / grouped["count"].sum()
        median_pos = np.median(positions)
        ci_lo, ci_hi = cluster_bootstrap_ci(
            grouped["sum"].to_numpy(), grouped["count"].to_numpy(), rng
        )
        deviates = not (ci_lo <= 0.5 <= ci_hi)
        rows.append({
            "label": label, "level": "main" if label in MAIN_LABELS else "sub",
            "n_sentences": n, "n_reviews_with_label": len(grouped),
            "mean_sentences_per_review": round(n / len(grouped), 2),
            "mean_rel_position": round(mean_pos, 4),
            "median_rel_position": round(median_pos, 4),
            "std_rel_position": round(positions.std(), 4),
            "frac_front_fifth": round((positions < FRONT_BACK_THRESHOLD).mean(), 4),
            "frac_back_fifth": round((positions > 1 - FRONT_BACK_THRESHOLD).mean(), 4),
            "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
            "deviates_from_center": deviates,
            "direction": ("early" if mean_pos < 0.5 else "late") if deviates else "none",
        })
    out = pd.DataFrame(rows).sort_values("mean_rel_position")
    out.to_csv(POSITION_OUT, index=False)
    return out


def onset_analysis(df):
    """Per label, the position of each review's *first* and *last* sentence
    carrying that label -- exactly one value per review regardless of how
    many times the label recurs, so unaffected by the multiplicity-driven
    compression toward 0.5 that the pooled mean shows."""
    rng = np.random.default_rng(RNG_SEED)
    multi_sentence = df[df["num_sentences"] > 1]
    rows = []
    for label in FEATURE_LABELS:
        active = multi_sentence.loc[multi_sentence[label] == 1]
        if active.empty:
            continue
        span = active.groupby("review_id")["rel_position"].agg(["min", "max"])
        first = span["min"].to_numpy()
        last = span["max"].to_numpy()
        first_ci_lo, first_ci_hi = bootstrap_mean_ci(first, rng)
        last_ci_lo, last_ci_hi = bootstrap_mean_ci(last, rng)
        rows.append({
            "label": label, "level": "main" if label in MAIN_LABELS else "sub",
            "n_reviews": len(span),
            "first_mean_position": round(first.mean(), 4),
            "first_median_position": round(np.median(first), 4),
            "first_ci_lo": round(first_ci_lo, 4), "first_ci_hi": round(first_ci_hi, 4),
            "last_mean_position": round(last.mean(), 4),
            "last_median_position": round(np.median(last), 4),
            "last_ci_lo": round(last_ci_lo, 4), "last_ci_hi": round(last_ci_hi, 4),
        })
    out = pd.DataFrame(rows).sort_values("first_mean_position")
    out.to_csv(ONSET_OUT, index=False)
    return out


def positional_density_by_length(df):
    """Positional density recomputed within review-length quartiles, as a
    robustness check on whether a label's early/late direction holds across
    review lengths or is an artifact of pooling very different review
    lengths (and rel_position granularities) together."""
    rng = np.random.default_rng(RNG_SEED)
    multi_sentence = df[df["num_sentences"] > 1].copy()
    review_lengths = multi_sentence.groupby("review_id")["num_sentences"].first()
    quartile = pd.qcut(review_lengths, 4, labels=LENGTH_QUARTILE_LABELS, duplicates="drop")
    multi_sentence["length_quartile"] = multi_sentence["review_id"].map(quartile)

    bounds = review_lengths.groupby(quartile, observed=True).agg(["min", "max", "count"])

    rows = []
    for q in quartile.cat.categories:
        qdf = multi_sentence[multi_sentence["length_quartile"] == q]
        if qdf.empty:
            continue
        for label in FEATURE_LABELS:
            active = qdf.loc[qdf[label] == 1]
            n = len(active)
            if n == 0:
                continue
            grouped = active.groupby("review_id")["rel_position"].agg(["sum", "count"])
            mean_pos = grouped["sum"].sum() / grouped["count"].sum()
            ci_lo, ci_hi = cluster_bootstrap_ci(
                grouped["sum"].to_numpy(), grouped["count"].to_numpy(), rng
            )
            deviates = not (ci_lo <= 0.5 <= ci_hi)
            rows.append({
                "length_quartile": q,
                "sentences_range": f"{bounds.loc[q, 'min']}-{bounds.loc[q, 'max']}",
                "label": label, "level": "main" if label in MAIN_LABELS else "sub",
                "n_sentences": n, "n_reviews_with_label": len(grouped),
                "mean_rel_position": round(mean_pos, 4),
                "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
                "deviates_from_center": deviates,
                "direction": ("early" if mean_pos < 0.5 else "late") if deviates else "none",
            })
    out = pd.DataFrame(rows)
    out.to_csv(LENGTH_STRATIFIED_OUT, index=False)
    return out, bounds


def transition_structure(df):
    n_labels = len(MAIN_LABELS_USED)
    idx = {label: i for i, label in enumerate(MAIN_LABELS_USED)}
    counts = np.zeros((n_labels, n_labels), dtype=np.int64)

    for rid, g in df.sort_values("sent_num").groupby("review_id"):
        mat = g[MAIN_LABELS_USED].to_numpy(dtype=np.int64)
        for t in range(len(mat) - 1):
            from_labels = np.nonzero(mat[t])[0]
            to_labels = np.nonzero(mat[t + 1])[0]
            for a in from_labels:
                for b in to_labels:
                    counts[a, b] += 1

    total = counts.sum()
    from_marginal = counts.sum(axis=1) / total
    to_marginal = counts.sum(axis=0) / total
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts, dtype=float), where=row_sums > 0)

    rows = []
    for i, a in enumerate(MAIN_LABELS_USED):
        for j, b in enumerate(MAIN_LABELS_USED):
            observed_p = counts[i, j] / total if total else 0.0
            expected_p = from_marginal[i] * to_marginal[j]
            pmi = np.log2(observed_p / expected_p) if observed_p > 0 and expected_p > 0 else float("nan")
            rows.append({
                "label_from": a, "label_to": b, "count": int(counts[i, j]),
                "transition_prob": round(probs[i, j], 4), "pmi": round(pmi, 4) if not np.isnan(pmi) else "",
            })
    long_out = pd.DataFrame(rows).sort_values("pmi", ascending=False)
    long_out.to_csv(TRANSITIONS_OUT, index=False)

    wide = pd.DataFrame(counts, index=MAIN_LABELS_USED, columns=MAIN_LABELS_USED)
    wide.to_csv(TRANSITION_MATRIX_OUT)
    return long_out, wide


def main():
    df = pd.read_csv(SENTENCE_TABLE)

    pos = positional_density(df)
    print(f"wrote {POSITION_OUT} ({len(pos)} labels)")
    print()
    print(pos.to_string(index=False))

    onset = onset_analysis(df)
    print()
    print(f"wrote {ONSET_OUT} ({len(onset)} labels)")
    print(onset.to_string(index=False))

    by_length, bounds = positional_density_by_length(df)
    print()
    print(f"wrote {LENGTH_STRATIFIED_OUT} ({len(by_length)} label x length-quartile rows)")
    print("length quartile boundaries (num_sentences min-max, n reviews):")
    print(bounds.to_string())
    print()
    print("main-category mean position by length quartile (direction flips flagged):")
    pivot = by_length[by_length["level"] == "main"].pivot(
        index="label", columns="length_quartile", values="mean_rel_position"
    )
    print(pivot.to_string())

    trans, wide = transition_structure(df)
    print()
    print(f"wrote {TRANSITIONS_OUT} ({len(trans)} label-pair transitions)")
    print(f"wrote {TRANSITION_MATRIX_OUT} ({wide.shape[0]}x{wide.shape[1]} matrix)")
    print()
    print("Transitions ranked by PMI (what follows what, more than base rates predict):")
    print(trans.head(15).to_string(index=False))


if __name__ == "__main__":
    main()

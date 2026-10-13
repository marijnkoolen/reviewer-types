"""Instance-level (multiple-imputation) correction at silver scale
(ANALYSIS_PLAN.md item 7), applied to co-occurrence, contiguity, order, and
typology clustering -- the four analyses that need the actual per-sentence
label *set*, not just a corpus-level rate (which the aggregate Rogan-Gladen
correction in 01c_occurrence.py already handles).

Last of the Phase C TODO items, deliberately: raw replication has already
strongly validated all four of these against gold (report_phase_c.md §2-5),
so this is a precision/rigor pass on findings already trusted, not a
gap-filling one. And the most expensive item -- M imputations x
permutation-heavy methods doesn't finish in reasonable time at the full
5.2M-sentence corpus (02c_cooccurrence.py's docstring flags this
explicitly as a separate, smaller-scale follow-up). Run on a random
subsample instead: 10,000 reviews (~8.7x gold's 1,152, ~2.1% of the fiction-
filtered corpus) -- big enough for a real precision gain over gold, small
enough that M=20 imputations x four analyses finishes in minutes, not hours.

Method: multiple_impute (src/correction/imputation.py, already validated on
gold's synthetic-corruption pipeline) draws M=20 plausible corrected
per-sentence label sets from each label's posterior, using the Rogan-Gladen
corrected sentence-level prevalence (silver_occurrence_sentence_level.csv)
as the prior and the real held-out confusion matrices (unconditional +
conditional). For each of the three statistics with a natural per-draw
variance (co-occurrence PMI is the exception, see below), Rubin's rules pool
the M point estimates into one corrected estimate + CI. Every corrected
number is reported alongside the RAW value on the *same subsample* (apples-
to-apples, isolates what correction itself changes) and the raw value
already published at full-corpus scale (context on subsampling variability).

Simplifications, stated rather than silently made:
- Co-occurrence PMI has no clean analytic per-imputation variance the way a
  simple mean (contiguity's clumpiness, order's position) does -- reported
  as the mean and SD *across* the M imputation draws instead of a full
  Rubin's-rules CI. This captures imputation-driven uncertainty, just not
  via the identical machinery as the other two statistics.
- Typology clustering isn't "pooled" across M imputations either -- there's
  no standard way to average M different KMeans assignments. Instead, each
  review's M imputed label sets are averaged into one soft "MI-mean"
  corrected composition vector (averaging M binary draws approximates the
  posterior probability, a smoother single input than any one hard draw),
  clustered once (k=8, matching the already-validated raw k), and compared
  against a clustering fit on the same subsample's RAW composition --
  cluster-profile similarity and a crosstab of who moves where under
  correction, not a formal pooled significance test.

Usage: python src/14c_imputed_correction.py
Writes results/14_imputed_correction/silver_imputed_cooccurrence.csv, silver_imputed_contiguity.csv,
  silver_imputed_order.csv, silver_imputed_typology_profile_{raw,corrected}.csv,
  silver_imputed_typology_crosstab.csv
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS, SUB_TO_MAIN, LABEL_TO_CODE, WEAK_MODEL_LABELS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "correction"))
from imputation import multiple_impute, rubin_pool  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
FEATURE_LABELS = [l for l in MAIN_LABELS_USED + SUB_LABELS if l not in WEAK_MODEL_LABELS]
PARENT_CHILD_PAIRS = {frozenset((sub, main)) for sub, main in SUB_TO_MAIN.items()}

N_SAMPLE_REVIEWS = 10_000
M_IMPUTATIONS = 20
K_TYPOLOGY = 8
N_NULL_DRAWS = 1000
RNG_SEED = 0


def load_sample():
    review = pd.read_parquet(RESULTS / "00_data_prep/silver_review_features.parquet")
    rng = np.random.default_rng(RNG_SEED)
    sample_ids = rng.choice(review["review_id"].to_numpy(), size=N_SAMPLE_REVIEWS, replace=False)
    review_sample = review[review["review_id"].isin(sample_ids)].reset_index(drop=True)

    sentence = pd.read_parquet(RESULTS / "00_data_prep/silver_sentence_sequences.parquet")
    sentence_sample = sentence[sentence["review_id"].isin(sample_ids)]
    # sorted once, here -- imputations preserve input row order, so every downstream
    # function can rely on sentence_sample and each imputed frame staying row-aligned
    # without re-sorting (re-sorting one but not the other silently scrambles the
    # label-to-review mapping, since imputed frames carry no review_id/sent_num of their own)
    sentence_sample = sentence_sample.sort_values(["review_id", "sent_num"]).reset_index(drop=True)
    return review_sample, sentence_sample


def load_priors_and_confusion():
    occ = pd.read_csv(RESULTS / "01_occurrence/silver_occurrence_sentence_level.csv")
    priors = dict(zip(occ["label"], occ["sentence_prevalence_corrected"]))
    confusion = pd.read_csv(RESULTS / "correction/confusion_matrix.csv")
    confusion_conditional = pd.read_csv(RESULTS / "correction/confusion_matrix_conditional.csv")
    code_of_label = {l: LABEL_TO_CODE[l] for l in FEATURE_LABELS if l in LABEL_TO_CODE}
    return priors, confusion, confusion_conditional, code_of_label


def run_imputation(sentence_sample, priors, confusion, confusion_conditional, code_of_label):
    priors_present = {l: priors[l] for l in FEATURE_LABELS if l in priors}
    return multiple_impute(sentence_sample, FEATURE_LABELS, code_of_label, confusion, priors_present,
                            SUB_TO_MAIN, m=M_IMPUTATIONS, seed=RNG_SEED,
                            conditional_confusion_df=confusion_conditional)


# --- co-occurrence (sentence-level PMI) ------------------------------------

def pairwise_pmi(X, labels):
    n = X.shape[0]
    joint = X.T @ X
    marginal = np.diag(joint)
    rows = []
    for i, j in combinations(range(len(labels)), 2):
        a, b = labels[i], labels[j]
        if frozenset((a, b)) in PARENT_CHILD_PAIRS:
            continue
        n_both = int(joint[i, j])
        n_a, n_b = int(marginal[i]), int(marginal[j])
        if n_both == 0 or n_a == 0 or n_b == 0:
            continue
        p_a, p_b, p_ab = n_a / n, n_b / n, n_both / n
        pmi = np.log(p_ab / (p_a * p_b))
        rows.append({"label_a": a, "label_b": b, "pmi": pmi})
    return pd.DataFrame(rows)


def cooccurrence_comparison(sentence_sample, imputations):
    raw_X = sentence_sample[FEATURE_LABELS].to_numpy(dtype=np.int64)
    raw = pairwise_pmi(raw_X, FEATURE_LABELS).rename(columns={"pmi": "raw_subsample_pmi"})

    imputed_pmis = []
    for imp in imputations:
        X = imp[FEATURE_LABELS].to_numpy(dtype=np.int64)
        imputed_pmis.append(pairwise_pmi(X, FEATURE_LABELS).set_index(["label_a", "label_b"])["pmi"])
    imputed_df = pd.concat(imputed_pmis, axis=1)
    pooled = pd.DataFrame({
        "mean_imputed_pmi": imputed_df.mean(axis=1),
        "sd_imputed_pmi": imputed_df.std(axis=1),
    }).reset_index()

    out = raw.merge(pooled, on=["label_a", "label_b"], how="outer")
    full_corpus = pd.read_csv(RESULTS / "02_cooccurrence/silver_cooccurrence_sentence.csv")[["label_a", "label_b", "pmi"]]
    full_corpus = full_corpus.rename(columns={"pmi": "raw_full_corpus_pmi"})
    out = out.merge(full_corpus, on=["label_a", "label_b"], how="left")
    out["shift"] = out["mean_imputed_pmi"] - out["raw_subsample_pmi"]
    return out.sort_values("mean_imputed_pmi", ascending=False)


# --- contiguity (mean clumpiness per label) --------------------------------

def observed_runs(sentence_df, label):
    grouped = sentence_df.groupby("review_id", sort=False)
    prev = grouped[label].shift(fill_value=0)
    starts = (sentence_df[label] == 1) & (prev != 1)
    return starts.groupby(sentence_df["review_id"], sort=False).sum()


def clumpiness_per_label(sentence_df, label):
    runs = observed_runs(sentence_df, label)
    k = sentence_df.groupby("review_id")[label].sum()
    k, runs = k.align(runs)
    has_k = k[k > 0]
    if len(has_k) == 0:
        return None, None
    clump = np.where(has_k == 1, 1.0, 1 - (runs.loc[has_k.index] - 1) / (has_k - 1))
    mean_clump = clump.mean()
    var_clump = clump.var(ddof=1) / len(clump) if len(clump) > 1 else 0.0
    return mean_clump, var_clump


def contiguity_comparison(sentence_sample, imputations):
    # sentence_sample is already sorted by (review_id, sent_num) -- see load_sample --
    # and every imputation preserves that same row order, so attaching review_id
    # straight from sentence_sample (no re-sort) keeps the two aligned.
    rows = []
    for label in FEATURE_LABELS:
        raw_mean, _ = clumpiness_per_label(sentence_sample, label)

        estimates, variances = [], []
        for imp in imputations:
            imp_sorted = imp.copy()
            imp_sorted["review_id"] = sentence_sample["review_id"].to_numpy()
            m, v = clumpiness_per_label(imp_sorted, label)
            if m is not None:
                estimates.append(m)
                variances.append(v)
        if not estimates:
            continue
        pooled = rubin_pool(estimates, variances)
        rows.append({
            "label": label, "raw_subsample_clumpiness": raw_mean,
            "pooled_imputed_clumpiness": pooled["pooled_estimate"],
            "ci_lo": pooled["ci_lo"], "ci_hi": pooled["ci_hi"],
        })

    out = pd.DataFrame(rows)
    full_corpus = pd.read_csv(RESULTS / "03_contiguity/silver_contiguity_label_summary.csv")[["label", "mean_clumpiness"]]
    full_corpus = full_corpus.rename(columns={"mean_clumpiness": "raw_full_corpus_clumpiness"})
    out = out.merge(full_corpus, on="label", how="left")
    out["shift"] = out["pooled_imputed_clumpiness"] - out["raw_subsample_clumpiness"]
    return out.sort_values("pooled_imputed_clumpiness", ascending=False)


# --- order (mean relative position per label) ------------------------------

def position_comparison(sentence_sample, imputations):
    rows = []
    for label in MAIN_LABELS_USED:
        raw_active = sentence_sample.loc[sentence_sample[label] == 1, "rel_position"]
        raw_mean = raw_active.mean()

        estimates, variances = [], []
        for imp in imputations:
            active = sentence_sample.loc[imp[label].to_numpy() == 1, "rel_position"]
            if len(active) > 1:
                estimates.append(active.mean())
                variances.append(active.var(ddof=1) / len(active))
        if not estimates:
            continue
        pooled = rubin_pool(estimates, variances)
        rows.append({
            "label": label, "raw_subsample_position": raw_mean,
            "pooled_imputed_position": pooled["pooled_estimate"],
            "ci_lo": pooled["ci_lo"], "ci_hi": pooled["ci_hi"],
        })

    out = pd.DataFrame(rows)
    full_corpus = pd.read_csv(RESULTS / "04_order/silver_order_position.csv")[["label", "mean_rel_position"]]
    full_corpus = full_corpus.rename(columns={"mean_rel_position": "raw_full_corpus_position"})
    out = out.merge(full_corpus, on="label", how="left")
    out["shift"] = out["pooled_imputed_position"] - out["raw_subsample_position"]
    return out.sort_values("pooled_imputed_position")


# --- typology (compositional clustering, raw vs. MI-mean corrected) --------

def cluster_profile(X_prop, review_ids, seed=RNG_SEED):
    X = StandardScaler().fit_transform(X_prop)
    km = KMeans(n_clusters=K_TYPOLOGY, n_init=10, random_state=seed).fit(X)
    labels = km.labels_
    profile_rows = []
    for c in range(K_TYPOLOGY):
        mask = labels == c
        row = {"cluster": c, "n_reviews": int(mask.sum()), "pct_of_sample": round(100 * mask.mean(), 1)}
        for j, label in enumerate(MAIN_LABELS_USED):
            row[f"mean_prop__{label}"] = round(X_prop[mask, j].mean(), 4)
        profile_rows.append(row)
    return pd.DataFrame(profile_rows), pd.Series(labels, index=review_ids, name="cluster")


def typology_comparison(review_sample, imputations, sentence_sample):
    review_ids = review_sample["review_id"].to_numpy()
    raw_X = review_sample[[f"prop__{l}" for l in MAIN_LABELS_USED]].to_numpy()
    raw_profile, raw_clusters = cluster_profile(raw_X, review_ids)

    n_sentences = sentence_sample.groupby("review_id")["num_sentences"].first().reindex(review_ids)
    sums = np.zeros((len(review_ids), len(MAIN_LABELS_USED)))
    id_index = {rid: i for i, rid in enumerate(review_ids)}
    row_idx = sentence_sample["review_id"].map(id_index).to_numpy()
    for imp in imputations:
        counts = np.zeros((len(review_ids), len(MAIN_LABELS_USED)))
        for j, label in enumerate(MAIN_LABELS_USED):
            np.add.at(counts[:, j], row_idx, imp[label].to_numpy())
        sums += counts / n_sentences.to_numpy()[:, None]
    mi_mean_X = sums / len(imputations)

    corrected_profile, corrected_clusters = cluster_profile(mi_mean_X, review_ids)

    crosstab = pd.crosstab(raw_clusters.rename("raw_cluster"), corrected_clusters.rename("corrected_cluster"))
    ari = adjusted_rand_score(raw_clusters.to_numpy(), corrected_clusters.to_numpy())
    return raw_profile, corrected_profile, crosstab, ari


def main():
    review_sample, sentence_sample = load_sample()
    print(f"subsample: {len(review_sample)} reviews, {len(sentence_sample)} sentences")

    priors, confusion, confusion_conditional, code_of_label = load_priors_and_confusion()
    print(f"running {M_IMPUTATIONS} imputations...")
    imputations = run_imputation(sentence_sample, priors, confusion, confusion_conditional, code_of_label)

    print("\n=== co-occurrence: raw subsample vs. imputed-corrected vs. raw full corpus ===")
    cooc = cooccurrence_comparison(sentence_sample, imputations)
    cooc.to_csv(RESULTS / "14_imputed_correction/silver_imputed_cooccurrence.csv", index=False)
    print(cooc.head(10).to_string(index=False))
    print(f"mean |shift| (corrected - raw subsample): {cooc['shift'].abs().mean():.4f}")

    print("\n=== contiguity: mean clumpiness, raw vs. corrected ===")
    contig = contiguity_comparison(sentence_sample, imputations)
    contig.to_csv(RESULTS / "14_imputed_correction/silver_imputed_contiguity.csv", index=False)
    print(contig.to_string(index=False))
    print(f"mean |shift|: {contig['shift'].abs().mean():.4f}")

    print("\n=== order: mean relative position, raw vs. corrected ===")
    order = position_comparison(sentence_sample, imputations)
    order.to_csv(RESULTS / "14_imputed_correction/silver_imputed_order.csv", index=False)
    print(order.to_string(index=False))
    print(f"mean |shift|: {order['shift'].abs().mean():.4f}")

    print("\n=== typology: cluster profile, raw subsample vs. MI-mean corrected ===")
    raw_profile, corrected_profile, crosstab, ari = typology_comparison(review_sample, imputations, sentence_sample)
    raw_profile.to_csv(RESULTS / "14_imputed_correction/silver_imputed_typology_profile_raw.csv", index=False)
    corrected_profile.to_csv(RESULTS / "14_imputed_correction/silver_imputed_typology_profile_corrected.csv", index=False)
    crosstab.to_csv(RESULTS / "14_imputed_correction/silver_imputed_typology_crosstab.csv")
    print(f"adjusted Rand index (raw cluster assignment vs. corrected): {ari:.4f}")
    print("\nraw subsample cluster profile:")
    print(raw_profile[["cluster", "n_reviews", "pct_of_sample"] + [f"mean_prop__{l}" for l in MAIN_LABELS_USED]]
          .to_string(index=False))
    print("\ncorrected (MI-mean) cluster profile:")
    print(corrected_profile[["cluster", "n_reviews", "pct_of_sample"] + [f"mean_prop__{l}" for l in MAIN_LABELS_USED]]
          .to_string(index=False))

    print(f"\nwrote results to {RESULTS} (silver_imputed_*.csv)")


if __name__ == "__main__":
    main()

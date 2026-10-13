"""Occurrence on the silver set (ANALYSIS_PLAN.md §2, Phase C), 466,976
fiction-genre reviews -- 405x the gold set.

IMPORTANT unit-of-analysis fix, caught by comparing against gold before
trusting the numbers: the confusion matrix (and Phase B's validated Rogan-
Gladen correction) is estimated at the SENTENCE level -- it answers "what
fraction of sentences carry this label." Gold's Phase A occurrence report
used a different, REVIEW-level statistic instead ("% of reviews containing
>=1 sentence of this category"), which runs much higher (e.g. Reader_response
hits 98% of reviews vs. only ~50% of sentences, since one review has ~10
sentences and only needs one hit). Applying the sentence-level Rogan-Gladen
correction directly to that review-level "at least one" rate is a real
mismatch, not a rounding issue -- it drove several labels to a nonsensical
100% corrected estimate on the first attempt (visible in git history / this
file's earlier version). Correcting "at least one true positive among ~10
correlated sentences" properly would need a different, harder model (it
depends on the per-review sentence-count distribution) that isn't built or
validated here.

So this script reports two different things, clearly separated:
- SENTENCE-level prevalence: raw AND Rogan-Gladen corrected (the validated
  quantity), compared against gold's own sentence-level prevalence.
- REVIEW-level "at least one" prevalence and multiplicity: RAW/uncorrected
  only, kept because it's what Phase A's report framing used and is still
  useful descriptively, but explicitly not corrected -- no validated method
  for it exists yet.

Usage: python src/01c_occurrence.py
Writes results/01_occurrence/silver_occurrence_sentence_level.csv (corrected),
results/01_occurrence/silver_occurrence_review_level.csv (raw only),
results/01_occurrence/silver_vs_gold_occurrence.csv (sentence-level replication check)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
from schema import MAIN_LABELS, SUB_LABELS, WEAK_MODEL_LABELS, LABEL_TO_CODE, SPARSE_GOLD_LABELS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "correction"))
from prevalence_correction import correct_labels  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
FEATURE_LABELS = [l for l in MAIN_LABELS_USED + SUB_LABELS if l not in WEAK_MODEL_LABELS]


def review_level_raw(df):
    rows = []
    for label in FEATURE_LABELS:
        col = f"count__{label}"
        present = df.loc[df[col] > 0, col]
        rows.append({
            "label": label,
            "review_prevalence_raw": round((df[col] > 0).mean(), 4),
            "multiplicity_mean_raw": round(present.mean(), 2) if len(present) else float("nan"),
            "multiplicity_median_raw": present.median() if len(present) else float("nan"),
        })
    return pd.DataFrame(rows)


def main():
    review_df = pd.read_parquet(RESULTS / "00_data_prep/silver_review_features.parquet")
    sent_df = pd.read_parquet(RESULTS / "00_data_prep/silver_sentence_sequences.parquet")
    confusion = pd.read_csv(RESULTS / "correction/confusion_matrix.csv")
    code_of_label = {l: LABEL_TO_CODE[l] for l in FEATURE_LABELS}

    print(f"n reviews: {len(review_df)}, n sentences: {len(sent_df)}")

    # sentence-level: the validated correction
    sent_df_prefixed = sent_df.rename(columns={l: f"count__{l}" for l in FEATURE_LABELS})
    corrected = correct_labels(sent_df_prefixed, "count__", confusion, code_of_label, n_boot=2000, seed=0)
    corrected = corrected.rename(columns={"p_observed": "sentence_prevalence_raw", "p_corrected": "sentence_prevalence_corrected"})
    corrected["sparse_gold_flag"] = corrected["label"].isin(SPARSE_GOLD_LABELS)
    corrected = corrected.sort_values("sentence_prevalence_corrected", ascending=False)
    corrected.to_csv(RESULTS / "01_occurrence/silver_occurrence_sentence_level.csv", index=False)
    print("\n=== sentence-level prevalence: raw vs. Rogan-Gladen corrected ===")
    print(corrected[["label", "n", "sentence_prevalence_raw", "sentence_prevalence_corrected",
                      "ci_lo", "ci_hi", "sparse_gold_flag"]].to_string(index=False))

    # review-level "at least one": raw only, no validated correction
    review_raw = review_level_raw(review_df)
    review_raw.to_csv(RESULTS / "01_occurrence/silver_occurrence_review_level.csv", index=False)
    print("\n=== review-level 'at least one' prevalence (RAW/uncorrected only) ===")
    print(review_raw.to_string(index=False))

    # replication check against gold, sentence-level (apples to apples)
    gold_sent = pd.read_csv(RESULTS / "00_data_prep/gold_sentence_sequences.csv")
    gold_sentence_prev = pd.DataFrame({
        "label": FEATURE_LABELS,
        "gold_sentence_prevalence": [round(gold_sent[l].mean(), 4) for l in FEATURE_LABELS],
    })
    cmp = corrected.merge(gold_sentence_prev, on="label")
    cmp["raw_gap_from_gold"] = (cmp["sentence_prevalence_raw"] - cmp["gold_sentence_prevalence"]).round(4)
    cmp["corrected_gap_from_gold"] = (cmp["sentence_prevalence_corrected"] - cmp["gold_sentence_prevalence"]).round(4)
    cmp["correction_moved_closer_to_gold"] = cmp["corrected_gap_from_gold"].abs() < cmp["raw_gap_from_gold"].abs()
    cmp["gold_within_corrected_ci"] = (cmp["gold_sentence_prevalence"] >= cmp["ci_lo"]) & (cmp["gold_sentence_prevalence"] <= cmp["ci_hi"])
    cmp = cmp[["label", "gold_sentence_prevalence", "sentence_prevalence_raw", "sentence_prevalence_corrected",
               "ci_lo", "ci_hi", "raw_gap_from_gold", "corrected_gap_from_gold",
               "correction_moved_closer_to_gold", "gold_within_corrected_ci", "sparse_gold_flag"]]
    cmp = cmp.sort_values("corrected_gap_from_gold", key=abs, ascending=False)
    cmp.to_csv(RESULTS / "01_occurrence/silver_vs_gold_occurrence.csv", index=False)

    print("\n=== silver (sentence-level, corrected) vs. gold replication check ===")
    print(cmp.to_string(index=False))
    print(f"\nmean |raw gap from gold|: {cmp['raw_gap_from_gold'].abs().mean():.4f}")
    print(f"mean |corrected gap from gold|: {cmp['corrected_gap_from_gold'].abs().mean():.4f}")
    print(f"correction moved closer to gold: {cmp['correction_moved_closer_to_gold'].sum()}/{len(cmp)} labels")
    print(f"gold prevalence within corrected 95% CI: {cmp['gold_within_corrected_ci'].sum()}/{len(cmp)} labels")
    print(f"\nwrote results to {RESULTS} (silver_occurrence_sentence_level.csv, "
          f"silver_occurrence_review_level.csv, silver_vs_gold_occurrence.csv)")


if __name__ == "__main__":
    main()

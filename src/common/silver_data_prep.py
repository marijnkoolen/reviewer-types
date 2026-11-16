"""Build the silver-set (model-predicted, 670K-review) review-level feature
table and sentence-level sequence table -- the Phase C counterpart of
common/data_prep.py, same output shape so every downstream script written
for gold works unchanged against either source (ANALYSIS_PLAN.md §1).

Two filters applied here, per ANALYSIS_PLAN.md §0b, baked into the data at
this stage rather than left to each downstream script to remember:
- Restricted to FICTION_GENRES. The confusion matrix (§0b) was built entirely
  on fiction-genre gold/test data, so RobBERT/MaChAmp's error rates on
  Non-fiction or missing-genre reviews are an untested domain shift -- those
  203,878 reviews (30% of the corpus) are dropped, not corrected.
- WEAK_MODEL_LABELS columns are dropped entirely (Reflection, Style--Structure,
  Content--Theme, Feelings) -- the classifier isn't reliable enough on these
  for any silver-set quantitative claim, corrected or not.

Correction itself (Rogan-Gladen, multiple imputation) is NOT applied in this
module -- it happens downstream, per analysis, using the same
src/correction/*.py functions Phase B already validated. This file only
produces the raw (uncorrected) tables, exactly like data_prep.py does for gold.

Output is parquet, not CSV: at ~467K reviews / ~4.7M sentences after
filtering, a CSV sentence table would be both slow to write/read and far
larger on disk than necessary.

Usage: python src/common/silver_data_prep.py
Writes results/00_data_prep/silver_review_features.parquet and results/00_data_prep/silver_sentence_sequences.parquet
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import MAIN_LABELS, SUB_LABELS, CODE_TO_LABEL, WEAK_MODEL_LABELS, FICTION_GENRES  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = ROOT / "metadata" / "review_metadata.tsv.gz"
PREDICTIONS_PATH = ROOT / "data" / "predctions-robbert-sentences-all.tsv.gz"
REVIEW_OUT = ROOT / "results" / "00_data_prep/silver_review_features.parquet"
SENTENCE_OUT = ROOT / "results" / "00_data_prep/silver_sentence_sequences.parquet"

MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
FEATURE_LABELS = [l for l in MAIN_LABELS_USED + SUB_LABELS if l not in WEAK_MODEL_LABELS]
CODE_COLS = [code for code, label in CODE_TO_LABEL.items() if label in FEATURE_LABELS]


def normalize_rating(series):
    val = pd.to_numeric(series, errors="coerce")
    return val.where(val != 0.0)


def main():
    print("loading metadata...")
    meta_full = pd.read_csv(METADATA_PATH, sep="\t", compression="gzip")
    meta = meta_full[meta_full["nur_genre"].isin(FICTION_GENRES)].copy()
    meta["review_date"] = meta["review_date"].fillna("").astype(str).str.slice(0, 10)
    meta["rating"] = normalize_rating(meta["rating"])
    n_dupe_ids = meta["review_id"].duplicated().sum()
    if n_dupe_ids:
        # 3 review_ids in the raw metadata are attached to two different rows
        # (one an exact duplicate, two with conflicting work_id/genre/author_id
        # -- a pre-existing upstream data issue, not introduced here). Can't
        # tell which row is "correct" for the conflicting pair, so keep the
        # first and drop the rest -- affects 6 of 670,924 rows either way,
        # not worth more care than that.
        print(f"  dropping {n_dupe_ids} duplicate review_id row(s) in metadata (keeping first)")
        meta = meta.drop_duplicates("review_id", keep="first")
    # user_id values are generated per platform (source_id), so the same string
    # on two platforms is two different people (920 ids / 27,286 reviews collide
    # across platforms). Qualify it so every reviewer-level groupby on user_id
    # downstream groups on (source_id, user_id) without further changes.
    meta["user_id"] = meta["source_id"].astype(str) + ":" + meta["user_id"].astype(str)
    valid_ids = set(meta["review_id"])
    print(f"  {len(meta)} fiction-genre reviews (of {len(meta_full)} total)")

    print("loading predictions...")
    usecols = ["review_id", "sent_num"] + CODE_COLS
    pred = pd.read_csv(PREDICTIONS_PATH, sep="\t", compression="gzip", usecols=usecols)
    pred = pred[pred["review_id"].isin(valid_ids)].copy()
    for c in CODE_COLS:
        pred[c] = pred[c].astype(np.int8)
    print(f"  {len(pred)} sentences in fiction-genre reviews")

    print("computing actual per-review sentence counts...")
    n_actual = pred.groupby("review_id")["sent_num"].transform("count")
    pred["num_sentences"] = n_actual
    pred["rel_position"] = np.where(
        pred["num_sentences"] > 1,
        (pred["sent_num"] - 1) / (pred["num_sentences"] - 1).clip(lower=1),
        0.0,
    )

    rename = {code: CODE_TO_LABEL[code] for code in CODE_COLS}
    sentence_table = pred.rename(columns=rename)[
        ["review_id", "sent_num", "num_sentences", "rel_position"] + FEATURE_LABELS
    ]
    SENTENCE_OUT.parent.mkdir(exist_ok=True, parents=True)
    sentence_table.to_parquet(SENTENCE_OUT, index=False)
    print(f"wrote {SENTENCE_OUT} ({len(sentence_table)} sentences)")

    print("building review-level feature table...")
    agg = sentence_table.groupby("review_id")[FEATURE_LABELS].sum()
    n_sentences = sentence_table.groupby("review_id")["sent_num"].count()
    review = meta.set_index("review_id").join(agg, how="inner").join(
        n_sentences.rename("num_sentences_actual"), how="inner"
    )
    review = review.rename(columns={"num_sentences": "num_sentences_meta"})
    review["num_sentences"] = review["num_sentences_actual"]
    for label in FEATURE_LABELS:
        review[f"count__{label}"] = review[label]
        review[f"prop__{label}"] = review[label] / review["num_sentences"]
        review = review.drop(columns=[label])
    review = review.reset_index()

    review.to_parquet(REVIEW_OUT, index=False)
    print(f"wrote {REVIEW_OUT} ({len(review)} reviews, {len(FEATURE_LABELS)} feature labels)")
    print(f"feature labels used: {FEATURE_LABELS}")


if __name__ == "__main__":
    main()

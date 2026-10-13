"""Build the gold-set review-level feature table and sentence-level sequence table
from data/review_sentence_annotations.tsv.

Review-level table: one row per review, with sentence counts and length-normalized
proportions per (sub)category, plus review metadata. This is what occurrence,
typology-clustering, genre/platform, book/author, reviewer, temporal, and ratings
analyses (ANALYSIS_PLAN.md sections 2, 6-11) are built on.

Sentence-level table: one row per sentence, preserving the full label matrix plus
derived fields (relative position in the review, whether the sentence carries no
label at all). This is what co-occurrence, contiguity, and order analyses
(sections 3-5) are built on.

Note: the file's own `num_sentences` metadata field disagrees with the actual
number of annotated sentence rows for 206/1152 reviews (off by 1-14, both
directions). `sent_num` itself is always a clean 1..N sequence with no gaps, so
N is taken from the actual row count per review, not the metadata field. The
metadata value is kept as `num_sentences_meta` for reference.

Usage: python src/common/data_prep.py
Writes results/00_data_prep/gold_review_features.csv and results/00_data_prep/gold_sentence_sequences.csv
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import MAIN_LABELS, SUB_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "data" / "review_sentence_annotations.tsv"
OUT_DIR = ROOT / "results"
REVIEW_OUT = OUT_DIR / "00_data_prep/gold_review_features.csv"
SENTENCE_OUT = OUT_DIR / "00_data_prep/gold_sentence_sequences.csv"

# All 21 label columns as they appear in the gold file (8 main incl. unused
# 'None', 13 sub). Used for reading; MAIN_LABELS_USED drops 'None' for the
# per-review count/proportion columns since it is never positive.
ALL_LABEL_COLS = MAIN_LABELS + SUB_LABELS
MAIN_LABELS_USED = [m for m in MAIN_LABELS if m != "None"]
FEATURE_LABELS = MAIN_LABELS_USED + SUB_LABELS  # 20 labels with real signal


def normalize_date(raw):
    """Both 'YYYY-MM-DD' and 'YYYY-MM-DD HH:MM:SS' appear; keep just the date part."""
    if not raw:
        return ""
    return raw[:10]


def normalize_rating(raw):
    """0.0 (and blank) are placeholders for 'no rating' / 'rating removed', not a real score."""
    if not raw:
        return ""
    val = float(raw)
    return "" if val == 0.0 else val


def main():
    with open(GOLD_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        all_rows = list(reader)

    rows_by_review = {}
    for row in all_rows:
        rows_by_review.setdefault(row["review_id"], []).append(row)

    n_sentences_actual = {rid: len(rows) for rid, rows in rows_by_review.items()}

    reviews = {}
    review_counts = {}
    review_unlabeled = {}
    sentence_rows = []

    for rid, rows in rows_by_review.items():
        n = n_sentences_actual[rid]
        first = rows[0]
        reviews[rid] = {
            "review_id": rid,
            "source_id": first["source_id"],
            "user_id": first["user_id"],
            "author_id": first.get("author_id", ""),
            "work_id": first["work_id"],
            "nur_genre": first["nur_genre"],
            "genre_id": first["genre_id"],
            "review_date": normalize_date(first["review_date"]),
            "rating": normalize_rating(first["rating"]),
            "review_num_words": int(first["review_num_words"]),
            "review_num_terms": int(first["review_num_terms"]),
            "num_sentences": n,
            "num_sentences_meta": int(first["num_sentences"]),
        }
        review_counts[rid] = {label: 0 for label in FEATURE_LABELS}
        review_unlabeled[rid] = 0

        for row in sorted(rows, key=lambda r: int(r["sent_num"])):
            active = [label for label in ALL_LABEL_COLS if row[label] == "1"]
            is_unlabeled = len(active) == 0
            if is_unlabeled:
                review_unlabeled[rid] += 1
            for label in active:
                if label in review_counts[rid]:
                    review_counts[rid][label] += 1

            sent_num = int(row["sent_num"])
            rel_position = (sent_num - 1) / (n - 1) if n > 1 else 0.0

            sentence_rows.append({
                "review_id": rid,
                "sent_num": sent_num,
                "sent_offset": row["sent_offset"],
                "sent_end": row["sent_end"],
                "num_sentences": n,
                "rel_position": round(rel_position, 4),
                "is_unlabeled": int(is_unlabeled),
                "active_labels": ";".join(active),
                **{label: row[label] for label in ALL_LABEL_COLS},
            })

    review_rows = []
    for rid, meta in reviews.items():
        n = meta["num_sentences"]
        row = dict(meta)
        row["n_unlabeled_sentences"] = review_unlabeled[rid]
        for label in FEATURE_LABELS:
            count = review_counts[rid][label]
            row[f"count__{label}"] = count
            row[f"prop__{label}"] = round(count / n, 4) if n else 0.0
        review_rows.append(row)

    REVIEW_OUT.parent.mkdir(exist_ok=True, parents=True)

    with open(REVIEW_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(review_rows[0].keys()))
        writer.writeheader()
        writer.writerows(review_rows)

    with open(SENTENCE_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sentence_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sentence_rows)

    n_mismatch = sum(1 for rid in reviews if reviews[rid]["num_sentences"] != reviews[rid]["num_sentences_meta"])
    print(f"wrote {REVIEW_OUT} ({len(review_rows)} reviews)")
    print(f"wrote {SENTENCE_OUT} ({len(sentence_rows)} sentences)")
    print(f"note: {n_mismatch} reviews had num_sentences_meta != actual annotated sentence count "
          f"(actual count used throughout)")


if __name__ == "__main__":
    main()

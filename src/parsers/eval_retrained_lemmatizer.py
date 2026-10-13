"""Score a retrained spaCy lemmatizer against its silver source, on held-out data.

The suspicion flags in compare_lemmas.py cannot cleanly evaluate a retrained
model: they judge a lemma partly by whether *other* tools propose it, and a
model trained on Alpino silver agrees with Alpino by construction. That inflates
its own scores and deflates the scores of tools that did not change (Frog's
"unattested" rate moves even though Frog did not).

This script avoids the circularity by asking one direct question on the reviews
the model never saw: how often does each spaCy variant reproduce the silver
lemma? Same tokens, same silver, only the model differs.

The dev split is recomputed exactly as build_lemma_trainset.py made it (same
seed, same fraction, split by review), so "held out" really means held out.

Usage:
  python src/parsers/eval_retrained_lemmatizer.py \
      --silver alpino --before spacy_lg --after spacy_retrained
Writes results/parser_comparison/retrained_lemma_eval.csv and prints examples.
"""
import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_fmt import OUT_DIR, iter_lines, read_tokens  # noqa: E402
from compare_util import coarse_pos, corpus_form_frequency, discover_tables, load_tables  # noqa: E402


def dev_docs(dev_frac, seed):
    """Reproduce build_lemma_trainset.py's split exactly."""
    docs = sorted({d for d, _, _ in iter_lines()})
    rng = random.Random(seed)
    rng.shuffle(docs)
    return set(docs[:max(1, int(len(docs) * dev_frac))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver", default="alpino")
    ap.add_argument("--before", default="spacy_lg")
    ap.add_argument("--after", default="spacy_retrained")
    ap.add_argument("--dev-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--examples", type=int, default=25)
    args = ap.parse_args()

    tools = [args.silver, args.before, args.after]
    tables = load_tables(tools=tools)
    freq = corpus_form_frequency(tables)
    dev = dev_docs(args.dev_frac, args.seed)

    rows = []
    fixed, broke = [], []
    stats = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # split -> bucket

    silver_tab = tables[args.silver]
    for key, spans in silver_tab.items():
        split = "dev" if key[0] in dev else "train"
        for span, s_row in spans.items():
            b_row = tables[args.before].get(key, {}).get(span)
            a_row = tables[args.after].get(key, {}).get(span)
            if b_row is None or a_row is None:
                continue
            pos = coarse_pos(s_row)
            if pos == "PUNCT":
                continue
            gold = s_row["lemma"]
            ok_b = b_row["lemma"] == gold
            ok_a = a_row["lemma"] == gold
            n = freq.get(s_row["form"].lower(), 0)
            bucket = "hapax" if n == 1 else ("rare (2-5)" if n <= 5 else "common (6+)")
            for label in ("all", bucket):
                stats[split][label][1] += 1
                stats[split][label][0] += 0  # placeholder, filled below
            stats[split][("before", "all")][1] += 1
            stats[split][("before", "all")][0] += ok_b
            stats[split][("after", "all")][1] += 1
            stats[split][("after", "all")][0] += ok_a
            stats[split][("before", bucket)][1] += 1
            stats[split][("before", bucket)][0] += ok_b
            stats[split][("after", bucket)][1] += 1
            stats[split][("after", bucket)][0] += ok_a

            if split == "dev" and ok_a and not ok_b and len(fixed) < 400:
                fixed.append({"form": s_row["form"], "silver": gold,
                              "before": b_row["lemma"], "after": a_row["lemma"],
                              "pos": pos, "form_freq": n, "doc": key[0]})
            if split == "dev" and ok_b and not ok_a and len(broke) < 400:
                broke.append({"form": s_row["form"], "silver": gold,
                              "before": b_row["lemma"], "after": a_row["lemma"],
                              "pos": pos, "form_freq": n, "doc": key[0]})

    for split in ("train", "dev"):
        for which in ("before", "after"):
            for bucket in ("all", "hapax", "rare (2-5)", "common (6+)"):
                hit, tot = stats[split].get((which, bucket), [0, 0])
                if tot:
                    rows.append({
                        "split": split,
                        "model": args.before if which == "before" else args.after,
                        "bucket": bucket, "tokens": tot,
                        "pct_matches_silver": round(100 * hit / tot, 2)})
    _write(OUT_DIR / "retrained_lemma_eval.csv", rows)
    _write(OUT_DIR / "retrained_lemma_fixed.csv", fixed)
    _write(OUT_DIR / "retrained_lemma_regressed.csv", broke)

    print(f"\n=== agreement with {args.silver} silver (PUNCT excluded) ===")
    print("dev = reviews held out of training; train = reviews it learned from\n")
    print(f"{'split':<7}{'bucket':<14}{args.before:>14}{args.after:>18}")
    for split in ("dev", "train"):
        for bucket in ("all", "hapax", "rare (2-5)", "common (6+)"):
            b = stats[split].get(("before", bucket))
            a = stats[split].get(("after", bucket))
            if not b or not b[1]:
                continue
            print(f"{split:<7}{bucket:<14}{100 * b[0] / b[1]:>13.2f}%"
                  f"{100 * a[0] / a[1]:>17.2f}%")
    print(f"\nheld-out tokens fixed by retraining:     {len(fixed)}+"
          if len(fixed) >= 400 else
          f"\nheld-out tokens fixed by retraining:     {len(fixed)}")
    print(f"held-out tokens regressed by retraining: {len(broke)}")

    print("\n=== examples fixed (held-out reviews) ===")
    for r in fixed[:args.examples]:
        print(f"  {r['form']:<22} {r['before']:<20} -> {r['after']:<20} "
              f"(silver: {r['silver']})")
    print("\n=== examples regressed (held-out reviews) ===")
    for r in broke[:10]:
        print(f"  {r['form']:<22} {r['before']:<20} -> {r['after']:<20} "
              f"(silver: {r['silver']})")


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

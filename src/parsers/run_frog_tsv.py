"""Parse a review-sentence TSV with Frog and emit a token TSV.

Same input, same output columns and the same sentence-id convention as
run_alpino_tsv.py, so the two can be benchmarked and diffed directly. Frog's
own sentence splitter is bypassed: each row is processed as one sentence and
any internal `eos` Frog proposes is ignored, because the point here is to hold
your existing split fixed.

Where the two differ in what they can emit:
  * Frog fills `morph` with MBMA's full morpheme decomposition
    (leesplezier -> [lees][plezier]); Alpino leaves it empty but encodes
    compound seams in the lemma itself (debuutroman -> debuut_roman).
  * Alpino fills `rel` and `frame` from the parse; Frog has no parser here
    (its CSI-DP is disabled) and leaves them empty.

Parallelism is by independent OS processes, one shard per core: python-frog
wraps a stateful C++ library that deadlocks inside a multiprocessing Pool.

Usage:
    python src/parsers/run_frog_tsv.py \\
        --input data/review_sentences/review_sentences-aggr.tsv.gz \\
        --shard 0 --n-shards 22 --out-dir results/frog_tsv
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_alpino_tsv import read_input, sanitise  # noqa: E402

COLUMNS = ["review_id", "sent_num", "tok_id", "begin", "end",
           "word", "lemma", "pos", "postag", "rel", "frame", "morph"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", default="results/frog_tsv")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--parser", action="store_true",
                    help="enable Frog's dependency parser (CSI-DP). Off by "
                         "default: it is Frog's slowest component and its "
                         "accuracy is well below Alpino's.")
    args = ap.parse_args()

    rows, n_reviews = read_input(args.input, args.shard, args.n_shards,
                                 limit=args.limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"shard {args.shard}/{args.n_shards}: {len(rows)} sentences from "
          f"{n_reviews} reviews", file=sys.stderr, flush=True)

    import frog
    tagger = frog.Frog(frog.FrogOptions(
        parser=args.parser, morph=True, lemma=True, mwu=False,
        ner=False, chunking=False,
    ))

    tok_path = out_dir / f"frog_tokens.shard{args.shard}.tsv"
    t0 = time.time()
    n_tokens = n_failed = 0
    with tok_path.open("w", encoding="utf-8", newline="") as tf:
        tw = csv.writer(tf, delimiter="\t", lineterminator="\n")
        tw.writerow(COLUMNS)
        for rid, snum, text in rows:
            try:
                out = tagger.process(sanitise(text))
            except Exception:
                n_failed += 1
                continue
            # Frog may propose internal sentence breaks; the fixed split wins,
            # so tokens are numbered continuously across the whole row.
            for tok_id, tok in enumerate(out, start=1):
                tw.writerow([
                    rid, snum, tok_id, tok_id - 1, tok_id,
                    tok.get("text", ""), tok.get("lemma", ""), "",
                    tok.get("pos", ""),
                    tok.get("parse_rel", "") if args.parser else "",
                    "", tok.get("morph", ""),
                ])
                n_tokens += 1
    elapsed = time.time() - t0

    meta = {
        "tool": "frog", "shard": args.shard, "n_shards": args.n_shards,
        "parser": args.parser, "sentences": len(rows), "tokens": n_tokens,
        "failed": n_failed, "seconds": round(elapsed, 1),
        "sents_per_sec": round(len(rows) / elapsed, 2) if elapsed else None,
        "tokens_per_sec": round(n_tokens / elapsed, 1) if elapsed else None,
    }
    (out_dir / f"frog_bench.shard{args.shard}.json").write_text(
        json.dumps(meta, indent=2))
    print(json.dumps(meta), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

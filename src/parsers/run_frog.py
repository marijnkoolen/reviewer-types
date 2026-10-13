"""Run Frog over the review sentences and write a common-format token table.

Frog is the only system in this comparison that produces a morphological
decomposition (MBMA), so its `morph` column is what compare_compounds.py reads.
Frog also re-runs its own sentence splitter, so one input line can come back as
several sentences; that is recorded in sent_id rather than suppressed, because
how often a tool overrides the existing sentence segmentation is one of the
things being measured.

Parallelism is by *sharding into separate OS processes*, not threads or a
multiprocessing Pool: python-frog wraps a stateful C++ library that deadlocks
when a Frog object is constructed inside Pool workers. Each shard is an
independent `python run_frog.py --shard i --n-shards N` writing its own TSV,
which is also how you would spread this across cores on a cluster. run_all.sh
launches the shards and concatenates them.

python-frog exposes no character offsets, so offsets come from
common_fmt.char_align().

Usage: python src/parsers/run_frog.py [--shard I --n-shards N] [--limit-docs N]
Writes results/parser_comparison/tokens_frog[.shardI].tsv and .meta.json
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_fmt import (  # noqa: E402
    OUT_DIR, char_align, iter_lines, open_writer, write_token,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit-docs", type=int, default=None)
    ap.add_argument("--doc-mode", action="store_true",
                    help="feed each review as one document so the tool's own\n                         sentence splitter runs; output goes to\n                         tokens_doc_<tool>.tsv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    by_doc = defaultdict(list)
    for doc_id, line_id, text in iter_lines(limit_docs=args.limit_docs, doc_mode=args.doc_mode):
        by_doc[doc_id].append((line_id, text))
    # Deterministic round-robin over sorted doc ids: every shard sees a mix of
    # long and short reviews, so shards finish at roughly the same time.
    docs = sorted(by_doc)
    mine = [d for i, d in enumerate(docs) if i % args.n_shards == args.shard]

    prefix = "tokens_doc_" if args.doc_mode else "tokens_"
    if args.out:
        out_path = Path(args.out)
    elif args.n_shards > 1:
        out_path = OUT_DIR / f"{prefix}frog.shard{args.shard}.tsv"
    else:
        out_path = OUT_DIR / f"{prefix}frog.tsv"

    print(f"frog shard {args.shard}/{args.n_shards}: {len(mine)} docs, "
          f"{sum(len(by_doc[d]) for d in mine)} lines -> {out_path.name}",
          file=sys.stderr, flush=True)

    import frog
    # Parser off: this comparison is about tokenisation, lemmas and morphology,
    # and Frog's dependency parser is by far its slowest component.
    tagger = frog.Frog(frog.FrogOptions(
        parser=False, morph=True, lemma=True, mwu=False,
        ner=False, chunking=False,
    ))

    fh, writer = open_writer(out_path)
    t0 = time.time()
    n_tokens = n_fail = 0

    for n_done, doc_id in enumerate(mine, start=1):
        for line_id, text in by_doc[doc_id]:
            try:
                out = tagger.process(text)
            except Exception as exc:
                print(f"  !! {doc_id}:{line_id} {type(exc).__name__}",
                      file=sys.stderr, flush=True)
                continue

            # Frog marks the last token of each of its own sentences with eos.
            sentences, current = [], []
            for tok in out:
                current.append(tok)
                if tok.get("eos"):
                    sentences.append(current)
                    current = []
            if current:
                sentences.append(current)

            spans, _ = char_align(text, [t["text"] for t in out])
            flat = 0
            for sent_id, sent in enumerate(sentences):
                for tok_id, tok in enumerate(sent, start=1):
                    start, end = spans[flat]
                    flat += 1
                    if start < 0:
                        n_fail += 1
                    write_token(
                        writer, doc_id, line_id, sent_id, tok_id, start, end,
                        tok["text"], tok["lemma"], "", tok.get("pos", ""), "",
                        tok.get("morph", ""), "ok" if start >= 0 else "fail",
                    )
                    n_tokens += 1
        if n_done % 25 == 0:
            print(f"  shard {args.shard}: {n_done}/{len(mine)} docs, "
                  f"{n_tokens} tokens, {time.time() - t0:.0f}s",
                  file=sys.stderr, flush=True)

    fh.close()
    elapsed = time.time() - t0
    meta = {
        "tool": "frog", "shard": args.shard, "n_shards": args.n_shards,
        "docs": len(mine), "tokens": n_tokens, "align_fail": n_fail,
        "seconds": round(elapsed, 1),
        "tokens_per_sec": round(n_tokens / elapsed, 1) if elapsed else None,
    }
    Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

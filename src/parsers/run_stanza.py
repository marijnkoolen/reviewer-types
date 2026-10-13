"""Run Stanza's Dutch pipeline over the review sentences.

Stanza exposes character offsets natively, so no alignment reconstruction is
needed here. Each input line is submitted as its own Document, which keeps
doc_id/line_id attribution exact while still letting Stanza's sentence splitter
re-split a line (recorded in sent_id).

Words rather than tokens are emitted: Dutch UD has no multi-word tokens in
practice, but if one appears, every word inherits the enclosing token's offsets
and the shared span makes the token-boundary comparison treat it correctly.

Usage: python src/parsers/run_stanza.py [--device mps|cpu|cuda] [--limit-docs N]
Writes results/parser_comparison/tokens_stanza.tsv and .meta.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_fmt import OUT_DIR, load_input, open_writer, write_token  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu",
                    help="cpu, mps or cuda; mps is often slower than cpu for "
                         "these small models")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit-docs", type=int, default=None)
    ap.add_argument("--doc-mode", action="store_true",
                    help="feed each review as one document so the tool's own\n                         sentence splitter runs; output goes to\n                         tokens_doc_<tool>.tsv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import stanza
    from stanza.models.common.doc import Document

    data = load_input(limit_docs=args.limit_docs, doc_mode=args.doc_mode)
    print(f"stanza: {len(data)} lines, device={args.device}", file=sys.stderr)

    nlp = stanza.Pipeline(
        "nl", processors="tokenize,pos,lemma", download_method=None,
        device=args.device, verbose=False,
    )

    prefix = "tokens_doc_" if args.doc_mode else "tokens_"
    out_path = Path(args.out or OUT_DIR / f"{prefix}stanza.tsv")
    fh, writer = open_writer(out_path)
    t0 = time.time()
    n_tokens = 0

    for i in range(0, len(data), args.batch_size):
        chunk = data[i:i + args.batch_size]
        docs = [Document([], text=text) for _, _, text in chunk]
        parsed = nlp.bulk_process(docs)
        for (doc_id, line_id, _text), doc in zip(chunk, parsed):
            for sent_id, sent in enumerate(doc.sentences):
                for tok_id, word in enumerate(sent.words, start=1):
                    parent = word.parent  # the Token, which carries the offsets
                    start = getattr(parent, "start_char", None)
                    end = getattr(parent, "end_char", None)
                    ok = start is not None and end is not None
                    write_token(
                        writer, doc_id, line_id, sent_id, tok_id,
                        start if ok else -1, end if ok else -1,
                        word.text, word.lemma or "", word.upos or "",
                        word.xpos or "", word.feats or "", "",
                        "ok" if ok else "fail",
                    )
                    n_tokens += 1
        if (i // args.batch_size) % 20 == 0:
            print(f"  {i + len(chunk)}/{len(data)} lines, {n_tokens} tokens, "
                  f"{time.time() - t0:.0f}s", file=sys.stderr)

    fh.close()
    elapsed = time.time() - t0
    meta = {
        "tool": "stanza", "device": args.device, "lines": len(data),
        "tokens": n_tokens, "seconds": round(elapsed, 1),
        "tokens_per_sec": round(n_tokens / elapsed, 1) if elapsed else None,
    }
    Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()

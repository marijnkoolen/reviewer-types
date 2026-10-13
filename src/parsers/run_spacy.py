"""Run a spaCy Dutch model over the review sentences.

spaCy is in this comparison as the incumbent whose lemmas we are trying to
improve, and as the target of build_lemma_trainset.py. Note that the
nl_core_news_* v3.7 pipelines lemmatise with an EditTreeLemmatizer (a trained
component), not a lookup table -- so its OOV failures are learned suffix-edit
rules misfiring, which is what the retraining path is meant to fix.

Usage: python src/parsers/run_spacy.py [--model nl_core_news_lg] [--limit-docs N]
Writes results/parser_comparison/tokens_spacy_<model>.tsv and .meta.json
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_fmt import OUT_DIR, load_input, open_writer, write_token  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nl_core_news_lg")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-process", type=int, default=1)
    ap.add_argument("--limit-docs", type=int, default=None)
    ap.add_argument("--doc-mode", action="store_true",
                    help="feed each review as one document so the tool's own\n                         sentence splitter runs; output goes to\n                         tokens_doc_<tool>.tsv")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    warnings.filterwarnings("ignore", message=r".*\[W095\].*")
    import spacy

    data = load_input(limit_docs=args.limit_docs, doc_mode=args.doc_mode)
    short = args.model.replace("nl_core_news_", "")
    prefix = "tokens_doc_" if args.doc_mode else "tokens_"
    out_path = Path(args.out or OUT_DIR / f"{prefix}spacy_{short}.tsv")
    print(f"spacy {args.model}: {len(data)} lines", file=sys.stderr)

    # The parser is kept because it supplies sentence boundaries; NER is not
    # needed by any comparison and is the most expensive remaining component.
    nlp = spacy.load(args.model, exclude=["ner"])

    # A retrained lemmatizer pipeline has no parser or senter, so doc.sents
    # raises. Falling back to one sentence per doc is correct in line mode
    # (the input is already one sentence) and is recorded in the metadata so a
    # sentence-splitting comparison cannot silently read it as "never splits".
    has_sents = bool({"parser", "senter", "sentencizer"} & set(nlp.pipe_names))

    fh, writer = open_writer(out_path)
    t0 = time.time()
    n_tokens = 0
    texts = [text for _, _, text in data]
    keys = [(doc_id, line_id) for doc_id, line_id, _ in data]

    for (doc_id, line_id), doc in zip(
            keys, nlp.pipe(texts, batch_size=args.batch_size,
                           n_process=args.n_process)):
        sentences = doc.sents if has_sents else [doc[:]]
        for sent_id, sent in enumerate(sentences):
            for tok_id, tok in enumerate(sent, start=1):
                write_token(
                    writer, doc_id, line_id, sent_id, tok_id,
                    tok.idx, tok.idx + len(tok.text),
                    tok.text, tok.lemma_, tok.pos_, tok.tag_,
                    str(tok.morph), "", "ok",
                )
                n_tokens += 1

    fh.close()
    elapsed = time.time() - t0
    meta = {
        "tool": f"spacy_{short}", "model": args.model, "lines": len(data),
        "sentence_boundaries": has_sents,
        "tokens": n_tokens, "seconds": round(elapsed, 1),
        "tokens_per_sec": round(n_tokens / elapsed, 1) if elapsed else None,
    }
    Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()

"""Run Alpino over the review sentences and write a common-format token table.

Two execution modes, same code path otherwise:

  --alpino-home PATH   a native Alpino install (Linux x86_64). This is the mode
                       to use on the GPU VMs: Alpino is CPU-only, so throughput
                       there comes from running one shard per core.
  --docker IMAGE       the image built by docker/Dockerfile.alpino, for machines
                       without a native install (e.g. Apple Silicon, where the
                       x86_64 binary runs emulated and is several times slower).

Pipeline per shard: raw lines -> Alpino's own tokeniser -> `Alpino -notk -parse`
with XML treebank output -> one <n>.xml per input line -> leaf nodes.

tokenize_no_breaks.sh is the default tokeniser because it guarantees exactly one
output line per input line, which keeps the (doc_id, line_id) mapping exact.
The consequence is that Alpino's *sentence splitter* is not exercised, so
Alpino contributes no sentence re-splitting figure to the tokenisation
comparison; pass --tokenizer tokenize.sh to measure that instead, at the cost of
the 1:1 line mapping.

Alpino XML leaves carry `lemma`, `postag` (full CGN), `pt` (coarse) and `begin`
(token index). Character offsets are reconstructed with common_fmt.char_align()
so that Alpino tokens join against the other tools' tokens.

Usage:
  python src/parsers/run_alpino.py --docker alpino-impfic:latest
  python src/parsers/run_alpino.py --alpino-home /opt/Alpino --shard 0 --n-shards 11
Writes results/parser_comparison/tokens_alpino[.shardI].tsv and .meta.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_fmt import (  # noqa: E402
    OUT_DIR, char_align, iter_lines, load_documents, open_writer, write_token,
)

# In document mode one input line is a whole review, and tokenize.sh splits it
# into several output lines -- which destroys the line-number mapping Alpino
# uses to name its XML. A sentinel line between documents restores it: it has no
# internal punctuation, so the tokeniser emits it as exactly one line (verified),
# and the output lines between two sentinels are one document's sentences.
SENTINEL = "ZZSPLITZZ ."


def build_script(tokenizer, user_max, max_len):
    """Shell run inside the container / on the native install.

    Kept as one script so that native and Docker modes execute byte-identical
    commands and cannot drift apart.
    """
    return f"""
set -e
cd /work
"$ALPINO_HOME/Tokenization/{tokenizer}" < raw.txt > tok.txt 2> tok.log
mkdir -p xml
"$ALPINO_HOME/bin/Alpino" -notk -veryfast \
    -flag treebank xml \
    -flag user_max {user_max} \
    -flag max_sentence_length {max_len} \
    end_hook=xml -parse < tok.txt > parse.out 2> parse.log
"""


def run_alpino(work, args):
    """Execute the Alpino pipeline over work/raw.txt, leaving work/xml/*.xml."""
    script = build_script(args.tokenizer, args.user_max, args.max_length)
    if args.docker:
        cmd = ["docker", "run", "--rm", "--platform", "linux/amd64",
               "-v", f"{work}:/work", args.docker, "bash", "-c", script]
    else:
        env = dict(os.environ, ALPINO_HOME=args.alpino_home)
        env.setdefault("LC_ALL", "en_US.utf8")
        return subprocess.run(["bash", "-c", script.replace("cd /work", f"cd {work}")],
                              env=env, capture_output=True, text=True)
    return subprocess.run(cmd, capture_output=True, text=True)


def parse_xml(path):
    """Return [(word, lemma, postag, pt)] for one sentence, in token order."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    leaves = {}
    for node in root.iter("node"):
        word = node.get("word")
        if word is None:
            continue
        begin = int(node.get("begin", -1))
        # Several nodes can share a begin index (index/co-indexed traces);
        # the first leaf encountered is the actual surface token.
        leaves.setdefault(begin, (
            word, node.get("lemma") or node.get("root") or word,
            node.get("postag") or "", node.get("pt") or "",
        ))
    return [leaves[b] for b in sorted(leaves)]


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--docker", metavar="IMAGE")
    mode.add_argument("--alpino-home", metavar="PATH")
    ap.add_argument("--tokenizer", default="tokenize_no_breaks.sh",
                    choices=["tokenize_no_breaks.sh", "tokenize.sh"])
    ap.add_argument("--user-max", type=int, default=60000,
                    help="per-sentence parse timeout in ms; Alpino returns a "
                         "partial parse rather than hanging")
    ap.add_argument("--max-length", type=int, default=100)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit-docs", type=int, default=None)
    ap.add_argument("--doc-mode", action="store_true",
                    help="feed each review as one document so Alpino's own "
                         "sentence splitter runs; implies --tokenizer "
                         "tokenize.sh and writes tokens_doc_alpino.tsv")
    ap.add_argument("--keep-work", action="store_true",
                    help="keep the Alpino XML instead of deleting it; the XML "
                         "carries the full dependency structure this table drops")
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.doc_mode:
        # Sentence splitting is the whole point here, so the no-breaks tokeniser
        # (which cannot split) would make the run meaningless.
        args.tokenizer = "tokenize.sh"
        all_docs = load_documents(limit_docs=args.limit_docs)
        mine = [d for i, d in enumerate(all_docs) if i % args.n_shards == args.shard]
        items = [(doc_id, 0, text) for doc_id, text, _ in mine]
    else:
        by_doc = defaultdict(list)
        for doc_id, line_id, text in iter_lines(limit_docs=args.limit_docs):
            by_doc[doc_id].append((line_id, text))
        docs = sorted(by_doc)
        mine = [d for i, d in enumerate(docs) if i % args.n_shards == args.shard]
        items = [(d, lid, txt) for d in mine for lid, txt in by_doc[d]]

    prefix = "tokens_doc_" if args.doc_mode else "tokens_"
    if args.out:
        out_path = Path(args.out)
    elif args.n_shards > 1:
        out_path = OUT_DIR / f"{prefix}alpino.shard{args.shard}.tsv"
    else:
        out_path = OUT_DIR / f"{prefix}alpino.tsv"

    work = Path(args.work_dir or tempfile.mkdtemp(prefix="alpino_"))
    work.mkdir(parents=True, exist_ok=True)
    # Alpino names output XML by 1-based input line number, so raw.txt line
    # order is the only thing tying results back to (doc_id, line_id).
    if args.doc_mode:
        raw = "".join(text.replace("\n", " ") + "\n" + SENTINEL + "\n"
                      for _, _, text in items)
    else:
        raw = "".join(text.replace("\n", " ") + "\n" for _, _, text in items)
    (work / "raw.txt").write_text(raw, encoding="utf-8")

    print(f"alpino shard {args.shard}/{args.n_shards}: {len(items)} lines, "
          f"work={work}", file=sys.stderr, flush=True)

    t0 = time.time()
    proc = run_alpino(str(work), args)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(proc.stdout[-2000:], file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"alpino failed (exit {proc.returncode})")

    tok_lines = (work / "tok.txt").read_text(encoding="utf-8").split("\n")
    while tok_lines and not tok_lines[-1]:
        tok_lines.pop()
    fh, writer = open_writer(out_path)
    n_tokens = n_missing = n_fail = 0

    if args.doc_mode:
        # Alpino names xml/<n>.xml by line number in tok.txt, sentinels
        # included, so grouping tok.txt lines on the sentinel gives each
        # document's XML indices directly.
        groups, current = [], []
        for idx, line in enumerate(tok_lines, start=1):
            if line.split() == SENTINEL.split():
                groups.append(current)
                current = []
            else:
                current.append(idx)
        if current:
            groups.append(current)
        if len(groups) < len(items):
            print(f"  !! {len(groups)} sentinel groups for {len(items)} docs; "
                  f"trailing documents will be skipped", file=sys.stderr)

        for k, (doc_id, line_id, text) in enumerate(items):
            idxs = groups[k] if k < len(groups) else []
            sents = []
            for idx in idxs:
                xml_path = work / "xml" / f"{idx}.xml"
                leaves = parse_xml(xml_path) if xml_path.exists() else None
                if not leaves:
                    n_missing += 1
                    continue
                sents.append(leaves)
            # Offsets are resolved once across the whole document, so sentence
            # boundaries land at real character positions in the joined text.
            spans, _ = char_align(text, [w for s in sents for w, _, _, _ in s])
            flat = 0
            for sent_id, leaves in enumerate(sents):
                for tok_id, (word, lemma, postag, pt) in enumerate(leaves, 1):
                    start, end = spans[flat]
                    flat += 1
                    if start < 0:
                        n_fail += 1
                    write_token(writer, doc_id, line_id, sent_id, tok_id,
                                start, end, word, lemma, "", postag, "", "",
                                "ok" if start >= 0 else "fail")
                    n_tokens += 1
    else:
        for idx, (doc_id, line_id, text) in enumerate(items, start=1):
            xml_path = work / "xml" / f"{idx}.xml"
            if not xml_path.exists():
                n_missing += 1
                continue
            leaves = parse_xml(xml_path)
            if not leaves:
                n_missing += 1
                continue
            spans, _ = char_align(text, [w for w, _, _, _ in leaves])
            for tok_id, ((word, lemma, postag, pt), (start, end)) in enumerate(
                    zip(leaves, spans), start=1):
                if start < 0:
                    n_fail += 1
                write_token(writer, doc_id, line_id, 0, tok_id, start, end,
                            word, lemma, "", postag, "", "",
                            "ok" if start >= 0 else "fail")
                n_tokens += 1
    fh.close()

    meta = {
        "tool": "alpino", "mode": "docker" if args.docker else "native",
        "doc_mode": args.doc_mode,
        "shard": args.shard, "n_shards": args.n_shards,
        "tokenizer": args.tokenizer, "lines": len(items),
        "tokens": n_tokens, "lines_missing": n_missing, "align_fail": n_fail,
        "seconds": round(elapsed, 1),
        "tokens_per_sec": round(n_tokens / elapsed, 1) if elapsed else None,
        "sents_per_sec": round(len(items) / elapsed, 2) if elapsed else None,
    }
    Path(str(out_path) + ".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), file=sys.stderr, flush=True)

    if not args.keep_work and not args.work_dir:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()

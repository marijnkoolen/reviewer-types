"""Parse a review-sentence TSV with Alpino and emit a token TSV.

Input is the fixed sentence split you already have:

    <review_id>\\t<sent_num>\\t<sent_text>

one sentence per row, optionally gzipped. That split is taken as given -- Alpino
is run with `tokenize_no_breaks.sh`, which tokenises without ever splitting a
line, so the sentence segmentation you bring is the segmentation you get back.

Output naming uses Alpino's own sentence-id convention. An input line of the
form `<id>|<tokens>` makes Alpino name its XML `<id>.xml`, so each sentence is
addressed as `<review_id>-<sent_num>` throughout: in the XML filenames (with
--keep-xml) and as the first two columns of the TSV. Parse `<review_id>` back
out with a single rsplit on "-", since sent_num is always numeric.

Alpino has no built-in CSV/TSV writer -- the CONLLU directory in the binary
distribution is an empty build artifact, and `end_hook=dependencies` emits
pipe-separated dependency *triples*, which is a different data model from a
token table. So XML is produced and converted here.

Token TSV columns:
    review_id sent_num tok_id begin end word lemma pos postag rel head_tok_id
    frame

`rel` + `head_tok_id` give a CoNLL-shaped dependency tree (head_tok_id 0 = ROOT).
Alpino does not emit token-to-token dependencies alongside XML -- `end_hook`
takes one value, and `end_hook=dependencies` writes no XML -- so heads are
reconstructed here by percolating them down the constituent tree.

WHAT THAT RECONSTRUCTION GUARANTEES. Measured against Alpino's own triples on
830 tokens (validate_heads.py): **96.5% head agreement**. The residual is not
random:
  ~1.6%  relative and wh pronouns (rhd/whd). Alpino resolves these through
         co-indexation, linking the fronted phrase to a trace inside the clause
         that carries the real function (su/obj1/mod); this conversion attaches
         them structurally and reports `rhd`/`whd` as the label.
  ~1.1%  discourse structure (dlink, dp, nucl) -- a long tail, no one pattern.
Clause structure, coordination, complementation and punctuation match.

So head_tok_id is sound for ordinary clause-internal dependency work and
systematically coarse on one construction. If that matters, either run a second
pass with end_hook=dependencies (exact, but doubles Alpino time) or keep the
XML with --keep-xml and redo the conversion later without re-parsing.
plus a per-sentence parse-quality file recording Alpino's `cats` and `skips`
(skips > 0 means Alpino could not build a single spanning analysis).

Work is done in chunks (default 1000 sentences) rather than one pass per shard.
That buys three things on a multi-day corpus run: output appears incrementally
instead of only at the end, progress is visible per chunk, and the job is
**resumable** -- re-running the same command picks up at the first unfinished
chunk. It also bounds disk, since only one chunk's XML exists at a time.

Resume is byte-exact rather than best-effort. After each chunk the output files
are flushed and fsynced and their sizes recorded in
`alpino_state.shard<i>.json`. On restart the outputs are truncated back to
those recorded sizes before appending, so a chunk that was half-written when
the process died leaves no partial rows behind. Use --restart to discard
progress and start over.

Usage (one shard per core; 22 on a dual-A10 VM):
    python src/parsers/run_alpino_tsv.py --alpino-home /opt/Alpino \\
        --input data/review_sentences/review_sentences-aggr.tsv.gz \\
        --shard 0 --n-shards 22 --out-dir results/alpino_tsv
Local smoke test via the Docker image:
    python src/parsers/run_alpino_tsv.py --docker alpino-impfic:latest ...
"""
import argparse
import csv
import gzip
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

COLUMNS = ["review_id", "sent_num", "tok_id", "begin", "end",
           "word", "lemma", "pos", "postag", "rel", "head_tok_id", "frame"]

# Order in which a constituent's children are considered its head. Alpino's XML
# is a hybrid constituency/dependency structure: `rel` gives each node's function
# within its parent, but token-to-token dependencies only emerge once the head
# child of every constituent is identified and heads are percolated down.
# `hd` covers most cases; the rest handle coordination (crd), subordinating
# conjunctions (cmp with the clause in body), relative and wh clauses (rhd/whd,
# whose lexical head sits inside body), discourse units (nucl/dp) and
# multiword units (mwp).
HEAD_PRIORITY = ("hd", "crd", "cmp", "rhd", "whd", "nucl", "body",
                 "dlink", "dp", "mwp", "svp")
# `body` is ranked last deliberately, and this was measured, not assumed.
# Promoting it above rhd/whd -- on the theory that a relative clause is headed
# by its verb -- drops agreement with Alpino's own triples from 96.5% to 95.1%:
# it fixes the 13 fronted-phrase attachments but breaks 17 others, because
# Alpino really does hang `body` off the rhd/cmp head as a dependent.
#
# The residual relative/wh mismatches are not a head-ordering problem. Alpino
# resolves the fronted phrase through co-indexation, linking it to a trace
# inside the clause that carries the real function (su/obj1/mod). Reproducing
# that means implementing index resolution; see the note in the module
# docstring on what this conversion does and does not guarantee.


def _kids(node):
    return [c for c in node if c.tag == "node"]


def _is_punct(node):
    """Punctuation never heads anything.

    Alpino attaches punctuation high in the tree -- often as a direct daughter
    of `top` with a low `begin` -- so any "leftmost child" fallback will pick a
    comma as the sentence head unless punctuation is excluded outright. Alpino's
    own dependency output gives punctuation no head at all (ROOT), which is what
    resolve_heads() reproduces.
    """
    return (node.get("pos") == "punct"
            or (node.get("postag") or "").startswith("LET"))


def _head_leaf(node, memo):
    """Token index of the lexical head of this subtree, or None if it has none.

    Nodes with neither `word` nor `cat` are empty co-indexation traces; they
    carry no surface token, so they can never be a head.
    """
    key = id(node)
    if key in memo:
        return memo[key]
    result = None
    if node.get("word") is not None:
        result = int(node.get("begin", -1))
    elif node.get("cat") is not None:
        kids = [c for c in _kids(node) if not _is_punct(c)]
        by_rel = {}
        for c in kids:
            by_rel.setdefault(c.get("rel"), c)
        for rel in HEAD_PRIORITY:
            c = by_rel.get(rel)
            if c is not None:
                result = _head_leaf(c, memo)
                if result is not None:
                    break
        if result is None:
            # No recognised head child -- `top` is the common case, whose
            # daughters are all `--`. Prefer a phrasal child over a bare leaf,
            # since the sentence's head lives inside the clause, not beside it.
            phrasal = [c for c in kids if c.get("cat") is not None]
            for c in sorted(phrasal or kids,
                            key=lambda n: int(n.get("begin", 0))):
                result = _head_leaf(c, memo)
                if result is not None:
                    break
    memo[key] = result
    return result


def resolve_heads(root):
    """{token index: (head token index, rel)} for one sentence.

    The head token of the sentence gets head index -1, written as head_tok_id 0
    in the TSV (the CoNLL convention for ROOT).
    """
    # The document element is <alpino_ds>, which wraps the actual tree in a
    # single <node rel="top">. Walking from the wrapper makes head resolution
    # return None at the top, and every token on the head path then inherits
    # None instead of ROOT -- which silently mis-heads the main verb of every
    # sentence. Descend to the tree first.
    if root.tag != "node":
        root = root.find("node")
        if root is None:
            return {}
    memo = {}
    heads = {}

    def walk(node, parent_head):
        own = _head_leaf(node, memo)
        if node.get("word") is not None:
            idx = int(node.get("begin", -1))
            # Punctuation attaches to ROOT, matching Alpino's own triples.
            head = -1 if (_is_punct(node) or parent_head is None) else parent_head
            heads[idx] = (head, node.get("rel") or "")
            return
        for c in _kids(node):
            # A child that carries this constituent's head inherits the
            # constituent's own head; every other child attaches to it.
            child_head = parent_head if _head_leaf(c, memo) == own else own
            # `own` is None when a constituent has no lexical head at all --
            # e.g. every daughter is punctuation or an empty co-indexation
            # trace. Such a token has nothing to attach to, so it becomes ROOT
            # rather than propagating None down the whole head path.
            walk(c, -1 if child_head is None else child_head)

    walk(root, -1)
    return heads


def read_input(path, shard, n_shards, limit=None):
    """Yield (review_id, sent_num, text) for this shard, streaming.

    Sharding is round-robin over reviews, not sentences, so a review's
    sentences stay in one shard and one output file.

    The file is streamed and non-shard rows are dropped as they are read. That
    matters at corpus scale: materialising all rows and filtering afterwards
    means every one of N concurrent shards holds the whole corpus in memory
    (~6.9M rows is on the order of a gigabyte per process, times N). Only the
    review-id-to-index map is kept whole, which is far smaller.

    With `limit`, reading stops as soon as this shard has enough rows, so a
    small test run does not pay for decompressing the entire file. The returned
    review count is then only the number seen so far, which is all the progress
    line uses it for.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    review_idx = {}
    n_reviews = 0
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            rid = parts[0]
            idx = review_idx.get(rid)
            if idx is None:
                idx = n_reviews
                review_idx[rid] = idx
                n_reviews += 1
            if idx % n_shards != shard:
                continue
            rows.append((rid, parts[1], "\t".join(parts[2:])))
            if limit and len(rows) >= limit:
                break
    return rows, n_reviews


def sanitise(text):
    """Alpino splits an input line on the first '|', so a pipe inside the
    sentence would truncate it. Nothing else about the text is touched."""
    return text.replace("|", "/").replace("\t", " ").strip()


SCRIPT = """
set -e
cd {work}
"$ALPINO_HOME/Tokenization/tokenize_no_breaks.sh" < raw.txt > tok.txt 2> tok.log
paste -d'|' ids.txt tok.txt > tok_id.txt
mkdir -p xml
"$ALPINO_HOME/bin/Alpino" -notk {speed} \
    -flag treebank xml \
    -flag user_max {user_max} \
    -flag max_sentence_length {max_len} \
    end_hook=xml -parse < tok_id.txt > parse.out 2> parse.log
"""


_CHILD = None
_CONTAINER = None


def _stop_child(signum, frame):
    """Stop Alpino before exiting, by whichever route actually works.

    Terminating our immediate child is not enough in either mode:

      docker  the child is the `docker run` client. Killing it leaves the
              container running -- this was verified, not assumed -- so the
              container is named and explicitly `docker kill`ed.
      native  the child is `bash -c`, whose own Alpino child survives it. The
              child is therefore started in its own session and the whole
              process group is signalled.

    SIGKILL to this process cannot be trapped and still orphans Alpino; see the
    cleanup section in docker/README.md for recovering from that.
    """
    global _CHILD, _CONTAINER
    child, _CHILD = _CHILD, None
    name, _CONTAINER = _CONTAINER, None
    if name:
        subprocess.run(["docker", "kill", name], capture_output=True)
    if child is not None and child.poll() is None:
        try:
            os.killpg(os.getpgid(child.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
    raise SystemExit(128 + signum)


def run(work, args):
    """Execute the Alpino pipeline over work/raw.txt, leaving work/xml/*.xml."""
    global _CHILD, _CONTAINER
    script = SCRIPT.format(work="/work" if args.docker else work,
                           speed=args.speed, user_max=args.user_max,
                           max_len=args.max_length)
    if args.docker:
        # A deterministic name is what makes the container killable from the
        # signal handler; without it only the client pid is reachable.
        _CONTAINER = f"alpino_tsv_s{args.shard}_{os.getpid()}"
        cmd = ["docker", "run", "--rm", "--name", _CONTAINER,
               "--platform", "linux/amd64",
               "-v", f"{work}:/work", args.docker, "bash", "-c", script]
        env = None
    else:
        _CONTAINER = None
        cmd = ["bash", "-c", script]
        env = dict(os.environ, ALPINO_HOME=args.alpino_home)
        env.setdefault("LC_ALL", "en_US.utf8")
    # Popen, not subprocess.run: run() exposes no handle, so a signal arriving
    # mid-parse could not reach Alpino. start_new_session puts the child in its
    # own process group so the group can be signalled as a unit.
    _CHILD = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True,
                              start_new_session=True)
    out, err = _CHILD.communicate()
    rc = _CHILD.returncode
    _CHILD = None
    _CONTAINER = None
    return subprocess.CompletedProcess(cmd, rc, out, err)


def parse_xml(path):
    """Return (leaves, cats, skips) for one sentence.

    Leaves are keyed by `begin` (the token index) and sorted by it: Alpino's XML
    is a tree, so document order is not surface order, and punctuation in
    particular floats to the front.
    """
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return None, None, None
    parser = root.find("parser")
    cats = parser.get("cats") if parser is not None else None
    skips = parser.get("skips") if parser is not None else None
    heads = resolve_heads(root)
    leaves = {}
    for node in root.iter("node"):
        word = node.get("word")
        if word is None:
            continue
        begin = int(node.get("begin", -1))
        head_idx, _ = heads.get(begin, (-1, ""))
        if head_idx is None:
            head_idx = -1
        leaves.setdefault(begin, (
            word,
            node.get("lemma") or node.get("root") or word,
            node.get("pos") or "",
            node.get("postag") or "",
            node.get("rel") or "",
            head_idx + 1,  # 1-based token id; 0 means ROOT
            node.get("frame") or "",
            int(node.get("end", begin + 1)),
        ))
    return [leaves[b] for b in sorted(leaves)], cats, skips


def state_path(out_dir, shard):
    return Path(out_dir) / f"alpino_state.shard{shard}.json"


def load_state(out_dir, shard, signature):
    """Return the saved progress for this shard, or a fresh state.

    The signature covers every setting that changes what a chunk *is* (input,
    sharding, chunk size). If any of them differ, resuming would interleave
    output from two different partitions of the data, so the run is refused
    rather than silently corrupting the file.
    """
    path = state_path(out_dir, shard)
    if not path.exists():
        return {"signature": signature, "chunks_done": 0,
                "tok_bytes": 0, "qual_bytes": 0, "tokens": 0,
                "no_parse": 0, "skips": 0, "seconds": 0.0}
    state = json.loads(path.read_text())
    if state.get("signature") != signature:
        raise SystemExit(
            f"{path} was written for a different configuration:\n"
            f"  saved:   {state.get('signature')}\n"
            f"  current: {signature}\n"
            "Re-run with --restart to discard it, or restore the old settings.")
    return state


def save_state(out_dir, shard, state):
    """Atomic state write: a crash mid-save must not leave a torn file."""
    path = state_path(out_dir, shard)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, path)


def run_chunk(chunk, args, out_dir, shard_tag):
    """Parse one chunk in its own work dir. Returns (parsed, work_dir)."""
    work = Path(tempfile.mkdtemp(prefix=f"alpino_tsv_{shard_tag}_"))
    ids, texts = [], []
    for rid, snum, text in chunk:
        ids.append(f"{sanitise(rid)}-{snum}")
        texts.append(sanitise(text))
    (work / "raw.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")
    (work / "ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    proc = run(str(work), args)
    if proc.returncode != 0:
        print(proc.stdout[-1500:], file=sys.stderr)
        print(proc.stderr[-1500:], file=sys.stderr)
        shutil.rmtree(work, ignore_errors=True)
        raise SystemExit(f"alpino failed on chunk (exit {proc.returncode})")
    return ids, work


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--docker", metavar="IMAGE")
    mode.add_argument("--alpino-home", metavar="PATH")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", default="results/alpino_tsv")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--speed", default="-veryfast",
                    choices=["-veryfast", "-fast", ""],
                    help="Alpino's heuristic setting; '' is the slowest and "
                         "most accurate")
    ap.add_argument("--user-max", type=int, default=60000,
                    help="per-sentence timeout in ms; Alpino returns a partial "
                         "parse rather than hanging on a pathological sentence")
    ap.add_argument("--max-length", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None,
                    help="only parse this many sentences (for benchmarking)")
    ap.add_argument("--chunk-size", type=int, default=1000,
                    help="sentences per chunk; output is flushed and progress "
                         "recorded after each one")
    ap.add_argument("--restart", action="store_true",
                    help="discard saved progress for this shard and start over")
    ap.add_argument("--keep-xml", action="store_true",
                    help="archive the full Alpino XML as one .tar.gz per chunk. "
                         "The TSV carries word/lemma/POS/rel/head; the XML is "
                         "the superset (constituents, frames, co-indexation) "
                         "and the only way back if the head conversion ever "
                         "needs revisiting without re-parsing.")
    args = ap.parse_args()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _stop_child)

    rows, n_reviews = read_input(args.input, args.shard, args.n_shards,
                                 limit=args.limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = [rows[i:i + args.chunk_size]
              for i in range(0, len(rows), args.chunk_size)]
    signature = {
        "input": str(args.input), "shard": args.shard,
        "n_shards": args.n_shards, "chunk_size": args.chunk_size,
        "limit": args.limit, "speed": args.speed,
    }
    if args.restart:
        state_path(out_dir, args.shard).unlink(missing_ok=True)
    state = load_state(out_dir, args.shard, signature)

    tok_path = out_dir / f"alpino_tokens.shard{args.shard}.tsv"
    qual_path = out_dir / f"alpino_parsequality.shard{args.shard}.tsv"
    resuming = state["chunks_done"] > 0

    if resuming:
        # Truncate back to the last byte offset that was fsynced, discarding
        # any rows from a chunk that was interrupted mid-write.
        for path, size in ((tok_path, state["tok_bytes"]),
                           (qual_path, state["qual_bytes"])):
            if path.exists():
                os.truncate(path, size)
            else:
                raise SystemExit(f"{path} is missing but progress says "
                                 f"{state['chunks_done']} chunks are done; "
                                 "re-run with --restart")
        print(f"shard {args.shard}: resuming at chunk "
              f"{state['chunks_done']}/{len(chunks)} "
              f"({state['tokens']} tokens already written)",
              file=sys.stderr, flush=True)

    print(f"shard {args.shard}/{args.n_shards}: {len(rows)} sentences from "
          f"{n_reviews} reviews, {len(chunks)} chunks of {args.chunk_size} "
          f"-> {out_dir}", file=sys.stderr, flush=True)

    mode = "a" if resuming else "w"
    t_start = time.time()
    with tok_path.open(mode, encoding="utf-8", newline="") as tf, \
            qual_path.open(mode, encoding="utf-8", newline="") as qf:
        tw = csv.writer(tf, delimiter="\t", lineterminator="\n")
        qw = csv.writer(qf, delimiter="\t", lineterminator="\n")
        if not resuming:
            tw.writerow(COLUMNS)
            qw.writerow(["review_id", "sent_num", "n_tokens", "cats", "skips"])
            tf.flush()
            qf.flush()

        for k in range(state["chunks_done"], len(chunks)):
            chunk = chunks[k]
            t0 = time.time()
            ids, work = run_chunk(chunk, args, out_dir, f"{args.shard}_{k}")
            n_tok = 0
            for (rid, snum, _), sid in zip(chunk, ids):
                leaves, cats, skips = parse_xml(work / "xml" / f"{sid}.xml")
                if not leaves:
                    state["no_parse"] += 1
                    qw.writerow([rid, snum, 0, "", "no_parse"])
                    continue
                if skips and skips != "0":
                    state["skips"] += 1
                qw.writerow([rid, snum, len(leaves), cats or "", skips or ""])
                for tok_id, (word, lemma, pos, postag, rel, head, frame,
                             end) in enumerate(leaves, start=1):
                    tw.writerow([rid, snum, tok_id, tok_id - 1, end,
                                 word, lemma, pos, postag, rel, head, frame])
                    n_tok += 1

            if args.keep_xml:
                # One archive per chunk, not loose files: at ~10 kB/sentence a
                # full corpus run is tens of GB spread over millions of files,
                # which is as much a filesystem problem as a disk one.
                import tarfile
                arc = out_dir / f"xml_shard{args.shard}_chunk{k}.tar.gz"
                with tarfile.open(arc, "w:gz") as tar:
                    tar.add(work / "xml", arcname=f"shard{args.shard}_chunk{k}")
            shutil.rmtree(work, ignore_errors=True)

            # Durability barrier: the chunk counts as done only once its rows
            # are on disk, and the recorded sizes are what a resume truncates to.
            tf.flush()
            qf.flush()
            os.fsync(tf.fileno())
            os.fsync(qf.fileno())
            elapsed = time.time() - t0
            state["chunks_done"] = k + 1
            state["tokens"] += n_tok
            state["seconds"] = round(state["seconds"] + elapsed, 1)
            state["tok_bytes"] = tok_path.stat().st_size
            state["qual_bytes"] = qual_path.stat().st_size
            save_state(out_dir, args.shard, state)

            done = min((k + 1) * args.chunk_size, len(rows))
            print(f"  shard {args.shard} chunk {k + 1}/{len(chunks)}: "
                  f"{done}/{len(rows)} sentences, {n_tok} tokens, "
                  f"{elapsed:.0f}s ({len(chunk) / max(1e-9, elapsed):.2f} sent/s)",
                  file=sys.stderr, flush=True)

    total = state["seconds"] if resuming else time.time() - t_start
    meta = {
        "shard": args.shard, "n_shards": args.n_shards,
        "mode": "docker" if args.docker else "native",
        "speed": args.speed, "chunk_size": args.chunk_size,
        "sentences": len(rows), "tokens": state["tokens"],
        "no_parse": state["no_parse"], "sentences_with_skips": state["skips"],
        "seconds": round(total, 1),
        "sents_per_sec": round(len(rows) / total, 3) if total else None,
        "tokens_per_sec": round(state["tokens"] / total, 1) if total else None,
        "resumed": resuming,
    }
    (out_dir / f"alpino_bench.shard{args.shard}.json").write_text(
        json.dumps(meta, indent=2))
    print(json.dumps(meta), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

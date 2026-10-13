"""Compare sentence splitters on reviews reassembled into running text.

The corpus ships one sentence per line, i.e. already segmented by some earlier
tool. `--doc-mode` in the runners joins those lines back with single spaces and
feeds each review as one document, so every system's own splitter has to find
the boundaries again. This script scores what they found.

The existing line segmentation is used as a reference, but it is **silver, not
gold, and it is not neutral**. Per the corpus owner it was produced by Trankit.
Trankit does not reuse Stanza's splitter -- it is a separate XLM-R-based joint
tokeniser/splitter -- but it is trained on the same UD Dutch treebanks (Alpino +
LassySmall) that Stanza's splitter is trained on, and spaCy's sentence
boundaries come from its UD-trained parser. So three of the systems being
scored here share segmentation *conventions* with the reference, and their
agreement with it is substantially circular.

The systems therefore fall into two camps, and the report prints them apart:

  UD-trained   stanza, spacy_lg   (same treebank conventions as the reference)
  rule-based   frog, alpino       (ucto and Alpino's tokenize.sh; independent)

Do not read "highest F1 against the reference" as "best splitter". The block
that carries real information is where the two rule-based systems agree with
each other and disagree with the reference: those boundaries are decided by
machinery that owes the reference nothing, and they are the cases worth hand
checking. That set is dumped to sentences_independent_vs_reference.csv and is
the natural seed for a gold segmentation set.

A boundary is the character offset at which a sentence starts (the first
sentence of a document is not a boundary, since every system agrees on it).
Offsets come from the token tables, so they are directly comparable.

Usage: python src/parsers/compare_sentences.py
Writes results/parser_comparison/sentences_{summary,pairwise,disagreements}.csv
"""
import argparse
import csv
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_fmt import OUT_DIR, document_line_spans, load_documents, read_tokens  # noqa: E402


def discover_doc_tables(out_dir=None):
    out_dir = Path(out_dir or OUT_DIR)
    tables = {}
    for path in sorted(out_dir.glob("tokens_doc_*.tsv")):
        if ".shard" in path.name:
            continue
        tables[path.name[len("tokens_doc_"):-len(".tsv")]] = path
    return tables


def boundaries_from_table(path):
    """{doc_id: (set_of_boundary_offsets, n_sentences)}."""
    sents = defaultdict(lambda: defaultdict(list))
    for row in read_tokens(path):
        if row["start"] < 0:
            continue
        sents[row["doc_id"]][row["sent_id"]].append(row["start"])
    out = {}
    for doc_id, by_sent in sents.items():
        starts = sorted(min(v) for v in by_sent.values())
        # The first sentence start is not a decision -- every system begins the
        # document at the same place -- so it is excluded from the boundary set.
        out[doc_id] = (set(starts[1:]), len(starts))
    return out


def prf(pred, ref):
    tp = len(pred & ref)
    p = tp / len(pred) if pred else (1.0 if not ref else 0.0)
    r = tp / len(ref) if ref else (1.0 if not pred else 0.0)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--convention-chars", default=":;",
                    help="characters after which splitting is a guideline "
                         "choice rather than a fact; a second, "
                         "convention-neutral score ignores every boundary "
                         "following one of them")
    ap.add_argument("--independent", nargs="*", default=["frog", "alpino"],
                    help="tools whose splitter is independent of the UD Dutch "
                         "treebanks the reference segmentation came from")
    args = ap.parse_args()

    tables = discover_doc_tables()
    if not tables:
        raise SystemExit(
            "no tokens_doc_*.tsv found. Run the runners with --doc-mode first "
            "(make parser-run-docs).")
    tools = sorted(tables)
    ref_spans = document_line_spans()
    texts = {d: t for d, t, _ in load_documents()}

    ref = {d: (set(s for s, _ in spans[1:]), len(spans))
           for d, spans in ref_spans.items()}
    tool_bounds = {t: boundaries_from_table(p) for t, p in tables.items()}

    # Boundaries that immediately follow a colon or semicolon. Whether to split
    # there is a guideline decision -- the reference does, both rule-based
    # splitters decline -- so scoring them conflates a convention difference
    # with a capability difference. They get their own, separate accounting.
    conv = {}
    for d, text in texts.items():
        marks = set()
        for spans in [ref_spans.get(d, [])]:
            for start, _ in spans[1:]:
                j = start - 1
                while j >= 0 and text[j].isspace():
                    j -= 1
                if j >= 0 and text[j] in args.convention_chars:
                    marks.add(start)
        conv[d] = marks

    # ---- vs the corpus's existing segmentation ------------------------------
    summary = []
    for tool in tools:
        tp = fp = fn = 0
        n_sent = n_ref = 0
        exact_docs = 0
        docs = [d for d in ref if d in tool_bounds[tool]]
        for d in docs:
            pred, n_p = tool_bounds[tool][d]
            gold, n_g = ref[d]
            tp += len(pred & gold)
            fp += len(pred - gold)
            fn += len(gold - pred)
            n_sent += n_p
            n_ref += n_g
            exact_docs += pred == gold
        # Convention-neutral pass: drop every boundary that follows a colon or
        # semicolon, from both sides, and rescore.
        tp2 = fp2 = fn2 = 0
        for d in docs:
            pred, _ = tool_bounds[tool][d]
            gold, _ = ref[d]
            skip = conv.get(d, set())
            pred = pred - skip
            gold = gold - skip
            tp2 += len(pred & gold)
            fp2 += len(pred - gold)
            fn2 += len(gold - pred)
        p2 = tp2 / max(1, tp2 + fp2)
        r2 = tp2 / max(1, tp2 + fn2)

        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        summary.append({
            "tool": tool, "docs": len(docs),
            "neutral_precision": round(100 * p2, 2),
            "neutral_recall": round(100 * r2, 2),
            "neutral_f1": round(100 * 2 * p2 * r2 / max(1e-9, p2 + r2), 2),
            "sentences": n_sent, "reference_sentences": n_ref,
            "ratio_vs_reference": round(n_sent / max(1, n_ref), 3),
            "boundary_precision": round(100 * p, 2),
            "boundary_recall": round(100 * r, 2),
            "boundary_f1": round(100 * 2 * p * r / max(1e-9, p + r), 2),
            "docs_exact": exact_docs,
            "pct_docs_exact": round(100 * exact_docs / max(1, len(docs)), 2),
            "extra_boundaries": fp, "missed_boundaries": fn,
        })

    # ---- tool vs tool -------------------------------------------------------
    pairwise = []
    for a, b in combinations(tools, 2):
        tp = fp = fn = 0
        exact = n = 0
        for d in ref:
            if d not in tool_bounds[a] or d not in tool_bounds[b]:
                continue
            pa, _ = tool_bounds[a][d]
            pb, _ = tool_bounds[b][d]
            tp += len(pa & pb)
            fp += len(pa - pb)
            fn += len(pb - pa)
            exact += pa == pb
            n += 1
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        pairwise.append({
            "tool_a": a, "tool_b": b, "docs": n,
            "shared_boundaries": tp, "only_a": fp, "only_b": fn,
            "boundary_f1": round(100 * 2 * p * r / max(1e-9, p + r), 2),
            "docs_identical": exact,
            "pct_docs_identical": round(100 * exact / max(1, n), 2),
        })

    # ---- what the disagreements look like -----------------------------------
    rows = []
    for tool in tools:
        for d in sorted(ref):
            if d not in tool_bounds[tool]:
                continue
            pred, _ = tool_bounds[tool][d]
            gold, _ = ref[d]
            text = texts.get(d, "")
            for off in sorted(pred - gold)[:3]:
                rows.append({
                    "tool": tool, "kind": "extra_boundary", "doc": d,
                    "offset": off,
                    "context": _context(text, off)})
            for off in sorted(gold - pred)[:3]:
                rows.append({
                    "tool": tool, "kind": "missed_boundary", "doc": d,
                    "offset": off,
                    "context": _context(text, off)})
    rows = rows[:args.top * len(tools)]

    # ---- where the UD-independent systems disagree with the reference -------
    indep = [t for t in args.independent if t in tool_bounds]
    indep_rows = []
    n_extra = n_missed = 0
    if len(indep) >= 2:
        for d in sorted(ref):
            if any(d not in tool_bounds[t] for t in indep):
                continue
            gold, _ = ref[d]
            sets = [tool_bounds[t][d][0] for t in indep]
            consensus = set.intersection(*sets)
            union = set.union(*sets)
            text = texts.get(d, "")
            # All independent splitters put a boundary here; the reference did
            # not. Either the reference under-splits, or they share a blind spot.
            for off in sorted(consensus - gold):
                n_extra += 1
                indep_rows.append({
                    "kind": "independent_consensus_not_in_reference",
                    "doc": d, "offset": off, "tools": "+".join(indep),
                    "context": _context(text, off)})
            # The reference split here and not one independent system agreed.
            for off in sorted(gold - union):
                n_missed += 1
                indep_rows.append({
                    "kind": "reference_only",
                    "doc": d, "offset": off, "tools": "none",
                    "context": _context(text, off)})
    _write(OUT_DIR / "sentences_independent_vs_reference.csv", indep_rows)

    _write(OUT_DIR / "sentences_summary.csv", summary)
    _write(OUT_DIR / "sentences_pairwise.csv", pairwise)
    _write(OUT_DIR / "sentences_disagreements.csv", rows)
    _print(summary, pairwise, rows, indep, n_extra, n_missed, indep_rows,
           conv, texts)


def _context(text, off, width=45):
    left = text[max(0, off - width):off].replace("\n", " ")
    right = text[off:off + width].replace("\n", " ")
    return f"...{left} || {right}..."


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


UD_TRAINED = {"stanza", "spacy_lg", "spacy_retrained"}


def _print(summary, pairwise, rows, indep, n_extra, n_missed, indep_rows,
           conv, texts):
    print("\n=== sentence splitting vs the corpus's existing segmentation ===")
    print("Reference = Trankit, trained on the same UD Dutch treebanks as")
    print("stanza and spacy. Their agreement with it is partly circular; the")
    print("rule-based splitters (frog, alpino) owe it nothing. Highest F1 here")
    print("does NOT mean best splitter.")
    n_conv = sum(len(v) for v in conv.values())
    print(f"({n_conv} reference boundaries follow a colon or semicolon; the "
          f"'neutral'\ncolumns drop those from both sides)\n")
    print(f"{'tool':<12}{'sents':>8}{'ratio':>7}{'prec':>7}{'rec':>7}{'F1':>7}"
          f"{'|':>3}{'n-prec':>8}{'n-rec':>7}{'n-F1':>7}")
    for camp in ("rule-based (independent)", "UD-trained (same as reference)"):
        want_ud = camp.startswith("UD")
        subset = [r for r in summary
                  if (r["tool"] in UD_TRAINED) == want_ud]
        if not subset:
            continue
        print(f"  -- {camp} --")
        for r in subset:
            print(f"{r['tool']:<12}{r['sentences']:>8}{r['ratio_vs_reference']:>7}"
                  f"{r['boundary_precision']:>7}{r['boundary_recall']:>7}"
                  f"{r['boundary_f1']:>7}{'|':>3}{r['neutral_precision']:>8}"
                  f"{r['neutral_recall']:>7}{r['neutral_f1']:>7}")

    print("\n=== sentence splitting: tool vs tool ===")
    print(f"{'pair':<28}{'boundary F1':>13}{'docs identical':>16}")
    for r in pairwise:
        pair = f"{r['tool_a']} vs {r['tool_b']}"
        print(f"{pair:<28}{r['boundary_f1']:>13}{r['pct_docs_identical']:>15}%")

    if len(indep) >= 2:
        print(f"\n=== where {' + '.join(indep)} (rule-based, independent) "
              f"disagree with the reference ===")
        print(f"  both split, reference did not: {n_extra}")
        print(f"  reference split, neither did:  {n_missed}")
        print("  These are the boundaries worth hand-checking first; the "
              "reference\n  has no special authority over them.")
        for r in indep_rows[:10]:
            print(f"    {r['kind']:<38} {r['context'][:70]}")

    print("\n=== sample disagreements with the existing segmentation ===")
    for r in rows[:10]:
        print(f"  {r['tool']:<10} {r['kind']:<16} {r['context'][:88]}")


if __name__ == "__main__":
    main()

"""Compare tokenisation and sentence splitting across tools, without a gold set.

Three questions, all answerable from the outputs alone:

1. How often do two tools produce exactly the same token sequence for a line?
   This is the strict measure -- one differing boundary fails the whole line.
2. At token level, what fraction of spans do they share (Jaccard / F1)?
3. How often does each tool override the corpus's existing one-sentence-per-line
   segmentation?

It also dumps an inventory of the actual disagreements, ranked by frequency,
because the useful output here is not a number but a short list of recurring
patterns (emoji runs, repeated punctuation, hyphenation, abbreviations) that you
can then decide about deliberately.

Usage: python src/parsers/compare_tokenisation.py [--tools frog stanza spacy_lg]
Writes results/parser_comparison/tokenisation_{summary,pairwise,disagreements}.csv
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_fmt import OUT_DIR  # noqa: E402
from compare_util import (  # noqa: E402
    load_source_lines, load_tables, sentence_counts,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", nargs="*", default=None)
    ap.add_argument("--top", type=int, default=60,
                    help="how many disagreement patterns to report per pair")
    args = ap.parse_args()

    tables = load_tables(tools=args.tools)
    tools = sorted(tables)
    sents = sentence_counts(tools=tools)
    lines = load_source_lines()
    all_keys = sorted(set().union(*(set(t) for t in tables.values())))

    # ---- per-system summary -------------------------------------------------
    per_system = []
    for tool in tools:
        by_line = tables[tool]
        n_tok = sum(len(s) for s in by_line.values())
        resplit = sum(1 for k, n in sents[tool].items() if n > 1)
        n_sent = sum(sents[tool].values())
        per_system.append({
            "tool": tool,
            "lines": len(by_line),
            "tokens": n_tok,
            "tokens_per_line": round(n_tok / max(1, len(by_line)), 2),
            "sentences": n_sent,
            "lines_resplit": resplit,
            "pct_lines_resplit": round(100 * resplit / max(1, len(by_line)), 2),
        })

    # ---- pairwise agreement -------------------------------------------------
    pairwise = []
    disagreements = defaultdict(Counter)
    examples = {}
    for a, b in combinations(tools, 2):
        exact_lines = 0
        shared = only_a = only_b = 0
        comparable = 0
        for key in all_keys:
            sa = set(tables[a].get(key, {}))
            sb = set(tables[b].get(key, {}))
            if not sa or not sb:
                continue
            comparable += 1
            inter = sa & sb
            shared += len(inter)
            only_a += len(sa - inter)
            only_b += len(sb - inter)
            if sa == sb:
                exact_lines += 1
                continue
            # Record the text covered by each side's non-shared spans. Grouping
            # by the covering substring is what turns thousands of individual
            # mismatches into a handful of recognisable patterns.
            text = lines.get(key, "")
            for label, spans in ((a, sa - inter), (b, sb - inter)):
                for span in spans:
                    surface = text[span[0]:span[1]] if text else ""
                    other = a if label == b else b
                    pattern = _pattern(surface)
                    disagreements[(a, b)][(pattern, label)] += 1
                    examples.setdefault((a, b, pattern, label),
                                        (key, text, sorted(sa), sorted(sb)))
        denom = shared + only_a + only_b
        prec = shared / max(1, shared + only_a)
        rec = shared / max(1, shared + only_b)
        pairwise.append({
            "tool_a": a, "tool_b": b,
            "lines_compared": comparable,
            "lines_identical": exact_lines,
            "pct_lines_identical": round(100 * exact_lines / max(1, comparable), 2),
            "shared_tokens": shared,
            "only_a": only_a, "only_b": only_b,
            "jaccard": round(shared / max(1, denom), 4),
            "f1": round(2 * prec * rec / max(1e-9, prec + rec), 4),
        })

    # ---- write --------------------------------------------------------------
    _write_csv(OUT_DIR / "tokenisation_summary.csv", per_system)
    _write_csv(OUT_DIR / "tokenisation_pairwise.csv", pairwise)

    rows = []
    for (a, b), counter in disagreements.items():
        for (pattern, label), n in counter.most_common(args.top):
            key, text, sa, sb = examples[(a, b, pattern, label)]
            rows.append({
                "tool_a": a, "tool_b": b, "split_by": label,
                "pattern": pattern, "count": n,
                "example_doc": key[0], "example_line": key[1],
                "example_text": text[:160],
            })
    rows.sort(key=lambda r: (-r["count"], r["tool_a"], r["tool_b"]))
    _write_csv(OUT_DIR / "tokenisation_disagreements.csv", rows)

    _print(per_system, pairwise, rows)


def _pattern(surface):
    """Collapse a surface string to a shape so that recurring disagreement types
    group together: digits to 9, letters to a, everything else literal."""
    if not surface:
        return "(empty)"
    if len(surface) > 24:
        surface = surface[:24] + "..."
    out = []
    for ch in surface:
        if ch.isdigit():
            out.append("9")
        elif ch.isalpha():
            out.append("a")
        else:
            out.append(ch)
    shape = "".join(out)
    # Collapse runs so that "aaaa" and "aaaaaa" are the same pattern.
    collapsed, prev = [], None
    for ch in shape:
        if ch != prev:
            collapsed.append(ch)
            prev = ch
    return f"{''.join(collapsed)}  e.g. {surface!r}"


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _print(per_system, pairwise, disagreements):
    print("\n=== tokenisation: per system ===")
    print(f"{'tool':<12} {'tokens':>8} {'tok/line':>9} {'sents':>7} "
          f"{'lines re-split':>15}")
    for r in per_system:
        print(f"{r['tool']:<12} {r['tokens']:>8} {r['tokens_per_line']:>9} "
              f"{r['sentences']:>7} {r['lines_resplit']:>8} "
              f"({r['pct_lines_resplit']:>5.2f}%)")

    print("\n=== tokenisation: pairwise ===")
    print(f"{'pair':<26} {'identical lines':>16} {'token F1':>9} {'jaccard':>8}")
    for r in pairwise:
        pair = f"{r['tool_a']} vs {r['tool_b']}"
        print(f"{pair:<26} {r['lines_identical']:>7} "
              f"({r['pct_lines_identical']:>5.2f}%) {r['f1']:>9} {r['jaccard']:>8}")

    print("\n=== top tokenisation disagreements ===")
    for r in disagreements[:18]:
        pair = f"{r['tool_a']}/{r['tool_b']}"
        print(f"  {r['count']:>5}  {pair:<22} split by {r['split_by']:<9} "
              f"{r['pattern'][:52]}")


if __name__ == "__main__":
    main()

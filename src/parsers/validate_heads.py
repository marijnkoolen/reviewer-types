"""Check the XML-derived dependency heads against Alpino's own triples.

run_alpino_tsv.py reconstructs token-to-token dependencies from Alpino's XML by
percolating heads down the constituent tree. That is reimplemented logic, and
Alpino's structures have real edge cases -- coordination, co-indexed traces,
gapping, wh and relative clauses whose lexical head sits inside `body`.

Alpino can also emit the dependencies itself, via `end_hook=dependencies`, in
the PaQu triple format. It cannot do that *and* write XML in the same pass, so
the triples are no use for production, but they are an exact reference to check
the conversion against. This script parses the same sentences both ways and
reports where the head assignments differ.

Anything below ~99% agreement means the head priority list is wrong somewhere;
the mismatch table shows which relations are involved.

Usage:
  python src/parsers/validate_heads.py --docker alpino-impfic:latest \\
      --input data/review_sentences/review_sentences-aggr.tsv.gz --limit 200
Writes results/parser_comparison/head_validation.csv
"""
import argparse
import csv
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_alpino_tsv import (  # noqa: E402
    parse_xml, read_input, resolve_heads, sanitise,
)


def alpino_triples(work, image, alpino_home):
    """Run Alpino with end_hook=dependencies and return raw triple lines."""
    script = f"""
set -e
cd {'/work' if image else work}
"$ALPINO_HOME/Tokenization/tokenize_no_breaks.sh" < raw.txt > tok.txt 2>/dev/null
paste -d'|' ids.txt tok.txt > tok_id.txt
"$ALPINO_HOME/bin/Alpino" -notk -veryfast end_hook=dependencies -parse \
    < tok_id.txt 2>/dev/null
"""
    if image:
        cmd = ["docker", "run", "--rm", "--platform", "linux/amd64",
               "-v", f"{work}:/work", image, "bash", "-c", script]
    else:
        import os
        env = dict(os.environ, ALPINO_HOME=alpino_home, LC_ALL="en_US.utf8")
        return subprocess.run(["bash", "-c", script], env=env,
                              capture_output=True, text=True).stdout
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def parse_triples(text):
    """{sentence_id: {dep_index: (head_index, rel)}} from PaQu triple lines.

    Fields are head(7)|rel|dependent(7)|id. The rel field is `headrel/deprel`;
    only the dependent's own relation is comparable with the XML's `rel`.
    A head root of "none" marks a token with no head, i.e. ROOT.
    """
    out = {}
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) < 16:
            continue
        head_root, head_begin = parts[0], parts[2]
        rel = parts[7]
        dep_begin = parts[10]
        sid = parts[15]
        try:
            dep_i = int(dep_begin)
            head_i = -1 if head_root == "none" else int(head_begin)
        except ValueError:
            continue
        deprel = rel.split("/")[-1].strip() if "/" in rel else rel.strip()
        out.setdefault(sid, {})[dep_i] = (head_i, deprel)
    return out


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--docker", metavar="IMAGE")
    mode.add_argument("--alpino-home", metavar="PATH")
    ap.add_argument("--input", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--out", default="results/parser_comparison/head_validation.csv")
    args = ap.parse_args()

    rows, _ = read_input(args.input, 0, 1)
    rows = rows[:args.limit]
    work = Path(tempfile.mkdtemp(prefix="alpino_headcheck_"))
    ids = [f"{sanitise(r)}-{s}" for r, s, _ in rows]
    (work / "raw.txt").write_text(
        "\n".join(sanitise(t) for _, _, t in rows) + "\n", encoding="utf-8")
    (work / "ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    print(f"parsing {len(rows)} sentences twice (XML and triples)...",
          file=sys.stderr, flush=True)

    # Pass 1: triples, straight from Alpino.
    ref = parse_triples(alpino_triples(str(work), args.docker, args.alpino_home))

    # Pass 2: XML, converted by resolve_heads().
    script = f"""
set -e
cd {'/work' if args.docker else work}
"$ALPINO_HOME/Tokenization/tokenize_no_breaks.sh" < raw.txt > tok.txt 2>/dev/null
paste -d'|' ids.txt tok.txt > tok_id.txt
mkdir -p xml
"$ALPINO_HOME/bin/Alpino" -notk -veryfast -flag treebank xml end_hook=xml \
    -parse < tok_id.txt >/dev/null 2>&1
"""
    if args.docker:
        subprocess.run(["docker", "run", "--rm", "--platform", "linux/amd64",
                        "-v", f"{work}:/work", args.docker, "bash", "-c", script],
                       capture_output=True, text=True)
    else:
        import os
        subprocess.run(["bash", "-c", script],
                       env=dict(os.environ, ALPINO_HOME=args.alpino_home,
                                LC_ALL="en_US.utf8"),
                       capture_output=True, text=True)

    agree = disagree = missing = 0
    rel_mismatch = Counter()
    examples = []
    for sid in ids:
        xml_path = work / "xml" / f"{sid}.xml"
        if not xml_path.exists() or sid not in ref:
            missing += 1
            continue
        import xml.etree.ElementTree as ET
        mine = resolve_heads(ET.parse(xml_path).getroot())
        theirs = ref[sid]
        for idx, (head, rel) in sorted(mine.items()):
            if idx not in theirs:
                continue
            ref_head, ref_rel = theirs[idx]
            if head == ref_head:
                agree += 1
            else:
                disagree += 1
                rel_mismatch[(rel, ref_rel)] += 1
                if len(examples) < 200:
                    examples.append({
                        "sent_id": sid, "token_index": idx,
                        "derived_head": head, "alpino_head": ref_head,
                        "derived_rel": rel, "alpino_rel": ref_rel})

    total = agree + disagree
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sent_id", "token_index",
                                           "derived_head", "alpino_head",
                                           "derived_rel", "alpino_rel"])
        w.writeheader()
        w.writerows(examples)

    print(f"\n=== head reconstruction vs Alpino's own triples ===")
    print(f"  sentences compared   {len(ids) - missing}")
    print(f"  tokens compared      {total}")
    print(f"  head agreement       {100 * agree / max(1, total):.2f}%  "
          f"({agree}/{total})")
    if rel_mismatch:
        print("\n  most common mismatches (derived rel -> alpino rel):")
        for (a, b), n in rel_mismatch.most_common(12):
            print(f"    {n:>5}  {a or '(none)':<12} vs {b or '(none)'}")
        print(f"\n  {len(examples)} examples written to {out}")


if __name__ == "__main__":
    main()

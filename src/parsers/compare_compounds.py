"""Analyse Frog's morphological decomposition (MBMA), and cross-check it.

Frog is the only tool here that segments words into morphemes, e.g.
    leesplezier  -> [lees][plezier]
    16-jarige    -> [16-jaar][ig][e]
    pageturner   -> [pa][ge][turn][er]
The third example is the failure mode worth quantifying: MBMA is trained on
Dutch and over-segments English loanwords, which online book reviews are full
of. There is no gold segmentation here, so plausibility is approximated by
*attestation*: a segment that is a real Dutch morpheme tends to show up
elsewhere in the corpus as a word form or as some tool's lemma, whereas "pa",
"ge" and "turn" as a decomposition of pageturner do not all survive that test.

Second, an independent cross-check: Stanza's Dutch lemmas mark compound and
separable-particle boundaries with underscores (lot_genoten, op_steken,
spel_fout), which is the UD Dutch Alpino convention. Where both tools split the
same token, their split points can be compared directly -- two systems built on
completely different principles agreeing on a boundary is decent evidence the
boundary is real.

Usage: python src/parsers/compare_compounds.py [--min-segment-len 3]
Writes results/parser_comparison/compound_{summary,examples,vs_stanza}.csv
"""
import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_fmt import OUT_DIR  # noqa: E402
from compare_util import (  # noqa: E402
    coarse_pos, corpus_form_frequency, load_tables,
)

SEGMENT_RE = re.compile(r"\[([^\]]*)\]")

# MBMA emits inflectional and derivational affixes as segments alongside
# compound members. Only segments that are not affixes count as evidence that
# the analysis is claiming a *compound*; without this, ordinary inflection like
# [verdien][en] or [schakel][t] is indistinguishable from over-segmentation.
AFFIXES = {
    # inflectional
    "e", "en", "s", "t", "d", "de", "te", "je", "tje", "pje", "kje", "etje",
    "er", "st", "ste", "n",
    # derivational prefixes
    "ge", "be", "ver", "ont", "her", "er", "ont", "aan", "her",
    # derivational suffixes
    "ing", "ig", "lijk", "heid", "baar", "loos", "achtig", "isch", "atie",
    "iteit", "sel", "dom", "schap", "nis", "aar", "ant", "ent", "eur", "ist",
    "ette", "eer", "ief", "air", "eus", "ish", "end", "ende",
    "elijk", "zaam", "tenis", "aat", "eel", "ator", "ement", "erij", "ering",
    "iek", "aal", "isme", "teit", "ling", "erig", "matig", "waardig",
}
# This list is a stand-in for Frog's `deepmorph` model, which labels each
# morpheme as stem/prefix/suffix and would make the distinction exactly. That
# model is not present in this Frog installation (deepmorph=True fails to
# initialise), so compound-vs-derivation separation here is heuristic and will
# keep leaking derivational suffixes it does not know about.


def segments(morph):
    return [s for s in SEGMENT_RE.findall(morph or "") if s]


def stem_segments(segs, min_len):
    """Segments that look like free-standing words rather than affixes."""
    return [s for s in segs
            if len(s) >= min_len and s.isalpha() and s.casefold() not in AFFIXES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-segment-len", type=int, default=3,
                    help="segments shorter than this are treated as affixes "
                         "and excluded from the attestation test")
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--wordlist", default=None,
                    help="optional external Dutch word list, one word per line. "
                         "The corpus alone is only ~100k tokens, so many real "
                         "compound members are simply absent from it; a lexicon "
                         "sharpens the attestation test considerably.")
    args = ap.parse_args()

    tables = load_tables()
    if "frog" not in tables:
        raise SystemExit("no tokens_frog.tsv: this analysis is Frog-specific")
    freq = corpus_form_frequency(tables)

    # Attestation universe: surface forms plus every lemma any tool produced.
    attested = set(freq)
    for tool, by_line in tables.items():
        for spans in by_line.values():
            for row in spans.values():
                if row["lemma"]:
                    attested.add(row["lemma"].casefold())
                    # Stanza-style underscore lemmas contribute their parts too.
                    for part in row["lemma"].casefold().split("_"):
                        if part:
                            attested.add(part)

    if args.wordlist:
        extra = Path(args.wordlist).read_text(encoding="utf-8", errors="ignore")
        attested.update(w.strip().casefold() for w in extra.split() if w.strip())
        print(f"attestation universe extended to {len(attested)} strings from "
              f"{args.wordlist}", file=sys.stderr)

    seg_counts = Counter()
    by_pos = defaultdict(Counter)
    examples = []
    implausible = []
    n_tokens = n_multi = n_compound = 0

    for key, spans in tables["frog"].items():
        for span, row in spans.items():
            segs = segments(row["morph"])
            if not segs:
                continue
            pos = coarse_pos(row)
            if pos == "PUNCT":
                continue
            n_tokens += 1
            seg_counts[len(segs)] += 1
            by_pos[pos][len(segs)] += 1
            if len(segs) < 2:
                continue
            n_multi += 1
            stems = stem_segments(segs, args.min_segment_len)
            # A compound claim needs at least two free-standing members. With
            # fewer, the analysis is inflection or derivation, and asking
            # whether its "parts" are attested is not a meaningful test.
            if len(stems) < 2:
                continue
            n_compound += 1
            hits = [s for s in stems if s.casefold() in attested]
            ratio = len(hits) / len(stems)
            rec = {
                "form": row["form"], "lemma": row["lemma"], "pos": pos,
                "morph": row["morph"], "n_segments": len(segs),
                "n_stems": len(stems), "n_attested": len(hits),
                "attested_ratio": round(ratio, 2),
                "unattested": " ".join(s for s in stems if s not in hits),
                "form_freq": freq.get(row["form"].lower(), 0),
                "doc": key[0], "line": key[1],
            }
            examples.append(rec)
            if ratio < 0.5:
                implausible.append(rec)

    # ---- cross-check against underscore-lemma systems -----------------------
    # Both Alpino and Stanza mark compound seams with underscores in the lemma
    # (Alpino natively, Stanza because UD Dutch Alpino inherits the convention),
    # so their lemmas are a second, independent opinion on where the seams are.
    # Alpino is the more interesting comparison of the two: it is lexicon-driven
    # rather than learned, and it normalises members to citation form
    # (paardenfans -> paard_fan), which MBMA does not do.
    vs_rows = []
    cross = {}
    for other in ("alpino", "stanza"):
        if other not in tables:
            continue
        agree = split_both = frog_only = other_only = 0
        for key, spans in tables["frog"].items():
            s_spans = tables[other].get(key, {})
            for span, row in spans.items():
                s_row = s_spans.get(span)
                if s_row is None or coarse_pos(row) == "PUNCT":
                    continue
                f_segs = stem_segments(segments(row["morph"]),
                                       args.min_segment_len)
                s_parts = [p for p in (s_row["lemma"] or "").split("_") if p]
                f_split, s_split = len(f_segs) >= 2, len(s_parts) >= 2
                if f_split and s_split:
                    split_both += 1
                    # Compare only the number of members, not their spelling:
                    # Alpino normalises members to citation form and drops
                    # linking morphemes (paardenfans -> paard_fan), so exact
                    # string equality would understate real agreement.
                    same_count = len(f_segs) == len(s_parts)
                    same = ([x.casefold() for x in f_segs]
                            == [p.casefold() for p in s_parts])
                    agree += same_count
                    vs_rows.append({
                        "other_tool": other, "form": row["form"],
                        "frog_morph": row["morph"],
                        "frog_stems": " ".join(f_segs),
                        "other_lemma": s_row["lemma"],
                        "same_member_count": int(same_count),
                        "same_exact": int(same),
                        "doc": key[0], "line": key[1],
                    })
                elif f_split:
                    frog_only += 1
                elif s_split:
                    other_only += 1
        cross[other] = (split_both, agree, frog_only, other_only)

    # ---- write --------------------------------------------------------------
    total_segs = sum(seg_counts.values())
    summary = [{"n_segments": n, "tokens": c,
                "pct": round(100 * c / max(1, total_segs), 2)}
               for n, c in sorted(seg_counts.items())]
    _write(OUT_DIR / "compound_summary.csv", summary)

    examples.sort(key=lambda r: (r["attested_ratio"], -r["n_segments"]))
    _write(OUT_DIR / "compound_examples.csv", examples[:args.top])
    _write(OUT_DIR / "compound_implausible.csv", implausible[:args.top])
    _write(OUT_DIR / "compound_vs_underscore_lemmas.csv", vs_rows[:args.top])

    # ---- report -------------------------------------------------------------
    print("\n=== Frog MBMA segmentation (non-punctuation tokens) ===")
    print(f"tokens with a morph analysis: {n_tokens}")
    print(f"multi-segment tokens:         {n_multi} "
          f"({100 * n_multi / max(1, n_tokens):.1f}%)")
    print("\nsegments per token:")
    for n, c in sorted(seg_counts.items()):
        print(f"  {n:>2}: {c:>7}  ({100 * c / max(1, total_segs):>5.2f}%)")

    print("\nmulti-segment rate by word class:")
    for pos in sorted(by_pos, key=lambda p: -sum(by_pos[p].values())):
        tot = sum(by_pos[pos].values())
        multi = sum(c for n, c in by_pos[pos].items() if n >= 2)
        print(f"  {pos:<8} {multi:>6}/{tot:<7} ({100 * multi / max(1, tot):>5.1f}%)")

    print(f"\ncompound claims (>=2 free-standing members): {n_compound} "
          f"({100 * n_compound / max(1, n_tokens):.1f}% of analysed tokens)")
    print(f"  of which <50% of members attested anywhere in the corpus: "
          f"{len(implausible)} ({100 * len(implausible) / max(1, n_compound):.1f}%)")
    print("  NOTE: attestation is against this ~100k-token corpus only, so real "
          "but rare\n  compound members are over-flagged. Pass --wordlist to "
          "sharpen this.")
    for r in implausible[:15]:
        print(f"  {r['form']:<22} {r['morph']:<34} unattested: {r['unattested']}")

    for other, (split_both, agree, frog_only, other_only) in cross.items():
        print(f"\n=== Frog MBMA vs {other} underscore lemmas ===")
        print(f"  both split the token:     {split_both}")
        print(f"    same member count:      {agree} "
              f"({100 * agree / max(1, split_both):.1f}%)")
        print(f"  only Frog splits:         {frog_only}")
        print(f"  only {other} splits:      {other_only}")


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

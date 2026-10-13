"""Compare lemmatisation across tools on tokens they tokenise identically.

Without a gold set there is no accuracy, only agreement -- so this script is
built to make agreement as informative as possible:

* Comparison is restricted to spans every tool segments identically, so a
  lemma disagreement is never really a tokenisation disagreement in disguise.
* Agreement is broken down by word class and by corpus frequency, because a
  single overall figure is dominated by punctuation and function words that
  everything gets right. The interesting behaviour is in the rare tail.
* With three or more tools, each tool's *singleton rate* is reported: how often
  it dissents while the others agree with each other. This is the closest thing
  to a per-tool error signal available without annotation.
* Two heuristics target the specific failure the project cares about -- a
  lemmatiser emitting a string that is not a Dutch word:
    uncorroborated  no other tool proposes this lemma for this token
    unattested      the lemma string occurs nowhere else in the corpus, neither
                    as a surface form nor as any other tool's lemma anywhere

IMPORTANT CAVEATS, both of which bias the numbers in knowable directions:

* Agreement is not correctness. Stanza and spaCy are both trained on UD Dutch
  (Alpino + LassySmall), so they share training data and therefore share
  errors; agreement between those two is inflated relative to agreement with a
  lexicon-driven system. Do not read a majority vote over these three as a
  silver standard -- two of the three voters are correlated.
* `unattested` over-flags correct lemmas whose citation form is genuinely
  absent from a 100k-token corpus. It is a triage signal for manual inspection,
  not a verdict.

Usage: python src/parsers/compare_lemmas.py [--tools frog stanza spacy_lg]
Writes results/parser_comparison/lemma_{pairwise,by_pos,by_freq,flags,
disagreements}.csv
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
    coarse_pos, corpus_form_frequency, freq_bucket, load_source_lines,
    load_tables, shared_spans,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", nargs="*", default=None)
    ap.add_argument("--top", type=int, default=400,
                    help="disagreement examples to dump for manual inspection")
    ap.add_argument("--suffix", default="",
                    help="suffix for the output CSV names, so two tool sets "
                         "(e.g. stock vs retrained spaCy) can be compared "
                         "without overwriting each other")
    ap.add_argument("--fold-case", action="store_true",
                    help="compare lemmas case-insensitively (tools differ on "
                         "whether sentence-initial words keep their capital)")
    args = ap.parse_args()

    tables = load_tables(tools=args.tools)
    tools = sorted(tables)
    if len(tools) < 2:
        raise SystemExit("need at least two token tables to compare")
    freq = corpus_form_frequency(tables)
    lines = load_source_lines()

    def norm(lemma):
        return lemma.casefold() if args.fold_case else lemma

    def canon(lemma):
        """Lemma stripped of tagset convention, for 'same analysis, different
        house style' comparison.

        Stanza follows UD Dutch Alpino and marks separable particles and
        compound seams with underscores (op_steken, spel_fout, 16_jarig);
        Frog and Alpino write plain citation forms (opsteken, spelfout,
        16-jarig). Scoring those as disagreements measures the annotation
        guideline, not the lemmatiser, so both figures are reported: exact, and
        modulo convention.
        """
        return lemma.casefold().replace("_", "").replace("-", "")

    # Type-level attestation universe: every surface form in the corpus, plus
    # every lemma any tool produces anywhere. A lemma outside this set is a
    # string the corpus gives no independent evidence for.
    lemma_vocab = defaultdict(set)
    for tool, by_line in tables.items():
        for spans in by_line.values():
            for row in spans.values():
                if row["lemma"]:
                    lemma_vocab[tool].add(row["lemma"].casefold())
    form_vocab = set(freq)

    shared = list(shared_spans(tables, tools))
    print(f"comparing {len(shared)} tokens segmented identically by all of: "
          f"{', '.join(tools)}", file=sys.stderr)

    pair_hits = Counter()
    pair_hits_canon = Counter()
    pair_total = Counter()
    unanimous = unanimous_canon = 0
    singleton = Counter()
    by_pos = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    by_freq = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    flags = defaultdict(Counter)
    flag_examples = defaultdict(list)
    disagreements = []

    for key, span, rows in shared:
        lemmas = {t: norm(rows[t]["lemma"]) for t in tools}
        canons = {t: canon(rows[t]["lemma"]) for t in tools}
        form = rows[tools[0]]["form"]
        pos = _consensus_pos(rows, tools)
        bucket = freq_bucket(freq.get(form.lower(), 0))

        for a, b in combinations(tools, 2):
            same = lemmas[a] == lemmas[b]
            pair_total[(a, b)] += 1
            pair_hits[(a, b)] += same
            pair_hits_canon[(a, b)] += canons[a] == canons[b]
            by_pos[(a, b)][pos][1] += 1
            by_pos[(a, b)][pos][0] += same
            by_freq[(a, b)][bucket][1] += 1
            by_freq[(a, b)][bucket][0] += same

        distinct = set(lemmas.values())
        unanimous_canon += len(set(canons.values())) == 1
        if len(distinct) == 1:
            unanimous += 1
        elif len(tools) >= 3:
            # A tool is a singleton when every other tool agrees and it does not.
            counts = Counter(lemmas.values())
            for tool, lemma in lemmas.items():
                if counts[lemma] == 1 and len(counts) == 2:
                    singleton[tool] += 1

        # Per-tool suspicion flags.
        for tool in tools:
            lemma = rows[tool]["lemma"]
            if not lemma or pos == "PUNCT":
                continue
            low = lemma.casefold()
            others = [t for t in tools if t != tool]
            # Corroboration is judged modulo convention, so that Stanza's
            # underscores do not make every other tool look uncorroborated.
            corroborated = any(canon(rows[t]["lemma"]) == canons[tool]
                               for t in others)
            bare = canon(lemma)
            attested = (low in form_vocab or bare in form_vocab
                        or any(low in lemma_vocab[t] or bare in lemma_vocab[t]
                               for t in others))
            flags[tool]["tokens"] += 1
            if not corroborated:
                flags[tool]["uncorroborated"] += 1
            if not attested:
                flags[tool]["unattested"] += 1
                if len(flag_examples[tool]) < 300:
                    flag_examples[tool].append({
                        "tool": tool, "form": form, "lemma": lemma,
                        "pos": pos, "form_freq": freq.get(form.lower(), 0),
                        "others": " | ".join(
                            f"{t}={rows[t]['lemma']}" for t in others),
                        "doc": key[0], "line": key[1],
                        "context": lines.get(key, "")[:120],
                    })

        if len(distinct) > 1 and pos != "PUNCT":
            disagreements.append({
                "form": form, "pos": pos,
                "form_freq": freq.get(form.lower(), 0),
                **{f"lemma_{t}": rows[t]["lemma"] for t in tools},
                "doc": key[0], "line": key[1],
                "context": lines.get(key, "")[:120],
            })

    _write(OUT_DIR / f"lemma_pairwise{args.suffix}.csv", [
        {"tool_a": a, "tool_b": b, "tokens": pair_total[(a, b)],
         "agree": pair_hits[(a, b)],
         "pct_agree": round(100 * pair_hits[(a, b)] / max(1, pair_total[(a, b)]), 2),
         "agree_canon": pair_hits_canon[(a, b)],
         "pct_agree_canon": round(
             100 * pair_hits_canon[(a, b)] / max(1, pair_total[(a, b)]), 2)}
        for a, b in combinations(tools, 2)])

    _write(OUT_DIR / f"lemma_by_pos{args.suffix}.csv", [
        {"tool_a": a, "tool_b": b, "pos": pos, "tokens": tot,
         "pct_agree": round(100 * hit / max(1, tot), 2)}
        for (a, b), d in by_pos.items() for pos, (hit, tot) in sorted(d.items())])

    order = ["1 (hapax)", "2-5", "6-25", "26-100", "100+"]
    _write(OUT_DIR / f"lemma_by_freq{args.suffix}.csv", [
        {"tool_a": a, "tool_b": b, "freq_bucket": bucket, "tokens": tot,
         "pct_agree": round(100 * hit / max(1, tot), 2)}
        for (a, b), d in by_freq.items()
        for bucket in order if bucket in d
        for hit, tot in [d[bucket]]])

    _write(OUT_DIR / f"lemma_flags{args.suffix}.csv", [
        {"tool": t, "tokens": flags[t]["tokens"],
         "singleton": singleton.get(t, 0),
         "pct_singleton": round(100 * singleton.get(t, 0) / max(1, flags[t]["tokens"]), 2),
         "uncorroborated": flags[t]["uncorroborated"],
         "pct_uncorroborated": round(100 * flags[t]["uncorroborated"] / max(1, flags[t]["tokens"]), 2),
         "unattested": flags[t]["unattested"],
         "pct_unattested": round(100 * flags[t]["unattested"] / max(1, flags[t]["tokens"]), 3)}
        for t in tools])

    ex_rows = [e for t in tools for e in flag_examples[t]]
    _write(OUT_DIR / f"lemma_unattested_examples{args.suffix}.csv", ex_rows)

    disagreements.sort(key=lambda r: r["form_freq"])
    _write(OUT_DIR / f"lemma_disagreements{args.suffix}.csv", disagreements[:args.top])

    _print(tools, shared, unanimous, unanimous_canon, pair_hits,
           pair_hits_canon, pair_total, by_freq, flags, singleton, ex_rows)


def _consensus_pos(rows, tools):
    """Coarse POS agreed by a majority of tools, else the first tool's."""
    votes = Counter(coarse_pos(rows[t]) for t in tools)
    top, n = votes.most_common(1)[0]
    return top if n > 1 else coarse_pos(rows[tools[0]])


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _print(tools, shared, unanimous, unanimous_canon, hits, hits_canon,
           total, by_freq, flags, singleton, ex):
    n = len(shared)
    print("\n=== lemma agreement (tokens segmented identically by all tools) ===")
    print(f"comparable tokens: {n}")
    print(f"unanimous lemma:   {unanimous}  ({100 * unanimous / max(1, n):.2f}%) "
          f"exact | {unanimous_canon} ({100 * unanimous_canon / max(1, n):.2f}%) "
          f"modulo convention")
    print("\npairwise (exact / modulo convention):")
    for (a, b), tot in total.items():
        print(f"  {a:<10} vs {b:<10} {100 * hits[(a, b)] / max(1, tot):>6.2f}%  "
              f"/ {100 * hits_canon[(a, b)] / max(1, tot):>6.2f}%")

    print("\nby corpus frequency of the surface form:")
    order = ["1 (hapax)", "2-5", "6-25", "26-100", "100+"]
    pairs = list(by_freq)
    header = "  " + "bucket".ljust(12) + "".join(
        f"{a[:6]}/{b[:6]:<8}".rjust(16) for a, b in pairs)
    print(header)
    for bucket in order:
        cells = []
        for pair in pairs:
            hit, tot = by_freq[pair].get(bucket, (0, 0))
            cells.append(f"{100 * hit / max(1, tot):>6.2f}% ({tot:>5})".rjust(16))
        print("  " + bucket.ljust(12) + "".join(cells))

    print("\n=== per-tool dissent and suspicion flags ===")
    print(f"{'tool':<12}{'singleton':>12}{'uncorrob.':>12}{'unattested':>12}")
    for t in tools:
        tot = max(1, flags[t]["tokens"])
        print(f"{t:<12}{100 * singleton.get(t, 0) / tot:>11.2f}%"
              f"{100 * flags[t]['uncorroborated'] / tot:>11.2f}%"
              f"{100 * flags[t]['unattested'] / tot:>11.3f}%")

    print("\n=== sample unattested lemmas (strings with no corpus evidence) ===")
    for row in ex[:20]:
        print(f"  {row['tool']:<10} {row['form']:<22} -> {row['lemma']:<22} "
              f"[{row['others']}]")


if __name__ == "__main__":
    main()

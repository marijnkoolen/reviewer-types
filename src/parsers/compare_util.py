"""Shared loading and alignment helpers for the compare_*.py scripts.

The central idea: a token is identified across tools by its character span
(doc_id, line_id, start, end) in the original review line. Two tools "agree on
a token" when they emit the same span; only for those shared spans is it
meaningful to ask whether they agree on the lemma. Everything downstream is
built on span sets.

Usage: imported by compare_*.py; not run directly.
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_fmt import OUT_DIR, iter_lines, normalise, read_tokens  # noqa: E402

# Coarse categories shared by CGN (Frog, Alpino) and UD (Stanza, spaCy), so
# agreement can be broken down by word class without committing to one tagset.
_CGN_COARSE = {
    "N": "NOUN", "WW": "VERB", "ADJ": "ADJ", "LID": "DET", "VNW": "PRON",
    "VZ": "ADP", "BW": "ADV", "VG": "CCONJ", "TW": "NUM", "TSW": "INTJ",
    "SPEC": "PROPN", "LET": "PUNCT",
}
_UD_COARSE = {
    "NOUN": "NOUN", "PROPN": "PROPN", "VERB": "VERB", "AUX": "VERB",
    "ADJ": "ADJ", "DET": "DET", "PRON": "PRON", "ADP": "ADP", "ADV": "ADV",
    "CCONJ": "CCONJ", "SCONJ": "CCONJ", "NUM": "NUM", "INTJ": "INTJ",
    "PUNCT": "PUNCT", "SYM": "PUNCT", "X": "X",
}


def coarse_pos(row):
    """Map a row's native tag to a shared coarse class, preferring UD when the
    tool has it and falling back to the CGN prefix otherwise."""
    upos = row.get("upos") or ""
    if upos:
        return _UD_COARSE.get(upos, "X")
    xpos = row.get("xpos") or ""
    head = xpos.split("(")[0].split("|")[0].strip()
    return _CGN_COARSE.get(head, "X")


def discover_tables(out_dir=None):
    """Return {tool_name: path} for every token table present."""
    out_dir = Path(out_dir or OUT_DIR)
    tables = {}
    for path in sorted(out_dir.glob("tokens_*.tsv")):
        if ".shard" in path.name:
            continue  # shards are concatenated by run_all.sh before comparison
        tables[path.name[len("tokens_"):-len(".tsv")]] = path
    return tables


def load_tables(out_dir=None, tools=None):
    """Load token tables into {tool: {(doc_id, line_id): {span: row}}}.

    Within a line, a span maps to the first row carrying it, so a tool that
    emits the same span twice cannot inflate its own agreement.
    """
    tables = discover_tables(out_dir)
    if tools:
        missing = [t for t in tools if t not in tables]
        if missing:
            raise SystemExit(f"no token table for: {', '.join(missing)}")
        tables = {t: tables[t] for t in tools}

    loaded = {}
    for tool, path in tables.items():
        by_line = defaultdict(dict)
        for row in read_tokens(path):
            if row["start"] < 0:
                continue  # unalignable token: cannot be compared across tools
            key = (row["doc_id"], row["line_id"])
            span = (row["start"], row["end"])
            by_line[key].setdefault(span, row)
        loaded[tool] = dict(by_line)
    return loaded


def load_source_lines():
    """{(doc_id, line_id): normalised text} for quoting context in reports."""
    return {(d, i): normalise(t) for d, i, t in iter_lines()}


def sentence_counts(out_dir=None, tools=None):
    """{tool: {(doc_id, line_id): n_sentences}} -- how often each tool
    overrode the existing one-sentence-per-line segmentation."""
    tables = discover_tables(out_dir)
    if tools:
        tables = {t: tables[t] for t in tools if t in tables}
    counts = {}
    for tool, path in tables.items():
        per_line = defaultdict(set)
        for row in read_tokens(path):
            per_line[(row["doc_id"], row["line_id"])].add(row["sent_id"])
        counts[tool] = {k: len(v) for k, v in per_line.items()}
    return counts


def corpus_form_frequency(tables_loaded):
    """Lowercased surface-form frequency over the union of all tools' tokens.

    Used to bucket tokens by rarity: lemmatiser disagreement concentrates in the
    rare tail, and reporting one overall agreement number hides that.
    """
    freq = defaultdict(int)
    seen = set()
    for tool, by_line in tables_loaded.items():
        for key, spans in by_line.items():
            for span, row in spans.items():
                ident = (key, span)
                if ident in seen:
                    continue
                seen.add(ident)
                freq[row["form"].lower()] += 1
    return freq


def freq_bucket(n):
    if n == 1:
        return "1 (hapax)"
    if n <= 5:
        return "2-5"
    if n <= 25:
        return "6-25"
    if n <= 100:
        return "26-100"
    return "100+"


def shared_spans(tables_loaded, tools):
    """Yield (key, span, {tool: row}) for spans every named tool agrees on."""
    first, rest = tools[0], tools[1:]
    for key, spans in tables_loaded[first].items():
        others = [tables_loaded[t].get(key) for t in rest]
        if any(o is None for o in others):
            continue
        for span, row in spans.items():
            rows = {first: row}
            ok = True
            for tool, other in zip(rest, others):
                got = other.get(span)
                if got is None:
                    ok = False
                    break
                rows[tool] = got
            if ok:
                yield key, span, rows

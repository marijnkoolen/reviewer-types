"""Shared token-table format for the parser comparison.

Every runner (Alpino, Frog, Stanza, spaCy) writes the same TSV so that the
comparison scripts never need to know which tool produced a file. One row per
token:

    doc_id   review file stem, e.g. impfic-review-1000
    line_id  0-based line number within that file (the input sentence unit)
    sent_id  0-based sentence index *within the line*; > 0 means the tool
             re-split a line the input already considered one sentence
    tok_id   1-based token index within the tool's sentence
    start    character offset of the token in the original line (-1 if unaligned)
    end      character offset just past the token (-1 if unaligned)
    form     surface form as the tool emitted it
    lemma    lemma
    upos     UD POS if the tool has one, else ''
    xpos     native tag (CGN for Frog/Alpino/spaCy/Stanza xpos)
    feats    morphological features, tool-native string, '' if none
    morph    morpheme segmentation, Frog MBMA bracket string, '' for others
    align    'ok' if start/end were established, 'fail' otherwise

Character offsets are the join key across tools: they are the only tool-neutral
way to ask "did these two systems produce the same token here?". Tools that
expose offsets natively (spaCy, Stanza) pass them straight through; tools that
do not (Frog, Alpino) get them reconstructed by char_align() below.

Usage: imported by run_*.py and compare_*.py; not run directly.
"""
import csv
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SENT_DIR = ROOT / "data" / "review_sentences" / "0"
OUT_DIR = ROOT / "results" / "parser_comparison"

COLUMNS = [
    "doc_id", "line_id", "sent_id", "tok_id", "start", "end",
    "form", "lemma", "upos", "xpos", "feats", "morph", "align",
]

# Single-character substitutions only, so that normalising a string never
# changes its length and character offsets stay valid in both directions.
# Tokenisers differ on which of these they pass through verbatim; normalising
# both sides keeps a quote-style difference from being scored as a token
# boundary disagreement.
_NORM_MAP = {
    "‘": "'", "’": "'", "‚": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ",
    "​": " ", "﻿": " ",
}


def normalise(text):
    """Length-preserving normalisation used on both sides of every comparison."""
    return "".join(_NORM_MAP.get(c, c) for c in text)


def char_align(text, forms, max_skip=100, resync_after=3):
    """Reconstruct (start, end) offsets for tokens that carry no offsets.

    Walks the text left to right, consuming each token in order. Returns
    (spans, ok) where ok is False if any token could not be located -- which
    happens when a tokeniser rewrites a token rather than just splitting it.
    Unlocatable tokens get (-1, -1).

    `max_skip` bounds how far ahead of the current position a match may be
    accepted. Without it, a single rewritten token matches a much later
    occurrence of itself -- a bare quote or full stop will always occur again --
    which drags the cursor past everything in between and desynchronises the
    rest of the document. That failure mode was observed on a review containing
    a line consisting of a lone full stop: 341 consecutive tokens were lost
    after one bad match.

    `resync_after` is the escape hatch: if that many tokens fail in a row the
    walk really is stuck (the tokeniser dropped a span), so one unbounded search
    is allowed to re-acquire the position.
    """
    ntext = normalise(text)
    spans, pos, ok = [], 0, True
    misses = 0
    for form in forms:
        nform = normalise(form)
        idx = ntext.find(nform, pos)
        if idx < 0:
            # Tokeniser rewrote the token. Try a whitespace-insensitive match
            # before giving up: some tokenisers strip internal spaces.
            idx = _skip_ws_find(ntext, nform, pos)
        if idx >= 0 and idx - pos > max_skip and misses < resync_after:
            idx = -1  # implausibly far ahead; do not jump the cursor
        if idx < 0:
            spans.append((-1, -1))
            ok = False
            misses += 1
        else:
            spans.append((idx, idx + len(nform)))
            pos = idx + len(nform)
            misses = 0
    return spans, ok


def _skip_ws_find(text, form, start):
    """Find `form` in `text` from `start`, allowing whitespace in `text` to be
    skipped. Returns the start offset of the match, or -1."""
    stripped = "".join(form.split())
    if not stripped:
        return -1
    i = start
    while i < len(text):
        j, k = i, 0
        while j < len(text) and k < len(stripped):
            if text[j].isspace():
                j += 1
                continue
            if text[j] != stripped[k]:
                break
            j += 1
            k += 1
        if k == len(stripped):
            return i
        i += 1
    return -1


def open_writer(path):
    """Open a token-table TSV for writing and emit the header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("w", encoding="utf-8", newline="")
    # QUOTE_MINIMAL, not QUOTE_NONE: review text contains bare double quotes
    # (and Frog emits them as tokens), which QUOTE_NONE cannot represent.
    # read_tokens() uses the matching dialect, so the round-trip is exact.
    writer = csv.writer(fh, delimiter="\t", lineterminator="\n",
                        quoting=csv.QUOTE_MINIMAL)
    writer.writerow(COLUMNS)
    return fh, writer


def clean(value):
    """Make a field safe for an unquoted TSV: no tabs, no newlines, never empty-
    ambiguous. None becomes ''."""
    if value is None:
        return ""
    return str(value).replace("\t", " ").replace("\n", " ").replace("\r", " ")


def write_token(writer, doc_id, line_id, sent_id, tok_id, start, end,
                form, lemma, upos="", xpos="", feats="", morph="", align="ok"):
    writer.writerow([
        clean(doc_id), line_id, sent_id, tok_id, start, end,
        clean(form), clean(lemma), clean(upos), clean(xpos),
        clean(feats), clean(morph), align,
    ])


def read_tokens(path):
    """Read a token table back as a list of dicts, ints restored."""
    rows = []
    with Path(path).open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            for key in ("line_id", "sent_id", "tok_id", "start", "end"):
                row[key] = int(row[key])
            rows.append(row)
    return rows


def _read_lines(path):
    return [ln for ln in (l.strip() for l in
                          path.read_text(encoding="utf-8").split("\n")) if ln]


def load_documents(sent_dir=None, limit_docs=None):
    """Yield (doc_id, joined_text, line_spans) -- the corpus reassembled.

    The corpus ships one sentence per line, i.e. already segmented by whatever
    produced it. Joining those lines back into a running text with single
    spaces recreates the input a sentence splitter would actually face, and
    line_spans records where each original line landed, which makes the
    existing segmentation a reference to score splitters against.

    That reference is silver, not gold: it is one tool's opinion, of unknown
    provenance. Treat "agreement with the existing segmentation" as agreement,
    not accuracy.
    """
    sent_dir = Path(sent_dir or SENT_DIR)
    paths = sorted(sent_dir.glob("*.txt"))
    if limit_docs:
        paths = paths[:limit_docs]
    docs = []
    for path in paths:
        lines = _read_lines(path)
        if not lines:
            continue
        spans, pos = [], 0
        for line in lines:
            spans.append((pos, pos + len(line)))
            pos += len(line) + 1  # the single joining space
        docs.append((path.stem, " ".join(lines), spans))
    return docs


def document_line_spans(sent_dir=None, limit_docs=None):
    """{doc_id: [(start, end), ...]} for the corpus's existing segmentation."""
    return {d: spans for d, _, spans in load_documents(sent_dir, limit_docs)}


def iter_lines(sent_dir=None, limit_docs=None, doc_mode=False):
    """Yield (doc_id, line_id, text) -- the single definition of "the input".

    With doc_mode, one whole review is yielded as a single unit with line_id 0,
    so a tool's own sentence splitter decides the segmentation and lands in
    sent_id. Every runner iterates this, so doc_id/line_id mean the same thing
    in every token table.
    """
    if doc_mode:
        for doc_id, text, _ in load_documents(sent_dir, limit_docs):
            yield doc_id, 0, text
        return
    sent_dir = Path(sent_dir or SENT_DIR)
    paths = sorted(sent_dir.glob("*.txt"))
    if limit_docs:
        paths = paths[:limit_docs]
    for path in paths:
        for line_id, line in enumerate(
                path.read_text(encoding="utf-8").split("\n")):
            line = line.strip()
            if line:
                yield path.stem, line_id, line


def load_input(sent_dir=None, limit_docs=None, doc_mode=False):
    """iter_lines() materialised, for runners that batch."""
    return list(iter_lines(sent_dir, limit_docs, doc_mode))

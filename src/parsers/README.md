# Parser comparison: Alpino / Frog / Stanza / spaCy

A gold-free comparison of Dutch tokenisation, lemmatisation and morphological
analysis over `data/review_sentences/0` (670 reviews, 7,125 sentences,
~103k words of user-generated Dutch book reviews).

There is no gold standard, so nothing here measures accuracy. What it measures
is **agreement**, structured so that agreement is actually informative, plus
targeted heuristics for the one failure mode this project cares about: a
lemmatiser emitting a string that is not a Dutch word.

## Layout

| file | role |
|---|---|
| `common_fmt.py` | the shared token-table TSV, and offset reconstruction |
| `run_frog.py` | Frog via `python-frog` (POS, lemma, MBMA morphology) |
| `run_stanza.py` | Stanza Dutch (UPOS/XPOS/feats, lemma) |
| `run_spacy.py` | spaCy `nl_core_news_*` |
| `run_alpino.py` | Alpino, native (`--alpino-home`) or Docker (`--docker`) |
| `docker/Dockerfile.alpino` | Alpino from the official RUG tarball |
| `compare_tokenisation.py` | token-boundary and sentence-splitting agreement |
| `compare_lemmas.py` | lemma agreement, stratified, plus suspicion flags |
| `compare_compounds.py` | Frog MBMA analysis, cross-checked against Alpino/Stanza |
| `compare_sentences.py` | sentence-splitter comparison in document mode |
| `build_lemma_trainset.py` | silver lemmas -> spaCy training data + config |
| `eval_retrained_lemmatizer.py` | scores a retrained lemmatizer on held-out reviews |
| `run_all.sh` | run everything, concatenate shards, compare |

## The join key

Every runner emits one TSV row per token with the token's **character offsets
into the original review line**. Offsets are the only tool-neutral identity a
token has: two systems "agree on a token" when they emit the same span, and
lemma agreement is only ever computed on spans every system segments
identically. Without that restriction a tokenisation difference silently shows
up as a lemma difference.

spaCy and Stanza expose offsets natively. Frog and Alpino do not, so
`common_fmt.char_align()` reconstructs them by walking the line; a
length-preserving normalisation (curly quotes, dashes, NBSP) keeps quote-style
differences from counting as boundary disagreements. Tokens that cannot be
located are marked `align=fail` and excluded from all comparisons. On this
sample, zero tokens failed for Frog.

## Two input modes

**Line mode** (default) feeds each corpus line — already one sentence — to each
tool. This is what the tokenisation and lemma comparisons use: it holds the
segmentation fixed so that token and lemma differences are the only variable.

**Document mode** (`--doc-mode` on every runner) joins a review's lines back
into running text with single spaces and feeds the whole review at once, so each
tool's own sentence splitter has to find the boundaries again. Output goes to
`tokens_doc_<tool>.tsv` and is read by `compare_sentences.py`.

Alpino needs a different tokeniser in each mode, and the runner switches
automatically. Line mode uses `tokenize_no_breaks.sh`, which emits exactly one
line per input line and so preserves the `(doc_id, line_id)` mapping. Document
mode uses `tokenize.sh`, which splits — destroying that mapping, since Alpino
names its XML by tokenised line number. A sentinel line (`ZZSPLITZZ .`) is
written between documents to restore it: it survives tokenisation as exactly one
line, so the output lines between two sentinels are one document's sentences.
This was verified directly against the tokeniser's output before use.

The corpus's existing line segmentation then serves as a reference to score
splitters against — but it is **silver, not gold**: one earlier tool's opinion,
of unknown provenance. Several of its boundaries (after colons and semicolons,
for instance) are ones a reasonable splitter would decline to make.

## Parallelism

Shards are **independent OS processes**, not threads or a multiprocessing Pool.
This is deliberate: `python-frog` wraps a stateful C++ library that deadlocks
when a `Frog` object is constructed inside `Pool` workers (it aborts with an
uncaught C++ exception, leaving `Python-*.ips` crash reports behind). Each shard
writes its own TSV and `run_all.sh` concatenates them.

On the dual-A10 VMs (11 cores each):
- **Alpino** is CPU-only: one shard per core, `--n-shards 11`.
- **Stanza** takes `--device cuda`; run two shards pinned with
  `CUDA_VISIBLE_DEVICES=0` / `=1`.
- **Frog** does ~4.4k tokens/s single-process and needs no sharding at this size.

## Reading the numbers

Three cautions, all of which bias results in knowable directions:

1. **Agreement is not accuracy.** Stanza and spaCy are both trained on UD Dutch
   (Alpino + LassySmall). They share training data and therefore share errors,
   so their agreement is inflated relative to agreement with a lexicon-driven
   system. A majority vote over these three is *not* a silver standard: two of
   the three voters are correlated.
2. **Lemma conventions differ.** Stanza follows UD Dutch Alpino and marks
   separable particles and compound seams with underscores (`op_steken`,
   `spel_fout`, `16_jarig`); Frog and Alpino write plain citation forms. Scoring
   these as disagreements measures the annotation guideline, not the
   lemmatiser, so every figure is reported twice: exact, and modulo convention.
   This matters a lot -- it moved Stanza's unattested-lemma rate from 3.11% to
   0.93%.
3. **`unattested` over-flags.** A lemma is called unattested when its string
   occurs nowhere else in the corpus, neither as a surface form nor as any
   other tool's lemma. In a ~100k-token corpus many correct citation forms are
   simply absent, so this is a triage signal for manual inspection, not a
   verdict. Pass `--wordlist` to `compare_compounds.py` to sharpen the
   equivalent test there with an external lexicon.

Agreement is reported stratified by **corpus frequency of the surface form**,
because a single overall number is dominated by punctuation and function words
that everything gets right. On this sample pairwise agreement is ~99.2-99.6% on
forms seen 100+ times and 67-82% on hapaxes -- the rare tail is where the
systems actually differ, and it is where review vocabulary lives.

Note that the correlated-errors caution applies to **two** pairs, not one:
Stanza and spaCy share UD Dutch training data, and Stanza and Alpino share
provenance because UD Dutch Alpino is derived from Alpino's own treebank. Frog
is the only system here that is independent of the other three.

## Compound analysis

Frog's MBMA is the only morpheme segmentation available here. Separating
*compounding* from *inflection/derivation* properly needs Frog's `deepmorph`
model, which this installation lacks (`deepmorph=True` fails to initialise), so
`compare_compounds.py` approximates it with an affix stoplist: a "compound
claim" is an analysis with >=2 segments that are not known affixes. Plausibility
is then approximated by attestation of those members.

An independent cross-check comes free: Alpino and Stanza both encode compound
seams as underscores in the lemma, so where they and Frog both split a token the
splits can be compared. Alpino is the more useful comparison -- it is
lexicon-driven and normalises members to citation form (`paardenfans` ->
`paard_fan`), which MBMA does not do, so the check counts *members* rather than
requiring identical strings.

## Retraining spaCy's lemmatizer

`nl_core_news_lg` already lemmatises with an `EditTreeLemmatizer` -- a trained
component, not a lookup table. Its OOV failures (`lotgenoat`, `opsteekken`,
`familienam`, `werkwijz`) are learned suffix-edit rules misfiring on unseen
words, which is a data problem. `build_lemma_trainset.py` turns any runner's
output into spaCy training data plus a validated config that retrains **only**
the lemmatizer on top of frozen sourced components, so tagging and morphology
are untouched.

Caveats worth stating before you invest in it: this distils the silver system
into a faster model, it cannot exceed that system's accuracy, and it inherits
its lemma conventions -- so do not mix silver sources in one training set.

**Evaluate it with `eval_retrained_lemmatizer.py`, not with the suspicion flags
in `compare_lemmas.py`.** Those flags judge a lemma partly by whether other
tools propose it, and a model trained on Alpino silver agrees with Alpino by
construction, which inflates its own numbers and moves other tools' numbers even
though those tools did not change. `eval_retrained_lemmatizer.py` reproduces the
train/dev split exactly and asks the one question that is not circular: on
reviews the model never saw, how often does it reproduce the silver lemma?

Also note what `spacy train` prints at step 0. The config creates the
lemmatizer with `factory`, not `source`, so step 0 is an *untrained* component
whose `backoff = "orth"` returns the surface form -- a copy-the-form baseline,
not stock spaCy. Compare against the stock model's real score instead.

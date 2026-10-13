# Dutch parsers for online book reviews: a gold-free comparison

**Corpus.** `data/review_sentences/0` — 670 reviews, 7,125 sentences, ~103k
words of user-generated Dutch, one sentence per line as shipped.

**Systems.** Alpino (official RUG build `git20260629`, run in Docker),
Frog (via `python-frog`, config `~/.config/frog/nld`), Stanza 1.14 (`nl`,
UD Dutch Alpino + LassySmall models), spaCy 3.8 (`nl_core_news_lg`).

**Method.** There is no gold standard for this corpus, so nothing below is an
accuracy figure. Everything is *agreement*, structured so that agreement is
informative, plus heuristics targeting one specific failure: a lemmatiser
emitting a string that is not a Dutch word. Method and code:
[`src/parsers/README.md`](../src/parsers/README.md).

---

## 1. How to read these numbers

Three biases, all knowable in direction, apply throughout.

**Agreement is not accuracy, and two of the four systems are not independent.**
Stanza and spaCy are both trained on UD Dutch (Alpino + LassySmall), so they
share training data and therefore share errors. Stanza and Alpino share
provenance, because UD Dutch Alpino is derived from Alpino's own treebank.
**Frog is the only system here independent of the other three**, which makes
Frog-vs-anything the most informative agreement number in this report, and makes
a majority vote over the four systems unusable as a silver standard.

**Lemma conventions differ, and the difference is large.** Stanza follows UD
Dutch Alpino and marks separable particles and compound seams with underscores
(`op_steken`, `spel_fout`, `16_jarig`). Alpino goes further: its lemmas are
*compound-decomposed citation forms* (`bestsellerslijsten → bestseller_lijst`,
`paardenfans → paard_fan`, `zwart-witfoto's → zwart_wit_foto`), it normalises
spelling (`sexmisbruik → seks_misbruik`), it emits `vanalles → van alles` with a
space, and it pulls particles in from context (`gaat → tegemoet_gaan`). Frog and
spaCy write plain citation forms. Scoring these against each other measures the
annotation guideline, not the lemmatiser, so every figure below is reported
twice: exact, and modulo convention (underscores and hyphens stripped,
casefolded). This matters: it moved Stanza's unattested-lemma rate from 3.11% to
0.93% in the three-way run.

**The `unattested` heuristic over-flags.** A lemma counts as unattested when its
string occurs nowhere else in the corpus — neither as a surface form nor as any
other tool's lemma anywhere. In ~100k tokens many correct citation forms are
simply absent. It is a triage signal for manual inspection, not a verdict.

**The join key.** Every runner emits character offsets into the original review
line. Two systems "agree on a token" when they emit the same span, and lemma
agreement is only ever computed on spans all four systems segment identically
(114,528 of ~115k tokens). Without that restriction a tokenisation difference
shows up silently as a lemma difference.

---

## 2. Tokenisation

| tool | tokens | tokens/line | sentences | lines re-split |
|---|---:|---:|---:|---:|
| alpino | 115,151 | 16.16 | 7,124 | 0.00% * |
| frog | 115,361 | 16.19 | 7,181 | 0.70% |
| stanza | 115,153 | 16.16 | 7,370 | 3.26% |
| spacy_lg | 115,290 | 16.18 | 7,528 | 4.60% |

\* by construction — Alpino was run with `tokenize_no_breaks.sh`, which cannot
split. See §5 for the sentence-splitting comparison that does exercise it.

Pairwise:

| pair | identical lines | token F1 | Jaccard |
|---|---:|---:|---:|
| alpino vs frog | 97.35% | 0.9969 | 0.9938 |
| alpino vs spacy_lg | 97.32% | 0.9971 | 0.9943 |
| alpino vs stanza | 96.87% | 0.9967 | 0.9934 |
| spacy_lg vs stanza | 96.67% | 0.9966 | 0.9933 |
| frog vs spacy_lg | 96.51% | 0.9961 | 0.9923 |
| frog vs stanza | 95.66% | 0.9952 | 0.9904 |

**Tokenisation is not where these systems differ.** Token F1 is 0.995–0.997
across every pair, and the residual disagreements are almost entirely
punctuation: `/`, apostrophes, `.`/`..`/`...` runs, and `!`-runs. Alpino is the
most *central* tokeniser — it has the highest agreement with every other system
despite none of them being trained to match it.

Practical consequence: if you only need tokens, this choice barely matters. Pick
on speed.

---

## 3. Lemmatisation

91.94% of comparable tokens get the same lemma from all four systems; 94.72%
modulo convention. Pairwise, exact / modulo convention:

| pair | exact | modulo conv. |
|---|---:|---:|
| alpino vs stanza | 96.43% | 97.50% |
| frog vs spacy_lg | 95.87% | 96.68% |
| spacy_lg vs stanza | 95.09% | 97.12% |
| frog vs stanza | 95.05% | 97.23% |
| alpino vs spacy_lg | 94.21% | 96.82% |
| alpino vs frog | 94.20% | 96.93% |

### The overall number hides everything interesting

Agreement stratified by corpus frequency of the surface form:

| bucket | tokens | alpino/frog | alpino/spacy | alpino/stanza | frog/spacy | frog/stanza | spacy/stanza |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100+ | 69,147 | 99.18% | 99.36% | 99.32% | 99.49% | 99.47% | 99.59% |
| 26–100 | 13,795 | 94.28% | 95.19% | 96.88% | 94.97% | 96.25% | 96.89% |
| 6–25 | 13,779 | 90.11% | 91.12% | 94.61% | 91.96% | 91.44% | 92.72% |
| 2–5 | 10,532 | 83.05% | 82.04% | 89.95% | 88.02% | 84.13% | 83.44% |
| **1 (hapax)** | **7,275** | **70.64%** | **66.98%** | **80.81%** | **81.98%** | **73.40%** | **70.31%** |

Agreement runs at ~99.3% on frequent forms and collapses to 67–82% on hapaxes.
Since 7,275 of 114,528 tokens are hapaxes and review vocabulary is
overwhelmingly rare — compounds, loanwords, names, spelling variants — a single
headline agreement figure describes the part of the corpus nobody worries about.

Note that alpino/frog hapax agreement (70.64%) reads *worse* than frog/spacy
(81.98%) largely because of Alpino's decompounding: `canon()` strips underscores
but cannot undo dropped linking morphemes (`paard_fan` vs `paardenfan`). That
row understates real agreement.

### Per-tool dissent and suspicion flags

Over 102,112 non-punctuation comparable tokens:

| tool | singleton | uncorroborated | unattested |
|---|---:|---:|---:|
| frog | 1.56% | 1.56% | **0.348%** |
| alpino | 1.69% | 1.40% | 0.539% |
| stanza | 0.80% | 1.07% | 0.626% |
| spacy_lg | 1.28% | 1.85% | **0.962%** |

*singleton* = the tool dissents while all others agree with each other;
*uncorroborated* = no other tool proposes this lemma for this token;
*unattested* = the lemma string occurs nowhere else in the corpus.

**spaCy's unattested rate is ~2.8× Frog's, and its flagged cases are genuine
non-words**: `lotgenoat`, `familienam`, `dierenverhaol`, `kauwgomplaa`,
`werkwijz`, `opsteekken`. Frog's flagged cases are mostly *correct* citation
forms the small corpus happens not to contain (`uitdenken`, `aftuigen`,
`gieten`). That asymmetry is the finding: the same heuristic flags real errors
for one system and false alarms for another.

Stanza's low singleton rate (0.80%) should not be read as accuracy. Stanza sits
between Alpino (shared provenance) and spaCy (shared training data), so it
rarely dissents alone by construction.

Worth noting: spaCy's Dutch lemmatizer is **not** a lookup table. Both
`nl_core_news_lg` and `nl_core_news_trf` run an `EditTreeLemmatizer` — a trained
component. Its OOV failures are learned suffix-edit rules misfiring on unseen
words, which is a data problem, and therefore fixable (§6).

---

## 4. Compounds and morphology

Frog's MBMA is the only morpheme segmentation available. Separating *compounding*
from *inflection and derivation* properly needs Frog's `deepmorph` model, which
this installation lacks (`deepmorph=True` fails to initialise), so the analysis
approximates it with an affix stoplist. Treat the split as heuristic.

Of 102,607 non-punctuation tokens with an analysis, 25.7% are multi-segment:

| segments | tokens | share |
|---|---:|---:|
| 1 | 76,210 | 74.27% |
| 2 | 18,712 | 18.24% |
| 3 | 6,266 | 6.11% |
| 4 | 1,219 | 1.19% |
| 5+ | 200 | 0.19% |

By word class, multi-segment rate: VERB 63.9%, ADJ 57.5%, NOUN 46.1%, ADV 13.4%,
NUM 21.8%, and 0% for closed classes.

**Compound claims** (≥2 free-standing members, i.e. not just inflection): 4,865
tokens, 4.7% of analysed tokens. Of those, 2.4% have under half their members
attested anywhere in the corpus — and they are overwhelmingly loanwords:

    podcast       [pod][cast]
    Metamorphosen [metamorp][hose][n]
    Endlösung     [endlös][ung]
    Ausweis       [aus][weis]
    Stromblazers  [strom][blaas][er][s]
    kauwgomplaatje[kauw][gomplaat][je]

This is MBMA's known weakness — it is trained on Dutch and forces Dutch
morphology onto English and German material, which online book reviews are full
of.

### Cross-check: Frog MBMA vs underscore lemmas

Alpino and Stanza both encode compound seams as underscores in the lemma, giving
two independent opinions on where the seams are. Because Alpino normalises
members to citation form and drops linking morphemes, the check counts *members*
rather than requiring identical strings.

| comparison | both split | same member count | only Frog | only other |
|---|---:|---:|---:|---:|
| Frog vs Alpino | 2,088 | **94.6%** | 2,763 | 1,898 |
| Frog vs Stanza | 1,650 | 93.5% | 3,206 | 1,118 |

Two systems built on completely different principles — memory-based
classification versus a hand-written lexicalised grammar — agree on the number
of compound members 94.6% of the time when both decide to split. The real
disagreement is about *whether* to split at all, and Frog splits considerably
more often.

**If you want compound splitting, Alpino gives it to you in the lemma field for
free**, and without MBMA's loanword failure mode, since it is lexicon-driven.
Frog's MBMA remains the richer analysis (full morpheme sequence including
affixes), but it needs the loanword cases filtered.

---

## 5. Sentence splitting

Document mode (`--doc-mode`) joins each review's lines back into running text
with single spaces and feeds the whole review to each system, so every splitter
has to find the boundaries again. Alpino switches from `tokenize_no_breaks.sh`
to `tokenize.sh` for this, with a sentinel line between documents to preserve
the mapping from its XML output back to reviews.

### The reference is not neutral

The corpus's existing line segmentation was produced by **Trankit**. Trankit
does not reuse Stanza's splitter — it is a separate XLM-R-based joint
tokeniser/splitter — but it is trained on the same UD Dutch treebanks (Alpino +
LassySmall) as Stanza's splitter, and spaCy's boundaries come from its UD-trained
parser. Three of the systems scored here therefore share segmentation
*conventions* with the reference, and their agreement with it is substantially
circular.

| camp | systems | relation to the reference |
|---|---|---|
| UD-trained | stanza, spacy_lg | same treebank conventions |
| rule-based | frog, alpino | ucto / `tokenize.sh`; independent |

**Highest F1 against the reference does not mean best splitter.** It partly
means "trained on the same data as whatever made the reference".

### Agreement with the existing segmentation

The `n-` columns are a convention-neutral rescoring that drops every boundary
following a colon or semicolon (370 of them) from **both** sides.

| tool | camp | sents | ratio | prec | rec | F1 | n-prec | n-rec | n-F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| alpino | rule-based | 6,477 | 0.909 | **99.10** | 89.16 | 93.87 | **99.10** | 94.58 | **96.79** |
| frog | rule-based | 6,533 | 0.917 | 97.83 | 88.86 | 93.13 | 97.83 | 94.26 | 96.02 |
| spacy_lg | UD-trained | 7,199 | 1.010 | 94.82 | 95.91 | 95.36 | 94.63 | 97.96 | 96.27 |
| stanza | UD-trained | 7,116 | 0.999 | 96.42 | 96.28 | 96.35 | 96.25 | 97.44 | **96.84** |

**The raw ranking is almost entirely a guideline artefact.** Remove the
colon/semicolon boundaries and all four systems sit within 0.8 points
(96.02–96.84); Alpino moves from last to a tie with Stanza, and has the highest
boundary precision of any system either way — when Alpino posits a boundary,
the reference agrees 99.1% of the time.

Tool to tool:

| pair | boundary F1 | docs identical |
|---|---:|---:|
| **alpino vs frog** | **97.60** | **82.69%** |
| spacy_lg vs stanza | 95.06 | 62.84% |
| alpino vs stanza | 93.85 | 61.64% |
| frog vs spacy_lg | 93.54 | 61.19% |
| alpino vs spacy_lg | 93.45 | 60.15% |
| frog vs stanza | 93.35 | 62.99% |

Two rule-based splitters that share neither code nor training data agree with
each other far more than either agrees with the UD-trained pair. That is the
strongest evidence here that the two camps are real and not an artefact of how
the table is grouped.

Note also that the spread between tools (93–97.6) is far wider than for
tokenisation (token F1 99.5–99.7). **Sentence splitting, not tokenisation, is
where these systems actually diverge.**

Frog's profile is high precision (97.83) and low recall (88.86) — it
under-splits, producing 8% fewer sentences than the reference. Inspecting
`sentences_disagreements.csv`, its misses are concentrated on:

* **Colons and semicolons.** The reference splits after `:` and `;`; neither
  rule-based system does. Of 624 boundaries the reference makes that neither
  Frog nor Alpino makes, 285 follow a colon and 85 a semicolon — 370, i.e.
  *every* colon/semicolon boundary in the reference is rejected by both. Against
  that, only 36 boundaries go the other way. This is one guideline decision, not
  a capability gap.
* **Closing quotes.** Frog puts the boundary *before* a closing quote
  (`Hoe ik het kon. || " Met dit...` rather than `kon." || Met dit...`). This
  one is a genuine Frog error.
* **URLs.** Frog splits inside them (`...held-zonder-vaderland. || html`).

### Alignment bug found here

Document mode initially reported 341 unalignable tokens, all in one review
containing a line consisting of a lone full stop. `char_align` had been
searching unboundedly forward, so a single rewritten token matched a much later
occurrence of itself and dragged the cursor past everything in between. The
search is now bounded with a resync escape hatch; failures dropped from 341 to
**1 in 115,142**. Worth knowing because it would have scaled badly: ~0.15% of
documents on a full-corpus run.

### What to actually trust

`sentences_independent_vs_reference.csv` isolates the boundaries where both
rule-based splitters agree with each other and disagree with the reference.
Those decisions are made by machinery that owes the reference nothing, so they
are the cases worth hand-checking first — and the natural seed for a gold
segmentation set, which is the only thing that would turn any of this into
accuracy.

---

## 6. Retraining spaCy's lemmatizer on silver

`nl_core_news_lg` lemmatises with an `EditTreeLemmatizer` — a trained
component, not a lookup table — so its OOV failures are a data problem and in
principle fixable by retraining on in-domain silver.
`build_lemma_trainset.py` builds the data (98,121 tokens train / 17,108 dev from
Alpino silver, 99.7% of spaCy tokens aligned, **split by review** so no
sentences leak between splits) and a config that retrains *only* the lemmatizer
on top of frozen sourced components.

Training converged quickly and plateaued. The result, however, is a negative
one worth stating plainly.

### Read the baseline carefully

`spacy train` reports `LEMMA_ACC = 73.92` at step 0. **That is not stock
spaCy.** The config declares `trainable_lemmatizer` with `factory`, so a fresh
component is initialised; at step 0 it is untrained and `backoff = "orth"` makes
it return the surface form unchanged. 73.92% is the copy-the-form baseline.
Stock `nl_core_news_lg` scores 93.17% against Alpino silver on held-out reviews.

### Held-out vs training performance

Agreement with Alpino silver, punctuation excluded, dev = reviews never seen in
training:

| split | bucket | spacy_lg | spacy_retrained |
|---|---|---:|---:|
| **dev** | all | 93.17% | **93.97%** |
| **dev** | hapax | 66.43% | **65.04%** |
| **dev** | rare (2–5) | 82.74% | 84.10% |
| **dev** | common (6+) | 96.99% | 97.91% |
| train | all | 93.53% | 96.68% |
| train | hapax | 67.09% | 77.48% |
| train | rare (2–5) | 81.85% | 91.95% |
| train | common (6+) | 97.27% | 98.93% |

**On reviews it trained on, the model gains 10.4 points on hapaxes. On reviews
it did not, it loses 1.4.** The overall held-out gain is +0.80 points. That gap
is memorisation, not generalisation: 98k tokens teaches the model the specific
rare words in the training reviews, and rare words by definition do not recur.

244 held-out tokens are fixed and 123 regressed. Inspecting them
(`retrained_lemma_fixed.csv`, `retrained_lemma_regressed.csv`):

* Most "fixes" are the model adopting Alpino's *convention* rather than
  repairing an error — `spelen → af_spelen`, `uitleggen → uit_leggen`,
  `verhaallijn → verhaal_lijn`. Valuable if you want Alpino-style lemmas,
  but it is not error correction.
* Genuine hallucination repairs do occur: `dialoag → dialoog`,
  `Kelly's → Kelly`.
* The regressions are `backoff = "orth"` firing when no learned edit tree
  matches, so the model returns the surface form: `gereden → gereden`,
  `schreven → schreven`, `staat → staat`.

### The flag-based view is circular here

Running `compare_lemmas.py` with the retrained model swapped in shows its
unattested rate dropping 0.962% → 0.303%. That number should not be trusted:
the flags judge a lemma partly by whether *other* tools propose it, and a model
trained on Alpino silver agrees with Alpino by construction. The tell is that
Frog's unattested rate moved from 0.348% to 0.639% in the same run even though
Frog did not change — the attestation universe shifted under it. Use the
held-out table above instead; that is what
`eval_retrained_lemmatizer.py` exists for.

### How much silver would actually be needed

`learning_curve.py` trains the same configuration at five sizes and scores every
model on the same held-out reviews, decomposing the result into coverage (test
tokens whose form was seen in training) and generalisation (accuracy on forms
never seen):

| train tokens | coverage | acc_all | acc_seen | acc_unseen | acc_hapax |
|---:|---:|---:|---:|---:|---:|
| 4,300 | 70.8% | 86.59 | 94.48 | **67.48** | 57.27 |
| 11,091 | 77.8% | 90.38 | 95.98 | **70.79** | 59.44 |
| 25,485 | 83.0% | 92.11 | 96.37 | **71.27** | 62.14 |
| 52,075 | 86.7% | 93.34 | 96.69 | **71.44** | 64.93 |
| 97,819 | 89.3% | 93.97 | 96.73 | **70.82** | 65.01 |

**`acc_unseen` is flat.** A 23× increase in training data moves it from 67.5 to
70.8 and the last step is negative. The entire headline gain comes from coverage
rising 70.8% → 89.3%: the model is memorising this corpus's vocabulary, not
learning Dutch morphology it can apply to new words.

Vocabulary growth on this corpus fits Heaps' law at `V = 3.33 * N^0.743`. That
exponent is high — newspaper Dutch is nearer 0.6 — because reviews keep minting
compounds, spelling variants, titles and names. Unseen-token mass therefore
falls only as `N^-0.257`: a 100× larger silver corpus cuts it by about 3×.
Projecting accuracy from the coverage model:

| train tokens | est. coverage | est. accuracy |
|---:|---:|---:|
| 97,819 | 89.3% | 93.97% |
| 978,190 (10×) | 94.1% | 95.20% |
| 9,781,900 (100×) | 96.7% | 95.88% |

### Conclusion: do not distil

Stock `nl_core_news_lg` already scores 93.17% against Alpino. Even 10M tokens of
silver — 100× what was used here — reaches only ~95.9%, so the whole exercise is
worth about 2.7 points of Alpino fidelity, and none of it on the rare vocabulary
that motivated it.

The break-even does not exist, because the full Alpino run is affordable — though
less comfortably than first estimated. The collection is 6,893,445 sentences over
670,748 reviews, ~106M tokens; at the **measured** 0.319 sentences/s per core it
is **4.2 days on two 30-core VMs** (§7), not the 1.5–3 days extrapolated from
emulated timings. Reaching ~95.9% by distillation would still require
Alpino-parsing 10M tokens — ~10% of the collection, around 10 hours of the same
compute — to save four days while giving up four points of accuracy and carrying
a second pipeline.

Parse everything with Alpino. Keep the distillation path only for a future case
where new reviews must be annotated interactively or at high volume, where a
~96% model at 9,500 tokens/s is a reasonable fast pass. The scripts remain so
the decision can be re-derived with different numbers:

    make lemma-train SILVER=alpino && make lemma-eval
    python src/parsers/learning_curve.py --silver alpino

---

## 7. Throughput

### Measured on the production VMs

Two VMs, 30 cores and 375 GB each, native x86_64 in Docker, `-veryfast`,
200 sentences per shard across 30 shards per VM (57 shards completed):

| | |
|---|---|
| pooled per-shard rate | **0.319 sentences/s** (11,400 sentences / 35,710 shard-seconds) |
| mean sentence length | 15.4 tokens |
| sentences with `skips > 0` | 3.6–3.7% |
| failed to parse | 2 of 11,400 |

| configuration | rate | full corpus (6.89M sentences) |
|---|---:|---|
| 1 VM, 30 shards | 9.6 sentences/s | 200 h (**8.3 days**) |
| 2 VMs, 60 shards | 19.2 sentences/s | 100 h (**4.2 days**) |

The full corpus is 6,893,445 sentences over 670,748 reviews, roughly 106M
tokens at the measured mean length.

An earlier estimate in this report of 1–3 sentences/s per core, extrapolated
from emulated timings on a Mac, was **optimistic by 3–10×**. Native throughput
is ~0.32 sentences/s per core. Do not size a corpus run from emulated numbers.

### The per-shard spread is small-sample noise

Shard times ranged 275–1397s for an identical 200 sentences each — a 5× spread
that looks alarming and is not. Neither obvious cause holds:
`corr(tokens, seconds) = 0.37`, and `corr(skips, seconds) = 0.06` (the fastest
shard had 6 timeouts, the slowest also 6).

Alpino's per-sentence cost is heavy-tailed, and `skips` only counts sentences
where it *abandoned* a spanning analysis; a sentence that burns 40s and parses
successfully does not appear in that count. At 200 sentences per shard two or
three such sentences dominate the total. At ~230k sentences per shard in a real
run this averages out, which is why the projections above use the pooled rate
rather than the slowest shard.

`--user-max` (default 60000 ms) is the lever if 4.2 days is too long: it bounds
what any single sentence can cost, at the price of more partial parses.

### The other systems, for comparison

Single process, 10-core Apple Silicon Mac:

| tool | tokens/s | notes |
|---|---:|---|
| spacy_lg | 9,456 | CPU |
| frog | 4,412 | CPU |
| stanza | 1,080 | CPU; GPU not used |

### Frog does not scale the same way everywhere

Frog's per-process rate is ~275–300 sentences/s, but how it scales across
concurrent processes turned out to be machine-dependent, and badly so on one
of the VMs. Measured with `scaling_frog.sh`, 150–200 sentences per shard:

| shards | Mac aggregate (sent/s) | Mac efficiency | VM3 aggregate | VM3 efficiency |
|---:|---:|---:|---:|---:|
| 1 | 300 | 100% | 275 | 100% |
| 2 | 600 | 100% | — | — |
| 4 | 1,091 | 90.9% | 167 | 15.2% |
| 6 | 1,543 | 85.7% | — | — |
| 8 | 1,882 | 78.4% | — | — |
| 22 | — | — | 36 | 0.6% |

On the Mac, throughput scales nearly linearly with a gentle tail-off. On VM3,
**aggregate throughput falls as shards are added** -- one process does more
total work than four, which do more than twenty-two. That is not
oversubscription, which flattens a curve rather than inverting it.

The shape fits cache and memory-bandwidth contention: Frog's tagger and MBMA
are TiMBL memory-based classifiers doing pointer-chasing nearest-neighbour
lookups over large in-memory models. One process keeps its working set largely
resident; several evict each other and every lookup falls through to RAM. The
degradation is uniform across shards, which is what a shared bottleneck looks
like and what initially made it easy to mistake for a slow installation.

The practical lesson generalises beyond Frog: **measure the scaling curve
before choosing a shard count, and pick the count with the highest aggregate
rather than the highest core count.** `make scaling-frog` does this in a few
minutes.

At 8 shards on the Mac -- 1,882 sentences/s aggregate -- the full 6.89M-sentence
corpus takes **about 1 hour**. Against Alpino's 4.2 days on two 30-core VMs that
is a factor of ~100, and it is what makes the hybrid in §8 a clear choice rather
than a compromise.

### Operational notes

Four things cost real time before any of the above was measurable, all of them
worth knowing before a long run:

* **Run queue, not memory, is the thing to watch.** A collapsed rate with zero
  swap and zero iowait but `r` several times `nproc` means oversubscription.
  This happened when killed runs left orphaned shards behind: `vmstat` showed
  `r = 155` on 30 cores and the rate fell to 0.106 sentences/s, a third of the
  clean figure.
* **Killing a driver orphans its shards.** The shards are backgrounded and each
  spawns Alpino; killing an ancestor leaves the children running and holding
  memory. The drivers now trap INT/TERM and signal their process group, and for
  the Docker path the container is named and explicitly killed — terminating
  the `docker run` client alone does *not* stop the container. `make kill-runs`
  recovers from a `kill -9`.
* **Loading the input must stream.** Materialising all rows before sharding
  costs 2.2 GB per process at corpus scale — 66 GB across 30 shards, before
  Alpino. Streaming and discarding non-shard rows brings that to ~195 MB.
* **Docker Desktop's memory ceiling** binds on a Mac but not on Linux. At the
  3.8 GB default, 8 Alpino containers OOM-killed each other and throughput fell
  to ~0.05 sentences/s; raising it to ~31 GB gave an ~8× speedup.

Alpino makes no use of the A10 GPUs — it is CPU-only. Those are only useful for
Stanza, spaCy-trf, and lemmatizer retraining.

---

## 8. Recommendation

For a corpus of Dutch book reviews where POS, lemma and morphology all matter:

1. **Alpino where syntax is actually needed**, sharded one process per core.
   At 4.2 days for the full corpus on two VMs it is affordable but not free, so
   if dependency structure is only needed for part of the analysis, parse that
   part. It has the most central tokenisation, the second-lowest unattested
   rate, the richest morphology, and it supplies compound decomposition in the
   lemma. The cost is ~3 orders of magnitude less throughput than spaCy, which
   is affordable for a one-off corpus build and not for anything interactive.
2. **Frog over everything.** At ~4,400 tokens/s per core, and ~1,880
   sentences/s aggregate on 8 cores, it covers the whole 106M-token corpus in
   about an hour — roughly 100× faster than Alpino — and supplies POS, lemma and full morpheme decomposition. With a
   fixed sentence split the case for preferring Alpino rests almost entirely on
   the dependency parse; on lemmas the two agree 94.2% exactly and 96.9% modulo
   convention, with no evidence either is better. Frog is also the only system
   here independent of the other three, making it the most useful second
   opinion.
3. **Stanza or spaCy on GPU** over the same corpus for a UD view and a cheap
   disagreement signal; sentences where the fast and slow pipelines disagree are
   the ones worth looking at.
4. **Do not retrain spaCy's lemmatizer on Alpino silver** for this corpus (§6).
   The learning curve shows the gain is coverage-driven and caps out ~2.7 points
   above stock spaCy, while the full Alpino run is affordable outright.
5. **DeepFrog was considered and rejected.** It is a separate project from
   Frog (Rust, `cargo install`, needs libtorch), its GitHub repository was
   archived in September 2026 and marked suspended, and it ships **POS and NER
   models only** — no lemmatiser, no morphological analysis, no parser. Its POS
   is ~97.4 F1 (BERTje) against Frog's ~96–98 on the same CGN tagset, so
   adopting it would trade the lemmas and MBMA morphology that Frog is here for
   against roughly a point of tagging accuracy. The models are on HuggingFace
   (`proycon/bert-pos-cased-deepfrog-nld`) and usable from `transformers`
   directly, should a POS-only need ever arise.
6. **Build a gold dev set.** Everything here is agreement. A few hundred
   hand-corrected sentences would convert all of it into accuracy.
   `lemma_disagreements.csv` and `lemma_unattested_examples.csv` are sorted
   rarest-first and make a reasonable sampling frame.

---

## Outputs

All under `results/parser_comparison/`:

| file | contents |
|---|---|
| `tokens_<tool>.tsv` | per-token tables, line mode |
| `tokens_doc_<tool>.tsv` | per-token tables, document mode (sentence splitting) |
| `tokenisation_{summary,pairwise,disagreements}.csv` | §2 |
| `lemma_{pairwise,by_pos,by_freq,flags,disagreements}.csv` | §3 |
| `lemma_unattested_examples.csv` | flagged lemmas with context, rarest first |
| `compound_{summary,examples,implausible}.csv` | §4 |
| `compound_vs_underscore_lemmas.csv` | Frog MBMA vs Alpino/Stanza seams |
| `sentences_{summary,pairwise}.csv` | §5 |
| `lemma_train_<silver>/` | spaCy training data, config and model |

Rebuild this report with `make parser-report`.

# Plan: improved ground truth for review-sentence annotations

## Status 2026-11-14
- Three adjudication batches done (348 sentences; `data/adjudication/batch{1,2,3}_final.tsv`). Codebook now `docs/codebook_v2.3.md`.
- Blind agreement between you and the Opus majority: kappa 0.72 (batch 2, 2 reviews) and 0.65 (batch 3, 4 reviews); 0.98 when the agent labels are visible.
- Your conclusion: Opus labels are better than the human ones. Adjudication is for sharpening the rules; human labels are mainly useful for finding unclear categories. So the ground truth becomes the Opus majority under the current codebook, with human adjudication aimed at rule boundaries rather than at labelling every sentence.
- Full run done (UA-012 option f, UA-014): opus_a and opus_b on all 1,250 reviews (13,473 sentences), opus_c as tie-breaker on the 1,935 sentences where they differ. Codebook v2.3, prompt `src/annotate/full_run_prompt.md`.
  - Output: `data/full_run/majority_labels.tsv` (`src/annotate/build_majority.py`); per sentence the three runs, the final labels and status (agree 11,538 / majority 1,935).
  - opus_a vs opus_b: identical label set for 85.6% of sentences; per-label F1 0.88–0.98.
  - Final vs your gold (batches 1–3, 348 sentences): pooled kappa 0.85, exact match 66% (seen 70%, blind 50%). 119 gold sentences differ: `data/full_run/gold_vs_final_diff.tsv` (input for the UA-013 revision page).
  - Final vs human majority (>=2 of 3): kappa >= 0.6 for most labels; low for THM 0.29, FEE 0.39, REF 0.21, SCX 0.08, STR 0.14, i.e. exactly the labels whose rules changed in v2.x.
  - Rare labels (final vs human majority): STR 804 vs 63, SCX 237 vs 13, RCP 202 vs 89, IDI 630 vs 343, MET 289 (new). So UA-004 (a) holds: STR and SCX have enough examples to keep as separate labels.
  - 5 of about 125 subagents were stopped once by a safety classifier; all passed on one plain retry.
- Revision page ready: `data/adjudication/gold_revision.html` (119 sentences, `src/annotate/make_gold_revision_html.py`).
- Phase 3 sample drawn (`data/phase3/sample.tsv`, `src/annotate/make_phase3_sample.py`): 2,600 reviews, ~16,000 sentences; 500 random + enriched strata RCP 300, REF 400, SCX 1,000, MET 400. Selection model = logistic regression on old RobBERT review-level predictions + length, rating, genre, trained on the 815 RobBERT test reviews (CV AUC 0.71–0.83). Ranked by expected label sentences per sentence (long reviews are not denser), enriched pool 3–30 sentences. Inclusion probabilities stored for weighted prevalence estimates. Waits for the review texts (UA-015).
- Gold revision done: `data/adjudication/gold_batches1-3.tsv` (348 sentences; columns gold_before, gold, v23_final). 82 of 119 revised sentences took the v2.3 label, 24 kept the old gold, 13 got other labels. Gold vs v2.3 final now kappa 0.96, exact 89% (was 0.85 / 66%). Candidate v2.4 rules: UA-016.
- Codebook v2.4 (UA-016 a): `docs/codebook_v2.4.md`, rules R1–R7 from the gold revision. Full-run re-check: 858 candidate sentences, two Opus passes, change applied only when both agree: 100 changed (R1 58, R6 19, R2 8, R3 6, R5 6, R4 2, R7 1), 744 kept, 14 split (kept v2.3). Current labels for the 1,250 reviews: `data/full_run/labels_v24.tsv` (column `labels`; `labels_v23` kept). Vs revised gold: kappa 0.969, exact 91.4% (v2.3: 0.961 / 89.4%).
- Phase 3 texts staged and chunked (UA-015): `data/phase3/chunks/`, 52 chunks, 15,989 sentences.
- Phase 3 done (2026-11-14): 2,600 reviews, 15,989 sentences, codebook v2.4; opus_a/opus_b exact agreement 81.5%, opus_c tie-breaker on 2,960 sentences. Labels: `data/phase3/majority_labels.tsv`.
- **Combined ground truth: `data/gt_v2/labels.tsv`** (`src/annotate/build_ground_truth.py`): 3,850 reviews, 29,462 sentences; columns include `set` (original/phase3), `stratum` and `inclusion_prob` for weighting. Rare labels now: RCP 918, REF 822, SCX 751, MET 721, QUO 707 (lowest), IDI 1,496.
- For corpus prevalence use only set=original or stratum=random, or weight by 1/inclusion_prob; enriched strata over-represent RCP/REF/SCX/MET.
- Your revised gold (`data/adjudication/gold_batches1-3.tsv`, 348 sentences) is kept separately as the evaluation set; it is not merged into gt_v2.
- Sentence classifier on gt_v2 (2026-11-14): TF-IDF baseline and RobBERT fine-tune done; RobBERT test micro-F1 0.80, on your gold 0.77. See section below.
- RobBERT fine-tune (UA-017 a): the first run crashed on 2026-11-14 (swap > 50 GB; MPS kept a graph per batch length). Rerun from scratch with batches padded to a multiple of 32 (UA-020 a): 4 epochs in 76 min, no swap. Crashed run kept in `results/gt_v2_classifier/robbert_crash_2ep/` (not in git).
- Phase 4 English pilot started (2026-11-14), see section below. English pilot labelled (opus_a/b/c); your check waits (UA-023). Italian and Russian: reviews parsed and pools drawn, Italian pilot chunked; see the section below (UA-021, UA-022).

## Phase 4: English pilot (2026-11-14)
- Source: `reader_profiles` DuckDB on the T7 drive (`/Volumes/T7_Data/Data/claude-projects-external/reader_profiles/db/reader_profiles.duckdb`, read-only): Goodreads 15.7M, Amazon 29.5M, LibraryThing 0.93M reviews.
- Pool `data/other_languages/en/` (`src/annotate/make_en_sample.py`): 1,800 reviews, 600 per platform; random rowids, langid English with confidence >= 0.9, genre known, 5–2,000 words, deduplicated; 85% fiction categories / 15% non-fiction (the Dutch corpus has 16% non-fiction, Amazon books 53%). Genre = Goodreads' 10 categories (Amazon via the reader_profiles crosswalk). Ratings 1–5 (LibraryThing's 1–10 halved; 0 = unrated → 'unknown'). Review ids `en-review-NNNNN`; platform and source keys (`source_keys.tsv`) are not shown to labellers. Amazon review titles are left out (Goodreads and LibraryThing have none).
- Pilot = first 100 per platform (column `pilot`), 300 reviews; the rest (1,500) is for later batches in the same order.
- Sentence split (`src/annotate/make_en_chunks.py`): line breaks, then spaCy `en_core_web_sm` parser; fragments without letters/digits merged into the previous sentence. Pool 14,700 sentences (median 5 per review, max 116); pilot 2,329 sentences in 12 chunks of 25 reviews (`chunks/`). Sentences per review: Amazon median 3, Goodreads 5, LibraryThing 5 (Dutch original set median 7).
- Codebook `docs/codebook_v2.4_en.md` = v2.4 with an English task line and a section mapping English platform conventions onto existing rules (ratings in text = EVA; ARC/NetGalley, DNF, reread, audiobook = CTX; 'relatable' = IDI; delivery = CTX, edition physical features = COO; bare headings/spoiler markers = empty). No new rules.
- Runs: opus_a and opus_b on the 12 pilot chunks, opus_c tie-breaker, same scripts with `RUN_BASE=data/other_languages/en`; prompt in `src/annotate/full_run_prompt.md` (section Phase 4 English pilot).
- Runs done (2026-11-14, resumed after the crash, UA-019 a): opus_a/opus_b exact label-set agreement 83.6% (Dutch full run 85.6%, Phase 3 81.5%); per-label F1 a vs b 0.83 (SCX) to 0.99 (RCP). opus_c tie-breaker on 383 sentences in 176 reviews, none unresolved. Labels: `data/other_languages/en/majority_labels.tsv` (same format as `data/full_run/majority_labels.tsv`).
- Label share per sentence, English pilot vs Dutch original set: close for most labels. Higher in English: CTX 11.6% vs 6.8% (Amazon 15.6%: delivery, ARC, audiobook), EVA 46.9 vs 41.4, FEE 16.2 vs 13.5, REF 2.8 vs 1.6, THM 7.3 vs 5.6. Lower: STR 3.6 vs 6.0, MET 1.1 vs 2.1, STY 11.7 vs 13.9. No label: 2.3% vs 2.8%; labels per sentence 1.86 vs 1.80.
- Your check (plan step 13): `data/adjudication/en_batch1_agents_adjudicate.html` (`src/annotate/make_en_check.py`, `make_agent_adjudication_html.py --meta-chunks`): 21 random pilot reviews (7 per platform, ≤ 20 sentences), 123 sentences, 29 of them disputed between opus_a and opus_b; the last 6 reviews are blind. Prefill = pilot label. UA-023.

## Phase 4: Italian and Russian (2026-11-14)
- Source: the Goodreads per-language book pages (`impfic-multilingual-reviews/data/Book_language_pages/{it,ru}`, read-only). Parsed with `src/annotate/parse_gr_lang_pages.py` (adapted from that repo's `scripts/parse_html.py`; keeps paragraph breaks, keeps no reviewer names): it 15,186 pages → 197,355 reviews from 15,137 books; ru 8,289 pages → 55,535 reviews from 8,263 books (the older extraction in `lang_reviews_all` has 26,670 / 9,487). Output `data/other_languages/{it,ru}/gr_reviews.tsv.gz`.
- The language pages are ~98% in the page language (langid on 1,000 reviews each).
- Pools (`src/annotate/make_gr_lang_sample.py`): the English filters for one platform. langid p ≥ 0.9 in the language, genre known (GR-10 from `gr_book_genre` by Goodreads book id; ~70% of the books), 5–2,000 words, deduplicated, at most 2 reviews per book, 85% fiction / 15% non-fiction. 1,800 reviews each; pilot = first 300.
  - it: 1,800 reviews from 1,716 books, median 91 words.
  - ru: 1,800 from 1,663 books, median 59 words. English pilot median: Goodreads 5 sentences.
- Italian split and chunks (`LANG_CODE=it src/annotate/make_en_chunks.py`, spaCy `it_core_news_lg`): pool 16,232 sentences (median 5 per review, max 102); pilot 2,718 sentences in 12 chunks of 25 reviews (`data/other_languages/it/chunks/`).
- Codebooks (UA-022 a): `docs/codebook_v2.4_it.md`, `docs/codebook_v2.4_ru.md`; prompt variant in `src/annotate/full_run_prompt.md`. Human check by colleagues after the pilots.
- Russian split and chunks (UA-021; `ru_core_news_sm` installed in my user's Python packages): pool 16,028 sentences (median 5 per review, max 188); pilot 2,598 sentences in 12 chunks (`data/other_languages/ru/chunks/`). Runs not started.
- **Paused 2026-11-14 ~12:10 (connection about to drop).** Italian opus_a: chunks 000–005 done (6 of 12); no agents running. To resume, run `RUN_BASE=data/other_languages/it venv/bin/python src/annotate/check_full_run.py opus_a opus_b`: it lists the chunks still to do. Delete nothing; incomplete part files are reported. Then opus_b on all 12 chunks, then the tie-breaker (`make_tiebreak_input.py opus_a opus_b`, opus_c 4 chunks per agent), then `build_majority.py`.
- RobBERT rerun done (2026-11-14), see the classifier section.

## Sentence classifier on gt_v2 (2026-11-14)
- Scripts `src/classifier/` (`gt_v2_data.py` split, `gt_v2_baseline.py`, `gt_v2_transformer.py`, `gt_v2_compare_old.py`); results and file guide `results/gt_v2_classifier/README.md`. Run with Homebrew `python3.11` (torch 2.12 + MPS); `venv/` has no torch.
- Split by review, stratified by stratum, 80/10/10: 3,056 / 382 / 382 reviews (22,778 / 3,521 / 2,815 sentences). The 30 reviews of your 348 gold sentences are held out completely (split `gold`).
- Input = labellers' context only: the sentence, its neighbouring review sentences, the `review_meta.py` header (genre, rating, year) and position. Per-label thresholds tuned on dev.
- **TF-IDF + one-vs-rest LR** (sentence word+char n-grams, previous/next sentence, header, position): micro-F1 test 0.72 (random stratum 0.67), macro-F1 0.63; on your gold micro 0.69, macro 0.56. Ceiling: the v2.4 labels vs your gold reach micro 0.97.
  - Good (F1 ≥ 0.7 on test): EVA, NAR, STY, REC, CTX. Weak: QUO 0.36, SCX 0.46, REF 0.48 (0.08 on gold, 18 sentences), STR 0.55, THM 0.58, RCP 0.58.
- **vs old RobBERT** on the same gold sentences (335 sentences / 29 reviews; one review has a different sentence split; 17 labels, no MET): old RobBERT micro 0.64 / macro 0.50, TF-IDF 0.69 / 0.54. Same picture on the 26 reviews that were in the old model's test set (0.63 vs 0.70). Old RobBERT is better on OTW, RCP, REF, CLS, COO, NAR, IDI; the baseline on FEE, STR, THM, SCX, REC, EVA, CTX, STY. Most of those are labels whose rules changed in v2.x. QUO is 0 for both.
- **RobBERT** (`pdelobelle/robbert-v2-dutch-base`, 4 epochs, lr 3e-5, batch 16, max 256 tokens, sentence + neighbours ~700 chars; MPS 76 min): dev micro-F1@0.5 per epoch 0.697 / 0.769 / 0.782 / 0.786. With dev-tuned thresholds: test micro 0.80 / macro 0.73 (original 0.81, random stratum 0.77, enriched 0.80); on your gold micro 0.77 / macro 0.69 (TF-IDF 0.69 / 0.56, old RobBERT 0.64 / 0.50; ceiling 0.97). Per-label test F1: EVA 0.87, NAR 0.87, STY 0.80, REC 0.80, FEE 0.80, CTX 0.79, CLS 0.77, QUO 0.76 (TF-IDF 0.36), COO 0.75, AUT 0.73, OTW 0.72, THM 0.70, IDI 0.69, RCP 0.69, MET 0.68, STR 0.63, SCX 0.52, REF 0.49. Next: the larger open model on the A10 GPUs (UA-025 a).

## Open model on the A10 GPUs (2026-11-16, UA-025 a, UA-028)
- Approach: an open-weight LLM (7–14B) fine-tuned with 4-bit QLoRA and an 18-label head. It reads the whole review (header, numbered sentences, then every sentence again) and scores each sentence at the end of its repeated copy, so every sentence sees the whole review, as the agents did. Large encoder baseline: XLM-R-large with the RobBERT setup.
- Scripts: `src/classifier/gt_v2_llm.py` (train + evaluate with the same split, thresholds and eval sets as RobBERT), `gt_v2_llm_predict.py` (corpus labelling, sharded over GPUs, resumable), `gt_v2_export.py` (one self-contained input file, `data/gt_v2/train_input.tsv.gz`, for the VMs), `requirements_gpu.txt`. Smoke-tested on the Mac with Qwen2.5-0.5B unquantised (training, dev scoring, saving, corpus labelling of 60 reviews); the 4-bit path is first tested on the VMs.
- Grouping check on all 3,850 reviews: every sentence labelled exactly once; at 2,048 tokens 4,003 sequences (153 long reviews split), 2.1M tokens per epoch.
- Runbook with the exact commands: `docs/GPU_RUNBOOK.md` (VM1: Qwen2.5-7B, EuroLLM-9B; VM2: Qwen2.5-14B, XLM-R-large). Your step.

## Findings from the current data (1,250 reviews, ~13.5k sentences, 3 annotators)
- Annotators: anno1 (12.8k sentences, 45 reviews missing), anno2, anno3 (13.5k each). Multi-label per sentence, 18 real labels; the parent columns `Content`/`Reader_response`/`Style`/`None` are essentially unused (<0.3%) and must be derived, not annotated.
- Pairwise kappa is good (>0.6) for Quote, Recommendations, Author, Narrative, Evaluation, Stylistic_features, Reception; **weak** for:

| Label | kappa range | Sentences with ≥1 / ≥2 / all 3 |
|---|---|---|
| Style--Context | 0.06-0.11 | 211 / 13 / 3 |
| Reader_response--Feelings | 0.14-0.39 | 2221 / 796 / 103 |
| Content--Theme | 0.31-0.46 | 534 / 178 / 65 |
| Style--Structure | 0.23-0.44 | 251 / 59 / 24 |
| Reader_response--Reflection | 0.38-0.44 | 2500 / 1052 / 384 |
| Content--Other | 0.45-0.66 | 564 / 266 / 133 |
| Other_works | 0.50-0.65 | 1029 / 664 / 262 |

- **Feelings vs Identification**: anno3 uses Feelings far less (3.9% vs 9.5-10.8%). Where anno1/anno2 tag Feelings, anno3 tagged Identification on the same sentence 56-74 times (the reverse is 14-21). Examples are empathy/involvement ("ik had het zo met Frans te doen", "ik ga de leden van Full Moon missen", "hoop je stiekem dat Tejo iets kan veranderen"). So anno1+2 read the guideline as "any emotion about the book", anno3 as "emotion about characters = identification". The guideline itself is ambiguous here (Feelings: "feelings related to the book... personal desires"; Identification: "identification with characters, immersion"). Also a second Feelings confusion: expectation/pleasure/curiosity sentences ("benieuwd wat ze nog meer...", "met plezier gelezen") tagged Feelings by two annotators although the guideline says pleasure is Evaluation.
- Rare labels (≥2-annotator agreement): Style--Context 13, Structure 59, Reception 89, Theme 178, Identification 335, Classification 359. Positive rate for several is 1-3%, so random sampling yields few positives.
- Pool for extra annotation: ~670k reviews in `metadata/review_metadata.tsv.gz` (genre, rating, date), sentence files in `data/review_sentences/`, plus the existing RobBERT predictions (`data/predctions-robbert-sentences-all.tsv.gz`) that can drive targeted sampling.

## Plan

### Phase 0: Fix the codebook (before any new annotation)
1. Write `docs/codebook_v2.md` (machine-readable, language-neutral, with Dutch/EN/IT/RU examples later). Decisions needed from you:
   - **Feelings vs Identification** rule. Proposal: *Feelings* = the reader's own affect toward the book/reading experience (sad, scared, bored, moved, disappointed, expectations unmet). *Identification & immersion* = affect/involvement directed at characters or fictional world, taking their perspective, being absorbed, or linking to own biography (empathy, "had het met hem te doen", "ik ga hen missen", "kon niet stoppen met lezen"). Allow both only when a sentence has a distinct generic emotion AND character involvement. Add a decision tree and 10+ contrastive pairs.
   - Handling of pleasure/expectation/curiosity (Evaluation vs Feelings), Reflection vs Identification, Theme vs Narrative, Context vs Classification, Structure, Other_works vs Author, Content--Other.
   - Drop parent labels from annotation (derive from children).
2. Mine the existing disagreements for the contrastive examples (e.g. all sentences with Feelings∩Identification conflict) and use them as the guideline's worked examples. I'd like you to adjudicate ~100-150 of these yourself: this becomes the **gold calibration set**.

### Phase 1: Agent annotators on existing data (calibrate)
3. Build an annotation harness `src/annotate/` : prompt = codebook v2 + few-shot examples + whole review for context, annotate per sentence, JSON output (labels per sentence_id), temperature 0, resumable, cached, batch API.
4. Run **3 agents** (different model/prompt variants, e.g. Sonnet, Opus, differently-ordered few-shots) on all 1,250 reviews (~13.5k sentences, cheap).
5. Evaluate: agent-vs-each-human kappa, agent-vs-human-majority, agent-agent consistency, and specifically whether agents separate Feelings/Identification. Compare against your adjudicated calibration set to tell which side (anno1/2 vs anno3) the agents side with and whether codebook v2 helps (run codebook v1 vs v2 as ablation).
6. Iterate codebook/prompt on a **development** split; hold out a **test** split of reviews (reuse the existing test set in `data/ground_truth-test_set.tsv` if suitable) never used for prompt tuning.

### Phase 2: Aggregation into improved ground truth
7. Combine 3 humans + 3 agents per label: per-label model (e.g. Dawid-Skene / MACE or simple weighted vote), treating anno3-style systematic bias on Feelings explicitly rather than by majority. Output both hard labels and soft probabilities, plus an `uncertain` flag when annotators split.
8. Human adjudication queue: only sentences where agents and humans disagree on rare/weak labels (budget e.g. 500-1000 sentences, ranked by disagreement). Keep adjudicated decisions as the top-precedence layer and as new few-shot examples.
9. Deliver `data/gt_v2/` with versioned TSVs (hard, soft, provenance), and the re-annotated labels remain traceable to raw ones.

### Phase 3: More positives for rare labels
10. Targeted sampling from the 670k pool: score sentences with the existing RobBERT predictions + keyword/regex heuristics per rare label (Context, Structure, Reception, Theme, Identification, Classification, Feelings, Content--Other), take high-score + a random control stratum per label (so precision/prevalence can still be estimated), stratify by genre/platform/date, keep whole reviews (not isolated sentences) so context is preserved, exclude the 1,250 already annotated.
11. Target: ≥300-500 positives (≥2 sources agree) per label; likely ~3-6k new reviews' worth only for the rarest. Annotate with the 3 agents (+ spot-check 10% by you). Report sampling weights so analyses (prevalence) are not biased by the enrichment.

### Phase 4: Other languages (English, Italian, Russian)
12. Translate codebook v2 into each language, with native-language examples (not just translations); a native reader (or you + agent translation check) validates ~100 sentences per language per rare label.
13. Pool: reviews in those languages (need sources; metadata currently has Dutch only), same sentence-split pipeline; start with the same 3-agent setup, 200-300 reviews per language, human-check 100-150 sentences, compute agent-vs-human kappa and compare with the Dutch numbers before scaling.

## Open questions
- Is there a budget/time limit (tokens, number of adjudicated sentences)? It changes how many reviews Phase 3 can cover.
- Can you adjudicate the calibration set, or are other annotators available?
- Which label set should the final scheme have (keep 18, merge Context/Structure into Stylistic if hopeless)?
- Where do the English/Italian/Russian reviews come from, and are they to be sentence-split already?

## Suggested first steps
1. You decide the Feelings/Identification rule and adjudicate a first batch (I'll extract the conflicting sentences).
2. I build the harness and run the 3-agent pass on the existing 1,250 reviews.

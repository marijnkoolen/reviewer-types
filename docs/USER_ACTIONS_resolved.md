# Resolved user actions reviewer-types

## Decisions taken in chat before USER_ACTIONS.md existed (2026-11-13)

- **Budget:** three agents may run on all 1,250 reviews; human adjudication starts with batches of 10 reviews (about 4 minutes per review).
- **Label set:** keep all labels; merge Style--Context and Style--Structure into Stylistic features only if too few examples (open as UA-004).
- **Context for labelling:** labellers see the whole review. Sentence-window context is a question for model training later.
- **Same context for everyone:** labellers and the trained model get the same input: review text plus genre, rating and year. No titles or outside knowledge. Implemented in `src/annotate/review_meta.py` and codebook v2.1.
- **Adjudicate on agent labels:** from batch 2 on, you adjudicate three Opus runs instead of the three human annotators (human labels pulled you in the wrong direction). Some reviews per batch stay blind to measure anchoring.
- **Codebook rulings** (in `docs/codebook_v2.md` / `docs/codebook_v2.1.md`):
  - Reflection is narrow: how the book makes one think about one's own life, society or the world outside the book. Plain opinion is not Reflection.
  - Compassion and similar emotions toward characters are Feelings, not Identification.
  - New label Metadata (MET): edition, identifier, price, publisher, translator.
  - Quality of a translation ("prachtig vertaald") is Style--Context.
  - "Moeite mee hebben" is Evaluation, not Feelings.
  - "Spectaculaire acties" is not Evaluation; Evaluation needs a good/bad judgement.
  - "Filmisch geschreven" is Style--Context: the list of contexts is open-ended.
  - A slow reveal is Structure; a back-cover blurb is Content--Other + Reception.
  - Hoping for a prize is the reviewer's opinion: Evaluation, not Reception.
  - "Prachtige gedichten" is Evaluation + Stylistic features.
- **Other languages:** you have English, Italian and Russian reviews elsewhere and will stage them later (open as UA-006).

## Resolved 2026-11-13

> **2026-11-14:** this file was emptied by mistake by a script in the session after the reboot, and rebuilt the same day. UA-001 to UA-016 (except UA-015) were recovered by the Claude app from your session backups (UA-027); the rest came from that session's transcript. The state before recovery is in `docs/USER_ACTIONS_resolved.before_recovery.md`.

### UA-001 Label for "left with questions" (2026-11-13, from batch 2 adjudication)
- **Action:** **Decide**.
  - Batch 2, review 1675, sentence 10: "Als uiteindelijk alles tot een eind komt, blijf je als lezer achter met vragen:". You labelled it NAR and noted that it is a reader response, but no subcategory fits.
  - Codebook v2.1 has no rule for it.
- **Options:**
  - (a) **Feelings.** Being left puzzled or unsatisfied counts as an emotional response to expectations (the FEE definition already covers expectations met or not).
  - (b) **Parent `Reader_response` as a fallback label**, for a reader response that fits no subcategory. Cost: a 19th label that will be rare and vague.
  - (c) **No Reader_response label.** Label only the content (here NAR).
  - Recommended: (a). It fits the existing FEE wording and adds no label.
- **Default if unanswered:** no rule; agents will probably pick FEE or nothing, inconsistently.
- **Blocks:** codebook v2.2; nothing now.
- **Response:** a
- **Outcome:** (a). Added to `docs/codebook_v2.2.md` under FEE: being left puzzled, unsatisfied or with open questions is FEE. Batch 3 agents used v2.1, so this applies from the next run.

### UA-005 Run Opus with codebook v1 on batch 2 as an ablation? (2026-11-13, from batch 2)
- **Action:** **Decide**.
  - It would show how much of the agreement gain comes from the v2 rules rather than from the model. It costs one agent run of 117 sentences.
- **Options:** (a) run it; (b) skip it.
- **Recommended:** (b) skip. Your gold for batch 2 was made with v2-era Opus labels visible, so the comparison would be biased toward v2 anyway.
- **Default if unanswered:** (b).
- **Blocks:** nothing.
- **Response:** b
- **Outcome:** (b). No ablation run.

## Resolved 2026-11-14

### UA-002 Adjudicate batch 3 (2026-11-13, from batch 3)
- **Action:** **Run** (the page is built). In `/Users/Shared/claude_projects/impfic/review_annotation/reviewer-types`:
  `open -a Safari data/adjudication/batch3_agents_adjudicate.html`
  - 10 reviews, 122 sentences. Your row starts as the majority of three Opus runs (codebook v2.1). The three runs agree pairwise at kappa 0.93–0.95 and give identical label sets on 92 sentences; the other 30 are tinted.
  - The last 4 reviews are **blind**: no agent labels, nothing prefilled (151224, 166964, 355158, 465480). Label them from scratch.
  - Use only the review text and the header (genre, rating, year); do not look up the book.
  - Press *Export TSV* and leave `batch3_agents_adjudicated.tsv` in `data/adjudication/`.
  - Estimated 40 minutes.
- **Default if unanswered:** no batch 3 gold; the v2.1 check rests on batch 2's 2 blind reviews.
- **Blocks:** the choice of agent setup for the full run (UA-003).
- **Response:** done
- **Outcome:** scored. Saved as `data/adjudication/batch3_final.tsv` (122 sentences, `blind` column). Seen reviews (75 sentences): kappa 0.98 with the Opus majority, 5 sentences changed. Blind reviews (47 sentences): kappa 0.65, 33 changed. Follow-ups: UA-008 to UA-011.

### UA-007 Which metadata goes into the labeller context and the model input? (2026-11-13, from metadata discussion)
- **Action:** **Decide**, and list the extra fields you have (per review and per book) under Response.
  - Rule so far: labellers and the trained model get the same context. Now: review text + genre, rating, year (`src/annotate/review_meta.py`).
  - Whether a field is "usable in a meaningful way" depends on the model:
    - A sentence classifier like the current RobBERT model (512-token limit) can take short fields as a text prefix ("Genre: Suspense · Rating: 2 · …") at almost no cost. Categorical fields like genre help most where a label depends on it (Narrative vs Theme in fiction vs non-fiction, Classification).
    - Long book-level text (blurb, plot summary) competes with the review for the 512 tokens. It would help Narrative vs Theme and Quote, but only with a long-context encoder or an LLM classifier.
    - Identifiers (title, author name, work id) mean nothing to a classifier unless it has learned about those books, so they help labellers but not the model. That breaks the same-context rule.
  - Whether the header helps at all can be measured in training: the same model with and without the prefix.
- **Options:**
  - (a) **Keep the current header** (genre, rating, year) for labelling, and test with/without prefix in training. If it doesn't help, it costs nothing: the labels don't depend much on it.
  - (b) **Add short categorical fields** from your extra metadata (e.g. fiction/non-fiction, series yes/no, original language/translated) to both the header and the model input.
  - (c) **Add book descriptions** too, and commit to a long-context or LLM classifier.
  - (d) **No header**: review text only.
  - Recommended: (a) now, with (b) for any short field that bears on a label (translated → SCX/MET, series → OTW/CTX). The ground truth then doesn't depend on fields the model can't use.
- **Default if unanswered:** (a).
- **Blocks:** nothing for batch 3; it fixes the context for the full run (UA-003).
- **Response:** training and labelling should have the same context. Any metadata that cannot easily be provided for training should not be provided for labelling.
- **Outcome:** principle recorded: labelling gets only metadata that can easily be provided for training and corpus-wide application. The current header (genre, rating, year) qualifies: these fields are in `metadata/review_metadata.tsv.gz` for all 670k reviews (86–94% filled, "unknown" otherwise). No titles, book descriptions or other per-book fields. Already in `CLAUDE.md`, `src/annotate/review_meta.py` and codebook v2.1/v2.2.

### UA-003 Which three agents for the full run on the 1,250 reviews (2026-11-13, from batches 1–2)
- **Action:** **Decide**, after batch 3 is scored.
  - On batch 2 (v2), Opus runs agree with each other at kappa 0.95, Sonnet with Opus at 0.89–0.90.
  - On the 2 blind reviews, you agreed with the Opus majority at 0.90 (poetry review) and 0.52 (review without book knowledge); Sonnet was lower on both.
- **Options:**
  - (a) **Three Opus runs** (two standard, one label-by-label), as in batches 2–3.
  - (b) **Two Opus + one Sonnet**, for more diversity in errors at lower cost.
  - (c) Decide after batch 3 numbers.
  - Recommended: (c), leaning to (a).
- **Default if unanswered:** (a).
- **Blocks:** Phase 1 full run.
- **Update 2026-11-14 (batch 3, codebook v2.1):**
  - The three Opus runs agree pairwise at kappa 0.93–0.95.
  - Your agreement with the Opus majority: 0.98 on the 75 sentences where you saw their labels (you changed 5), but **0.65 on the 47 blind sentences** (33 changed). Batch 2 blind: 0.72. So seen-mode gold is close to the Opus majority, and the honest agreement is at the level of the best human pairs.
  - Most blind differences are codebook boundaries (UA-008, UA-009, UA-010), plus a few sentences where the codebook text points the other way from your label (UA-011). The three runs make the same choices there, so a Sonnet run would not add useful diversity.
  - New recommendation: (a), three Opus runs, after UA-008 to UA-011 are settled in codebook v2.3.
- **Response (2026-11-13):** c
- **Response:** a
- **Outcome (2026-11-14):** (a). The full run uses three Opus runs (two standard, one label-by-label) with codebook v2.3. How to run it at this scale: UA-012.

### UA-004 Merge Style--Context and Style--Structure into Stylistic features? (2026-11-13, from planning)
- **Action:** **Decide**, after the full agent run gives counts.
  - Your earlier answer: keep all labels if there are enough examples, else merge these two into STY.
  - So far: Style--Context 3 sentences in your batch 1 gold, 0–2 per batch 2 run; Structure 11 (batch 1) and 13–15 per batch 2 run.
- **Options:** (a) keep both; (b) merge Context only; (c) merge both. Threshold proposal: at least 150 positive sentences in the full ground truth after targeted sampling (Phase 3).
- **Default if unanswered:** keep both until Phase 3 counts are in.
- **Blocks:** nothing now; model training.
- **Update 2026-11-14:** your gold over batches 1–3 (348 sentences, selected for disagreement on rare labels, so enriched): Structure 34 sentences, Style--Context 6, Metadata 3. Structure is clearly frequent enough. For Style--Context only the full run on 1,250 reviews will give a usable count.
  - Recommended: keep Structure; decide on Style--Context after the full run.
- **Response (2026-11-13):** after batch 3
- **Response:** a
- **Outcome (2026-11-14):** (a). Keep both Style--Context and Style--Structure as separate labels.

### UA-008 Imagining oneself in the character's situation: Identification or Reflection? (2026-11-14, from batch 3 blind)
- **Action:** **Decide**.
  - Review 166964 (blind), sentences 7–10: "erg interessant om te bedenken wat ik zelf zou doen als ik in de situatie…", "Want zou ik bijvoorbeeld zelf bij het verzet durven te gaan?", "En dan op een gevaarlijke missie.", "(Stiekem denk ik van wel want ik kan heel slecht tegen onrecht)".
  - You: REF (+ NAR / THM). All three Opus runs: IDI.
  - The codebook leans to IDI: v1 lists "connection of the content of the book with the reader's biography" and "It made me think about how children have experienced the war" under IDI. The v2 clarification says taking the characters' perspective is IDI.
- **Options:**
  - (a) **IDI**: putting yourself in a character's situation, or linking it to your own character or biography. REF only when the thinking moves beyond the story to one's own life, society or the world in general.
  - (b) **REF**: any thinking about oneself prompted by the book. IDI only for absorption in the story and feeling with or as a character.
  - Recommended: (a). It matches the v1 examples and is the narrower REF you chose in batch 1.
- **Default if unanswered:** (a).
- **Blocks:** codebook v2.3, and with it the full run (UA-003).
- **Response:** a
- **Outcome (2026-11-14):** (a). Added to `docs/codebook_v2.3.md` under IDI, with the four review-166964 sentences as examples.

### UA-009 Historical setting and learning about the past: Narrative or Theme? (2026-11-14, from batch 3 blind)
- **Action:** **Decide**.
  - Review 166964 (blind):
    - "We worden meegenomen naar 1930 een tijd waarin er grote verschillen zijn tussen rijk en arm." You: NAR. Opus: NAR + THM.
    - "erg leuk om meer te weten te komen over Monte Carlo hoe dat een Casino werd…" and "En wat daar allemaal in het verleden heeft afgespeeld." You: EVA + NAR, NAR. Opus: EVA + THM, THM.
  - Codebook v2.1 says the story's setting is NAR, and real-world history discussed as a subject is THM. The agents took the second reading; you took the first. The rule did not separate these cases.
- **Options:**
  - (a) **Setting is NAR**, including the historical period and place and what happened there. THM only when the reviewer names what the book is *about* as a topic or issue ("het boek gaat over klassenverschil", "een boek over de oorlog").
  - (b) **Real history is THM**: whenever the reviewer refers to historical facts or a real period or place, also tag THM.
  - Recommended: (a). It is closest to your labels across batches 2–3 and easier to apply consistently. I will add these sentences as worked examples.
- **Default if unanswered:** (a).
- **Blocks:** codebook v2.3, and with it the full run (UA-003).
- **Response:** a
- **Outcome (2026-11-14):** (a). Replaced the v2.1 NAR vs THM rule in `docs/codebook_v2.3.md`: the setting, including the historical period and place and what happened there, is NAR; THM only when the reviewer names what the book is about. The batch 3 sentences are the worked examples.

### UA-010 Is suspense (or its absence) Feelings? (2026-11-14, from batch 3 blind)
- **Action:** **Decide**.
  - Review 355158: "Spanning is zeker ver te zoeken." You: EVA. Opus: EVA + FEE.
  - Review 151224: "Geen luchtig boek, geen spannend boek, wel een boek met diepgang…". You: EVA NAR STY. Opus: + FEE.
  - Review 166964: "…en dat houdt het boek boeiend." You: EVA NAR. Opus: + FEE.
  - The v1 guideline has "Een spannend verhaal (Feelings and Evaluation)" and "Drie intrigerende verhalen (Feeling + Evaluation)". The agents follow those examples, and they also apply them to negations.
- **Options:**
  - (a) **Keep the v1 examples:** spannend / boeiend / intrigerend = FEE + EVA, also when negated ("geen spannend boek").
  - (b) **Only experienced suspense is FEE:** "ik zat op het puntje van mijn stoel" = FEE + EVA; "spannend", "boeiend" and negations as a description of the book = EVA only.
  - (c) **Never FEE** for these words: EVA only, and drop the v1 examples.
  - Recommended: (b). It matches your labels, and keeps FEE for the reader reporting a feeling. The cost: the v1 examples change, and the old human labels follow (a).
- **Default if unanswered:** (a), the guideline as written.
- **Blocks:** codebook v2.3, and with it the full run (UA-003).
- **Response:** a
- **Outcome (2026-11-14):** (a). Added to `docs/codebook_v2.3.md` under FEE: spannend / boeiend / intrigerend are FEE + EVA, also when negated or absent. Note: your batch 1–3 gold has these as EVA only in some sentences (see UA-013).

### UA-011 Five batch 3 labels where the codebook points the other way (2026-11-14, from batch 3 blind)
- **Action:** **Decide** per sentence, all in blind reviews 355158 and 465480. In each, the three Opus runs follow the codebook text quoted.
  1. 465480-003 "Anders dan de Crimezone-recensie van 15 december doet vermoeden, …". You: OTW. Codebook RCP: "In another review I read …". → RCP + OTW?
  2. 465480-004 "De zwerm is het zesde boek van Schatzing, maar is wel als eerste vertaald in het Nederlands." You: AUT. Codebook: other books mentioned without discussion of the author → OTW; translation information → MET. → AUT + OTW + MET?
  3. 465480-005 "Waarschijnlijk wegens het succes reden voor de uitgever om dan ook zijn debuutroman uit 1995 … [te laten vertalen]". You: AUT SCX. Codebook: SCX is about style (incl. translation *quality*); a translation decision is MET; "het succes" is reception → AUT + MET + RCP?
  4. 465480-012 "Maar dat is geen reden om al zijn vorige boeken alsnog te gaan vertalen." You: EVA. Codebook: earlier books → OTW; EVA only if it refers to the reviewed book → OTW (+ MET)?
  5. 355158-001 "Zo ondertussen heb ik bijna alle feist boeken (met veel plezier) gelezen." You: AUT CTX. Codebook: "If the other books of the author are listed/mentioned without the discussion of the author — use OTW" → CTX + OTW?
- **Options:** (a) take the codebook reading (after the →) for all five; (b) keep your labels for all five; (c) mixed: write the numbers to keep under Response.
- **Recommended:** (a), unless one of them shows the codebook rule itself is wrong.
- **Default if unanswered:** (a).
- **Blocks:** batch 3 gold; nothing else.
- **Response:** See the next few lines:
1. this has nothing to do with reception, but with a factual error in another review.
2. my mistake, it should be AUT + MET + OTW
3. "het succes" refers to a different book in the series. Again, this has nothing to do with reception of the reviewed book. I don't know why I picked SCX, but it's clearly wrong. MET is possible because it states it is his debut novel and that it's from 1995. 
4. This mainly refers to the reviewed book (and indirectly also to other books). So I would argue that OTW is not warranted. This is a sarcastic sentence to indicate that this earlier novel that is being reviewed was not worth translating. The problem is that it's indirect. So, most defensible is no label at all. If anything, it's EVA.
5. The author is explicitly mentioned and is the key reason for referring to other works. Since no other work is mentioned explicitly, I take it that the author connection is much more important. So AUT+CTX.

But overall, I find the Opus labels much better than my own. My take is that the adjudication is good for sharpening the rules, but that human labels are mostly useful for identifying categories that are unclear. Human labellers clearly struggle with this task.
- **Outcome (2026-11-14):** applied to `data/adjudication/batch3_final.tsv`: 1 keeps OTW; 2 → AUT MET OTW; 3 → AUT MET; 4 → no label; 5 keeps AUT CTX. The reasons became v2.3 rules: RCP only for the reception of the reviewed book (not factual corrections of another review, not another book's success); AUT rather than OTW when no other specific work is named; 'debuutroman uit 1995' is MET; indirect or sarcastic remarks with unclear target get no label.
- **Your overall conclusion (recorded in `docs/GROUND_TRUTH_PLAN.md`):** the Opus labels are better than the human ones. Adjudication is for sharpening the rules; human labels mainly show which categories are unclear.

### UA-013 Bring batch 1–3 gold in line with codebook v2.3 (2026-11-14, from UA-008 to UA-010)
- **Action:** **Decide**.
  - Your gold for batches 1–3 (348 sentences) was made under v1–v2.1. The UA-008/009/010 rulings change some of it: for example the suspense sentences you labelled EVA only, and REF on "what would I do in her place".
  - The full run re-annotates these 30 reviews with v2.3, so the conflicts can be listed automatically.
- **Options:**
  - (a) **After the full run, a revision page** with only the gold sentences where your label differs from the v2.3 Opus majority, showing both, the rule that applies and the agents' reasons. You accept or keep per sentence.
  - (b) **Apply the rulings automatically** where the v2.3 majority agrees and the difference is only in the labels affected by UA-008/009/010; a revision page for the rest.
  - (c) Leave the gold as it is.
  - Recommended: (a). It keeps every change to your gold in your hands, and it is likely under 50 sentences.
- **Default if unanswered:** (a).
- **Blocks:** final batch 1–3 gold; nothing before the full run.
- **Response:** a
- **Outcome (2026-11-14):** (a). Planned: after the full run, a revision page with only the batch 1–3 gold sentences where your label differs from the v2.3 Opus majority.

### UA-012 How to run three Opus annotators on all 1,250 reviews (2026-11-14, from UA-003)
- **Action:** **Decide**.
  - Volume per run: 1,250 reviews, 13,473 sentences. One call per review: the codebook (about 5k tokens, the same every call) plus the review and its header, and about 25 output tokens per sentence. That is about 6M input tokens per run (mostly the repeated codebook) and 0.35M output tokens. Three runs triple that.
  - So far each batch was annotated by Claude Code subagents (about 80k tokens per 10 reviews). At 125 chunks × 3 runs that is ~375 subagent calls.
  - This environment has no `ANTHROPIC_API_KEY` set and no `anthropic` package in the venv.
- **Options:**
  - (a) **Message Batches API with prompt caching.** I write `src/annotate/run_agents.py`: one request per review, cached codebook, JSON output, resumable, with all raw responses saved. First I check it on batch 3 against the subagent labels, then run all three. You set the key yourself (I never handle it): `export ANTHROPIC_API_KEY=...` in the shell you start the script from, or a `.env` file that is gitignored. I add `anthropic` to `requirements.txt`. Billed to the API account; batch pricing is about half of normal, and cached input much less.
  - (b) **Claude Code headless (`claude -p`) in a script**, one review per call. Same prompt and resumability as (a), billed to your Claude plan's usage, no API key needed. Slower; plan limits may pause it.
  - (c) **Claude Code subagents as before**, orchestrated in this session. You need to opt in to a multi-agent workflow explicitly. It is the most expensive in tokens and the least reproducible.
  - Recommended: (a). It is cheapest, gives exact model ids and raw outputs per review, and can be rerun on other languages later.
- **Default if unanswered:** nothing runs.
- **Blocks:** Phase 1 full run; and with it UA-013.
- **Update 2026-11-14 (from your question in chat):**
  - The API is billed separately from the subscription, so (a) needs API credits. You don't want to add credits, so (a) is out.
  - The app's usage card shows the **Pro** plan (not Max): 5-hour window 19% used, weekly (all models) **77% used, resetting 2026-11-18 20:00 UTC**. Extra usage is disabled, so hitting a limit stops the work; nothing is billed extra.
  - So the weekly limit, not the 5-hour window, is what binds. Splitting over two 5-hour windows works mechanically (the run is resumable per chunk), but the remaining 23% of the week may not cover three full Opus runs. I can't convert tokens to plan percent without measuring.
  - Revised options, all within the subscription:
    - (d) **Calibrate first:** one Opus pass on 25 reviews (one chunk), and read the usage card before and after. Then choose (e), (f) or (g) on real numbers.
    - (e) **Three full Opus runs**, spread over windows and weeks: the run stops at a limit and resumes from the last finished chunk.
    - (f) **Two full Opus runs, the third only on sentences where they disagree** (about 20–25% of sentences, judging by batches 2–3). It gives the same majority vote at about two-thirds of the cost.
    - (g) As (f), but the tie-breaker run on Sonnet (lower usage per token; agreed with Opus at 0.89–0.90 in batch 2).
  - Mechanism: bigger chunks (25 reviews per call instead of 10) to spread the fixed per-agent overhead, which was most of the ~80k tokens per 10-review run. Either as subagents from a session, or a `claude -p` script you start yourself, which needs no session open. One output file per chunk; finished chunks are skipped on restart.
  - Recommended: (d), then probably (f).
- **Calibration result 2026-11-14 (option d done):**
  - One Opus subagent annotated chunk 0 (25 reviews, 286 sentences) in 3.3 minutes and about 112k tokens. Output `data/full_run/opus_a/chunk_000.json` is complete: all 286 sentences, valid codes only, 8 with no label.
  - Usage card before → after: 5-hour window 19% → 20%, weekly 77% → 77%. The card reports whole percents, and my own turns in this session are in that 1% as well. So one chunk costs at most about 1% of a 5-hour window and well under 1% of the week.
  - Projection (50 chunks per run): one full run is at most about 50% of one 5-hour window. Option (e), three runs, is about 150%, so two windows. Option (f) is about 110–115%, and (g) is a bit less. I can't project the weekly cost precisely because it didn't move. If a 5-hour window is a fraction of the week (typical), one run is a few to ~10% of the week, so (f) likely fits in the 23% left. If it doesn't, the run stops at the limit and resumes after your weekly reset.
  - (The usage tool still labels the plan "Pro"; you see Max in the app. The low cost per chunk fits Max.)
  - Wall clock: 3.3 minutes per chunk, so about 2.75 hours per run sequentially, or about 35 minutes with 5 subagents at a time. Parallel runs fill the 5-hour window faster but cost the same in total.
  - In the one chunk, the rare labels are much more frequent than in the human majority: STR 22, SCX 11, MET 7, RCP 7, IDI 9. That suggests the full run gives Phase 3 a better starting point.
- **Revised recommendation:** (f), run in steps of 10 chunks with a usage reading after each step. If the weekly reading goes past 95%, stop and continue after the weekly reset (2026-11-18 20:00 UTC), which is the reset use you described. Order: opus_a chunks 1–49, then opus_b chunks 0–49, then the opus_c tie-breaker on the sentences where a and b disagree. Up to 5 subagents in parallel, unless you prefer them one at a time so the 5-hour window lasts longer. Answer with (e), (f) or (g), plus "parallel" or "sequential".
- **Response:**
- **Answer (in chat, 2026-11-14):** "(f), parallel"
- **Outcome (2026-11-14):** (f), 5 subagents at a time, steps of 10 chunks with a usage check after each; stop above 95% weekly and resume after the reset. Runs opus_a and opus_b on all 50 chunks, then opus_c on the sentences where they disagree. Prompt: `src/annotate/full_run_prompt.md`; outputs `data/full_run/<run>/chunk_NNN.json`.

### UA-014 Chunk 3 stopped by a safety classifier (2026-11-14, from full run opus_a)
- **Action:** **Decide**.
  - The opus_a annotator for `data/full_run/chunks/chunk_003.json` (25 reviews, 275 sentences) was stopped by a safety classifier while writing its labels. No output was written. It doesn't say which review set it off; the chunk looks like ordinary Dutch book reviews.
  - I haven't rerun it, because the stop said not to produce that content again, and a new annotator would be asked for the same thing.
- **Options:**
  - (a) Leave chunk 3's 25 reviews out of the agent labels. They keep only their human labels, and I mark them `held` in the full-run index. 1,225 of the 1,250 reviews keep the full agent treatment.
  - (b) You read the chunk (`data/full_run/chunks/chunk_003.json`) and name the review(s) you think are the cause. I then leave only those out and run the remaining reviews of the chunk as a smaller chunk.
  - Recommended: (b) if you have a few minutes, otherwise (a). Either way the rest of the run goes on.
- **Update 2026-11-14:** you checked chunk 3 and found only ordinary reviews; the only unusual thing is two sentences with "❤️❤️❤️❤️❤️". So I started one retry with the same input; if it is stopped again, chunk 3 gets (a).
- **Update 2026-11-14, chunk 22:** the opus_a annotator for `data/full_run/chunks/chunk_022.json` (25 reviews, 252 sentences) was also stopped: "Opus 5.5's safeguards flagged this message … This sometimes happens with safe, normal conversations" (request id req_011Ch1pUQrypSUYkZA9U4CEs). Not retried. The same options apply to chunk 22: (a) leave it out, or (b) you name the review(s) you suspect, and I run the rest of the chunk.
- **Update 2026-11-14, opus_b chunk 0:** also stopped by a safety classifier, although opus_a annotated the same chunk 0 without trouble. So the stops look random, not caused by a particular review. The chunk 22 error names `[reasoning_extraction]` as the reason, and the prompt asks for a short "why" per label. That request may look like extracting the model's reasoning.
  - Extra option (c): for the held chunks only, retry once with a prompt without the "why" field (labels only). It's your call whether that is an acceptable change; I won't do it without your answer.
  - Chunk 3 retry (same prompt, same input) succeeded: opus_a chunk 3 is done. Held now: opus_a 22 and 38, opus_b 0, and the opus_c tie-breaker for chunks 5–8 (one subagent, stopped before writing). For comparison: opus_b did chunks 22 and 38 without trouble. Over the run, 5 of about 115 subagents were stopped. Option (d): one plain retry per held chunk, as for chunk 3.
- **Default if unanswered:** (a) at the end of the run, for each held chunk.
- **Blocks:** nothing else; only the held chunks (25 reviews each).
- **Response:**
- **Answer (in chat, 2026-11-14):** "UA 14 d"
- **Outcome (2026-11-14):** (d) one plain retry, same prompt and input, for each held piece: opus_a chunks 22 and 38, opus_b chunk 0, opus_c tie-breaker chunks 5–8. All four succeeded, so no reviews are left out. Then a tie-breaker ran on chunks 0, 22 and 38. In total 5 of about 125 subagents were stopped once by a safety classifier; every retry passed. The stops look random, not tied to particular reviews.

### UA-016 Codebook v2.4 from your gold revision (2026-11-14, from UA-013 revision)
- **Action:** **Decide** whether these rules, taken from your 37 non-v2.3 choices and your notes in `data/adjudication/gold_revision_adjudicated.tsv`, go into `docs/codebook_v2.4.md`. Answer per rule (yes / no / edit), or (a) for all.
  - **R1 Other books get no content labels.** A sentence about another book, including the author's earlier book, gets OTW (+ AUT when it is about the author's development) but no NAR, THM or MET. Those are only for the reviewed book. Conversely, a sentence about the reviewed book is not OTW (for example its film deal is RCP only). Your cases: 356-005/006/007, 168-003/004. Full run: OTW+NAR 263, OTW+THM 23, OTW+MET 49 sentences to re-check.
  - **R2 A publisher's blurb is not reception.** The cover description by the publisher ('de achterflap spreekt van…') is COO only; RCP stays for opinions of others, including a third party's quote printed on the cover (v2.1 rule unchanged). Your case: 337-007. Full run: RCP+COO 18 sentences.
  - **R3 Wondering what you would do is IDI + REF.** Imagining oneself in a character's situation stays IDI (v2.3), and when the reviewer weighs their own behaviour ('zou ik zelf bij het verzet durven te gaan?') it is also REF. Your cases: 964-007…010. Full run: IDI 630 sentences, of which only 10 have REF.
  - **R4 Frustration with a character is FEE, not IDI.** 'Je zou het boek in willen kruipen om haar streng toe te spreken', 'Mens, ga wat uitvinden' = FEE. Your cases: 261-007, 522-012.
  - **R5 EVA needs a judgement of quality.** Curiosity raised by the title or being made to think is not EVA (878-002/003). Wondering why others praise the book is indirect EVA + RCP (867-002). Full run: EVA+COO 248 sentences, mostly about titles and covers.
  - **R6 STY includes how the author renders characters and story.** 'Personages goed neerzetten', 'een prachtige historische young adult neerzetten', 'leest vlot weg' = STY. Your cases: 356-008, 964-001, 811-006.
  - **R7 A sarcastic conditional is not REC.** 'Als je graag wordt aangesproken alsof je twaalf bent, dan zul je dit boek geweldig vinden' = EVA (+ STY), no REC. Your case: 480-009.
- **Options:** (a) adopt R1–R7 as written, then re-check only the affected full-run sentences with one Opus pass under v2.4 (about 1,000 sentences: the counts above plus IDI, EVA+COO and REC candidates; roughly 1% of a week), and run Phase 3 with v2.4. (b) adopt with your edits, same follow-up. (c) keep v2.3; the revised gold stands as is.
  - Recommended: (a). The rules are your own rulings, and fixing them before Phase 3 keeps all 3,850 reviews on one codebook.
- **Default if unanswered:** (c); Phase 3 runs with v2.3.
- **Blocks:** codebook v2.4 and the re-check; Phase 3 should wait for this answer.
- **Response:** a
- **Answer (in chat, 2026-11-14):** "UA-016 a"
- **Outcome (2026-11-14):** (a). `docs/codebook_v2.4.md` written with R1–R7 inline at the affected labels (the v2.1 blurb rule narrowed by R2). 858 full-run sentences in 506 reviews selected as candidates (`src/annotate/make_v24_recheck_input.py`), re-checked by two independent Opus passes; a change is applied only when both passes agree (`src/annotate/apply_v24_recheck.py` -> `data/full_run/labels_v24.tsv`). Phase 3 will use v2.4.

### UA-015 Stage the texts of the 2,600 Phase 3 reviews (2026-11-14, from Phase 3 preparation)
- **Action:** **Edit**: put the sentence-split texts of the reviews listed in `data/phase3/review_ids.txt` (2,600 ids, one per line) in `data/phase3/texts.tsv.gz`: tab-separated, header `review_id	sent_num	sent_text`, one row per sentence, split the same way as the RobBERT predictions (sent_num 1..n). Only 1 of these reviews has text in the project now (`data/review_sentences/` holds every 1,000th review).
  - The sample is drawn and fixed (`data/phase3/sample.tsv`, script `src/annotate/make_phase3_sample.py`): 500 random reviews (prevalence) plus 300 / 400 / 1,000 / 400 reviews enriched for RCP / REF / SCX / MET, about 16,000 sentences. The model expects about 550–670 extra sentences per rare label, which brings each to about 750–950.
  - Estimated cost of the same setup as the full run (two Opus passes plus tie-breaker): about 13–14% of a week.
  - After you stage the texts: `venv/bin/python src/annotate/make_phase3_chunks.py` builds the chunks and reports missing reviews.
- **Default if unanswered:** Phase 3 waits; nothing else is blocked.
- **Blocks:** Phase 3 annotation. I'd start it after your gold revision (in case the codebook changes to v2.4) and after the weekly reset of 2026-11-18.
- **Response:**
- **Answer (in chat, 2026-11-14):** the whole sentence-split corpus staged at `data/review_sentences.tsv.gz` (review_id, num_sentences, sent_num, sent_text).
- **Outcome (2026-11-14):** the 2,600 sampled reviews extracted to `data/phase3/texts.tsv.gz` (15,989 sentences, all reviews present, sentence counts identical to the RobBERT split). Chunks: `data/phase3/chunks/` (52 chunks of 50 reviews, mean 307 sentences; strata mixed in every chunk).

### UA-017 Fine-tune RobBERT on the v2.4 ground truth (2026-11-14, from gt_v2 classifier session)
- **Action:** **Decide** whether to run the transformer step. It downloads `pdelobelle/robbert-v2-dutch-base` (~500 MB, Hugging Face) and trains 4 epochs on the M1 Max GPU (MPS), ~1.5–2 h (measured ~1 s/step, 1,424 steps/epoch, plus ~5 min to predict all 29k sentences). Command, in the project root:
  `/opt/homebrew/opt/python@3.11/bin/python3.11 src/classifier/gt_v2_transformer.py --model pdelobelle/robbert-v2-dutch-base --epochs 4 --name robbert`
- **Options:** (a) run it as above — recommended; (b) run 3 epochs (`--epochs 3`, ~1.2 h); (c) use RobBERT-2023 (`--model DTAI-KULeuven/robbert-2023-dutch-base`, same size, newer pretraining); (d) stop at the TF-IDF baseline.
- **Default if unanswered:** nothing runs; the TF-IDF baseline (test micro-F1 0.72, human gold 0.69) stays the only new model.
- **Blocks:** the RobBERT rows in `results/gt_v2_classifier/` and in the summary in `docs/GROUND_TRUTH_PLAN.md`.
- **Response:**
- **Answer (in chat, 2026-11-14):** "Yes, run 4 epochs" (option a).
- **Outcome (2026-11-14):** run started; results in `results/gt_v2_classifier/robbert/` and the summary in `docs/GROUND_TRUTH_PLAN.md`.

### UA-006 Stage the English, Italian and Russian reviews (2026-11-13, from planning)
- **Action:** **Edit** later: put the reviews somewhere in the project (for example `data/other_languages/<lang>/`) with review id, text and, if available, genre, rating and date, and tell me the path.
- **Default if unanswered:** Phase 4 does not start.
- **Blocks:** Phase 4 (other languages).
- **Response:** There are two main datasets, see below.

Amazon + Goodreads + LibraryThing:
- `/Volumes/T7_Data/Data/claude-projects-external/reader_profiles` this is a related project where I have 46 million reviews, mostly in English, from three large datasets.

Goodreads scrape:
- `/Volumes/T7_Data/Code/impact-of-fiction/impfic-multilingual-reviews/data/Canonical_book_pages` this contains the canonical pages for books based on book lists curatd by Goodreads members. Most of the reviews in these book pages are in English, but there are also reviews in other languages.
- `/Volumes/T7_Data/Code/impact-of-fiction/impfic-multilingual-reviews/data/Book_language_pages/` this contains book pages scraped from Goodreads, for a range of languages, organised by language code. 
- **Outcome (2026-11-14):** paths received (reader_profiles: 46M reviews, mostly English; Goodreads canonical book pages; Goodreads book pages per language). Not readable from this session: macOS denies access to /Volumes/T7_Data and the app refused to add reader_profiles. Follow-up: UA-018.

### UA-018 Give access to the review data on the T7 drive (2026-11-14, from UA-006)
- **Action:** **Decide** / **Run**. I can't read the folders you named in UA-006: macOS answers "Operation not permitted" for everything on `/Volumes/T7_Data`, also outside my sandbox, so the Claude app lacks the macOS privacy permission for that drive. In addition, the app's safety check refused to add `/Volumes/T7_Data/Data/claude-projects-external/reader_profiles` to this session.
- **Options:**
  - (a) In System Settings → Privacy & Security → Files and Folders, allow Claude access to "Removable Volumes" (or add Claude under Full Disk Access), then restart the app. For `reader_profiles`, also add the folder to the session yourself (or a permission rule for it), since I may not request it.
  - (b) Copy a sample into the project instead, e.g. `data/other_languages/<lang>/`, with review id, text and (if available) rating, date and genre. For a first pilot 1,000–2,000 reviews per language (en, it, ru) is enough; I then sentence-split them the same way as the Dutch reviews.
  - Recommended: (b) for the pilot. It keeps the project self-contained and the data you choose is the data that gets labelled; (a) is better once we sample at scale from the 46M reviews.
- **Update (2026-11-14):** `reader_profiles` is now readable (reviews appear to be in `db/reader_profiles.duckdb`; `data/amazon` and `data/librarything` are empty). `/Volumes/T7_Data/Code/impact-of-fiction` still gives "Permission denied" (`/Volumes/T7_Data/Code` is mode `drwxr----x`), so the Goodreads scrape (canonical and per-language book pages) is still unreadable. Option (a) now only concerns that folder: Run in Terminal, any directory: `chmod o+x /Volumes/T7_Data/Code/impact-of-fiction /Volumes/T7_Data/Code/impact-of-fiction/impfic-multilingual-reviews && chmod -R o+rX /Volumes/T7_Data/Code/impact-of-fiction/impfic-multilingual-reviews/data` (this session runs as a different macOS user, so it only gets the "others" permissions) (or copy a sample as in (b)).
- **Default if unanswered:** Phase 4 waits for the Italian and Russian data; English could start from `reader_profiles`.
- **Blocks:** Phase 4 for it and ru (the Goodreads per-language pages are the only non-English source named).
- **Response:**
- **Answer (in chat, 2026-11-14):** "I've updated the permissions" (option a); also fixed the project's group-write permissions.
- **Outcome (2026-11-14):** both T7 sources readable. English pilot drawn from reader_profiles (`data/other_languages/en/`, see the Phase 4 section of `docs/GROUND_TRUTH_PLAN.md`). Goodreads per-language pages (it, ru) readable, not yet used.

### UA-019 Resume the English pilot runs after the crash (2026-11-14, from session after reboot)
- **Action:** **Decide** whether I resume the Phase 4 English pilot annotation. On disk: opus_a has done chunks 000–010 of 12 (checked: every sentence present), chunk_011 is missing; opus_b and the opus_c tie-breaker have not started. Resuming means opus_a on chunk_011, opus_b on all 12 chunks (2,329 sentences), then opus_c on the sentences where they differ. Same prompt and codebook as before (`src/annotate/full_run_prompt.md`, section Phase 4 English pilot; `docs/codebook_v2.4_en.md`).
- **Options:** (a) resume now, keeping opus_a's 11 chunks; about 1.5–2% of a week (scaled from Phase 3) — recommended; (b) resume after the weekly reset of 2026-11-18; (c) redo opus_a from scratch too (the existing chunks are complete, so I see no reason to).
- **Default if unanswered:** nothing runs; the pilot stays at opus_a 11/12.
- **Blocks:** the English agreement and prevalence comparison and your check of 100–150 English sentences (plan step 13).
- **Response:** a
- **Outcome (2026-11-14):** resumed: opus_a chunk_011 and opus_b chunks 000–011 (up to 5 subagents at a time), then opus_c tie-breaker; results in the Phase 4 section of `docs/GROUND_TRUTH_PLAN.md`.
- **Outcome (2026-11-14, later):** done. opus_a/opus_b exact agreement 83.6%; opus_c on 383 sentences, none unresolved; labels in `data/other_languages/en/majority_labels.tsv`. Your check: UA-023.

### UA-020 Restart the RobBERT fine-tune (2026-11-14, from session after reboot)
- **Action:** **Decide** / **Run**. The UA-017 run stopped in epoch 3 of 4 when the system crashed. It finished 2 epochs (dev micro-F1@0.5 0.691 → 0.764) and saved that checkpoint, but it did not predict, so there are no test or gold results. `gt_v2_transformer.py` cannot resume from a checkpoint. Measured speed: ~1 s/step, 47 min for 2 epochs, so 4 epochs take ~1.6 h plus ~5 min to predict. You saw more than 50 GB of swap before the crash. A likely cause in this script: it padded each batch to its own length, so there were up to ~256 tensor shapes, and MPS compiles and keeps a graph per shape (the crash log shows an `mpsgraph` file failing to save). I changed `collate()` in `src/classifier/gt_v2_transformer.py` to pad to a multiple of 32 (at most 8 lengths). Results are unchanged; padding is masked. The command below also caps MPS memory, so an overrun stops the run with an error instead of swapping, and logs to a new file so the log of the crashed run is kept. Run it with no other heavy Python process alongside. Command, in the project root (the model is already downloaded):
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 /opt/homebrew/opt/python@3.11/bin/python3.11 src/classifier/gt_v2_transformer.py --model pdelobelle/robbert-v2-dutch-base --epochs 4 --name robbert 2>&1 | tee results/gt_v2_classifier/robbert_train_rerun.log`
  While it runs, check memory in another Terminal window: `sysctl vm.swapusage` (any directory). Swap that keeps growing past a few GB means stop the run (Ctrl-C).
- **Options:** (a) rerun 4 epochs from scratch as above — recommended, since dev F1 was still rising fast between epochs 1 and 2; (b) rerun with `--epochs 3` (~1.2 h); (c) I add an evaluate-only flag and score the saved 2-epoch checkpoint (~5 min to predict). This gives a lower bound now, and (a) can still run later.
- **Default if unanswered:** nothing runs; the TF-IDF baseline stays the only gt_v2 model with results.
- **Blocks:** the RobBERT rows in `results/gt_v2_classifier/` and in the classifier summary in `docs/GROUND_TRUTH_PLAN.md`.
- **Response:** a
- **Outcome (2026-11-14):** the crashed run was moved to `results/gt_v2_classifier/robbert_crash_2ep/` (2-epoch checkpoint and its log). The rerun started from this session with a watchdog that stops it if swap passes 8 GB or free disk drops below 20 GB (`robbert_rerun_watch.log`). It loads the base model from your Hugging Face cache, offline. Log: `results/gt_v2_classifier/robbert_train_rerun.log`. Correction to the command above: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0` alone fails ("invalid low watermark ratio 1.4"), so it also needs `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8`.

### UA-022 Codebook and human check for Italian and Russian (2026-11-14, from Phase 4 it/ru parsing)
- **Action:** **Decide** which codebook the Italian and Russian pilots use, and who does the human check of 100–150 sentences per language (plan step 13).
- **Options:**
  - (a) Use `docs/codebook_v2.4_en.md` (its English-platform section already covers Goodreads conventions), plus a short per-language file I write: task line, and how common Italian/Russian phrasings map onto existing rules. No new rules. The human check comes after the pilot. Recommended for the pilot: the agents read English rules fine, and the pilot shows which labels need native examples.
  - (b) First translate the codebook with native-language examples (plan step 12) and have it checked by a native reader, then run the pilot.
  - (c) Like (a), but with no per-language notes at all.
  - Human check: (i) you, with agent translations next to the original; (ii) a native reader you name; (iii) skip for the pilot and use only agent-agent agreement.
- **Default if unanswered:** I prepare chunks for Italian (and Russian after UA-021) but do not start the annotation runs.
- **Blocks:** the Italian and Russian pilot runs and their agent-vs-human comparison.
- **Response:** a, the check will be done by colleagues
- **Outcome (2026-11-14):** codebooks `docs/codebook_v2.4_it.md` and `docs/codebook_v2.4_ru.md`: v2.4 plus the platform conventions from v2.4-en plus a section mapping common Italian / Russian phrasings onto existing rules (no new rules). Prompt variant in `src/annotate/full_run_prompt.md`. Italian pilot runs (opus_a, opus_b, opus_c) started; Russian follows once UA-021 is done. Human check by colleagues after the pilots: I'll prepare a check page per language like `en_batch1` (with the English codebook, since the colleagues may not read Dutch examples).

### UA-024 Measure cheaper Claude models for corpus-scale labelling (2026-11-14, from the "larger models" question)
- **Action:** **Decide** whether to run a small API pilot. Haiku 5.5, Sonnet 5.5 and Opus 5.5 (low effort) would label your 348 gold sentences plus ~300 random gt_v2 reviews (~2,500 sentences), with the v2.4 codebook as a cached system prompt and one request per 10 reviews. I then compare each model with the gold and with gt_v2. This needs API credentials: none are set for my user (`ANTHROPIC_API_KEY` is empty, and the `ant` on the PATH is Apache Ant). Edit: put a key in `~/.anthropic_key` readable by the shared group, or tell me another way to get one. It is billed separately from the Claude subscription.
- **Options:** (a) run the pilot on all three models; estimated cost under $15 — recommended; (b) Haiku 5.5 and Sonnet 5.5 only (under $5); (c) no API work; stay with subagents for labelling and the RobBERT-style classifier for the corpus.
- **Default if unanswered:** nothing is spent; the corpus labelling plan stays with the trained classifier.
- **Blocks:** choosing between direct LLM labelling of the 670k Dutch reviews and distilling into a larger encoder.
- **Response:**
- **Answer (in chat, 2026-11-14):** "No, I don't want to spend any money. I have access to multiple GPUs so taking several days to run is no issue." (option c)
- **Outcome (2026-11-14):** no API work. Corpus-scale labelling goes through open models on your GPUs: UA-025.

### UA-021 Sentence splitter for Russian (2026-11-14, from Phase 4 it/ru parsing)
- **Action:** **Decide** / **Run**. Italian can be split like English, with spaCy (`it_core_news_lg` is installed). Russian has no splitter installed: no spaCy `ru_core_news_*` and no NLTK punkt data, and stanza/trankit fail to import.
- **Options:**
  - (a) install the small spaCy Russian model (~15 MB; `pymorphy3` is already there). Same parser-based splitting as English and Italian. Run in Terminal, any directory: `/opt/homebrew/opt/python@3.11/bin/python3.11 -m spacy download ru_core_news_sm` — recommended.
  - (b) I install the same model into my own user's Python packages. Your environment stays untouched, but my installs also get loaded by the shared Homebrew Python (that is already the case for `huggingface_hub`).
  - (c) NLTK punkt for Russian (`python3.11 -c "import nltk; nltk.download('punkt_tab')"`). Rule-based; weaker on informal text without capitals or punctuation.
- **Default if unanswered:** the Russian pool is drawn but not sentence-split; the Italian pilot can go ahead on its own.
- **Blocks:** Russian chunks and the Russian pilot run.
- **Response:** a
- **Answer (in chat, 2026-11-14):** "Can you install the Russian spaCy model?"
- **Outcome (2026-11-14):** installed by me with `python3.11 -m spacy download ru_core_news_sm --user`. Option (b) in effect: the Homebrew site-packages are not writable for my user, so the model is in `/Users/claude-agent/Library/Python/3.11/lib/python/site-packages` and only my user sees it. Russian pool split (16,028 sentences, median 5 per review, max 188); pilot 300 reviews, 2,598 sentences in 12 chunks (`data/other_languages/ru/chunks/`). Runs not started yet (paused).

### UA-025 Open model on your GPUs for corpus-scale labelling (2026-11-14, from UA-024)
- **Action:** **Decide** the approach, and **Edit** this entry with the GPU details: GPU type and memory, how many, where they are, and how jobs run there (ssh, SLURM, …). Also say whether I can reach the machine from here or whether you run the jobs. I write the scripts and job files either way; they read the same split as `results/gt_v2_classifier/` and evaluate on the same test set and on your 348 gold sentences.
- **Options:**
  - (a) Fine-tune an open-weight LLM of 7–14B (multilingual: Qwen, Gemma, Llama, EuroLLM; I pick the newest at download time) with LoRA and an 18-label multi-label head. One forward pass reads the whole review with its header and scores every sentence, read at the end of each sentence. That is the same context the agents get, which the 256-token RobBERT input cannot hold. Training on gt_v2 (29k sentences; English, Italian and Russian added as their pilots finish) takes roughly a day on one 40–80 GB GPU. Inference is one pass per review: hours for the 670k Dutch reviews, days for the 46M English — recommended.
  - (b) A larger encoder (XLM-R-large or mDeBERTa-v3; Dutch-only: RobBERT-2023-large) with the current sentence-plus-neighbours input. Simplest, fits any GPU, smaller gain.
  - (c) Prompt a 70B-class open model with the codebook, no training. A cross-check on the gold rather than the main labeller; it needs 2–4 large GPUs and is slow.
  - Recommended: (a), with (b) as the baseline in the same run.
- **Default if unanswered:** the RobBERT-base result (running now) stays the best corpus model; nothing is set up for GPUs.
- **Blocks:** corpus-scale labelling of the Dutch corpus and the other languages.
- **Response:** a
- **Outcome (2026-11-14):** approach (a), a fine-tuned open LLM with a LoRA multi-label head over the whole review, with (b), a large encoder, as the baseline in the same run. The GPU details are still needed: UA-028.

### UA-026 What goes into the GitHub repository (2026-11-14, from "initialise a github repo")
- **Action:** **Decide**. The project is already a git repository: 2 commits by you (2026-10-13), remote `origin` = `git@github.com:marijnkoolen/reviewer-types.git`, and that GitHub repository is **public** (the GitHub API answers without login). Working tree: 8 tracked files modified (`report/*`, `src/common/*.py`, `src/figures.py`); untracked: `docs/`, `src/annotate/`, `src/classifier/`, eight `src/2[7-9]c_*/30c_*` scripts, `CLAUDE.md`, `ANALYSIS_PLAN.md`, `Makefile`, `requirements.txt`, `notebooks/` (15 MB), `metadata/` (13 MB), `data/` (423 MB, full review texts: the Dutch corpus, the Goodreads/Amazon/LibraryThing pools, the it/ru parses, the adjudication pages), `results/` (23 GB, of which `frog_tsv` 15 GB and `parser_comparison` 7.1 GB). I can commit locally but cannot push: my user has no SSH key and no `gh` login. You push.
- **Options:**
  - (a) Keep the repository public and commit only code, docs and aggregate results. `.gitignore` excludes `data/`, `metadata/` and `results/`, except an allowlist of small summary files (`README.md`, `summary.json`, `per_label.tsv`, `train_log.json`, logs) with no review text. I make the commits in logical groups, with you as author and a Co-Authored-By line for me. Afterwards Run, in the project root: `git push origin main` — recommended.
  - (b) First make the GitHub repository private (GitHub → Settings → Danger Zone → Change visibility), then as (a) plus `notebooks/` and the sentence-level results. Still no `data/`: too big for git, and the review texts should not leave the project.
  - (c) A new, separate repository for the ground-truth and annotation work (docs, `src/annotate`, `src/classifier`); this one stays as it is.
  - Also say whether the 8 modified tracked files go in as they are (they came from earlier sessions; I have not reviewed them).
- **Default if unanswered:** nothing is committed.
- **Blocks:** nothing else.
- **Response:** a
- **Outcome (2026-11-14):** `.gitignore` extended (data/, metadata/, notebooks/, results/ except the summary files of `results/gt_v2_classifier/`); code and docs committed locally in groups, with you as author. The 8 modified tracked files and the other `results/` folders are left out for now (UA-028 asks about the modified files). You push.

### UA-027 Recover the lost text of UA-001 to UA-016 (2026-11-14, from the resolved-file accident)
- **Action:** **Run** in Terminal, as yourself, any directory. This lists backup copies of `docs/USER_ACTIONS_resolved.md` that your Claude sessions made, newest first:
  `grep -rl '^### UA-016 Codebook v2.4' ~/.claude/file-history 2>/dev/null | xargs ls -lt | head -5`
  Then copy the newest hit into the project (replace `<path>` with the first path printed):
  `cp <path> /Users/Shared/claude_projects/impfic/review_annotation/reviewer-types/docs/USER_ACTIONS_resolved.recovered.md`
  If the grep prints nothing, use the prompt I gave in chat in the Claude app instead; it also searches the session transcripts.
- **Options:** (a) run the two commands above, and I merge the recovered entries into the stubs — recommended; (b) send the chat prompt to the Claude app and let it do the merge; (c) leave the stubs as they are.
- **Default if unanswered:** the stubs stay; each one says where its outcome is recorded.
- **Blocks:** nothing.
- **Response:**
- **Answer (2026-11-14):** done via the Claude app (option b): every stub was replaced with recovered text; the copy from before recovery is `docs/USER_ACTIONS_resolved.before_recovery.md`.

### UA-028 GPU details, and the 8 modified tracked files (2026-11-14, from UA-025 and UA-026)
- **Action:** **Edit** this entry under Response:
  - (1) for UA-025 a: GPU type and memory, how many, where they are, how jobs run there (ssh, SLURM, …), and whether I can reach the machine from here or you start the jobs.
  - (2) for UA-026 a: whether the 8 tracked files changed by earlier sessions (`report/report.html`, `report/report.md`, `report/report_phase_c.html`, `report/report_phase_c.md`, `src/common/data_prep.py`, `src/common/silver_data_prep.py`, `src/common/similarity.py`, `src/figures.py`) go into a commit as they are. I have not reviewed those changes; `git diff --stat` shows the size.
  - (3) whether `docs/Annotation-categories-book_reviews.pdf` (the original guidelines; codebook v1 was transcribed from it) may go into the public repository. I left it out with `docs/*.pdf` in `.gitignore`.
- **Options:** (1) free text. (3) yes / no. (2) (a) commit them as one "state of earlier sessions" commit; (b) I review the diffs first and propose commits; (c) leave them uncommitted — recommended until you have looked, since I don't know their origin.
- **Default if unanswered:** (1) I write the training script generic (one CUDA GPU, settings by flag) and smoke-test it here on a tiny model; (2) the files stay uncommitted.
- **Blocks:** (1) the GPU job files and batch sizes; (2) nothing.
- **Response:** (1) 2 VMs, each with 2 A10 GPUs, each A10 has 22GB memory. (2) I've commited the files. (3) no, let's add a link to the original, which is published on Zenodo: https://zenodo.org/records/23335376
- **Outcome (2026-11-16):** (1) the open-model scripts target 2 VMs × 2 A10 (22 GB): 4-bit LoRA (QLoRA) so a 7–9B model trains on one A10, one run per GPU, inference sharded over all four; you start the jobs. (2) noted, committed by you (05d5b56). (3) the PDF stays out of git; the codebook headers now link to https://zenodo.org/records/23335376.

### UA-029 Resume the Italian and Russian pilot runs? (2026-11-16, from the pause of 2026-11-14)
- **Action:** **Decide**. On 2026-11-14 you paused new tasks because the connection was about to drop. State: Italian opus_a 6 of 12 chunks done (000–005), opus_b and opus_c not started; Russian chunked (12 chunks, 2,598 sentences), no runs. Resuming means Italian opus_a 006–011, opus_b 000–011, tie-breaker; then the same for Russian. Codebooks `docs/codebook_v2.4_it.md`, `docs/codebook_v2.4_ru.md`.
- **Options:** (a) resume both now, Italian first, up to 5 agents at a time; roughly 2% of a week per language, as for English — recommended; (b) Italian only; (c) wait (say until when).
- **Default if unanswered:** nothing runs.
- **Blocks:** the Italian and Russian agreement figures and the check pages for your colleagues.
- **Response:**
- **Answer (in chat, 2026-11-16):** "UA 29 yes, let's continue." (option a)
- **Outcome (2026-11-16):** Italian runs resumed (opus_a 006–011, then opus_b, tie-breaker), Russian after; progress in the Phase 4 Italian and Russian section of `docs/GROUND_TRUTH_PLAN.md`.

### UA-031 Same books across languages, or targeted samples per language? (2026-11-16, from the label-share differences)
- **Action:** **Decide** the direction, and answer one factual question. The two directions answer different questions:
  - Same books in all languages (matched): separates "what is reviewed" from "how people review in this language". Needed for any cross-language comparison.
  - Targeted samples per language (recent books, more genre fiction): gives each language a pool like what is actually reviewed in it now. This is what a classifier for that language should be trained and checked on.
  - The genre check (plan, Phase 4 Italian and Russian) shows both matter: within crime/thriller the four languages look alike, within literary fiction it/ru still differ.
- **Options:**
  - (a) Matched first, then targeted — recommended. Matched: about 150 books stratified by genre and publication decade, 2 reviews per language per book, so ~300 reviews per language like the pilots, labelled with the same setup. For it/ru/en this can be drawn from the pages already scraped: 2,590 books have ≥ 5 reviews in both it and ru, and the scrape has English pages for the same books (on the T7 drive). Targeted: I write selection criteria per language (publication year, genre quotas, minimum reviews in the language) and a sampling script over whatever book lists you scrape.
  - (b) Matched only.
  - (c) Targeted only.
- **Question (Edit, under Response):** can the Dutch corpus be linked to Goodreads books? The Dutch work ids are anonymised (`impfic-work-N`). The T7 notebook `impfic-multilingual-reviews/scripts/Link-crawled-and-impfic-reviews.ipynb` suggests you made such a link before; if so, where is it? Without it, Dutch can only be matched by genre and publication year, not by book.
- **Default if unanswered:** nothing is drawn; the pilots stay as they are.
- **Blocks:** the cross-language comparison, and the next Italian/Russian/English annotation rounds.
- **Response:** a. There is a mapping file of book identifiers: `/Users/marijnkoolen/Code/impact-of-fiction/data/book_metadata/work_isbn_title_genre.tsv.gz`.
- **Outcome (2026-11-16):** Dutch works linked to Goodreads by ISBN (`src/annotate/link_dutch_goodreads.py`, 25,366 works); matched sample of 152 books drawn (`data/matched/`). Before annotating, two follow-up choices on the Dutch source and the likes ranking: UA-032.

### UA-032 Matched-books sample: Dutch source and the likes ranking (2026-11-16, from UA-031 a; revised the same day)
- **Action:** **Decide** two things before the matched sample is annotated. Nothing has run yet.
  - Facts:
    - The current draw (`data/matched/`) takes Dutch reviews from the Dutch corpus, not from the 14,487 Dutch Goodreads pages.
    - Goodreads sorts each language page by likes. Of the 152 matched books, the page shows only the 30 most-liked reviews for every English page, 107 Italian, 78 of 151 Dutch and 39 Russian.
    - On such pages, length falls with rank: Italian median 170 words at positions 1–5, 80 at 21–30, and 76 on pages that are not cut off. Russian is similar, weaker. Rank barely depends on review age (Spearman 0.03–0.07).
    - The Italian and Russian pilots were drawn at random from these pages, so they lean towards liked, longer reviews; their labels are fine, but their label shares are not shares of all reviews.
  - (1) Dutch source:
    - (a) Goodreads Dutch pages for the cross-language comparison (same platform, ranking and years as en/it/ru), plus the corpus reviews as a fifth arm (Dutch Goodreads vs Dutch corpus = platform effect), ~3,600 more sentences — recommended;
    - (b) Goodreads Dutch only;
    - (c) corpus only (current draw).
  - (2) Likes ranking:
    - (a) Per book and language, take one review from the top 5 positions and one from the bottom of the page (positions 21–30, or the last third of a shorter page). Every language then gets the same mix of high- and low-ranked reviews. Likes and position are recorded as covariates — recommended;
    - (b) random within the page (current draw);
    - (c) only reviews with 0–1 likes (closest to an ordinary review; drops books whose pages hold only liked reviews).
  - For the later targeted samples, the clean fix is in the scraping: Goodreads' review listing sorted by newest instead of the book page. That is your call; I would only use what you scrape.
- **Default if unanswered:** nothing runs; the current draw stays unannotated.
- **Blocks:** the book-controlled comparison across languages.
- **Response:** 1 a, 2 a.
- **Outcome (2026-11-16):** redrawn with Goodreads Dutch, English, Italian and Russian (one top-5 and one bottom-third review per book and language) plus the Dutch corpus as a fifth arm; 152 books, 304 reviews per arm, 16,863 sentences. Codebook for Dutch Goodreads: `docs/codebook_v2.4_nl_gr.md`. English bottom-band reviews are still liked (median 27.5): every English page is cut off at 30. Running the annotation: UA-033.

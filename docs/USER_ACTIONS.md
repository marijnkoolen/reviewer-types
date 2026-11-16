# Open user actions reviewer-types

Format and rules: `CLAUDE.md`, section "When something is left for the user". Answer under **Response**.
Answered items move to `docs/USER_ACTIONS_resolved.md`.

### UA-023 Check 123 English pilot sentences (2026-11-14, from UA-019)
- **Action:** **Edit**. Open `data/adjudication/en_batch1_agents_adjudicate.html` in a browser (Run, in the project root: `open data/adjudication/en_batch1_agents_adjudicate.html`). It works like the batch 2/3 pages: 21 random English pilot reviews (7 Amazon, 7 Goodreads, 7 LibraryThing), 123 sentences, prefilled with the pilot label; the 29 sentences where opus_a and opus_b disagreed are tinted. The last 6 reviews are blind (no labels shown, no prefill). Correct the labels under codebook v2.4-en, click Export TSV, and save the file as `data/adjudication/en_batch1_agents_adjudicated.tsv`. At about 4 minutes per review this takes roughly 1.5 h.
- **Options:** (a) do the whole page — recommended; (b) only the 6 blind reviews (~25 min; gives the agent-vs-human kappa, but nothing on the disputed sentences); (c) skip, and scale English on agent agreement alone.
- **Default if unanswered:** English stays at the 300-review pilot; the rest of the pool (1,500 reviews) is not annotated.
- **Blocks:** English agent-vs-human kappa, and scaling English to the rest of the pool.
- **Response:** a
- **Status (2026-11-14):** waiting for your exported file. When `data/adjudication/en_batch1_agents_adjudicated.tsv` is there, I compute agent-vs-you agreement (blind and seen) and compare it with the Dutch batches.

### UA-030 Colleagues check the Italian and Russian pilot labels (2026-11-16, from UA-022)
- **Action:** **Edit** (by your colleagues). Send each colleague their page and codebook. The pages hold review text, so send them privately, not through the public repository.
  - Italian: `data/adjudication/it_batch1_agents_adjudicate.html` + `docs/codebook_v2.4_it.md` (21 reviews, 142 sentences, ~1.5 h).
  - Russian: `data/adjudication/ru_batch1_agents_adjudicate.html` + `docs/codebook_v2.4_ru.md` (21 reviews, 106 sentences, ~1.5 h).
  The pages work like your batch 2/3 pages: prefilled with the pilot labels; sentences where the two agent passes disagreed are tinted; the last 6 reviews are blind (no labels, no prefill). Ask them to label every sentence of the blind reviews themselves, correct the rest under the codebook, click Export TSV and send you the file. Save the files as `data/adjudication/it_batch1_agents_adjudicated.tsv` and `data/adjudication/ru_batch1_agents_adjudicated.tsv`.
- **Options:** (a) as above — recommended; (b) only the 6 blind reviews per language (~30 min each; gives the agent-vs-human agreement only); (c) no check.
- **Default if unanswered:** Italian and Russian stay at the 300-review pilots, and there is no agent-vs-human agreement for them.
- **Blocks:** the Italian and Russian agent-vs-human comparison, and scaling them to the rest of their pools (1,500 reviews each).
- **Response:**

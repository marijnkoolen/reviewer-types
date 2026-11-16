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

### UA-032 Run the matched-books annotation? (2026-11-16, from UA-031 a)
- **Action:** **Decide**. The matched sample is ready (`data/matched/`): 152 books, the same books in Dutch, English, Italian and Russian, 304 reviews per language, 13,821 sentences in all (English 5,144 — English reviews are long, median 12 sentences — Dutch 3,653, Russian 2,546, Italian 2,478). Same setup as the pilots: opus_a, opus_b, opus_c tie-breaker, per-language codebooks (Dutch `codebook_v2.4.md`). Estimated cost: about 11–12% of a week (scaled from Phase 3: 16k sentences ≈ 13–14%), ~40% more than four pilot-sized sets.
- **Options:** (a) run all four languages — recommended; it gives the book-controlled comparison and ~14k more training sentences; (b) run without English, or with only the English reviews up to 400 words (~60% of the English sentences), to save ~3–5% of a week; (c) wait until after the weekly reset (2026-11-18).
- **Default if unanswered:** nothing runs.
- **Blocks:** the book-controlled comparison across languages.
- **Response:**

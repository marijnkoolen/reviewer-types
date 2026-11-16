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

### UA-028 GPU details, and the 8 modified tracked files (2026-11-14, from UA-025 and UA-026)
- **Action:** **Edit** this entry under Response:
  - (1) for UA-025 a: GPU type and memory, how many, where they are, how jobs run there (ssh, SLURM, …), and whether I can reach the machine from here or you start the jobs.
  - (2) for UA-026 a: whether the 8 tracked files changed by earlier sessions (`report/report.html`, `report/report.md`, `report/report_phase_c.html`, `report/report_phase_c.md`, `src/common/data_prep.py`, `src/common/silver_data_prep.py`, `src/common/similarity.py`, `src/figures.py`) go into a commit as they are. I have not reviewed those changes; `git diff --stat` shows the size.
  - (3) whether `docs/Annotation-categories-book_reviews.pdf` (the original guidelines; codebook v1 was transcribed from it) may go into the public repository. I left it out with `docs/*.pdf` in `.gitignore`.
- **Options:** (1) free text. (3) yes / no. (2) (a) commit them as one "state of earlier sessions" commit; (b) I review the diffs first and propose commits; (c) leave them uncommitted — recommended until you have looked, since I don't know their origin.
- **Default if unanswered:** (1) I write the training script generic (one CUDA GPU, settings by flag) and smoke-test it here on a tiny model; (2) the files stay uncommitted.
- **Blocks:** (1) the GPU job files and batch sizes; (2) nothing.
- **Response:**

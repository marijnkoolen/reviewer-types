# reviewer-types

## When something is left for the user

Never leave a user action as prose in the chat ("you may want to…", "if it
matters…", "worth checking…"). Every item that needs the user — a decision,
a command, a file edit, a confirmation — takes one of three forms:

- **Run** — a copy-pasteable command, with the directory it runs in.
- **Edit** — the file path, the location (section heading or line), and the
  exact text to add or replace.
- **Decide** — an entry in `docs/USER_ACTIONS.md` (format below); give its
  ID in the chat.

Each item also says **what happens if the user does nothing** and **what it
blocks** ("nothing" is a valid answer). End any message that leaves items
open with a short **For you** list of the items and their IDs.

### `docs/USER_ACTIONS.md`

The user answers here, not in the chat. One entry per item:

    ### UA-NNN <short title> (YYYY-MM-DD, from <step or session>)
    - **Action:** what to do, in Run / Edit / Decide form
    - **Options:** (a) … (b) … — recommended: …
    - **Default if unanswered:** …
    - **Blocks:** …
    - **Response:**

- At the start of a session, run `grep -n -A1 '^### UA-\|Response:'
  docs/USER_ACTIONS.md` rather than reading the whole file.
- Act on every item whose **Response** is filled in and touches your task.
  Record the outcome where it belongs (plan, codebook, step file). Then
  move the entry to `docs/USER_ACTIONS_resolved.md` with the date and what
  was done.
- Never fill in a **Response** yourself. Number IDs sequentially and never
  reuse one.
- Keep `docs/USER_ACTIONS.md` to open items only, so it stays cheap to read.

## Ground-truth annotation work

- Plan: `docs/GROUND_TRUTH_PLAN.md`. Current codebook: `docs/codebook_v2.4.md` (full run used v2.3; batch 3 agents used v2.1).
- Adjudication batches, pages and agent runs: `data/adjudication/`; scripts: `src/annotate/`.
- Gold (your adjudication, batches 1–3, revised): `data/adjudication/gold_batches1-3.tsv`. Combined ground truth (v2.4, 3,850 reviews): `data/gt_v2/labels.tsv`; parts: `data/full_run/labels_v24.tsv` (1,250 original reviews), `data/phase3/majority_labels.tsv` (2,600 sampled reviews). Phase 3 sample: `data/phase3/`.
- Labellers (humans, agents) and the model trained on the ground truth get the same context:
  the whole review plus the header from `src/annotate/review_meta.py` (genre, rating, year). No outside knowledge of the book.
- Adjudication exports are tab-separated without quoting: read them with `quoting=csv.QUOTE_NONE` (a sentence can start with a `"`).

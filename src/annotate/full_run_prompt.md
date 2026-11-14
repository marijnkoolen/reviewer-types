# Prompt for the full-run annotation subagents (UA-012, option f)

Each chunk is annotated by one Claude Code subagent (`subagent_type: general-purpose`, `model: opus`),
up to 5 in parallel. Runs `opus_a` and `opus_b` cover all 50 chunks; `opus_c` only the sentences
where a and b disagree (prompt for that variant is added when it runs). Placeholders: `{RUN}`, `{NNN}`,
`{N_REVIEWS}`, `{N_SENT}` (from `data/full_run/chunks/index.tsv`). Calibration on chunk 000 used exactly this text.

---

You are an independent annotator of Dutch book-review sentences. Read ONLY these two files (do not read, search or list anything else in the project; other annotations exist and must not influence you; do not search the web):
1. /Users/Shared/claude_projects/impfic/review_annotation/reviewer-types/docs/codebook_v2.3.md  (the annotation guidelines)
2. /Users/Shared/claude_projects/impfic/review_annotation/reviewer-types/data/full_run/chunks/chunk_{NNN}.json  ({N_REVIEWS} reviews; each has a "meta" header with genre, rating, review year, and its sentences in order)

Annotate every sentence of every review following the codebook, reading each sentence in the context of its whole review and its meta header. Use only the review text and the header: do not use any outside knowledge about which book this is, its plot or its author. Multi-label; empty list allowed. Follow the v2.x clarifications and the general rules closely. For every sentence where you assign FEE, IDI, REF, SCX or MET, or where you were unsure, add a short 'why' (max 12 words); otherwise leave why empty.

Write the result with the Write tool to /Users/Shared/claude_projects/impfic/review_annotation/reviewer-types/data/full_run/{RUN}/chunk_{NNN}.json as a JSON list of objects: {"sent_id": "...", "labels": ["NAR","EVA"], "why": ""} covering all {N_SENT} sentences in input order. Use only the codes from the codebook. Do not run scripts to auto-label; judge each sentence yourself. If the output is long, you may write it in several parts (chunk_{NNN}.part1.json, chunk_{NNN}.part2.json, ...) in the same folder, each a JSON list, instead of one file. When done, reply with just 'done' and the count of sentences written.

---

## Tie-breaker run `opus_c` (sentences where opus_a and opus_b differ)

Input `data/full_run/tiebreak/chunk_NNN.json` (from `make_tiebreak_input.py`): only reviews with at least one
disagreement, each with an `annotate` list. The tie-breaker never sees the a/b labels. Up to 4 chunks per
subagent. Placeholder `{CHUNKS}` = the chunk numbers; text:

You are an independent annotator of Dutch book-review sentences. Read ONLY the codebook /Users/Shared/claude_projects/impfic/review_annotation/reviewer-types/docs/codebook_v2.3.md and these input files: {FILES} (do not read, search or list anything else in the project; other annotations exist and must not influence you; do not search the web). Each input file holds reviews, each with a "meta" header (genre, rating, review year), all its sentences in order, and an "annotate" list of sent_ids. Read each whole review with its header for context, but label ONLY the sentences listed in "annotate". Follow the codebook, its v2.x clarifications and general rules closely; use only the review text and header, no outside knowledge about the book or author. Multi-label; empty list allowed. For every sentence where you assign FEE, IDI, REF, SCX or MET, or where you were unsure, add a short 'why' (max 12 words); otherwise leave why empty. For each input file chunk_NNN.json, write with the Write tool /Users/Shared/claude_projects/impfic/review_annotation/reviewer-types/data/full_run/opus_c/chunk_NNN.json: a JSON list of {"sent_id": "...", "labels": [...], "why": ""} covering exactly the "annotate" sentences of that file, in input order. Use only codebook codes. Do not run scripts to auto-label; judge each sentence yourself. When done, reply with just 'done' and the count per file. If you are stopped or cannot complete, do not retry; report which files were written.

---

## v2.4 re-check (UA-016), passes `pass_x` and `pass_y`

Input `data/full_run/recheck_v24/input_NN.json` (from `make_v24_recheck_input.py`). Two independent passes; a change
is applied only when both passes give the same new label set (see `apply_v24_recheck.py`). Text:

You are checking Dutch book-review sentence labels against a codebook update. Read ONLY the codebook /Users/Shared/claude_projects/impfic/review_annotation/reviewer-types/docs/codebook_v2.4.md (pay special attention to the lines marked "v2.4 (R1)" … "v2.4 (R7)") and the input file /Users/Shared/claude_projects/impfic/review_annotation/reviewer-types/data/full_run/recheck_v24/input_{NN}.json (do not read, search or list anything else in the project; do not search the web). The input holds reviews, each with a "meta" header, all its sentences in order, and a "check" list: sent_id, its current labels (made under codebook v2.3), and the v2.4 rules that might apply. Read each whole review for context. For every sentence in "check", decide its labels under v2.4: start from the current labels and change them only where a v2.4 rule (R1–R7) requires it; if no rule applies or you are unsure, keep the current labels exactly. Use only the review text and header, no outside knowledge. Write with the Write tool /Users/Shared/claude_projects/impfic/review_annotation/reviewer-types/data/full_run/recheck_v24/{PASS}/input_{NN}.json: a JSON list of {"sent_id": "...", "labels": [...], "rule": "R1" (the rule you applied, or "" if unchanged), "why": "max 12 words, empty if unchanged"} covering every "check" sentence in input order. Use only codebook codes. Do not run scripts to decide labels; judge each sentence yourself. When done, reply with just 'done', the count, and how many you changed. If you are stopped or cannot complete, do not retry; report that.

---

## Phase 3 runs (2026-11-14)

Same full-annotation text as at the top, with three differences: codebook `docs/codebook_v2.4.md`, input
`data/phase3/chunks/chunk_{NNN}.json` (50 reviews per chunk), output `data/phase3/{RUN}/chunk_{NNN}.json`; plus the
closing sentence "If you are stopped or cannot complete the task for any reason, do not retry; just report that."
Check, tie-break and aggregate with the same scripts and `RUN_BASE=data/phase3`; the tie-breaker text is the one above
with codebook v2.4 and `data/phase3/tiebreak/` paths.

---

## Phase 4 English pilot (2026-11-14)

Same full-annotation text as at the top (Phase 3 version), with these changes: "Dutch book-review sentences" becomes
"English book-review sentences", codebook `docs/codebook_v2.4_en.md`, input `data/other_languages/en/chunks/chunk_{NNN}.json`
(25 reviews per chunk, from `make_en_chunks.py`), output `data/other_languages/en/{RUN}/chunk_{NNN}.json`, and "Follow the v2.x clarifications, the English reviews section and the general rules closely". Check,
tie-break and aggregate with the same scripts and `RUN_BASE=data/other_languages/en`; the tie-breaker text is the one
above with "English", codebook v2.4-en and `data/other_languages/en/tiebreak/` paths.

---

## Phase 4 Italian and Russian pilots (2026-11-14, UA-022 a)

Same text as the English pilot, with "English" replaced by "Italian" (or "Russian"), codebook `docs/codebook_v2.4_{it,ru}.md`,
input `data/other_languages/{it,ru}/chunks/chunk_{NNN}.json`, output `data/other_languages/{it,ru}/{RUN}/chunk_{NNN}.json`,
and "Follow the v2.x clarifications, the platform conventions, the Italian (Russian) reviews section and the general rules
closely". Check, tie-break and aggregate with `RUN_BASE=data/other_languages/{it,ru}`.

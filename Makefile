# Project make targets.
#
# The parser-comparison pipeline lives in src/parsers/ and writes to
# results/parser_comparison/. See src/parsers/README.md for the method.

PYTHON      ?= python3
OUT         := results/parser_comparison
ALPINO_IMAGE ?= alpino-impfic:latest
ALPINO_SHARDS ?= 8

.PHONY: help parser-report parser-run parser-run-docs parser-compare \
        alpino-image lemma-trainset lemma-train lemma-eval \
        bench-alpino bench-alpino-docker bench-frog validate-heads \
        list-runs kill-runs scaling-frog clean-parser-report

help:
	@echo "parser-report    build report/parser_comparison.html from its .md"
	@echo "parser-run       run all parsers per sentence line"
	@echo "parser-run-docs  run all parsers per whole review (sentence splitting)"
	@echo "parser-compare   re-run the three comparison scripts"
	@echo "alpino-image     build the Alpino docker image from the RUG tarball"
	@echo "lemma-trainset   build spaCy lemmatizer training data (SILVER=alpino)"
	@echo "lemma-train      train the lemmatizer on it"
	@echo "lemma-eval       score the retrained lemmatizer on held-out reviews"
	@echo "bench-alpino     benchmark Alpino on the fixed sentence split (SHARDS=22)"
	@echo "bench-alpino-docker  same, inside the Alpino image (no local install)"
	@echo "                     needs sudo? use DOCKER_CMD=\"sudo docker\""
	@echo "bench-frog       same input and shards, with Frog"
	@echo "validate-heads   check the TSV dependency heads against Alpino triples"
	@echo "scaling-frog     sweep shard counts to find Frog's throughput knee"
	@echo "list-runs        show this project's running shards and containers"
	@echo "kill-runs        stop them all (use after a kill -9 orphaned some)"

# --- report ---------------------------------------------------------------
# build_report_html.py needs pandoc on PATH; it inlines figures as data URIs
# so the HTML is self-contained.
report/parser_comparison.html: report/parser_comparison.md src/build_report_html.py
	$(PYTHON) src/build_report_html.py parser_comparison \
	  "Dutch parsers for online book reviews: a gold-free comparison"

parser-report: report/parser_comparison.html
	@echo "wrote report/parser_comparison.html"

clean-parser-report:
	rm -f report/parser_comparison.html

# --- running the parsers --------------------------------------------------
parser-run:
	bash src/parsers/run_all.sh

parser-run-docs:
	$(PYTHON) src/parsers/run_frog.py --doc-mode
	$(PYTHON) src/parsers/run_stanza.py --doc-mode
	$(PYTHON) src/parsers/run_spacy.py --doc-mode
	@for i in $$(seq 0 $$(($(ALPINO_SHARDS) - 1))); do \
	  $(PYTHON) src/parsers/run_alpino.py --docker $(ALPINO_IMAGE) --doc-mode \
	    --shard $$i --n-shards $(ALPINO_SHARDS) \
	    > $(OUT)/_alpino_doc_shard$$i.log 2>&1 & \
	done; wait
	@head -1 $(OUT)/tokens_doc_alpino.shard0.tsv > $(OUT)/tokens_doc_alpino.tsv
	@for s in $(OUT)/tokens_doc_alpino.shard*.tsv; do tail -n +2 $$s >> $(OUT)/tokens_doc_alpino.tsv; done
	@rm -f $(OUT)/tokens_doc_alpino.shard*.tsv

parser-compare:
	$(PYTHON) src/parsers/compare_tokenisation.py
	$(PYTHON) src/parsers/compare_lemmas.py
	$(PYTHON) src/parsers/compare_compounds.py
	$(PYTHON) src/parsers/compare_sentences.py

# --- benchmarks on the fixed sentence split -------------------------------
# Alpino is CPU-only, so on the dual-A10 VM the 22 cores are the resource that
# matters and the GPUs sit idle. Override with SHARDS=, LIMIT=, SPEED=.
SENT_TSV ?= data/review_sentences/review_sentences-aggr.tsv.gz
SHARDS   ?= 22

CHUNK ?= 1000

# KEEP_XML=1 archives the full Alpino XML (one .tar.gz per chunk). Recommended
# for the real corpus run: head_tok_id is a 96.5%-fidelity reconstruction, and
# the XML is the only way to revise it without re-parsing.
KEEP_XML ?=
# Empty by default: the drivers treat an empty LIMIT/SPEED as "not set" and use
# their own defaults (no cap, -veryfast). Both are forwarded explicitly rather
# than relying on make's command-line export, because the docker target starts
# a fresh environment and only sees what -e passes in.
LIMIT ?=
SPEED ?=

bench-alpino:
	INPUT=$(SENT_TSV) SHARDS=$(SHARDS) CHUNK=$(CHUNK) KEEP_XML=$(KEEP_XML) \
	  LIMIT=$(LIMIT) SPEED=$(SPEED) bash src/parsers/bench_alpino.sh

# Same benchmark, run inside the Alpino image: one container, SHARDS processes.
# The runner is stdlib-only, so no venv is needed in the container.
#
# --user keeps output files owned by you rather than root. When this target is
# invoked through sudo (because the Docker socket needs it), `id -u` inside the
# recipe would report root, so SUDO_UID/SUDO_GID -- which sudo sets to the
# *invoking* user -- take precedence when present. Better still, add yourself
# to the docker group so sudo is not needed at all:
#     sudo usermod -aG docker "$USER" && newgrp docker
# Set DOCKER_CMD="sudo docker" to keep running unprivileged while sudo'ing only
# the docker call itself, which also keeps $(id -u) correct.
DOCKER_CMD ?= docker

bench-alpino-docker:
	$(DOCKER_CMD) run --rm -v "$$PWD":/work -w /work \
	  --user "$${SUDO_UID:-$$(id -u)}:$${SUDO_GID:-$$(id -g)}" -e HOME=/tmp \
	  -e SHARDS=$(SHARDS) -e CHUNK=$(CHUNK) -e INPUT=$(SENT_TSV) \
	  -e KEEP_XML=$(KEEP_XML) -e LIMIT=$(LIMIT) -e SPEED=$(SPEED) \
	  $(ALPINO_IMAGE) bash src/parsers/bench_alpino.sh

bench-frog:
	INPUT=$(SENT_TSV) SHARDS=$(SHARDS) LIMIT=$(LIMIT) \
	  bash src/parsers/bench_frog.sh

# Parses a sample twice -- once to XML through our converter, once through
# Alpino's own end_hook=dependencies -- and compares head assignments.
VALIDATE_LIMIT ?= 200
validate-heads:
	$(PYTHON) src/parsers/validate_heads.py --docker $(ALPINO_IMAGE) \
	  --input $(SENT_TSV) --limit $(VALIDATE_LIMIT)

# Orphan recovery. The drivers now trap INT/TERM and clean up after themselves,
# but a kill -9 cannot be trapped, and runs started before that fix leave shards
# behind. Patterns are specific to this project's scripts so nothing else on a
# shared machine is touched.
RUN_PATTERN = run_alpino_tsv\.py|run_frog_tsv\.py|bench_alpino\.sh|bench_frog\.sh

# Frog does not scale linearly with cores: one process does ~275 sentences/s,
# but 22 concurrent processes fell to 1.6 each. This finds the knee.
STEPS ?= 1 2 4 8 16 22
scaling-frog:
	INPUT=$(SENT_TSV) STEPS="$(STEPS)" LIMIT=$(if $(LIMIT),$(LIMIT),200) \
	  bash src/parsers/scaling_frog.sh

list-runs:
	@echo "--- processes ---"; pgrep -af "$(RUN_PATTERN)" || echo "(none)"
	@echo "--- containers ---"; docker ps --filter ancestor=$(ALPINO_IMAGE) \
	  --format '{{.ID}} {{.Names}} {{.Status}}' 2>/dev/null || true
	@echo "--- load ---"; echo "cores: $$(nproc 2>/dev/null || sysctl -n hw.ncpu)"; \
	  vmstat 1 2 2>/dev/null | tail -1 || true

kill-runs:
	@n=$$(pgrep -f "$(RUN_PATTERN)" | wc -l | tr -d ' '); \
	  echo "stopping $$n process(es)"; \
	  pkill -f "$(RUN_PATTERN)" 2>/dev/null || true; \
	  sleep 3; \
	  pkill -9 -f "$(RUN_PATTERN)" 2>/dev/null || true
	@c=$$(docker ps -q --filter ancestor=$(ALPINO_IMAGE) 2>/dev/null); \
	  if [ -n "$$c" ]; then echo "killing containers: $$c"; docker kill $$c >/dev/null; \
	  else echo "no containers"; fi
	@echo "--- remaining ---"; pgrep -af "$(RUN_PATTERN)" || echo "(clean)"

alpino-image:
	docker build --platform linux/amd64 \
	  -f src/parsers/docker/Dockerfile.alpino -t $(ALPINO_IMAGE) src/parsers/docker

# --- spaCy lemmatizer retraining ------------------------------------------
SILVER ?= alpino
LEMMA_DIR = $(OUT)/lemma_train_$(SILVER)

lemma-trainset:
	$(PYTHON) src/parsers/build_lemma_trainset.py --silver $(SILVER) \
	  --out-dir $(LEMMA_DIR)

lemma-train: lemma-trainset
	$(PYTHON) -m spacy train $(LEMMA_DIR)/config.cfg \
	  --paths.train $(LEMMA_DIR)/train.spacy \
	  --paths.dev $(LEMMA_DIR)/dev.spacy \
	  --output $(LEMMA_DIR)/model

# Runs the retrained model over the corpus, then scores it on the held-out
# reviews. Do not judge a retrained model by compare_lemmas.py's flags: they are
# circular when the model was trained on one of the tools being compared.
lemma-eval:
	$(PYTHON) src/parsers/run_spacy.py --model $(LEMMA_DIR)/model/model-best \
	  --out $(OUT)/tokens_spacy_retrained.tsv
	$(PYTHON) src/parsers/eval_retrained_lemmatizer.py --silver $(SILVER)

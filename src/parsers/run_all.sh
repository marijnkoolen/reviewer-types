#!/usr/bin/env bash
# Run every available parser over data/review_sentences/0 and concatenate shards.
#
# Sharding model: each shard is an independent OS process writing its own TSV,
# concatenated afterwards. This is deliberate -- python-frog deadlocks inside a
# multiprocessing Pool, and Alpino is a separate binary anyway -- and it is the
# same shape you want on a many-core VM: one shard per core.
#
# On the dual-A10 VMs:
#   ALPINO_SHARDS=11  (Alpino is CPU-only; one shard per core)
#   STANZA_DEVICE=cuda  and run two Stanza shards, one per GPU, with
#   CUDA_VISIBLE_DEVICES=0 / =1.
#
# Usage: bash src/parsers/run_all.sh [frog|stanza|spacy|alpino ...]
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=results/parser_comparison
mkdir -p "$OUT"

ALPINO_SHARDS="${ALPINO_SHARDS:-8}"
ALPINO_IMAGE="${ALPINO_IMAGE:-alpino-impfic:latest}"
STANZA_DEVICE="${STANZA_DEVICE:-cpu}"
SPACY_MODEL="${SPACY_MODEL:-nl_core_news_lg}"
TOOLS=("$@")
[ ${#TOOLS[@]} -eq 0 ] && TOOLS=(frog stanza spacy alpino)

concat_shards () {  # $1 = tool name
  local tool="$1"
  shopt -s nullglob
  local shards=("$OUT/tokens_${tool}".shard*.tsv)
  [ ${#shards[@]} -eq 0 ] && return 0
  # Header from the first shard, bodies from all of them.
  head -1 "${shards[0]}" > "$OUT/tokens_${tool}.tsv"
  for s in "${shards[@]}"; do tail -n +2 "$s" >> "$OUT/tokens_${tool}.tsv"; done
  rm -f "${shards[@]}"
  echo "  concatenated ${#shards[@]} shards -> tokens_${tool}.tsv"
}

for tool in "${TOOLS[@]}"; do
  echo "=== $tool ==="
  case "$tool" in
    frog)
      # Frog runs ~4.4k tokens/s single-process; sharding is unnecessary here.
      python3 src/parsers/run_frog.py 2>"$OUT/_frog.log" || echo "  FAILED"
      ;;
    stanza)
      python3 src/parsers/run_stanza.py --device "$STANZA_DEVICE" \
        2>"$OUT/_stanza.log" || echo "  FAILED"
      ;;
    spacy)
      python3 src/parsers/run_spacy.py --model "$SPACY_MODEL" \
        2>"$OUT/_spacy.log" || echo "  FAILED"
      ;;
    alpino)
      pids=()
      for i in $(seq 0 $((ALPINO_SHARDS - 1))); do
        python3 src/parsers/run_alpino.py --docker "$ALPINO_IMAGE" \
          --shard "$i" --n-shards "$ALPINO_SHARDS" \
          > "$OUT/_alpino_shard${i}.log" 2>&1 &
        pids+=($!)
      done
      fail=0
      for p in "${pids[@]}"; do wait "$p" || fail=1; done
      [ "$fail" -eq 1 ] && echo "  one or more shards FAILED (see $OUT/_alpino_shard*.log)"
      concat_shards alpino
      ;;
    *) echo "  unknown tool: $tool" ;;
  esac
done

echo
echo "=== comparisons ==="
python3 src/parsers/compare_tokenisation.py
python3 src/parsers/compare_lemmas.py
python3 src/parsers/compare_compounds.py

#!/usr/bin/env bash
# Benchmark Alpino on the fixed sentence split, one shard per core.
#
# On the dual-A10 VM (2 x 11 = 22 cores) Alpino uses no GPU at all -- it is
# CPU-only -- so the GPUs are idle here and the core count is what matters.
#
#   bash src/parsers/bench_alpino.sh                      # native, 22 shards
#   SHARDS=4 LIMIT=200 bash src/parsers/bench_alpino.sh   # quick smoke test
#   DOCKER=alpino-impfic:latest bash src/parsers/bench_alpino.sh   # local
#
# Env:
#   ALPINO_HOME  native install (default /opt/Alpino); ignored if DOCKER is set
#   DOCKER       image name; use instead of a native install
#   SHARDS       parallel shards (default 22)
#   LIMIT        sentences per shard, for a quick timing run
#   CHUNK        sentences per chunk (default 1000). Output is flushed and
#                progress saved after each chunk, so this is also the resume
#                granularity. Each chunk pays one Alpino startup (~5s), so keep
#                it large enough that startup is noise: at 1-3 sentences/s a
#                1000-sentence chunk runs 5-15 minutes and startup is under 2%.
#   KEEP_XML     set to 1 to archive the full Alpino XML, one .tar.gz per
#                chunk. The TSV's head_tok_id is a reconstruction (96.5% vs
#                Alpino's own triples); the XML is the only way to redo that
#                without re-parsing. Enable it from the START of a run --
#                turning it on mid-run only archives chunks from that point,
#                because it is not part of the resume signature.
#   SPEED        -veryfast (default), -fast, or empty for the slowest/best
#                (passed as --speed=VALUE: a bare "-veryfast" would be read
#                 by argparse as an option name, not a value)
#   INPUT/OUT    paths
set -uo pipefail
cd "$(dirname "$0")/../.."

INPUT="${INPUT:-data/review_sentences/review_sentences-aggr.tsv.gz}"
OUT="${OUT:-results/alpino_tsv}"
SHARDS="${SHARDS:-22}"
SPEED="${SPEED:--veryfast}"
CHUNK="${CHUNK:-1000}"
ALPINO_HOME="${ALPINO_HOME:-/opt/Alpino}"
mkdir -p "$OUT"

if [ -n "${DOCKER:-}" ]; then
  BACKEND=(--docker "$DOCKER")
else
  BACKEND=(--alpino-home "$ALPINO_HOME")
  if [ ! -x "$ALPINO_HOME/bin/Alpino" ]; then
    echo "no Alpino at $ALPINO_HOME (set ALPINO_HOME or DOCKER)" >&2; exit 1
  fi
fi
EXTRA=()  # expanded below with the bash-3.2-safe idiom
[ -n "${LIMIT:-}" ] && EXTRA+=(--limit "$LIMIT")
[ -n "${KEEP_XML:-}" ] && EXTRA+=(--keep-xml)

echo "input=$INPUT shards=$SHARDS chunk=$CHUNK speed=$SPEED keep_xml=${KEEP_XML:-0} backend=${BACKEND[0]}"
echo "(re-running this same command resumes any shard that did not finish)"
rm -f "$OUT"/alpino_bench.shard*.json

# Without this, killing the driver orphans every shard: the shells exit but the
# backgrounded Python processes -- and the Alpino/Frog children they spawned --
# keep running and keep their memory. Several interrupted runs then pile up,
# oversubscribing the machine (watch `r` in vmstat) while each run looks idle.
# Killing the negative pid signals the whole process group, children included.
cleanup() {
  trap - INT TERM EXIT
  echo "interrupted: stopping shards and their children" >&2
  kill -- -$$ 2>/dev/null
}
trap cleanup INT TERM

START=$(date +%s)
pids=()
for i in $(seq 0 $((SHARDS - 1))); do
  python3 src/parsers/run_alpino_tsv.py "${BACKEND[@]}" \
    --input "$INPUT" --out-dir "$OUT" --speed="$SPEED" \
    --shard "$i" --n-shards "$SHARDS" --chunk-size "$CHUNK" "${EXTRA[@]+"${EXTRA[@]}"}" \
    > "$OUT/_bench_shard${i}.log" 2>&1 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
WALL=$(( $(date +%s) - START ))
[ "$fail" -eq 1 ] && echo "WARNING: at least one shard failed; see $OUT/_bench_shard*.log" >&2

python3 - "$OUT" "$WALL" "$SHARDS" <<'PY'
import glob, json, sys
out, wall, shards = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
metas = [json.load(open(f)) for f in sorted(glob.glob(f"{out}/alpino_bench.shard*.json"))]
if not metas:
    sys.exit("no shard results")
sents = sum(m["sentences"] for m in metas)
toks = sum(m["tokens"] for m in metas)
noparse = sum(m["no_parse"] for m in metas)
skips = sum(m["sentences_with_skips"] for m in metas)
slowest = max(m["seconds"] for m in metas)
per = [m["sents_per_sec"] for m in metas]
resumed = [m for m in metas if m.get("resumed")]
print(f"\n=== Alpino benchmark: {len(metas)}/{shards} shards ===")
if resumed:
    print(f"  NOTE: {len(resumed)} shard(s) resumed. Wall clock below covers "
          f"only this\n        invocation, so the aggregate rate is not a "
          f"clean benchmark number;\n        use the per-shard 'seconds' "
          f"(cumulative) instead.")
print(f"  sentences          {sents:,}")
print(f"  tokens             {toks:,}")
print(f"  no parse           {noparse:,}")
print(f"  partial (skips>0)  {skips:,}  ({100*skips/max(1,sents):.2f}%)")
print(f"  wall clock         {wall}s   (slowest shard {slowest:.0f}s)")
print(f"  per shard          {min(per):.2f}-{max(per):.2f} sentences/s")
print(f"  AGGREGATE          {sents/max(1,wall):.2f} sentences/s, "
      f"{toks/max(1,wall):,.0f} tokens/s")
# Project to a full corpus using the measured aggregate rate.
rate = sents / max(1, wall)
print(f"\n  projection at this rate:")
for label, n in [("1M sentences", 1e6), ("6.4M sentences (~100M words)", 6.4e6)]:
    h = n / rate / 3600
    print(f"    {label:<32} {h:8.1f} h  ({h/24:.1f} days)")
PY

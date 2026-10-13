#!/usr/bin/env bash
# Benchmark Frog on the fixed sentence split, one shard per core.
#
# Mirrors bench_alpino.sh: same input, same sharding, same output shape, so the
# two benchmarks are directly comparable. Frog is CPU-only, so on the dual-A10
# VM the 22 cores are what matters and the GPUs sit idle.
#
#   bash src/parsers/bench_frog.sh                  # 22 shards
#   SHARDS=4 LIMIT=200 bash src/parsers/bench_frog.sh
#
# Env: SHARDS, LIMIT, INPUT, OUT, PARSER=1 (enable Frog's own dependency parser)
set -uo pipefail
cd "$(dirname "$0")/../.."

INPUT="${INPUT:-data/review_sentences/review_sentences-aggr.tsv.gz}"
OUT="${OUT:-results/frog_tsv}"
SHARDS="${SHARDS:-22}"
PYTHON="${PYTHON:-python3}"
mkdir -p "$OUT"           # must exist before any shard redirects its log here

EXTRA=()  # expanded below with the bash-3.2-safe idiom
[ -n "${LIMIT:-}" ] && EXTRA+=(--limit "$LIMIT")
[ -n "${PARSER:-}" ] && EXTRA+=(--parser)

# Preflight. Frog is a C++ library with its own model files; pip installing
# python-frog does not bring either. Failing here with an explanation beats 22
# shards each dying with the same traceback.
if ! "$PYTHON" - <<'PY'
import sys
try:
    import frog
except ImportError as exc:
    sys.exit(f"cannot import frog: {exc}\n"
             "  python-frog is a binding for the Frog C++ library, which must be\n"
             "  installed separately (Debian/Ubuntu: apt install frog libfrog-dev,\n"
             "  or use LaMachine), then: pip install python-frog")
try:
    frog.Frog(frog.FrogOptions(parser=False, morph=True, lemma=True,
                               mwu=False, ner=False, chunking=False))
except Exception as exc:
    sys.exit(f"frog imported but would not initialise: {exc}\n"
             "  This is normally missing model data. Frog looks for frog.cfg in\n"
             "  ~/.config/frog/nld/ or <prefix>/etc/frog/; install the Dutch\n"
             "  models (apt install frog-data, or LaMachine) and check that\n"
             "  frog.cfg exists there.")
PY
then
  echo "preflight failed; not launching shards" >&2
  exit 1
fi

echo "input=$INPUT shards=$SHARDS out=$OUT"
rm -f "$OUT"/frog_bench.shard*.json

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
  "$PYTHON" src/parsers/run_frog_tsv.py --input "$INPUT" --out-dir "$OUT" \
    --shard "$i" --n-shards "$SHARDS" "${EXTRA[@]+"${EXTRA[@]}"}" \
    > "$OUT/_bench_shard${i}.log" 2>&1 &
  pids+=($!)
done
failed=0
for p in "${pids[@]}"; do wait "$p" || failed=$((failed + 1)); done
WALL=$(( $(date +%s) - START ))

if [ "$failed" -gt 0 ]; then
  echo "ERROR: $failed/$SHARDS shard(s) failed. Last lines of the first failure:" >&2
  for i in $(seq 0 $((SHARDS - 1))); do
    if [ ! -f "$OUT/frog_bench.shard${i}.json" ]; then
      echo "--- $OUT/_bench_shard${i}.log ---" >&2
      tail -15 "$OUT/_bench_shard${i}.log" >&2
      break
    fi
  done
fi

"$PYTHON" - "$OUT" "$WALL" "$SHARDS" <<'PY'
import glob, json, sys
out, wall, shards = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
files = sorted(glob.glob(f"{out}/frog_bench.shard*.json"))
if not files:
    sys.exit(f"no shard results in {out}/. Every shard failed -- see "
             f"{out}/_bench_shard*.log for the reason.")
metas = [json.load(open(f)) for f in files]
sents = sum(m["sentences"] for m in metas)
toks = sum(m["tokens"] for m in metas)
bad = sum(m["failed"] for m in metas)
slowest = max(m["seconds"] for m in metas)
per = [m["sents_per_sec"] for m in metas]
print(f"\n=== Frog benchmark: {len(metas)}/{shards} shards ===")
print(f"  sentences          {sents:,}")
print(f"  tokens             {toks:,}")
print(f"  failed sentences   {bad:,}")
print(f"  wall clock         {wall}s   (slowest shard {slowest:.1f}s)")
print(f"  per shard          {min(per):.1f}-{max(per):.1f} sentences/s")
print(f"  AGGREGATE          {sents/max(1,wall):,.1f} sentences/s, "
      f"{toks/max(1,wall):,.0f} tokens/s")
rate = sents / max(1, wall)
print("\n  projection at this rate:")
for label, n in [("1M sentences", 1e6), ("6.4M sentences (~100M words)", 6.4e6)]:
    h = n / rate / 3600
    print(f"    {label:<32} {h:8.2f} h")
PY

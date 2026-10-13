#!/usr/bin/env bash
# Measure how Frog's per-shard throughput degrades as shards are added.
#
# A single Frog process does ~275 sentences/s. If that scaled with cores, 22
# shards would give ~6000 sentences/s and the whole corpus in ~20 minutes. It
# does not: at 22 shards each shard fell to 1.6 sentences/s, a 170x collapse,
# uniform across every shard.
#
# Uniform degradation points at a *shared* bottleneck rather than CPU
# oversubscription -- most likely memory bandwidth and cache, since Frog's
# tagger and MBMA are TiMBL memory-based classifiers doing nearest-neighbour
# lookups over large in-memory models, and every process streams its own copy.
# This sweep finds the knee so you can pick the shard count that actually
# maximises throughput, which may be far below the core count.
#
#   bash src/parsers/scaling_frog.sh
#   STEPS="1 2 4 8" LIMIT=200 bash src/parsers/scaling_frog.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

INPUT="${INPUT:-data/review_sentences/review_sentences-aggr.tsv.gz}"
STEPS="${STEPS:-1 2 4 8 16 22}"
LIMIT="${LIMIT:-200}"
OUT="${OUT:-results/frog_scaling}"
PYTHON="${PYTHON:-python3}"
mkdir -p "$OUT"

echo "sweep: shards = $STEPS, $LIMIT sentences each"
echo
printf '%8s %12s %14s %16s %10s\n' shards per-shard aggregate corpus-hours efficiency
for n in $STEPS; do
  rm -f "$OUT"/frog_bench.shard*.json
  pids=()
  for i in $(seq 0 $((n - 1))); do
    "$PYTHON" src/parsers/run_frog_tsv.py --input "$INPUT" --out-dir "$OUT" \
      --shard "$i" --n-shards "$n" --limit "$LIMIT" \
      > "$OUT/_scale_${n}_${i}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || true; done
  "$PYTHON" - "$OUT" "$n" <<'PY'
import glob, json, sys, statistics as st
out, n = sys.argv[1], int(sys.argv[2])
m = [json.load(open(f)) for f in glob.glob(f"{out}/frog_bench.shard*.json")]
if not m:
    print(f"{n:>8} {'FAILED':>12}"); sys.exit()
# Pooled per-shard rate: total work over total busy time, so a straggler does
# not distort it the way a wall-clock aggregate would.
sents = sum(x["sentences"] for x in m)
secs = sum(x["seconds"] for x in m)
per = sents / secs
agg = per * len(m)
hours = 6_893_445 / agg / 3600
print(f"{n:>8} {per:>12.1f} {agg:>14.1f} {hours:>16.2f} {'':>4}", end="")
PY
  # Efficiency relative to one shard, written on the same line.
  "$PYTHON" - "$OUT" "$n" <<'PY'
import glob, json, sys, os
out, n = sys.argv[1], int(sys.argv[2])
m = [json.load(open(f)) for f in glob.glob(f"{out}/frog_bench.shard*.json")]
per = sum(x["sentences"] for x in m) / sum(x["seconds"] for x in m)
ref_file = f"{out}/.baseline"
if n == 1 or not os.path.exists(ref_file):
    open(ref_file, "w").write(str(per))
ref = float(open(ref_file).read())
print(f"{100*per/ref:>9.1f}%")
PY
done
echo
echo "efficiency = per-shard rate as % of the single-shard rate."
echo "Pick the shard count with the highest aggregate, not the highest shard count."

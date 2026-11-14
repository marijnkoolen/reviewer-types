"""Compare two full runs and build the tie-breaker input (UA-012 option f).

usage: python src/annotate/make_tiebreak_input.py RUN_A RUN_B      e.g. opus_a opus_b
For every chunk done in both runs (see check_full_run.py), finds the sentences whose label sets
differ. Writes data/full_run/tiebreak/chunk_NNN.json with only the reviews that contain at least
one such sentence, in the chunk format plus a per-review "annotate" list of sent_ids. The
tie-breaker sees the whole review but labels only those sentences. Also prints agreement stats.
Existing tiebreak files are not overwritten.
"""
import glob, json, os, sys
from collections import Counter

dry = '--dry' in sys.argv          # stats only, write nothing
ra, rb = [a for a in sys.argv[1:] if a != '--dry'][:2]
BASE = os.environ.get('RUN_BASE', 'data/full_run')   # e.g. RUN_BASE=data/phase3
CH, OUT = f'{BASE}/chunks', f'{BASE}/tiebreak'
if not dry:
    os.makedirs(OUT, exist_ok=True)


def load(run, n):
    f = f'{BASE}/{run}/chunk_{n:03d}.json'
    return {o['sent_id']: frozenset(o['labels']) for o in json.load(open(f))} if os.path.exists(f) else None


n_chunks = len(glob.glob(f'{CH}/chunk_*.json'))
tot = same = 0
lab_a, lab_b, lab_both = Counter(), Counter(), Counter()
written, missing = [], []
for n in range(n_chunks):
    a, b = load(ra, n), load(rb, n)
    if a is None or b is None:
        missing.append(n); continue
    chunk = json.load(open(f'{CH}/chunk_{n:03d}.json'))
    if set(a) != set(b):
        missing.append(n); continue
    out = {}
    for rid, r in chunk.items():
        diff = []
        for s in r['sentences']:
            sid = s['sent_id']; tot += 1
            for l in a[sid]: lab_a[l] += 1
            for l in b[sid]: lab_b[l] += 1
            for l in a[sid] & b[sid]: lab_both[l] += 1
            if a[sid] == b[sid]:
                same += 1
            else:
                diff.append(sid)
        if diff:
            out[rid] = {**r, 'annotate': diff}
    f = f'{OUT}/chunk_{n:03d}.json'
    if not dry and not os.path.exists(f):
        json.dump(out, open(f, 'w'), ensure_ascii=False, indent=1)
    written.append((n, sum(len(r['annotate']) for r in out.values()), len(out)))

print(f'{len(written)} chunks compared, {tot} sentences, exact label-set agreement {same / max(tot, 1):.3f}')
print(f'tie-breaker: {sum(w[1] for w in written)} sentences in {sum(w[2] for w in written)} reviews')
print('not comparable yet:', ' '.join(f'{n:03d}' for n in missing) or '-')
print('label   a     b     both  agreement(F1 a vs b)')
for l in sorted(set(lab_a) | set(lab_b), key=lambda x: -(lab_a[x] + lab_b[x])):
    f1 = 2 * lab_both[l] / (lab_a[l] + lab_b[l]) if lab_a[l] + lab_b[l] else 0
    print(f'{l:6} {lab_a[l]:5} {lab_b[l]:5} {lab_both[l]:5}  {f1:.2f}')

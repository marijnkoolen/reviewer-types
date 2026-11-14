"""Check full-run outputs: which chunks are complete, merge part files, list what is left.

usage: python src/annotate/check_full_run.py RUN [RUN ...]     e.g. opus_a opus_b
A chunk counts as done when data/full_run/RUN/chunk_NNN.json covers exactly the chunk's sent_ids
with valid codes. Part files (chunk_NNN.partK.json) are merged into chunk_NNN.json when together
they are complete; incomplete or invalid outputs are reported and listed under 'todo' (rerun them).
"""
import glob, json, os, re, sys

CODES = set('AUT CLS OTW REC NAR COO QUO THM EVA FEE IDI CTX RCP REF SCX STR STY MET'.split())
BASE = os.environ.get('RUN_BASE', 'data/full_run')   # e.g. RUN_BASE=data/phase3
CH = f'{BASE}/chunks'


def expected(n):
    d = json.load(open(f'{CH}/chunk_{n:03d}.json'))
    return [s['sent_id'] for r in d.values() for s in r['sentences']]


def load(run, n):
    f = f'{BASE}/{run}/chunk_{n:03d}.json'
    parts = sorted(glob.glob(f'{BASE}/{run}/chunk_{n:03d}.part*.json'),
                   key=lambda p: int(re.search(r'part(\d+)', p).group(1)))
    if os.path.exists(f):
        return json.load(open(f)), False
    if parts:
        return [x for p in parts for x in json.load(open(p))], True
    return None, False


n_chunks = len(glob.glob(f'{CH}/chunk_*.json'))
for run in sys.argv[1:]:
    done, todo, problems = [], [], []
    for n in range(n_chunks):
        try:
            out, from_parts = load(run, n)
        except json.JSONDecodeError as e:
            problems.append(f'{n:03d}: invalid JSON ({e})'); todo.append(n); continue
        if out is None:
            todo.append(n); continue
        exp = expected(n)
        ids = [o.get('sent_id') for o in out]
        bad = [o for o in out if not isinstance(o.get('labels'), list) or not set(o['labels']) <= CODES]
        if sorted(ids) != sorted(exp) or bad:
            problems.append(f'{n:03d}: {len(set(ids) & set(exp))}/{len(exp)} sentences, '
                            f'{len(set(ids) - set(exp))} unknown ids, {len(bad)} bad label lists')
            todo.append(n); continue
        if from_parts:
            order = {s: i for i, s in enumerate(exp)}
            json.dump(sorted(out, key=lambda o: order[o['sent_id']]),
                      open(f'{BASE}/{run}/chunk_{n:03d}.json', 'w'), ensure_ascii=False, indent=1)
        done.append(n)
    print(f'{run}: {len(done)}/{n_chunks} chunks done')
    for p in problems:
        print('  problem', p)
    print('  todo:', ' '.join(f'{n:03d}' for n in todo) or '-')

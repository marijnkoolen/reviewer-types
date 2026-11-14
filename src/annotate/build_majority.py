"""Combine the full-run passes into one label set per sentence (UA-012 option f).

usage: python src/annotate/build_majority.py
input : data/full_run/{opus_a,opus_b,opus_c}/chunk_NNN.json, data/full_run/chunks/chunk_NNN.json
output: data/full_run/majority_labels.tsv  (tab-separated, no quoting; read with quoting=csv.QUOTE_NONE)
        columns: review_id sent_id sent_text opus_a opus_b opus_c labels status
status:
  agree       opus_a and opus_b gave the same label set -> labels = that set
  majority    they differed and opus_c labelled it -> per-label vote, kept when >= 2 of 3
  unresolved  they differed, no opus_c label (held tie-breaker) -> labels = intersection of a and b
  single      only one of opus_a/opus_b exists (held chunk) -> labels = that run
Label lists are comma-separated codes in codebook order; empty string = no label.
"""
import csv, glob, json, os
from collections import Counter

CODES = 'AUT CLS OTW REC NAR COO QUO THM EVA FEE IDI CTX RCP REF SCX STR STY MET'.split()
ORDER = {c: i for i, c in enumerate(CODES)}
FR = os.environ.get('RUN_BASE', 'data/full_run')   # e.g. RUN_BASE=data/phase3


def run(name, n):
    f = f'{FR}/{name}/chunk_{n:03d}.json'
    return {o['sent_id']: set(o['labels']) for o in json.load(open(f))} if os.path.exists(f) else {}


def fmt(s):
    return '' if s is None else ','.join(sorted(s, key=ORDER.get))


rows, status = [], Counter()
for f in sorted(glob.glob(f'{FR}/chunks/chunk_*.json')):
    n = int(f[-8:-5])
    a, b, c = run('opus_a', n), run('opus_b', n), run('opus_c', n)
    for rid, r in json.load(open(f)).items():
        for s in r['sentences']:
            sid = s['sent_id']
            la, lb, lc = a.get(sid), b.get(sid), c.get(sid)
            if la is not None and lb is not None:
                if la == lb:
                    lab, st = la, 'agree'
                elif lc is not None:
                    votes = Counter(list(la) + list(lb) + list(lc))
                    lab, st = {l for l, v in votes.items() if v >= 2}, 'majority'
                else:
                    lab, st = la & lb, 'unresolved'
            elif la is not None or lb is not None:
                lab, st = (la if la is not None else lb), 'single'
            else:
                lab, st = None, 'missing'
            status[st] += 1
            text = s['text'].replace('\t', ' ').replace('\n', ' ')
            rows.append([rid, sid, text, fmt(la), fmt(lb), fmt(lc), fmt(lab), st])

with open(f'{FR}/majority_labels.tsv', 'w', newline='') as fh:
    w = csv.writer(fh, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
    w.writerow(['review_id', 'sent_id', 'sent_text', 'opus_a', 'opus_b', 'opus_c', 'labels', 'status'])
    w.writerows(rows)

print(len(rows), 'sentences;', dict(status))
cnt = Counter(l for r in rows for l in r[6].split(',') if l)
print('label counts (final):', ' '.join(f'{c}:{cnt[c]}' for c in CODES))
print('no label:', sum(1 for r in rows if r[6] == '' and r[7] != 'missing'))

"""Label a review corpus with a model trained by gt_v2_llm.py, sharded over GPUs and resumable (UA-025 a).

usage (one process per GPU; shard K of N takes every N-th review):
  CUDA_VISIBLE_DEVICES=0 python src/classifier/gt_v2_llm_predict.py --run llm_qwen25_7b --shard 0 --num_shards 4
input:  --sentences data/review_sentences.tsv.gz (review_id, num_sentences, sent_num, sent_text; sentences of a review
        on consecutive rows) and --meta metadata/review_metadata.tsv.gz (nur_genre, rating, review_date), turned into
        the same review_meta header the model was trained with. The base model is read from the run's train_log.json.
output: results/corpus_labels/<run>/shard_KK_of_NN/part_PPPPP.tsv.gz: review_id, sent_num, p_<label> (3 decimals) and
        pred (labels at the dev-tuned thresholds of results/gt_v2_classifier/<run>/summary.json). A part holds
        --part_size reviews; parts that exist are skipped, so a stopped job continues where it stopped.
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'annotate'))
from gt_v2_data import LABELS, OUT  # noqa: E402
from gt_v2_llm import PAD_MULTIPLE, Encoder  # noqa: E402
from review_meta import format_header  # noqa: E402


def reviews(path, shard, num_shards):
    """Yield (review_id, [(sent_num, text)]) for this shard, streaming the file."""
    cur, rows, k = None, [], -1
    for chunk in pd.read_csv(path, sep='\t', quoting=csv.QUOTE_NONE, keep_default_na=False, chunksize=200_000,
                             usecols=['review_id', 'sent_num', 'sent_text'], dtype={'sent_num': int}):
        for rid, n, t in zip(chunk.review_id, chunk.sent_num, chunk.sent_text):
            if rid != cur:
                if cur is not None and k % num_shards == shard:
                    yield cur, rows
                cur, rows, k = rid, [], k + 1
            rows.append((n, t))
    if cur is not None and k % num_shards == shard:
        yield cur, rows


def load(run, device, bits):
    from peft import PeftModel
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    cfg = json.load(open(f'{OUT}/{run}/train_log.json'))['args']
    adapter = f'{OUT}/{run}/adapter'
    tok = AutoTokenizer.from_pretrained(adapter)
    kw = {}
    if bits == 4:
        kw = {'quantization_config': BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                        bnb_4bit_compute_dtype=torch.bfloat16,
                                                        bnb_4bit_use_double_quant=True), 'device_map': {'': 0}}
    base = AutoModel.from_pretrained(cfg['model'], dtype=torch.bfloat16 if device == 'cuda' else torch.float32, **kw)
    if bits != 4:
        base = base.to(device)
    model = PeftModel.from_pretrained(base, adapter).eval()
    head = torch.nn.Linear(base.config.hidden_size, len(LABELS))
    head.load_state_dict(torch.load(f'{adapter}/head.pt', map_location='cpu'))
    sp = f'{OUT}/{run}/summary.json'
    if os.path.exists(sp):
        th = json.load(open(sp))['thresholds']
    else:                                                       # smoke runs write no thresholds
        print(f'warning: {sp} missing, using 0.5 for every label')
        th = {l: 0.5 for l in LABELS}
    return tok, model, head.to(device).float(), np.array([th[l] for l in LABELS]), cfg


@torch.no_grad()
def score(model, head, seqs, device, pad_id, batch_tokens):
    """seqs: list of (ids, pos). Batches by length; returns one probability array per sequence."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    out = [None] * len(seqs)
    i = 0
    while i < len(order):
        L = -(-len(seqs[order[i]][0]) // PAD_MULTIPLE) * PAD_MULTIPLE
        j = i + 1
        while j < len(order):                                   # grow the batch while it fits the token budget
            Lj = -(-len(seqs[order[j]][0]) // PAD_MULTIPLE) * PAD_MULTIPLE
            if Lj * (j - i + 1) > batch_tokens:
                break
            L, j = Lj, j + 1
        idx = order[i:j]
        x = torch.full((len(idx), L), pad_id, dtype=torch.long)
        m = torch.zeros((len(idx), L), dtype=torch.long)
        for r, k in enumerate(idx):
            ids = seqs[k][0]
            x[r, :len(ids)] = torch.tensor(ids); m[r, :len(ids)] = 1
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
            h = model(input_ids=x.to(device), attention_mask=m.to(device)).last_hidden_state
        for r, k in enumerate(idx):
            out[k] = torch.sigmoid(head(h[r, seqs[k][1]].float())).cpu().numpy()
        i = j
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='name of the gt_v2_llm.py run')
    ap.add_argument('--sentences', default='data/review_sentences.tsv.gz')
    ap.add_argument('--meta', default='metadata/review_metadata.tsv.gz')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--part_size', type=int, default=2000)
    ap.add_argument('--batch_tokens', type=int, default=8192, help='padded tokens per forward pass')
    ap.add_argument('--bits', type=int, default=4, choices=[4, 16])
    ap.add_argument('--limit', type=int, default=0, help='>0: stop after this many reviews (test)')
    a = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if a.bits == 4 and device != 'cuda':
        sys.exit('--bits 4 needs CUDA (bitsandbytes); use --bits 16 elsewhere')
    tok, model, head, th, cfg = load(a.run, device, a.bits)
    enc = Encoder(tok, cfg['max_len'])
    m = pd.read_csv(a.meta, sep='\t', usecols=['review_id', 'nur_genre', 'rating', 'review_date'],
                    dtype={'review_date': str})
    hdr = {r.review_id: format_header(r.nur_genre, r.rating, r.review_date) for r in m.itertuples()}
    unknown = 'Genre: unknown · Rating: unknown · Review year: unknown'
    out = f'results/corpus_labels/{a.run}/shard_{a.shard:02d}_of_{a.num_shards:02d}'
    os.makedirs(out, exist_ok=True)
    cols = ['review_id', 'sent_num'] + [f'p_{l}' for l in LABELS] + ['pred']
    t0, done, part, buf = time.time(), 0, 0, []

    def flush(buf, part):
        path = f'{out}/part_{part:05d}.tsv.gz'
        if os.path.exists(path):
            return
        seqs, keys = [], []
        for rid, rows in buf:
            for ids, idx, pos in enc.groups(hdr.get(rid, unknown), [t for _, t in rows]):
                seqs.append((ids, pos)); keys.append((rid, [rows[i][0] for i in idx]))
        P = score(model, head, seqs, device, tok.pad_token_id, a.batch_tokens)
        recs = []
        for (rid, nums), p in zip(keys, P):
            for n, row in zip(nums, p):
                recs.append([rid, n] + [round(float(v), 3) for v in row]
                            + [','.join(l for l, v, t in zip(LABELS, row, th) if v >= t)])
        tmp = path + '.tmp'
        pd.DataFrame(recs, columns=cols).sort_values(['review_id', 'sent_num']).to_csv(
            tmp, sep='\t', index=False, compression='gzip')
        os.replace(tmp, path)                                   # a part is either complete or absent

    for rid, rows in reviews(a.sentences, a.shard, a.num_shards):
        buf.append((rid, rows)); done += 1
        if len(buf) == a.part_size:
            flush(buf, part); part += 1; buf = []
            print(f'{done} reviews, {done / (time.time() - t0):.1f}/s', flush=True)
        if a.limit and done >= a.limit:
            break
    if buf:
        flush(buf, part)
    print(f'shard {a.shard}/{a.num_shards}: {done} reviews in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()

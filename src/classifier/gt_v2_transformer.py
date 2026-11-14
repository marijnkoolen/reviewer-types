"""Fine-tune a Dutch transformer (default RobBERT v2) as a multi-label sentence classifier on the v2.4 ground truth.

Input = the labellers' context only:
    segment 1: "<header> | <target sentence>"     header from src/annotate/review_meta.py (genre, rating, year)
    segment 2: the surrounding review sentences, growing outward from the target until a character budget,
               with "[…]" at the target's place. No outside knowledge of the book.
18 sigmoid outputs, BCE loss; best epoch by dev micro-F1; per-label thresholds tuned on dev.

Run (needs torch + transformers; Homebrew python3.11 has them, venv/ does not):
    /opt/homebrew/opt/python@3.11/bin/python3.11 src/classifier/gt_v2_transformer.py
    ... --model pdelobelle/robbert-v2-dutch-base --epochs 4 --name robbert
Smoke test (cached model, few steps):  ... --model emanjavacas/GysBERT --max_steps 20 --name smoke
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from gt_v2_data import LABELS, OUT, load, to_matrix
from gt_v2_eval import evaluate_all, tune_thresholds


def build_inputs(d, ctx_chars):
    """d must have a 0..n-1 index and sentences in review order (as gt_v2_data.load() returns them)."""
    first, second = [None] * len(d), [None] * len(d)
    for _, grp in d.groupby('review_id', sort=False):
        sents = grp.sent_text.tolist()
        header = grp.header.iloc[0]
        for i, (row, s) in enumerate(zip(grp.index, sents)):
            left, right, budget = [], [], ctx_chars
            lo, hi = i - 1, i + 1
            while budget > 0 and (lo >= 0 or hi < len(sents)):
                if lo >= 0:
                    left.insert(0, sents[lo]); budget -= len(sents[lo]); lo -= 1
                if hi < len(sents) and budget > 0:
                    right.append(sents[hi]); budget -= len(sents[hi]); hi += 1
            first[row] = f'{header} | {s}'
            second[row] = ' '.join(left + ['[…]'] + right)
    return first, second


def batches(enc, ids, bs, shuffle, rng):
    """Length-bucketed batches (dynamic padding)."""
    ids = sorted(ids, key=lambda i: len(enc['input_ids'][i]))
    chunks = [ids[k:k + bs] for k in range(0, len(ids), bs)]
    if shuffle:
        rng.shuffle(chunks)
    return chunks


def collate(enc, chunk, pad_id, device):
    # Pad to a multiple of 32: MPS caches a compiled graph per tensor shape, and one shape per length
    # grew memory until the machine swapped >50 GB and crashed (UA-020).
    L = -(-max(len(enc['input_ids'][i]) for i in chunk) // 32) * 32
    ii = torch.full((len(chunk), L), pad_id, dtype=torch.long)
    am = torch.zeros((len(chunk), L), dtype=torch.long)
    for r, i in enumerate(chunk):
        x = enc['input_ids'][i]
        ii[r, :len(x)] = torch.tensor(x); am[r, :len(x)] = 1
    return ii.to(device), am.to(device)


@torch.no_grad()
def predict(model, enc, ids, bs, pad_id, device):
    model.eval()
    P = np.zeros((len(enc['input_ids']), len(LABELS)), dtype=np.float32)
    for chunk in batches(enc, ids, bs * 2, False, None):
        ii, am = collate(enc, chunk, pad_id, device)
        P[chunk] = torch.sigmoid(model(input_ids=ii, attention_mask=am).logits.float()).cpu().numpy()
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='pdelobelle/robbert-v2-dutch-base')
    ap.add_argument('--name', default='robbert')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--max_len', type=int, default=256)
    ap.add_argument('--ctx_chars', type=int, default=700)
    ap.add_argument('--max_steps', type=int, default=0, help='>0: stop early (smoke test)')
    ap.add_argument('--seed', type=int, default=13)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

    d = load().reset_index(drop=True)
    s1, s2 = build_inputs(d, a.ctx_chars)
    tok = AutoTokenizer.from_pretrained(a.model)
    enc = tok(s1, s2, truncation='only_second', max_length=a.max_len)
    Y = to_matrix(d.labels).astype(np.float32)
    tr = np.flatnonzero(d.split == 'train'); dev = np.flatnonzero(d.split == 'dev')
    if a.max_steps:
        dev = dev[:200]

    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=len(LABELS), problem_type='multi_label_classification').to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    n_steps = a.max_steps or a.epochs * int(np.ceil(len(tr) / a.bs))
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * n_steps), n_steps)
    lossf = torch.nn.BCEWithLogitsLoss()
    Yt = torch.tensor(Y)
    out = f'{OUT}/{a.name}'; os.makedirs(out, exist_ok=True)
    best, best_ep, step, log, t0 = -1, -1, 0, [], time.time()
    for ep in range(a.epochs):
        model.train()
        for chunk in batches(enc, tr, a.bs, True, rng):
            ii, am = collate(enc, chunk, tok.pad_token_id, device)
            loss = lossf(model(input_ids=ii, attention_mask=am).logits, Yt[chunk].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 100 == 0:
                print(f'ep {ep} step {step}/{n_steps} loss {loss.item():.4f} {time.time() - t0:.0f}s', flush=True)
            if a.max_steps and step >= a.max_steps:
                break
        Pd = predict(model, enc, dev, a.bs, tok.pad_token_id, device)[dev]
        f = f1_score(Y[dev], Pd >= 0.5, average='micro', zero_division=0)
        log.append({'epoch': ep, 'step': step, 'dev_micro_f1@0.5': float(f), 'secs': time.time() - t0})
        print(log[-1], flush=True)
        if f > best:
            best, best_ep = f, ep
            model.save_pretrained(f'{out}/model'); tok.save_pretrained(f'{out}/model')
        if a.max_steps and step >= a.max_steps:
            break

    model = AutoModelForSequenceClassification.from_pretrained(f'{out}/model').to(device)
    P = predict(model, enc, np.arange(len(d)), a.bs, tok.pad_token_id, device)
    th = tune_thresholds(Y[dev].astype(int), P[dev])
    summary, pl = evaluate_all(a.name, d, P, th)
    with open(f'{out}/train_log.json', 'w') as fh:
        json.dump({'args': vars(a), 'device': device, 'best_epoch': best_ep, 'log': log}, fh, indent=2)
    for k, s in summary.items():
        print(f"{k:24s} n={s['n_sentences']:5d} micro-F1={s['micro_f1']:.3f} macro-F1={s['macro_f1']:.3f}")


if __name__ == '__main__':
    main()

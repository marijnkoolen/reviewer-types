"""Fine-tune an open-weight LLM as a whole-review sentence classifier on the v2.4 ground truth (UA-025 a).

usage (one GPU per run; on the A10 VMs, 22 GB each):
  CUDA_VISIBLE_DEVICES=0 python src/classifier/gt_v2_llm.py --model Qwen/Qwen2.5-7B --name llm_qwen25_7b
  smoke test without CUDA:  python3.11 src/classifier/gt_v2_llm.py --model Qwen/Qwen2.5-0.5B --name smoke_llm \
                                --bits 16 --max_steps 20 --max_len 512 --eval_limit 40

Input: the same context the labellers had. One sequence per review: the review_meta header, all sentences numbered,
then every sentence again ('Sentences to label'). The hidden state at the end of each repeated sentence feeds an
18-label sigmoid head. A causal model's state at the first copy of a sentence has not seen what follows it; at the
repeated copy it has seen the whole review. A review longer than --max_len is cut into groups of consecutive
sentences, each with as much surrounding review as fits.
Model: the base model in 4-bit NF4 (bitsandbytes; --bits 16 loads it unquantised), LoRA adapters on all linear
layers, a float32 linear head. One sequence per step, gradients accumulated over --grad_accum sequences.
Data: gt_v2_data.load() (or data/gt_v2/train_input.tsv.gz on another machine). Best epoch by dev micro-F1@0.5;
then dev-tuned thresholds and gt_v2_eval.evaluate_all, as for the other classifiers.
output: results/gt_v2_classifier/<name>/ (summary.json, per_label.tsv, predictions.tsv.gz, train_log.json) and
        <name>/adapter/ (LoRA adapter, head.pt, tokenizer) for gt_v2_llm_predict.py.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gt_v2_data import LABELS, OUT, load, to_matrix  # noqa: E402
from gt_v2_eval import evaluate_all, tune_thresholds  # noqa: E402

PAD_MULTIPLE = 64   # pad sequences to a few lengths only (MPS caches a graph per shape; see UA-020)


class Encoder:
    """Turns one review (header + sentences) into token groups with the positions to read the labels from."""

    def __init__(self, tok, max_len):
        self.tok, self.max_len = tok, max_len

    def ids(self, text):
        return self.tok(text, add_special_tokens=False)['input_ids']

    def groups(self, header, sents):
        """-> list of (input_ids, [sentence index], [position]) covering every sentence once."""
        bos = [self.tok.bos_token_id] if self.tok.bos_token_id is not None else []
        head = bos + self.ids(f'{header}\nReview:')
        mid = self.ids('\nSentences to label:')
        cap = max(16, self.max_len // 4)                        # one very long sentence is cut to this length
        line = [self.ids(f'\n[{i + 1}] {s}')[:cap] for i, s in enumerate(sents)]
        fixed = len(head) + len(mid)
        if fixed + 2 * sum(map(len, line)) <= self.max_len:
            return [self._build(head, mid, line, range(len(sents)), range(len(sents)))]
        out, i, copy_budget = [], 0, (self.max_len - fixed) // 3
        while i < len(sents):                                   # consecutive sentences whose copies fit the budget
            j, used = i, 0
            while j < len(sents) and (j == i or used + len(line[j]) <= copy_budget):
                used += len(line[j]); j += 1
            lo, hi, ctx = i, j, used                            # context: the group, then widen both ways
            while True:
                grown = False
                for k in (lo - 1, hi):
                    if 0 <= k < len(sents) and not lo <= k < hi and fixed + ctx + len(line[k]) + used <= self.max_len:
                        ctx += len(line[k]); lo, hi = min(lo, k), max(hi, k + 1); grown = True
                if not grown:
                    break
            out.append(self._build(head, mid, line, range(lo, hi), range(i, j)))
            i = j
        return out

    @staticmethod
    def _build(head, mid, line, ctx_idx, lab_idx):
        ids = head + [t for k in ctx_idx for t in line[k]] + mid
        pos = []
        for k in lab_idx:
            ids = ids + line[k]
            pos.append(len(ids) - 1)
        return ids, list(lab_idx), pos


def build_examples(d, enc):
    """Per review: groups with the row numbers of d they label."""
    ex = []
    for rid, g in d.groupby('review_id', sort=False):
        rows = g.index.to_numpy()
        for ids, idx, pos in enc.groups(g.header.iloc[0], g.sent_text.tolist()):
            ex.append({'review_id': rid, 'split': g.split.iloc[0], 'ids': ids, 'rows': rows[idx], 'pos': pos})
    return ex


def load_model(a, device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kw = {}
    if a.bits in (4, 8):
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
        kw['quantization_config'] = (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                        bnb_4bit_compute_dtype=torch.bfloat16,
                                                        bnb_4bit_use_double_quant=True)
                                     if a.bits == 4 else BitsAndBytesConfig(load_in_8bit=True))
        kw['device_map'] = {'': 0}
    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    base = AutoModel.from_pretrained(a.model, dtype=dtype, **kw)
    if a.bits in (4, 8):
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    else:
        base = base.to(device)
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    base.config.use_cache = False
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, bias='none',
                      target_modules='all-linear')
    model = get_peft_model(base, lcfg)
    head = torch.nn.Linear(base.config.hidden_size, len(LABELS)).to(device=device, dtype=torch.float32)
    return tok, model, head


def forward(model, head, ex, device, pad_id):
    ids = ex['ids']
    L = -(-len(ids) // PAD_MULTIPLE) * PAD_MULTIPLE
    x = torch.full((1, L), pad_id, dtype=torch.long)
    m = torch.zeros((1, L), dtype=torch.long)
    x[0, :len(ids)] = torch.tensor(ids); m[0, :len(ids)] = 1
    x, m = x.to(device), m.to(device)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device == 'cuda'):
        h = model(input_ids=x, attention_mask=m).last_hidden_state[0, ex['pos']]
    return head(h.float())


@torch.no_grad()
def predict(model, head, examples, n_rows, device, pad_id):
    model.eval()
    P = np.full((n_rows, len(LABELS)), np.nan, dtype=np.float32)
    for ex in examples:
        P[ex['rows']] = torch.sigmoid(forward(model, head, ex, device, pad_id)).cpu().numpy()
    model.train()
    return P


def save(model, head, tok, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    torch.save(head.state_dict(), f'{path}/head.pt')
    tok.save_pretrained(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen2.5-7B')
    ap.add_argument('--name', default='llm')
    ap.add_argument('--bits', type=int, default=4, choices=[4, 8, 16])
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-4, help='LoRA learning rate')
    ap.add_argument('--head_lr', type=float, default=1e-3)
    ap.add_argument('--lora_r', type=int, default=16)
    ap.add_argument('--grad_accum', type=int, default=8)
    ap.add_argument('--max_len', type=int, default=2048)
    ap.add_argument('--max_steps', type=int, default=0, help='>0: stop after this many sequences (smoke test)')
    ap.add_argument('--eval_limit', type=int, default=0, help='>0: score only this many dev sequences per epoch')
    ap.add_argument('--seed', type=int, default=13)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if a.bits < 16 and device != 'cuda':
        sys.exit('--bits 4/8 needs CUDA (bitsandbytes); use --bits 16 elsewhere')

    d = load().reset_index(drop=True)
    Y = to_matrix(d.labels).astype(np.float32)
    tok, model, head = load_model(a, device)
    ex = build_examples(d, Encoder(tok, a.max_len))
    train = [e for e in ex if e['split'] == 'train']
    dev = [e for e in ex if e['split'] == 'dev']
    dev_eval = dev[:a.eval_limit] if a.eval_limit else dev
    print(f'{len(ex)} sequences ({len(train)} train, {len(dev)} dev); max length {max(len(e["ids"]) for e in ex)}; '
          f'device {device}', flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{'params': params, 'lr': a.lr}, {'params': head.parameters(), 'lr': a.head_lr}],
                            weight_decay=0.01)
    n_seq = a.max_steps or a.epochs * len(train)
    n_upd = math.ceil(n_seq / a.grad_accum)
    from transformers import get_linear_schedule_with_warmup
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * n_upd), n_upd)
    lossf = torch.nn.BCEWithLogitsLoss()
    out = f'{OUT}/{a.name}'
    best, best_ep, seen, log, t0 = -1.0, -1, 0, [], time.time()
    model.train()
    for ep in range(a.epochs):
        random.shuffle(train)
        run = 0.0
        for k, e in enumerate(train):
            loss = lossf(forward(model, head, e, device, tok.pad_token_id),
                         torch.tensor(Y[e['rows']], device=device)) / a.grad_accum
            loss.backward()
            run += loss.item() * a.grad_accum
            seen += 1
            if seen % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params + list(head.parameters()), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if seen % 200 == 0:
                print(f'ep {ep} seq {seen}/{n_seq} loss {run / (k + 1):.4f} {time.time() - t0:.0f}s', flush=True)
            if a.max_steps and seen >= a.max_steps:
                break
        P = predict(model, head, dev_eval, len(d), device, tok.pad_token_id)
        rows = np.concatenate([e['rows'] for e in dev_eval])
        f = f1_score(Y[rows], P[rows] >= 0.5, average='micro', zero_division=0)
        log.append({'epoch': ep, 'sequences': seen, 'dev_micro_f1@0.5': float(f), 'secs': time.time() - t0})
        print(log[-1], flush=True)
        if f > best:
            best, best_ep = f, ep
            save(model, head, tok, f'{out}/adapter')
        if a.max_steps and seen >= a.max_steps:
            break

    if best_ep != log[-1]['epoch']:                             # reload the best epoch's adapter and head
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(f'{out}/adapter/adapter_model.safetensors'))
        head.load_state_dict(torch.load(f'{out}/adapter/head.pt', map_location=device))
    with open(f'{out}/train_log.json', 'w') as fh:
        json.dump({'args': vars(a), 'device': device, 'best_epoch': best_ep, 'log': log}, fh, indent=2)
    if a.max_steps or a.eval_limit:                             # smoke test: no full evaluation
        print('smoke test done; dev micro-F1@0.5', round(best, 3))
        return
    P = predict(model, head, ex, len(d), device, tok.pad_token_id)
    dev_m = (d.split == 'dev').to_numpy()
    th = tune_thresholds(Y[dev_m].astype(int), P[dev_m])
    summary, _ = evaluate_all(a.name, d, P, th)
    for k, s in summary.items():
        print(f"{k:24s} n={s['n_sentences']:5d} micro-F1={s['micro_f1']:.3f} macro-F1={s['macro_f1']:.3f}")


if __name__ == '__main__':
    main()

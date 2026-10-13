"""Estimate how much silver a distilled spaCy lemmatizer actually needs.

The question this answers: distilling Alpino into spaCy is only worth doing if
the silver corpus N you must Alpino-parse is a small fraction of the target
corpus M. If N approaches M you have already parsed everything with Alpino and
the distilled model is redundant -- it has learned, at best, to copy a system
whose output you already hold.

Rather than guess, this trains the same configuration at several training-set
sizes and evaluates every model on the *same* held-out reviews, then decomposes
the result into the two effects that drive it:

  coverage       a test token whose surface form appeared in training is easy;
                 more data mainly buys more of these
  generalisation accuracy on forms never seen in training, which is what the
                 model actually learned about Dutch morphology

Accuracy at any N is then approximately

    acc(N) = C(N) * acc_seen + (1 - C(N)) * acc_unseen

where C(N) is the fraction of test tokens whose form was seen. C(N) is governed
by vocabulary growth (Heaps' law, V = K * N^beta), which is fitted on the corpus
itself and extrapolates far beyond the sizes we can afford to train. That makes
a projection to millions of tokens defensible in a way that extrapolating the
raw accuracy curve alone would not be.

If acc_unseen is flat across sizes, the model is not learning generalisable
morphology and the ceiling is set by coverage -- which is the case where more
data helps but only logarithmically, and where the honest answer may be "use
Alpino directly".

Usage:
  python src/parsers/learning_curve.py --silver alpino \
      --fractions 0.0625 0.125 0.25 0.5 1.0
Writes results/parser_comparison/learning_curve{,_heaps}.csv
"""
import argparse
import csv
import math
import random
import subprocess
import sys
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_fmt import OUT_DIR, iter_lines, normalise, read_tokens  # noqa: E402
from compare_util import corpus_form_frequency, discover_tables, freq_bucket, load_tables  # noqa: E402


def load_silver(tool):
    tables = discover_tables()
    if tool not in tables:
        raise SystemExit(f"no tokens_{tool}.tsv")
    silver = defaultdict(dict)
    for row in read_tokens(tables[tool]):
        if row["start"] >= 0 and row["lemma"]:
            silver[(row["doc_id"], row["line_id"])][(row["start"], row["end"])] = \
                row["lemma"]
    return silver


def split_docs(dev_frac, seed):
    """Reproduce build_lemma_trainset.py's split exactly."""
    docs = sorted({d for d, _, _ in iter_lines()})
    rng = random.Random(seed)
    rng.shuffle(docs)
    n_dev = max(1, int(len(docs) * dev_frac))
    return set(docs[:n_dev]), docs[n_dev:]


def build_docbin(tokenizer, silver, lines, keep_docs, out_path):
    """Write a DocBin of the given reviews; returns (tokens, labelled, forms)."""
    from spacy.tokens import DocBin
    db = DocBin(store_user_data=False)
    n_tok = n_lab = 0
    forms = set()
    for doc_id, line_id, text in lines:
        if doc_id not in keep_docs:
            continue
        gold = silver.get((doc_id, line_id))
        if not gold:
            continue
        doc = tokenizer(text)
        labelled = 0
        for tok in doc:
            lemma = gold.get((tok.idx, tok.idx + len(tok.text)))
            if lemma:
                tok.lemma_ = lemma
                labelled += 1
                forms.add(tok.text.lower())
            else:
                tok.lemma_ = ""
        if labelled:
            db.add(doc)
            n_tok += len(doc)
            n_lab += labelled
    db.to_disk(out_path)
    return n_tok, n_lab, forms


def evaluate(model_path, silver, lines, dev_docs, train_forms, freq):
    """Accuracy against silver on held-out reviews, split by seen/unseen form."""
    import spacy
    nlp = spacy.load(model_path)
    stats = defaultdict(lambda: [0, 0])
    for doc_id, line_id, text in lines:
        if doc_id not in dev_docs:
            continue
        gold = silver.get((doc_id, line_id))
        if not gold:
            continue
        doc = nlp(text)
        for tok in doc:
            lemma = gold.get((tok.idx, tok.idx + len(tok.text)))
            if not lemma or tok.is_punct:
                continue
            ok = tok.lemma_ == lemma
            seen = tok.text.lower() in train_forms
            for key in ("all", "seen" if seen else "unseen",
                        freq_bucket(freq.get(tok.text.lower(), 0))):
                stats[key][1] += 1
                stats[key][0] += ok
    return {k: (h, t) for k, (h, t) in stats.items()}


def fit_heaps(lines, all_docs, samples=12):
    """Fit V = K * N^beta on the corpus by sampling nested prefixes."""
    order = list(all_docs)
    per_doc = defaultdict(list)
    for doc_id, _, text in lines:
        per_doc[doc_id].extend(w.lower() for w in normalise(text).split())
    pts = []
    vocab = set()
    n = 0
    step = max(1, len(order) // samples)
    for i, doc_id in enumerate(order, 1):
        toks = per_doc.get(doc_id, [])
        n += len(toks)
        vocab.update(toks)
        if i % step == 0 and n > 0:
            pts.append((n, len(vocab)))
    xs = [math.log(n) for n, _ in pts]
    ys = [math.log(v) for _, v in pts]
    k = len(xs)
    mx, my = sum(xs) / k, sum(ys) / k
    beta = (sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            / max(1e-9, sum((x - mx) ** 2 for x in xs)))
    logK = my - beta * mx
    return math.exp(logK), beta, pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver", default="alpino")
    ap.add_argument("--base-model", default="nl_core_news_lg")
    ap.add_argument("--fractions", nargs="*", type=float,
                    default=[0.0625, 0.125, 0.25, 0.5, 1.0])
    ap.add_argument("--dev-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work-dir", default=str(OUT_DIR / "learning_curve"))
    ap.add_argument("--skip-training", action="store_true",
                    help="only fit and report the vocabulary model")
    args = ap.parse_args()

    warnings.filterwarnings("ignore", message=r".*\[W095\].*")
    import spacy

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    silver = load_silver(args.silver)
    lines = [(d, i, normalise(t)) for d, i, t in iter_lines()]
    dev, train_docs = split_docs(args.dev_frac, args.seed)
    freq = corpus_form_frequency(load_tables(tools=[args.silver]))

    K, beta, pts = fit_heaps(lines, train_docs)
    _write(OUT_DIR / "learning_curve_heaps.csv",
           [{"tokens": n, "types": v} for n, v in pts])
    print(f"\nHeaps' law on this corpus: V = {K:.2f} * N^{beta:.3f}")
    print("  (type count grows sublinearly, so coverage of a fixed test set "
          "rises\n   only logarithmically in training size)")

    if args.skip_training:
        return

    nlp = spacy.load(args.base_model, exclude=["ner", "parser"])
    # Dev set is built once and reused, so every size is scored identically.
    build_docbin(nlp.tokenizer, silver, lines, dev, work / "dev.spacy")
    cfg = OUT_DIR / f"lemma_train_{args.silver}" / "config.cfg"
    if not cfg.exists():
        raise SystemExit(f"missing {cfg}; run build_lemma_trainset.py first")

    rows = []
    for frac in sorted(args.fractions):
        n_docs = max(1, int(len(train_docs) * frac))
        subset = set(train_docs[:n_docs])  # nested: each size contains the last
        train_path = work / f"train_{frac}.spacy"
        n_tok, n_lab, forms = build_docbin(
            nlp.tokenizer, silver, lines, subset, train_path)
        out_model = work / f"model_{frac}"
        print(f"\n=== fraction {frac}: {n_docs} reviews, {n_lab} labelled tokens ===",
              flush=True)
        proc = subprocess.run(
            [sys.executable, "-m", "spacy", "train", str(cfg),
             "--paths.train", str(train_path),
             "--paths.dev", str(work / "dev.spacy"),
             "--output", str(out_model)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr[-1500:], file=sys.stderr)
            continue
        stats = evaluate(str(out_model / "model-best"), silver, lines, dev,
                         forms, freq)
        row = {"fraction": frac, "reviews": n_docs, "train_tokens": n_lab,
               "train_types": len(forms)}
        for key in ("all", "seen", "unseen", "1 (hapax)", "2-5", "100+"):
            hit, tot = stats.get(key, (0, 0))
            row[f"acc_{key}"] = round(100 * hit / tot, 2) if tot else None
            row[f"n_{key}"] = tot
        row["coverage"] = round(
            100 * stats.get("seen", (0, 0))[1]
            / max(1, stats.get("all", (0, 1))[1]), 2)
        rows.append(row)
        print(f"  held-out acc {row['acc_all']}%  "
              f"(seen {row['acc_seen']}% / unseen {row['acc_unseen']}%), "
              f"coverage {row['coverage']}%", flush=True)

    _write(OUT_DIR / "learning_curve.csv", rows)
    _project(rows, K, beta)


def _project(rows, K, beta):
    """Project accuracy to larger N from the coverage model."""
    if len(rows) < 2:
        return
    last = rows[-1]
    acc_seen = last["acc_seen"]
    acc_unseen = last["acc_unseen"]
    if acc_seen is None or acc_unseen is None:
        return
    n0, c0 = last["train_tokens"], last["coverage"] / 100.0
    print("\n=== projection ===")
    print(f"acc_seen={acc_seen}%  acc_unseen={acc_unseen}%  "
          f"(gap {acc_seen - acc_unseen:.1f} points)")
    print("Coverage is modelled as rising with the log of training size, with "
          "the\nfitted Heaps exponent setting the rate. Treat as an order-of-"
          "magnitude guide.")
    print(f"{'train tokens':>14}{'est. coverage':>15}{'est. accuracy':>15}")
    for mult in (1, 2, 5, 10, 25, 50, 100):
        n = n0 * mult
        # Unseen-token mass falls as the new-type rate does: N^(beta-1).
        cov = 1 - (1 - c0) * (n / n0) ** (beta - 1)
        acc = cov * acc_seen + (1 - cov) * acc_unseen
        print(f"{n:>14,}{100 * cov:>14.1f}%{acc:>14.2f}%")


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
